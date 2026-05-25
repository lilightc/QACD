"""Local CPU sanity tests for QACD ops + attention mask logic.

These cover the parts that don't need a GPU or the LLaVA package:
    - all 7 operations at all 3 intensities (shape/dtype/finiteness)
    - region masking confines changes to the masked area
    - attention -> heatmap -> mask pipeline
    - center fallback

Run:  python src/utils/test_qacd_local.py   (from repo root, src on PYTHONPATH)
"""
import torch

from utils.qacd_ops import apply_operation, OPERATION_SET
from utils.qacd_attention import (
    heatmap_from_attention,
    mask_from_heatmap,
    center_region_mask,
)
from utils.qacd_planner import build_planner_prompt, parse_recipe
from utils.qacd_debug import save_debug

# CLIP normalization stats (same as llava_model.py)
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
H = W = 336


def _rand_image():
    px = torch.rand(1, 3, H, W)
    return (px - MEAN) / STD  # normalized, like the image processor output


def test_ops_shapes():
    img = _rand_image()
    for op in OPERATION_SET:
        for lvl in (1, 2, 3):
            out = apply_operation(op, lvl, img, MEAN, STD)
            assert out.shape == img.shape, (op, lvl, out.shape)
            assert out.dtype == img.dtype
            assert torch.isfinite(out).all(), f'{op}/{lvl} produced non-finite'
    print('PASS test_ops_shapes (7 ops x 3 intensities)')


def test_region_confinement():
    img = _rand_image()
    mask = torch.zeros(1, 1, H, W)
    mask[:, :, :H // 2, :] = 1.0  # top half only
    out = apply_operation('invert', 3, img, MEAN, STD, region_mask=mask)
    bottom_unchanged = torch.allclose(out[:, :, H // 2:, :], img[:, :, H // 2:, :], atol=1e-5)
    top_changed = not torch.allclose(out[:, :, :H // 2, :], img[:, :, :H // 2, :], atol=1e-3)
    assert bottom_unchanged, 'masked-out region was modified'
    assert top_changed, 'masked-in region was not modified'
    print('PASS test_region_confinement')


def test_intensity_monotonic():
    # stronger desat intensity should move further from the original
    img = _rand_image()
    d = [
        (apply_operation('desat', lvl, img, MEAN, STD) - img).abs().mean().item()
        for lvl in (1, 2, 3)
    ]
    assert d[0] < d[1] < d[2], f'desat not monotonic in intensity: {d}'
    print(f'PASS test_intensity_monotonic (desat deltas={[round(x,4) for x in d]})')


def test_attention_pipeline():
    heads, q_len = 8, 60
    img_start, n_img = 5, 576  # 24x24
    k_len = img_start + n_img + 10
    attn = torch.rand(heads, q_len, k_len) * 0.01
    # plant a hot spot: target tokens attend strongly to a patch block
    target_idx = [40, 41, 42]
    hot = img_start + 12 * 24 + 12  # center-ish patch
    attn[:, target_idx, hot:hot + 5] = 1.0

    heat = heatmap_from_attention(attn, target_idx, img_start, n_img, (24, 24))
    assert heat.shape == (24, 24)
    mask, degenerate = mask_from_heatmap(heat, (H, W), lam=0.5)
    assert mask.shape == (1, 1, H, W)
    assert not degenerate
    assert mask.max() == 1.0 and mask.min() == 0.0
    frac = mask.mean().item()
    assert 0.0 < frac < 1.0, f'mask covers {frac:.2%} (expected partial)'
    print(f'PASS test_attention_pipeline (mask covers {frac:.1%} of image)')


def test_mask_denoising_keeps_multiple_regions():
    # Component filter logic (decoupled from smoothing): two genuine clusters
    # + a single-cell speck. min_region=2 must drop ONLY the 1-cell speck and
    # keep BOTH clusters (not collapse to one blob).
    from utils.qacd_attention import _connected_components, _filter_small_components
    gm = torch.zeros(24, 24)
    gm[4:7, 4:7] = 1.0                # cluster A (9 cells)
    gm[16:19, 16:19] = 1.0            # cluster B (9 cells)
    gm[1, 22] = 1.0                   # isolated 1-cell speck
    assert len(_connected_components(gm)) == 3

    filtered = _filter_small_components(gm, min_size=2)
    comps = _connected_components(filtered)
    assert len(comps) == 2, f'expected 2 regions kept, got {len(comps)}'
    assert filtered[1, 22] == 0.0, 'single-cell speck was not dropped'
    assert filtered[4:7, 4:7].sum() == 9 and filtered[16:19, 16:19].sum() == 9

    # full path still produces a sane partial mask
    heat = torch.zeros(24, 24)
    heat[4:7, 4:7] = 1.0
    heat[16:19, 16:19] = 1.0
    mask, deg = mask_from_heatmap(heat, (H, W), lam=0.5,
                                  smooth_sigma=0.8, min_region=2, dilate=1)
    assert not deg and 0.0 < mask.mean().item() < 1.0
    print('PASS test_mask_denoising_keeps_multiple_regions '
          '(2 clusters kept, 1-cell speck dropped)')


def test_sink_norm_removes_baseline():
    # one patch is a SINK (every query row attends to it) and one is the true
    # OBJECT (only TARGET tokens attend to it). sink_norm must suppress the sink
    # and keep the object.
    heads, q_len = 4, 50
    img_start, n_img = 5, 576
    k_len = img_start + n_img + 5
    attn = torch.zeros(heads, q_len, k_len)
    sink = img_start + 100      # attended by ALL rows
    obj = img_start + 300       # attended only by TARGET rows
    target_idx = [40, 41]
    attn[:, :, sink] = 0.5                 # universal sink
    attn[:, target_idx, obj] = 0.8         # query-specific object

    raw = heatmap_from_attention(attn, target_idx, img_start, n_img, (24, 24),
                                 sink_norm=False).reshape(-1)
    normed = heatmap_from_attention(attn, target_idx, img_start, n_img, (24, 24),
                                    sink_norm=True).reshape(-1)
    # raw: sink is strong; normed: sink suppressed, object dominates
    assert raw[100] > 0.4, 'raw heat should see the sink'
    assert normed[100] < 1e-4, f'sink not removed by baseline subtraction ({normed[100]})'
    assert normed[300] > normed[100], 'object should dominate after sink removal'
    print('PASS test_sink_norm_removes_baseline (sink suppressed, object kept)')


def test_center_fallback():
    mask = center_region_mask((H, W), frac=0.5)
    assert mask.shape == (1, 1, H, W)
    assert mask[:, :, H // 2, W // 2] == 1.0  # center is inside
    assert mask[:, :, 0, 0] == 0.0            # corner is outside
    print('PASS test_center_fallback')


def test_parse_recipe_clean():
    r = parse_recipe('TARGET: the red car\nOPERATION: invert\nINTENSITY: 2')
    assert r.parsed_ok and r.op == 'invert' and r.intensity == 2
    assert r.target == 'the red car'
    print('PASS test_parse_recipe_clean')


def test_parse_recipe_aliases_and_noise():
    # loose spelling + surrounding prose
    r = parse_recipe('Here is my plan.\nTARGET: bird on branch\n'
                     'OPERATION: Color Inversion.\nINTENSITY: 3\nDone.')
    assert r.parsed_ok and r.op == 'invert' and r.intensity == 3
    r2 = parse_recipe('TARGET: text\nOPERATION: pixelate\nINTENSITY: 1')
    assert r2.op == 'downsample'
    print('PASS test_parse_recipe_aliases_and_noise')


def test_parse_recipe_fallback():
    r = parse_recipe('I cannot help with that.')
    assert not r.parsed_ok and r.op == 'noise' and r.intensity == 2
    print('PASS test_parse_recipe_fallback')


def test_planner_prompt_variants():
    adv = build_planner_prompt('Is there a dog?', 'adversarial')
    neu = build_planner_prompt('Is there a dog?', 'neutral')
    assert 'adversary' in adv.lower() and 'Is there a dog?' in adv
    assert 'analyst' in neu.lower()
    assert 'TARGET:' in adv and 'OPERATION:' in adv and 'INTENSITY:' in adv
    print('PASS test_planner_prompt_variants')


def test_planner_fewshot_toggle():
    from utils.qacd_planner import parse_recipe as _pr
    with_icl = build_planner_prompt('Is there a dog?', 'adversarial', icl=True)
    no_icl = build_planner_prompt('Is there a dog?', 'adversarial', icl=False)
    assert '## Examples ##' in with_icl and '## Examples ##' not in no_icl
    assert len(with_icl) > len(no_icl)
    # every exemplar must itself parse cleanly to a valid recipe.
    # match per-example (groups can't span newlines), so the format-spec
    # line `INTENSITY: <1, 2...>` is excluded (no bare digit after the colon).
    import re
    blocks = re.findall(
        r'TARGET: ([^\n]+)\nOPERATION: ([^\n]+)\nINTENSITY: ([123])', with_icl
    )
    assert len(blocks) >= 7, f'expected >=7 exemplars, found {len(blocks)}'
    for tgt, op, inten in blocks:
        r = _pr(f'TARGET: {tgt}\nOPERATION: {op}\nINTENSITY: {inten}')
        assert r.parsed_ok, f'exemplar did not parse: {op}'
    print(f'PASS test_planner_fewshot_toggle ({len(blocks)} exemplars all parse)')


def test_save_debug():
    import json
    import os
    import tempfile
    img = _rand_image()
    mask = center_region_mask((H, W), frac=0.5)
    corrupted = apply_operation('invert', 2, img, MEAN, STD, region_mask=mask)
    meta = {
        'op': 'invert', 'intensity': 2, 'target': 'the red car',
        'parsed_ok': True, 'parse_fallback': False,
        'requested_region': 'attention', 'used_region': 'attention',
        'region_fallback': False, 'mask_coverage': 0.25,
    }
    with tempfile.TemporaryDirectory() as d:
        rec = save_debug(d, 7, 'What color is the car?', img, corrupted, mask,
                         MEAN, STD, meta)
        png = os.path.join(d, '7.png')
        assert os.path.exists(png), 'composite png not written'
        # composite is original | gap | overlay | gap | corrupted => width ~ 3W
        from PIL import Image as _Im
        w, h = _Im.open(png).size
        assert h == H and w > 3 * W - 50, (w, h)
        recs = [json.loads(x) for x in open(os.path.join(d, 'recipes.jsonl'))]
        assert recs[0]['op'] == 'invert' and recs[0]['question_id'] == 7
        assert recs[0]['region_fallback'] is False
    print(f"PASS test_save_debug (coverage={rec['mask_coverage']})")


if __name__ == '__main__':
    torch.manual_seed(0)
    test_ops_shapes()
    test_region_confinement()
    test_intensity_monotonic()
    test_attention_pipeline()
    test_mask_denoising_keeps_multiple_regions()
    test_sink_norm_removes_baseline()
    test_center_fallback()
    test_parse_recipe_clean()
    test_parse_recipe_aliases_and_noise()
    test_parse_recipe_fallback()
    test_planner_prompt_variants()
    test_planner_fewshot_toggle()
    test_save_debug()
    print('\nAll local QACD tests passed.')
