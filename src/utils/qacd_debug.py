"""QACD intermediate-result logging for verification / qualitative analysis.

For each question (when a debug dir is configured), saves:
    {dir}/{qid}.png       -- [ original | attended-region overlay | corrupted ]
    {dir}/recipes.jsonl   -- planner target/op/intensity, region, mask coverage

Lets you eyeball whether the planner picks sensible ops and whether the
attention region lands on the queried object, without running a full POPE split.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image


def _to_uint8_hwc(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    """Denormalize a [1,3,H,W] CLIP-normalized tensor to an HWC uint8 RGB array."""
    mean = mean.float().cpu()
    std = std.float().cpu()
    px = torch.clamp(t.float().cpu() * std + mean, 0.0, 1.0)
    return (px[0].permute(1, 2, 0).numpy() * 255).astype('uint8')


def _overlay(arr: np.ndarray, mask: torch.Tensor | None) -> np.ndarray:
    """Tint the masked region red on a copy of `arr`."""
    if mask is None:  # region == 'full'
        return arr
    m = mask[0, 0].detach().cpu().numpy() > 0.5
    out = arr.astype('float32')
    red = np.array([255.0, 0.0, 0.0], dtype='float32')
    out[m] = out[m] * 0.55 + red * 0.45
    return out.astype('uint8')


def save_debug(
    out_dir: str,
    qid,
    question: str,
    image: torch.Tensor,
    corrupted: torch.Tensor,
    mask: torch.Tensor | None,
    mean: torch.Tensor,
    std: torch.Tensor,
    meta: dict,
) -> dict:
    """Save the composite image + append a recipe record built from `meta`.

    `meta` is the dict produced by LlavaModel._qacd_build_cd_image (op,
    intensity, target, parse/region fallback flags, mask coverage).
    """
    os.makedirs(out_dir, exist_ok=True)

    orig = _to_uint8_hwc(image, mean, std)
    corr = _to_uint8_hwc(corrupted, mean, std)
    ovl = _overlay(orig, mask)

    h = orig.shape[0]
    gap = np.full((h, 8, 3), 255, dtype='uint8')
    composite = np.concatenate([orig, gap, ovl, gap, corr], axis=1)
    Image.fromarray(composite).save(os.path.join(out_dir, f'{qid}.png'))

    record = {'question_id': qid, 'question': question, **meta}
    with open(os.path.join(out_dir, 'recipes.jsonl'), 'a') as f:
        f.write(json.dumps(record) + '\n')
    return record
