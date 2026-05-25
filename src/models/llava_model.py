import json
import os
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import GenerationConfig

from models.base_models import ModelWrapper
from models.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)
from models.llava.conversation import conv_templates, SeparatorStyle
from models.llava.model.builder import load_pretrained_model
from models.llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria
)
from utils.utils import timer
from utils.qacd_planner import build_planner_prompt, parse_recipe
from utils.qacd_ops import apply_operation
from utils.qacd_attention import (
    heatmap_from_attention,
    mask_from_heatmap,
    center_region_mask,
)

# LLaVA-1.5 uses CLIP ViT-L/14-336 -> 24x24 = 576 image patch tokens.
QACD_GRID = (24, 24)
QACD_N_IMAGE_TOKENS = QACD_GRID[0] * QACD_GRID[1]


AUG_LIST = (
    'random_crop',
    'color_inversion',
    'horizontal_flip',
    'vertical_flip',
    'random_mask',
    'noise'
)


class LlavaModel(ModelWrapper):
    def __init__(self, model_path, cd_config):
        super().__init__(cd_config, "llava")
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name
        )
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor

        self.sampling_config = GenerationConfig.from_dict(
            {
                **self.base_config.to_dict(),
                'do_sample': True,
            }
        )

        self.image_mean = torch.tensor(
            [(0.48145466, 0.4578275, 0.40821073)],
            device=self.model.device
        ).view(1, 3, 1, 1)
        self.image_std = torch.tensor(
            [(0.26862954, 0.26130258, 0.27577711)],
            device=self.model.device
        ).view(1, 3, 1, 1)

    @timer
    def generate_sentence(
        self,
        query: str,
        append_txt: str = None,
        image_path: str = None,
        mode: str = None,
        sas: bool = False,
        oracle: tuple = None,
        qid=None,
    ) -> dict:
        original_query = query
        if append_txt is not None:
            query = query + append_txt

        if image_path is None:                               # text-only inference
            images, image_sizes = None, None
        else:
            if self.model.config.mm_use_im_start_end:
                query = DEFAULT_IM_START_TOKEN + \
                    DEFAULT_IMAGE_TOKEN + \
                    DEFAULT_IM_END_TOKEN + '\n' + query
            else:
                query = DEFAULT_IMAGE_TOKEN + '\n' + query
            images = self.image_processor.preprocess(
                Image.open(image_path),
                return_tensors='pt'
            )['pixel_values'].to('cuda', dtype=torch.float16)

        conv = conv_templates['llava_v1'].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to('cuda')

        applied_aug, reason, qacd_meta = None, None, None
        if mode == 'vcd':
            image_tensor_cd, applied_aug = self.apply_augmentation(
                aug='noise', tensor=images
            )
        elif mode == 'selfaug':
            out = self.get_self_aug(original_query)
            reason, aug = out['reason'], out['aug']
            image_tensor_cd, applied_aug = self.apply_augmentation(
                aug=aug, tensor=images
            )
        elif mode in AUG_LIST:
            image_tensor_cd, applied_aug = self.apply_augmentation(
                aug=mode, tensor=images
            )
        elif mode == 'vacode':
            with torch.inference_mode():
                logits = F.softmax(self.model.generate(
                    input_ids,
                    images=images,
                    output_scores=True,
                    generation_config=self.sampling_config,
                ).scores[0])
                image_tensor_cd, max_dist = None, -1
                for aug in AUG_LIST:
                    tensor_cd, selected_aug = self.apply_augmentation(
                        aug=aug, tensor=images
                    )
                    score = F.softmax(self.model.forward(
                        input_ids,
                        images=tensor_cd,
                    ).get('logits')[:,-1,:])
                    l2_norm = torch.linalg.vector_norm(logits - score, ord=2)
                    if l2_norm > max_dist:
                        max_dist = l2_norm
                        image_tensor_cd = tensor_cd
                        applied_aug = selected_aug
        elif mode == 'qacd':
            if images is None:
                image_tensor_cd = None
            else:
                image_tensor_cd, recipe, qacd_meta = self._qacd_build_cd_image(
                    original_query, images, qid=qid
                )
                applied_aug = f'{recipe.op}:{recipe.intensity}'
                reason = recipe.target if recipe.parsed_ok else \
                    f'[fallback] target={recipe.target}'
        else:
            image_tensor_cd = None

        with torch.inference_mode():
            if not sas:
                # hand CD params to vcd_sample.sample via attributes (see base_models)
                self.model.cd_alpha = self.cd_alpha
                self.model.cd_beta = self.cd_beta
                self.model.cd_tau = self.cd_tau
                output_dict = self.model.generate(
                    input_ids,
                    images=images,
                    images_cd=image_tensor_cd,
                    generation_config=self.sampling_config,
                    **self.cd_param
                )
            else:                                             # greedy decoding
                output_dict = self.model.generate(
                    input_ids,
                    generation_config=self.greedy_config,
                )
        input_token_len = input_ids.shape[1]
        output_token_len = output_dict.sequences.shape[1]
        n_diff_input_output = (input_ids != output_dict.sequences[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_dict.sequences[:, input_token_len:], skip_special_tokens=True)[0]
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        return {
            'applied_aug': applied_aug,
            'reason': reason,
            'text': outputs.strip(),
            'threshold': output_dict.get('threshold', None),
            'qacd': qacd_meta,
        }

    # ------------------------------------------------------------------ QACD
    def _build_image_prompt_ids(self, text: str) -> torch.Tensor:
        """Build LLaVA input_ids (with the image placeholder) for `text`."""
        if self.model.config.mm_use_im_start_end:
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                DEFAULT_IM_END_TOKEN + '\n' + text
        else:
            text = DEFAULT_IMAGE_TOKEN + '\n' + text
        conv = conv_templates['llava_v1'].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to('cuda')

    @torch.inference_mode()
    def _qacd_build_cd_image(self, query: str, images: torch.Tensor, qid=None):
        """QACD Stages 1-3: plan a recipe, ground a region, corrupt the image.

        Returns (corrupted_image_tensor, recipe, meta). `meta` records what
        actually happened (op/intensity applied, parse + region fallbacks,
        mask coverage) for the fallback-rate metric.
        """
        cfg = self.cd_config
        h, w = images.shape[-2], images.shape[-1]
        device = images.device

        # Stage 1: image-conditioned planner -> recipe text.
        prompt = build_planner_prompt(
            query,
            getattr(cfg, 'qacd_prompt', 'adversarial'),
            icl=getattr(cfg, 'qacd_icl', True),
        )
        planner_ids = self._build_image_prompt_ids(prompt)
        planner_cfg = GenerationConfig.from_dict(
            {**self.greedy_config.to_dict(), 'max_new_tokens': 64}
        )
        gen = self.model.generate(
            planner_ids, images=images, generation_config=planner_cfg
        )
        comp_ids = gen.sequences[0, planner_ids.shape[1]:]
        recipe = parse_recipe(
            self.tokenizer.decode(comp_ids, skip_special_tokens=True)
        )

        # Ablation overrides.
        op = recipe.op
        allowed = tuple(getattr(cfg, 'qacd_ops', ()) or ())
        if allowed and op not in allowed:
            op = 'noise'
        intensity = getattr(cfg, 'qacd_intensity', 0) or recipe.intensity

        # Stage 2: region mask. Track which region method actually got used and
        # whether attention grounding fell back. A failed grounding degrades to
        # full-image corruption (mask=None), i.e. standard VCD/SA-VCD behavior.
        # `center` remains available only as an explicit ablation choice.
        region = getattr(cfg, 'qacd_region', 'attention')
        region_fallback = False
        if region == 'full':
            mask, used_region = None, 'full'
        elif region == 'center':
            mask = center_region_mask((h, w), cfg.qacd_center_frac, device)
            used_region = 'center'
        elif recipe.target is None:                # wanted attention, no target
            mask = None
            used_region, region_fallback = 'full', True
        else:
            mask, region_fallback = self._qacd_attention_mask(
                planner_ids, comp_ids, images, (h, w)
            )
            used_region = 'full' if region_fallback else 'attention'

        # Stage 3: confined pixel-space corruption.
        image_cd = apply_operation(
            op, intensity, images, self.image_mean, self.image_std, mask
        )
        recipe.op, recipe.intensity = op, intensity

        coverage = float((mask[0, 0] > 0.5).float().mean()) if mask is not None else 1.0
        meta = {
            'op': op,
            'intensity': intensity,
            'target': recipe.target,
            'parsed_ok': recipe.parsed_ok,
            'parse_fallback': not recipe.parsed_ok,
            'requested_region': region,
            'used_region': used_region,
            'region_fallback': region_fallback,
            'mask_coverage': round(coverage, 4),
        }

        debug_dir = getattr(cfg, 'qacd_debug_dir', '') or ''
        if debug_dir and qid is not None:
            try:
                from utils.qacd_debug import save_debug
                save_debug(
                    debug_dir, qid, query, images, image_cd, mask,
                    self.image_mean, self.image_std, meta,
                )
            except Exception as e:  # noqa: BLE001 - debugging must never break a run
                print(f'[QACD] debug save failed ({e})')

        return image_cd, recipe, meta

    @torch.inference_mode()
    def _qacd_attention_mask(self, planner_ids, comp_ids, images, image_hw):
        """Build a region mask from the planner's mid-layer cross-attention.

        Returns (mask, fell_back). On a degenerate mask or any exception the
        mask is None (full-image corruption, like VCD/SA-VCD) and fell_back is
        True, so the pipeline always produces a usable corruption while
        recording that grounding failed.
        """
        cfg = self.cd_config
        device = images.device
        try:
            full = torch.cat([planner_ids[0], comp_ids]).unsqueeze(0)
            out = self.model(
                full, images=images, use_cache=False,
                output_attentions=True, return_dict=True,
            )
            layer = max(0, min(cfg.qacd_layer, len(out.attentions) - 1))
            attn = out.attentions[layer][0]  # [heads, q_len, k_len]

            img_tok = (planner_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            img_pos = int(img_tok[0])
            # the single image placeholder expands to QACD_N_IMAGE_TOKENS tokens,
            # shifting every later position by (N - 1).
            comp_start = planner_ids.shape[1] + (QACD_N_IMAGE_TOKENS - 1)
            comp_positions = list(range(comp_start, comp_start + comp_ids.shape[0]))
            tgt = self._qacd_target_positions(comp_ids, comp_start) or comp_positions

            heat = heatmap_from_attention(
                attn, tgt, img_pos, QACD_N_IMAGE_TOKENS, QACD_GRID,
                sink_norm=getattr(cfg, 'qacd_sink_norm', True),
            )
            mask, degenerate = mask_from_heatmap(
                heat, image_hw, cfg.qacd_lam,
                thresh_mode=getattr(cfg, 'qacd_thresh_mode', 'std'),
                grow_ratio=getattr(cfg, 'qacd_grow_ratio', 0.5),
                smooth_sigma=getattr(cfg, 'qacd_smooth_sigma', 0.8),
                min_region=getattr(cfg, 'qacd_min_region', 2),
                dilate=getattr(cfg, 'qacd_dilate', 1),
            )
            if degenerate:
                return None, True
            return mask.to(device), False
        except Exception as e:  # noqa: BLE001 - defensive: never break decoding
            print(f'[QACD] attention localization failed ({e}); using full image')
            return None, True

    def _qacd_target_positions(self, comp_ids, comp_start):
        """Locate the generated TARGET-line tokens in expanded-sequence coords.

        Returns None if the TARGET line can't be isolated (caller then averages
        attention over the whole completion instead).
        """
        try:
            toks = [self.tokenizer.decode([int(t)]) for t in comp_ids]
            joined, spans = '', []
            for s in toks:
                spans.append((len(joined), len(joined) + len(s)))
                joined += s
            ti = joined.lower().find('target:')
            if ti < 0:
                return None
            start = ti + len('target:')
            nl = joined.find('\n', start)
            nl = nl if nl >= 0 else len(joined)
            pos = [comp_start + i for i, (a, b) in enumerate(spans)
                   if b > start and a < nl]
            return pos or None
        except Exception:
            return None
