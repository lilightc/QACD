"""QACD attention-derived region localization (Stage 2).

Turns the planner's mid-layer cross-attention (from the generated TARGET
tokens to the image patch tokens) into a binary region mask over the image.

Pipeline:
    raw attention [heads, q_len, k_len]
      -> select TARGET query rows, image-token key columns
      -> average over heads and TARGET tokens            -> [n_patches]
      -> reshape to patch grid                            -> [gh, gw]
      -> threshold at mean + lambda * std                 -> binary grid
      -> upsample to image resolution                     -> [1, 1, H, W]

The threshold is intentionally adaptive (mean + lambda*std) so the selected
region scales with how peaked the attention is for a given image/query.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def heatmap_from_attention(
    attn: torch.Tensor,
    target_token_idx: torch.Tensor | list[int],
    image_token_start: int,
    n_image_tokens: int,
    grid_hw: tuple[int, int],
) -> torch.Tensor:
    """Reduce a layer's attention to a [gh, gw] heatmap over image patches.

    Args:
        attn: attention weights for one layer, shape [heads, q_len, k_len]
            (batch dim already squeezed).
        target_token_idx: query positions of the generated TARGET tokens.
        image_token_start: column index where the 576 image tokens begin.
        n_image_tokens: number of image patch tokens (e.g., 576).
        grid_hw: (gh, gw) patch grid, e.g., (24, 24).

    Returns:
        Heatmap of shape grid_hw, non-negative, on attn's device.
    """
    if isinstance(target_token_idx, list):
        target_token_idx = torch.tensor(target_token_idx, device=attn.device)

    img_cols = slice(image_token_start, image_token_start + n_image_tokens)
    # [heads, n_target, n_image]
    sub = attn[:, target_token_idx, img_cols]
    # average over heads and target tokens -> [n_image]
    heat = sub.mean(dim=(0, 1)).float()

    gh, gw = grid_hw
    if heat.numel() != gh * gw:
        raise ValueError(
            f'attention has {heat.numel()} image tokens but grid is {gh}x{gw}'
        )
    return heat.reshape(gh, gw)


def mask_from_heatmap(
    heatmap: torch.Tensor,
    image_hw: tuple[int, int],
    lam: float = 0.5,
) -> tuple[torch.Tensor, bool]:
    """Threshold a patch heatmap and upsample to a full-resolution mask.

    Threshold = mean(heatmap) + lam * std(heatmap).

    Args:
        heatmap: [gh, gw] non-negative attention heatmap.
        image_hw: (H, W) target image resolution.
        lam: std multiplier for the adaptive threshold.

    Returns:
        (mask, degenerate) where mask is [1, 1, H, W] in {0, 1} and
        `degenerate` is True if thresholding selected no patches (caller
        should fall back to a center region).
    """
    thresh = heatmap.mean() + lam * heatmap.std()
    grid_mask = (heatmap > thresh).float()

    degenerate = bool(grid_mask.sum() == 0)
    if degenerate:
        # keep the single most-attended patch so we never return all-zeros
        idx = torch.argmax(heatmap)
        grid_mask = torch.zeros_like(heatmap).reshape(-1)
        grid_mask[idx] = 1.0
        grid_mask = grid_mask.reshape(heatmap.shape)

    mask = F.interpolate(
        grid_mask[None, None],
        size=image_hw,
        mode='nearest',
    )
    return mask, degenerate


def center_region_mask(
    image_hw: tuple[int, int],
    frac: float = 0.5,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Centered square mask covering `frac` of the shorter side.

    Used as the parse-failure / degenerate-attention fallback (report Sec. 3.3)
    and as the `--qacd-region center` ablation baseline.
    """
    h, w = image_hw
    side = int(min(h, w) * frac)
    top = (h - side) // 2
    left = (w - side) // 2
    mask = torch.zeros(1, 1, h, w, device=device, dtype=dtype)
    mask[:, :, top:top + side, left:left + side] = 1.0
    return mask
