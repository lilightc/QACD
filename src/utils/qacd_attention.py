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
    sink_norm: bool = True,
) -> torch.Tensor:
    """Reduce a layer's attention to a [gh, gw] heatmap over image patches.

    With `sink_norm`, subtract a query-agnostic baseline (the average attention
    each patch receives from ALL query positions) from the TARGET-token
    attention, keeping only the query-specific excess:

        heat(patch) = relu( attn_from_TARGET(patch) - attn_from_all_rows(patch) )

    This cancels attention sinks / positional bias (patches that everything
    attends to) and isolates what the queried concept specifically looks at
    (cf. IAVA, Li et al. 2025).

    Args:
        attn: attention weights for one layer, shape [heads, q_len, k_len]
            (batch dim already squeezed).
        target_token_idx: query positions of the generated TARGET tokens.
        image_token_start: column index where the 576 image tokens begin.
        n_image_tokens: number of image patch tokens (e.g., 576).
        grid_hw: (gh, gw) patch grid, e.g., (24, 24).
        sink_norm: subtract the query-agnostic baseline.

    Returns:
        Heatmap of shape grid_hw, non-negative, on attn's device.
    """
    if isinstance(target_token_idx, list):
        target_token_idx = torch.tensor(target_token_idx, device=attn.device)

    img_cols = slice(image_token_start, image_token_start + n_image_tokens)
    # target signal: average over heads and TARGET tokens -> [n_image]
    heat = attn[:, target_token_idx, img_cols].mean(dim=(0, 1)).float()

    if sink_norm:
        # baseline: average attention each patch receives from every query row
        baseline = attn[:, :, img_cols].mean(dim=(0, 1)).float()
        heat = torch.clamp(heat - baseline, min=0.0)

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
    smooth_sigma: float = 0.8,
    min_region: int = 2,
    dilate: int = 1,
) -> tuple[torch.Tensor, bool]:
    """Threshold a patch heatmap into a clean region mask and upsample.

    Raw LVLM attention is noisy (scattered hot patches + attention sinks), so
    the heatmap is denoised before/after thresholding:
      1. Gaussian smoothing on the patch grid    -> merges signal, kills specks
      2. threshold at mean + lam * std
      3. drop connected components smaller than `min_region` cells -> removes
         scattered noise while KEEPING multiple genuine regions (counting,
         relations, multi-object / open-ended queries, not just one object)
      4. dilate                                    -> ensure full object coverage

    Args:
        heatmap: [gh, gw] non-negative attention heatmap.
        image_hw: (H, W) target image resolution.
        lam: std multiplier for the adaptive threshold.
        smooth_sigma: Gaussian sigma on the grid (0 disables smoothing).
        min_region: drop connected components smaller than this many grid cells
            (1 or 0 keeps everything; does NOT force a single blob).
        dilate: grid-cell dilation iterations (0 disables).

    Returns:
        (mask, degenerate) where mask is [1, 1, H, W] in {0, 1} and
        `degenerate` is True if thresholding selected no patches.
    """
    h = heatmap.float()
    if smooth_sigma and smooth_sigma > 0:
        h = _gaussian_blur_grid(h, smooth_sigma)

    thresh = h.mean() + lam * h.std()
    grid_mask = (h > thresh).float()

    degenerate = bool(grid_mask.sum() == 0)
    if degenerate:
        # keep the single most-attended patch so we never return all-zeros
        idx = torch.argmax(h)
        grid_mask = torch.zeros_like(h).reshape(-1)
        grid_mask[idx] = 1.0
        grid_mask = grid_mask.reshape(h.shape)
    else:
        if min_region and min_region > 1:
            grid_mask = _filter_small_components(grid_mask, min_region)
        if dilate and dilate > 0:
            grid_mask = _dilate_grid(grid_mask, dilate)

    mask = F.interpolate(
        grid_mask[None, None],
        size=image_hw,
        mode='nearest',
    )
    return mask, degenerate


def _gaussian_blur_grid(grid: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on a [gh, gw] grid (reflect padding)."""
    radius = max(1, int(round(2 * sigma)))
    coords = torch.arange(2 * radius + 1, dtype=torch.float32, device=grid.device) - radius
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :])[None, None]
    x = F.pad(grid[None, None], (radius, radius, radius, radius), mode='reflect')
    return F.conv2d(x, kernel)[0, 0]


def _dilate_grid(grid: torch.Tensor, iters: int) -> torch.Tensor:
    """Binary dilation via 3x3 max-pool, `iters` times."""
    x = grid[None, None]
    for _ in range(iters):
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x[0, 0]


def _connected_components(grid_mask: torch.Tensor) -> list[list[tuple[int, int]]]:
    """All 4-connected components of a binary grid, as lists of (i, j) cells."""
    m = grid_mask.detach().cpu().numpy() > 0.5
    gh, gw = m.shape
    visited = [[False] * gw for _ in range(gh)]
    comps: list[list[tuple[int, int]]] = []
    for i in range(gh):
        for j in range(gw):
            if not m[i][j] or visited[i][j]:
                continue
            stack, comp = [(i, j)], []
            visited[i][j] = True
            while stack:
                a, b = stack.pop()
                comp.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 0 <= na < gh and 0 <= nb < gw and m[na][nb] and not visited[na][nb]:
                        visited[na][nb] = True
                        stack.append((na, nb))
            comps.append(comp)
    return comps


def _filter_small_components(grid_mask: torch.Tensor, min_size: int) -> torch.Tensor:
    """Drop components smaller than `min_size` cells, keeping all others.

    Keeps a varying number of regions (good for multi-object / counting /
    relational queries). If every component is below the threshold, keep the
    single largest so we never return an empty mask.
    """
    comps = _connected_components(grid_mask)
    kept = [c for c in comps if len(c) >= min_size]
    if not kept and comps:
        kept = [max(comps, key=len)]
    out = torch.zeros_like(grid_mask)
    for comp in kept:
        for a, b in comp:
            out[a, b] = 1.0
    return out


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
