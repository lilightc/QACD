"""QACD region-aware corruption operations (Stage 3 execution).

Each operation maps an op name + intensity level (1/2/3) to a pixel-space
transform, applied only inside an attention-derived region mask:

    out = mask * corrupted + (1 - mask) * original

All ops operate on CLIP-normalized image tensors of shape [1, 3, H, W]
(the same tensors LLaVA's image processor produces). They denormalize to
[0, 1] pixel space, transform, then renormalize, so the result is a drop-in
replacement for the `images_cd` tensor consumed by the VCD decoding loop.

Design principle (report Sec. 3.2): every op here is either a content-neutral
degrader or an evidence flipper that cannot reproduce a plausible ground truth.
Zero-masking / mean-fill are deliberately excluded.
"""
from __future__ import annotations

import torch
import torchvision.transforms.functional as F


# Operation registry. "type" mirrors Table 1 in the report.
DEGRADERS = ('blur', 'downsample', 'noise', 'obscure', 'r-noise', 'desat')
FLIPPERS = ('invert',)
OPERATION_SET = DEGRADERS + FLIPPERS

# Per-operation intensity schedules, indexed by level 1/2/3.
_INTENSITY = {
    'blur':       {'sigma': {1: 1.5, 2: 3.0, 3: 5.0}},
    'downsample': {'factor': {1: 2, 2: 4, 3: 8}},
    'noise':      {'std': {1: 0.05, 2: 0.12, 3: 0.25}},
    'obscure':    {'sigma': {1: 1.5, 2: 3.0, 3: 5.0},
                   'darken': {1: 0.6, 2: 0.4, 3: 0.2}},
    'r-noise':    {'std': {1: 0.3, 2: 0.6, 3: 1.0}},
    'desat':      {'amount': {1: 0.5, 2: 0.8, 3: 1.0}},
    'invert':     {'amount': {1: 0.6, 2: 0.8, 3: 1.0}},
}


def _kernel_for_sigma(sigma: float) -> int:
    """Odd kernel size covering ~2 sigma on each side."""
    k = int(2 * round(2 * sigma) + 1)
    return max(3, k)


def _denorm(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.clamp(t * std + mean, 0.0, 1.0)


def _renorm(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (t - mean) / std


def _op_blur(px: torch.Tensor, lvl: int) -> torch.Tensor:
    sigma = _INTENSITY['blur']['sigma'][lvl]
    return F.gaussian_blur(px, kernel_size=_kernel_for_sigma(sigma), sigma=sigma)


def _op_downsample(px: torch.Tensor, lvl: int) -> torch.Tensor:
    factor = _INTENSITY['downsample']['factor'][lvl]
    _, _, h, w = px.shape
    small = F.resize(px, [max(1, h // factor), max(1, w // factor)], antialias=True)
    return F.resize(small, [h, w], antialias=False)  # blocky upscale = pixelation


def _op_noise(px: torch.Tensor, lvl: int) -> torch.Tensor:
    std = _INTENSITY['noise']['std'][lvl]
    return torch.clamp(px + torch.randn_like(px) * std, 0.0, 1.0)


def _op_obscure(px: torch.Tensor, lvl: int) -> torch.Tensor:
    sigma = _INTENSITY['obscure']['sigma'][lvl]
    darken = _INTENSITY['obscure']['darken'][lvl]
    blurred = F.gaussian_blur(px, kernel_size=_kernel_for_sigma(sigma), sigma=sigma)
    return torch.clamp(blurred * darken, 0.0, 1.0)


def _op_rnoise(px: torch.Tensor, lvl: int) -> torch.Tensor:
    """Replace content with strong noise (region becomes noise), vs. `noise`
    which only adds mild Gaussian noise on top of visible content."""
    std = _INTENSITY['r-noise']['std'][lvl]
    noise = torch.clamp(0.5 + torch.randn_like(px) * std, 0.0, 1.0)
    return noise


def _op_desat(px: torch.Tensor, lvl: int) -> torch.Tensor:
    amount = _INTENSITY['desat']['amount'][lvl]
    # Rec. 601 luma
    gray = (0.299 * px[:, 0] + 0.587 * px[:, 1] + 0.114 * px[:, 2]).unsqueeze(1)
    gray = gray.expand_as(px)
    return torch.lerp(px, gray, amount)


def _op_invert(px: torch.Tensor, lvl: int) -> torch.Tensor:
    amount = _INTENSITY['invert']['amount'][lvl]
    return torch.lerp(px, 1.0 - px, amount)


_DISPATCH = {
    'blur': _op_blur,
    'downsample': _op_downsample,
    'noise': _op_noise,
    'obscure': _op_obscure,
    'r-noise': _op_rnoise,
    'desat': _op_desat,
    'invert': _op_invert,
}


def apply_operation(
    op: str,
    intensity: int,
    image_tensor: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    region_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply `op` at `intensity` to `image_tensor`, confined to `region_mask`.

    Args:
        op: one of OPERATION_SET.
        intensity: 1, 2, or 3.
        image_tensor: normalized image, shape [1, 3, H, W].
        mean, std: CLIP normalization stats, shape [1, 3, 1, 1].
        region_mask: float mask in [0, 1], shape broadcastable to [1, 1, H, W].
            None applies the op globally (whole image).

    Returns:
        Corrupted normalized tensor, same shape/device/dtype as input.
    """
    if op not in _DISPATCH:
        raise ValueError(f'unknown op {op!r}; expected one of {OPERATION_SET}')
    if intensity not in (1, 2, 3):
        raise ValueError(f'intensity must be 1/2/3, got {intensity}')

    orig_dtype = image_tensor.dtype
    px = _denorm(image_tensor.float(), mean.float(), std.float())
    corrupted_px = _DISPATCH[op](px, intensity)
    corrupted = _renorm(corrupted_px, mean.float(), std.float())

    if region_mask is not None:
        m = region_mask.to(corrupted.dtype)
        corrupted = m * corrupted + (1.0 - m) * image_tensor.float()

    return corrupted.to(orig_dtype)
