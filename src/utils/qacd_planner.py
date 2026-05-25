"""QACD recipe planner (Stage 1): prompt construction + output parsing.

The planner is the LVLM itself, prompted (with the image) to act as an
adversary proposing the single most misleading edit for the query. It emits a
parseable recipe:

    TARGET: <one-line description of the concept to attack>
    OPERATION: <one of the allowed ops>
    INTENSITY: <1, 2, or 3>

TARGET is used only to localize attention (Stage 2); OPERATION + INTENSITY
drive the pixel-space corruption (Stage 3). On any parse failure we fall back
to a fixed recipe (noise / intensity 2 / center region), per report Sec. 3.3.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from utils.qacd_ops import OPERATION_SET

# Maps loose model spellings to canonical op names.
_OP_ALIASES = {
    'blur': 'blur', 'gaussian blur': 'blur', 'gaussianblur': 'blur',
    'downsample': 'downsample', 'pixelate': 'downsample', 'pixelation': 'downsample',
    'noise': 'noise', 'gaussian noise': 'noise',
    'obscure': 'obscure', 'blur+darken': 'obscure', 'darken': 'obscure',
    'r-noise': 'r-noise', 'rnoise': 'r-noise', 'region noise': 'r-noise',
    'regional noise': 'r-noise',
    'desat': 'desat', 'desaturate': 'desat', 'grayscale': 'desat',
    'greyscale': 'desat',
    'invert': 'invert', 'color inversion': 'invert', 'color invert': 'invert',
    'inversion': 'invert',
}

_OP_MENU = """- blur: Gaussian blur. Removes fine detail and texture.
- downsample: Heavy pixelation. Destroys small structures and text.
- noise: Adds Gaussian noise. Obscures details while leaving content visible.
- obscure: Blur combined with darkening. Hides a region's contents.
- r-noise: Replaces the region with strong noise. Erases the region's evidence.
- desat: Collapses color to grayscale. Defeats color-dependent questions.
- invert: Inverts colors to their complement. Flips color evidence to a wrong value."""

_ADVERSARIAL_HEADER = """You are a red-team adversary attacking a vision-language model. \
Given an image and a question, propose the single image edit that would most \
effectively mislead another model trying to answer that question, by attacking \
the specific visual evidence the question depends on. This is a prompt-engineering \
device for analysis, not real adversarial training."""

_NEUTRAL_HEADER = """You are an image-augmentation analyst. Given an image and a \
question, choose the single image edit that would most disrupt the visual \
evidence the question depends on."""

_BODY = """
## Available operations ##
{menu}

## Output format (exactly three lines) ##
TARGET: <one-line description of the concept or region the question depends on>
OPERATION: <one operation name from the list above>
INTENSITY: <1, 2, or 3, where 3 is strongest>

## Guidance ##
Pick the operation whose effect most directly defeats the question. Do not choose
an edit whose result would coincidentally match a correct answer. Output only the
three lines, nothing else.
{examples}
Question: "{query}"
"""

# Few-shot exemplars. Chosen to (a) lock the exact three-line format and
# (b) demonstrate matching question type -> operation under the anti-correlation
# principle (Sec. 3.2). Covers all 7 operations.
_FEWSHOT = """
## Examples ##
Question: "What color is the umbrella?"
TARGET: the umbrella
OPERATION: invert
INTENSITY: 2

Question: "Are the flowers yellow?"
TARGET: the flowers
OPERATION: desat
INTENSITY: 2

Question: "Is there a dog in the image?"
TARGET: the dog
OPERATION: obscure
INTENSITY: 2

Question: "How many people are in the image?"
TARGET: the people
OPERATION: r-noise
INTENSITY: 3

Question: "What does the sign say?"
TARGET: the text on the sign
OPERATION: downsample
INTENSITY: 3

Question: "Is the cat wearing a collar?"
TARGET: the cat's neck
OPERATION: blur
INTENSITY: 2

Question: "Is the table surface smooth?"
TARGET: the table surface
OPERATION: noise
INTENSITY: 2
"""

# Fallback recipe used on parse failure (report Sec. 3.3).
FALLBACK = {'target': None, 'op': 'noise', 'intensity': 2}


@dataclass
class Recipe:
    target: str | None
    op: str
    intensity: int
    parsed_ok: bool


def build_planner_prompt(
    query: str, variant: str = 'adversarial', icl: bool = True
) -> str:
    """Construct the planner prompt.

    Args:
        query: the question to plan a corruption for.
        variant: 'adversarial' (GAN-style framing) or 'neutral'.
        icl: include the few-shot exemplars (set False for the zero-shot ablation).
    """
    header = _ADVERSARIAL_HEADER if variant == 'adversarial' else _NEUTRAL_HEADER
    examples = _FEWSHOT if icl else ''
    return header + _BODY.format(menu=_OP_MENU, examples=examples, query=query)


def _canon_op(raw: str) -> str | None:
    key = raw.strip().lower().strip('.').strip()
    if key in _OP_ALIASES:
        return _OP_ALIASES[key]
    # substring fallback: pick the first known op name appearing in the string
    for name in OPERATION_SET:
        if name in key:
            return name
    return None


def parse_recipe(text: str) -> Recipe:
    """Parse a planner output into a Recipe, falling back on malformed output."""
    target = op = None
    intensity = None

    m = re.search(r'TARGET:\s*(.+)', text, re.IGNORECASE)
    if m:
        target = m.group(1).strip().splitlines()[0].strip() or None

    m = re.search(r'OPERATION:\s*([^\n]+)', text, re.IGNORECASE)
    if m:
        op = _canon_op(m.group(1))

    m = re.search(r'INTENSITY:\s*([123])', text, re.IGNORECASE)
    if m:
        intensity = int(m.group(1))

    if op is None or intensity is None:
        return Recipe(
            target=target,  # keep target if we got it, for attention grounding
            op=FALLBACK['op'],
            intensity=FALLBACK['intensity'],
            parsed_ok=False,
        )
    return Recipe(target=target, op=op, intensity=intensity, parsed_ok=True)
