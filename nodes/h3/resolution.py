# APNext H3 Resolution - the vendor's adapt_canvas rule, per aspect ratio
#
# H3's released pipelines size every canvas with one rule (adapt_canvas):
#
#     1. scale so the SHORT edge is 768
#     2. if the area exceeds 1344x768 = 1,032,192 px, scale down by
#        sqrt(cap / area)
#     3. round each side to the nearest multiple of 32
#
# That yields the 95 canvases H3 was trained on - and it means aspects do NOT
# share a pixel count: 16:9 is 1344x768 (1008 tokens/frame) while 1:1 is
# 768x768 (576 tokens/frame, a third of the attention cost). An equal-pixels
# model hands H3 off-distribution canvases (1:1 at 1024x1024) that no vendor
# pipeline would ever produce. Only the widest aspects hit the area cap; from
# ~3:4 through 7:4 the short edge stays a full 768.
#
# 1344x768 is exactly 7:4 (1.750), not 16:9 (1.778) - no H3 canvas is a true
# 16:9; a 16:9 request lands on the trained 1344x768 with a 1.6% squeeze.
#
# The frames input snaps onto the VAE's 17n+5 frame grid (latent frames
# 5n+2; trained range ~124-362, ceiling 362) and reports the video token
# count: (5n+2) x (W/32) x (H/32). At 1344x768 the fused-qkv int32 kernel
# crossing (99,864 tokens) sits between 311 and 328 frames - sage/quant
# builds without int64 offsets overflow past it, and references on top of
# the target only bring it closer.

import math
import re

from ...utils.constants import CUSTOM_CATEGORY

SHORT_EDGE = 768                          # adapt_canvas step 1
AREA_CAP = 1344 * 768                     # 1,032,192 - adapt_canvas step 2
GRID = 32                                 # VAE 16x spatial x DiT patchify 2
TOKENS_16_9 = (1344 // 32) * (768 // 32)  # 1008 - the cost yardstick

FRAME_GRID = 17                           # frame counts are 17n + 5
FRAME_GRID_OFFSET = 5
MAX_FRAMES = 362                          # longest H3 was trained on
MIN_TRAINED_FRAMES = 124
INT32_TOKEN_CROSSING = 99_864             # fused-qkv stride 21504: 2^31 / 21504


def adapt_canvas(aspect_w, aspect_h, scale=1.0, multiple=GRID):
    """
    (width, height) for aspect w:h under the vendor's adapt_canvas rule,
    optionally scaled: `scale` multiplies the AREA (linear sides by sqrt),
    with 1.0 the exact vendor recipe.
    """
    a = float(aspect_w) / float(aspect_h)
    lin = math.sqrt(max(scale, 1e-6))
    short = SHORT_EDGE * lin
    if a >= 1.0:
        h, w = short, short * a
    else:
        w, h = short, short / a
    cap = AREA_CAP * scale
    if w * h > cap:
        k = math.sqrt(cap / (w * h))
        w, h = w * k, h * k
    m = int(multiple)
    return (max(m, round(w / m) * m), max(m, round(h / m) * m))


def snap_frames(frames):
    """`frames` snapped UP onto the 17n+5 grid, like the H3 nodes do."""
    n = max(1, math.ceil((int(frames) - FRAME_GRID_OFFSET) / FRAME_GRID))
    return FRAME_GRID * n + FRAME_GRID_OFFSET


# (label prefix, aspect_w, aspect_h) - the label shows the real 1.0 canvas.
_ASPECT_DEFS = [
    ("16:9 widescreen - the trained canvas", 16, 9),
    ("9:16 portrait widescreen", 9, 16),
    ("21:9 ultrawide", 21, 9),
    ("9:21 portrait ultrawide", 9, 21),
    ("1:1 square - a third of 16:9's attention cost", 1, 1),
    ("3:2 photo", 3, 2),
    ("2:3 portrait photo", 2, 3),
    ("4:3 standard", 4, 3),
    ("3:4 portrait standard", 3, 4),
    ("5:4 near-square", 5, 4),
    ("4:5 portrait near-square", 4, 5),
]

ASPECTS = [
    (f"{prefix} ({adapt_canvas(w, h)[0]}x{adapt_canvas(w, h)[1]})", w, h)
    for prefix, w, h in _ASPECT_DEFS
]

_ASPECT_RE = re.compile(r"(\d+)\s*:\s*(\d+)")


def _parse_aspect(label):
    """w:h from any current or legacy dropdown label; 16:9 when unreadable."""
    match = _ASPECT_RE.search(str(label or ""))
    if match:
        w, h = int(match.group(1)), int(match.group(2))
        if w > 0 and h > 0:
            return w, h
    return 16, 9


class H3ResolutionSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": ([label for label, _, _ in ASPECTS], {
                    "default": ASPECTS[0][0],
                    "tooltip": (
                        "The frame's shape. Each label shows the trained canvas the vendor's "
                        "adapt_canvas rule gives that aspect (short edge 768, area capped at "
                        "1344x768). Aspects do NOT cost the same: squarer is cheaper - 1:1 is "
                        "a third of 16:9's attention."
                    ),
                }),
                "megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 2.0, "step": 0.05, "round": 0.01,
                    "display": "slider",
                    "tooltip": (
                        "Canvas scale. 1.0 is the exact vendor recipe (the trained canvas for "
                        "the aspect) - leave it there for renders. Below 1.0 shrinks the area "
                        "for cheap previews; above 1.0 exceeds the vendor's area cap, which no "
                        "released H3 pipeline ever does."
                    ),
                }),
            },
            "optional": {
                "multiple": ("INT", {
                    "default": GRID, "min": 16, "max": 128, "step": 16,
                    "tooltip": (
                        "Grid every side must sit on. H3 needs 32; 128 also aligns the token "
                        "grid for Morton-ordered sparse attention."
                    ),
                }),
                "frames": ("INT", {
                    "default": 192, "min": 0, "max": 1000, "step": 1,
                    "tooltip": (
                        "Target frame count, snapped UP onto H3's 17n+5 grid for the frames / "
                        "duration outputs and the token report. 192 = exactly 8.000s (the only "
                        "common integer duration on the grid); trained range 124-362. 0 skips "
                        "the report."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("width", "height", "megapixels", "info", "frames", "duration_seconds")
    OUTPUT_TOOLTIPS = (
        "Frame width - wire into the render's `width`.",
        "Frame height - wire into the render's `height`.",
        "The real megapixel count of the chosen canvas (1344x768 = 1.03).",
        "Canvas, tokens/frame, attention cost vs 16:9, snapped length and video tokens.",
        "The frame count snapped up onto the 17n+5 grid - wire into the render's `length`.",
        "The snapped length in seconds at 24fps - wire into a writer's `duration_seconds`.",
    )
    FUNCTION = "pick"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Frame size for the H3 renders using the vendor's adapt_canvas rule: short edge "
        "768, area capped at 1344x768, sides rounded to 32 - the canvases H3 was trained "
        "on, so 1:1 is 768x768 at a third of 16:9's attention cost. Also snaps the frame "
        "count onto the 17n+5 grid and reports the video token budget."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, aspect_ratio=None):
        # Legacy labels (older equal-pixel builds of this node) parse fine and
        # anything unreadable falls back to 16:9 - never fail a saved workflow.
        return True

    def pick(self, aspect_ratio, megapixels, multiple=GRID, frames=192):
        aw, ah = _parse_aspect(aspect_ratio)
        scale = float(megapixels)
        width, height = adapt_canvas(aw, ah, scale, multiple)

        if scale > 1.001:
            print(
                f"⚠️ H3 Resolution | scale {scale:.2f} exceeds the vendor's area cap "
                f"(1,032,192 px) - no released H3 pipeline renders above it; expect "
                "off-distribution output."
            )

        real_mp = width * height / 1_000_000.0
        tokens_per_frame = (width // 32) * (height // 32)
        attention = (tokens_per_frame / TOKENS_16_9) ** 2
        info = (
            f"{width}x{height} | {aw}:{ah} | {real_mp:.2f} MP | "
            f"{tokens_per_frame} tok/frame | attention {attention:.2f}x vs 16:9"
        )

        snapped, duration = 0, 0.0
        if int(frames) > 0:
            snapped = snap_frames(frames)
            duration = snapped / 24.0
            latent_frames = 5 * ((snapped - FRAME_GRID_OFFSET) // FRAME_GRID) + 2
            video_tokens = latent_frames * tokens_per_frame
            info += f" | {snapped}f = {duration:.3f}s | {video_tokens:,} video tokens"
            if snapped != int(frames):
                info += f" (snapped from {int(frames)})"
            if snapped > MAX_FRAMES:
                print(
                    f"⚠️ H3 Resolution | {snapped} frames is past the trained ceiling of "
                    f"{MAX_FRAMES} (15.083s) - step down to 362 or less."
                )
            elif snapped < MIN_TRAINED_FRAMES:
                print(
                    f"ℹ️ H3 Resolution | {snapped} frames is below the ~{MIN_TRAINED_FRAMES}-"
                    "frame start of the trained range."
                )
            # The whole packed sequence counts against the kernel limit, not just
            # video: target audio is ~80 rows/second and text ~1,000 tokens, which
            # is what puts the crossing between 311f and 328f at 1344x768.
            est_sequence = video_tokens + round(duration * 80) + 1000
            if est_sequence > INT32_TOKEN_CROSSING:
                print(
                    f"⚠️ H3 Resolution | est. packed sequence ~{est_sequence:,} tokens "
                    f"({video_tokens:,} video + audio + text, before any reference rows) "
                    f"crosses the fused-qkv int32 kernel limit ({INT32_TOKEN_CROSSING:,}) - "
                    "sage/quant kernels without int64 offsets overflow here."
                )

        print(f"🖼️ H3 Resolution | {info}")
        return (width, height, round(real_mp, 3), info, snapped, round(duration, 5))
