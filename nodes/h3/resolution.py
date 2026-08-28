# APNext H3 Resolution - frame sizes with the trained frame as the unit
#
# H3 is trained at 1344x768: 16:9, 1,032,192 pixels. Core's Resolution
# Selector counts megapixels from 1 MP = 1024x1024, so its "1.0" is 3% under
# the trained frame and 16:9 lands on 1344x768 only by rounding luck; nudge
# the slider and the aspect drifts. Here 1.0 MP IS 1344x768. Every other
# aspect gets the same pixel count, the slider scales all of them together,
# and every side is a multiple of 32 - the grid the H3 nodes take.
#
# The size is chosen on that grid, not rounded onto it: of the four grid
# corners around the ideal (float) size, the one with the smallest error
# wins, the aspect weighing twice the pixel count - a user who asks for 4:3
# wants 4:3 framing, the megapixels are a slider anyway. 1.0 therefore gives
# 1344x768 / 768x1344 / 1024x1024 / 1248x832 / 832x1248 / 1152x864 / 864x1152.

import math

from ...utils.constants import CUSTOM_CATEGORY

BASE_W, BASE_H = 1344, 768
BASE_PIXELS = BASE_W * BASE_H            # 1,032,192 - what "1.0" means here
GRID = 32                                 # the H3 nodes' width / height step

ASPECTS = [
    ("16:9 widescreen - what H3 is trained on (1344x768 at 1.0)", 16, 9),
    ("9:16 portrait widescreen (768x1344 at 1.0)", 9, 16),
    ("1:1 square", 1, 1),
    ("2:3 portrait photo", 2, 3),
    ("3:2 photo", 3, 2),
    ("3:4 portrait standard", 3, 4),
    ("4:3 standard", 4, 3),
]
_ASPECT_BY_LABEL = {label: (w, h) for label, w, h in ASPECTS}


def pick_size(pixels, aspect_w, aspect_h, multiple=GRID):
    """(width, height) on the `multiple` grid nearest `pixels` at aspect w:h.

    Candidates are the grid corners around the ideal size; the score is the
    log error of the pixel count plus twice the log error of the aspect.
    """
    pixels = max(float(pixels), float(multiple * multiple))
    a = float(aspect_w) / float(aspect_h)
    w0 = math.sqrt(pixels * a)
    h0 = w0 / a
    m = int(multiple)
    cands = set()
    for w in (math.floor(w0 / m) * m, math.ceil(w0 / m) * m):
        for h in (math.floor(h0 / m) * m, math.ceil(h0 / m) * m):
            if w >= m and h >= m:
                cands.add((int(w), int(h)))

    def score(wh):
        w, h = wh
        return abs(math.log((w * h) / pixels)) + 2.0 * abs(math.log((w / h) / a))

    return min(cands, key=score)


class H3ResolutionSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": ([label for label, _, _ in ASPECTS], {
                    "default": ASPECTS[0][0],
                    "tooltip": (
                        "The frame's shape. 16:9 is what H3 is trained on; the others keep the same "
                        "pixel count at that shape, so the render costs the same."
                    ),
                }),
                "megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 2.0, "step": 0.05, "round": 0.01,
                    "display": "slider",
                    "tooltip": (
                        "Size, in units of the trained frame: 1.0 = 1344x768 (1.03 real megapixels), "
                        "0.5 = half the pixels (960x544 at 16:9), 2.0 = twice (1920x1088). The aspect "
                        "is kept while the slider scales the frame; sides stay multiples of 32."
                    ),
                }),
            },
            "optional": {
                "multiple": ("INT", {
                    "default": GRID, "min": 16, "max": 128, "step": 16,
                    "tooltip": "Grid every side must sit on. The H3 nodes take multiples of 32.",
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("width", "height", "megapixels", "info")
    OUTPUT_TOOLTIPS = (
        "Frame width - wire into the render's `width`.",
        "Frame height - wire into the render's `height`.",
        "The real megapixel count of the chosen size (1344x768 = 1.03).",
        "`1344x768 | 16:9 | 1.03 MP (1.00 of the trained frame)`.",
    )
    FUNCTION = "pick"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Frame size for the H3 renders, with the trained 1344x768 frame as the unit: "
        "megapixels 1.0 is exactly 1344x768, every aspect gets the same pixel count, the "
        "slider scales them together and every side is a multiple of 32. Drop-in for "
        "core's Resolution Selector (same width / height outputs)."
    )

    def pick(self, aspect_ratio, megapixels, multiple=GRID):
        aw, ah = _ASPECT_BY_LABEL.get(aspect_ratio, (BASE_W, BASE_H))
        width, height = pick_size(float(megapixels) * BASE_PIXELS, aw, ah, multiple)
        real_mp = width * height / 1_000_000.0
        units = width * height / float(BASE_PIXELS)
        info = f"{width}x{height} | {aw}:{ah} | {real_mp:.2f} MP ({units:.2f} of the trained frame)"
        print(f"🖼️ H3 Resolution | {info}")
        return (width, height, round(real_mp, 3), info)
