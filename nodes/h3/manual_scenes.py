# APNext H3 Manual Scenes
#
# The fully hand-authored counterpart of the multi-scene writers: paste (or
# type) the finished H3 scene prompts yourself and get the same matching lists
# the writers emit - `scenes`, `durations`, `lengths` (frame counts on H3's
# 5 + 17k grid) and `scenes_text` - so the render side of any writer workflow
# plugs in unchanged. No model is called; the script IS the film.
#
# The script accepts two shapes:
#   - `=== SCENE NN | duration: S.S ===` ... `=== END SCENE NN ===` envelopes
#     (the same contract the writers emit; duration is optional per scene), or
#   - bare scene blocks, split wherever a line starts with
#     `subject_definitions:` - the first section of every H3 prompt.
#
# Per-scene durations: the envelope header wins, then the `durations` box
# (comma/space separated, one value per scene), then `default_duration`.

from ...utils.constants import CUSTOM_CATEGORY
from .music_support import frames_for_seconds, seconds_for_frames
from .scenes_support import parse_scenes, scenes_to_text

_SPLIT_KEY = "subject_definitions:"


def _split_bare_blocks(text):
    """Scene bodies from a script without envelopes: each block starts at a
    line beginning with `subject_definitions:`."""
    blocks, current = [], []
    for line in (text or "").splitlines():
        if line.strip().lower().startswith(_SPLIT_KEY):
            if any(l.strip() for l in current):
                blocks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if any(l.strip() for l in current):
        blocks.append("\n".join(current).strip())
    return blocks


def _parse_durations(text):
    out = []
    for token in (text or "").replace(",", " ").split():
        try:
            out.append(float(token))
        except ValueError:
            raise ValueError(f"durations: '{token}' is not a number")
    return out


class H3ManualScenes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "The finished H3 scene prompts, in order. Either `=== SCENE NN | "
                        "duration: S.S ===` envelopes (the writers' contract), or bare "
                        "prompts split wherever a line starts with `subject_definitions:`. "
                        "Each scene is one rendered clip."
                    ),
                }),
                "default_duration": ("FLOAT", {
                    "default": 12.0, "min": 5.2, "max": 15.1, "step": 0.1,
                    "tooltip": (
                        "Seconds per scene when neither the envelope header nor the "
                        "durations box names one. Snapped to H3's frame grid."
                    ),
                }),
            },
            "optional": {
                "durations": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Optional per-scene durations, comma or space separated in scene "
                        "order (e.g. `12, 10, 11.5`). An envelope's own `duration:` wins "
                        "over this list; scenes beyond the list fall back to "
                        "default_duration."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("scenes", "durations", "lengths", "scenes_text", "scene_count", "total_seconds")
    OUTPUT_IS_LIST = (True, True, True, False, False, False)
    FUNCTION = "build"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Hand-authored scenes instead of a writer: paste finished H3 prompts (envelopes "
        "or blocks starting at `subject_definitions:`) and get the writers' matching "
        "`scenes` / `durations` / `lengths` lists for the render side. No model is called."
    )

    def build(self, script, default_duration, durations=""):
        script = (script or "").strip()
        if not script:
            raise ValueError("H3 Manual Scenes: the script is empty - paste at least one scene prompt.")

        wanted = _parse_durations(durations)

        if "=== SCENE" in script.upper():
            _synopsis, parsed = parse_scenes(script, default_duration)
            bodies = [p for _no, _d, p in parsed]
            header_durs = [d for _no, d, _p in parsed]
        else:
            bodies = _split_bare_blocks(script)
            header_durs = [None] * len(bodies)

        if not bodies:
            raise ValueError(
                "H3 Manual Scenes: no scenes found - use `=== SCENE NN ===` envelopes or "
                "start each scene at a `subject_definitions:` line."
            )

        durs = []
        for i, header in enumerate(header_durs):
            d = header if header is not None else (wanted[i] if i < len(wanted) else default_duration)
            durs.append(max(5.2, min(15.1, float(d))))
        frames = [frames_for_seconds(d) for d in durs]
        durs = [round(seconds_for_frames(fr), 4) for fr in frames]

        scenes_text = scenes_to_text("", [(i + 1, d, p) for i, (d, p) in enumerate(zip(durs, bodies))])
        total = float(sum(durs))
        print(f"🎬 H3 Manual Scenes | {len(bodies)} scene(s), {total:.1f}s total")
        return (bodies, durs, frames, scenes_text, len(bodies), total)
