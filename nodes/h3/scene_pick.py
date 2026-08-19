# APNext H3 Scene Pick
#
# Pulls one scene out of the list the H3 Crossover Writer emits. ComfyUI would
# otherwise run every downstream node once per element; this collapses the list
# to a single prompt (and its duration) chosen by index.

from ...utils.constants import CUSTOM_CATEGORY


class H3ScenePick:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenes": ("STRING", {
                    "forceInput": True,
                    "tooltip": "The `scenes` list from H3 Crossover Writer (or any STRING list).",
                }),
                "index": ("INT", {
                    "default": 0, "min": 0, "max": 999,
                    "tooltip": "0-based scene index. Values past the end are clamped to the last scene.",
                }),
            },
            "optional": {
                "durations": ("FLOAT", {
                    "forceInput": True,
                    "tooltip": "The matching `durations` list; the picked scene's duration comes out.",
                }),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("scene", "duration", "index", "count")
    FUNCTION = "pick"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Selects one scene (and its duration) from the list produced by the H3 Crossover "
        "Writer, so a single scene can be rendered or refined on its own."
    )

    def pick(self, scenes, index, durations=None):
        scenes = list(scenes or [])
        if not scenes:
            return ("", 0.0, 0, 0)

        idx = int(index[0]) if isinstance(index, (list, tuple)) else int(index)
        idx = max(0, min(idx, len(scenes) - 1))

        duration = 0.0
        if durations:
            durations = list(durations)
            duration = float(durations[min(idx, len(durations) - 1)])

        return (scenes[idx], duration, idx, len(scenes))
