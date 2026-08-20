# APNext H3 Scene Counter
#
# Output node that shows render progress across a scene list: "scene X of N,
# M remaining". Wire H3 Scene Pick's `index` and `count` outputs (or any INT
# pair) into it; every run updates the big readout on the node (rendered by
# web/js/h3_scene_counter.js). The status line and the remaining count come
# out as sockets too, so they can feed filenames or notes.

from ...utils.constants import CUSTOM_CATEGORY


class H3SceneCounter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {
                    "forceInput": True,
                    "tooltip": "0-based index of the scene being rendered (H3 Scene Pick's `index` output).",
                }),
                "count": ("INT", {
                    "forceInput": True,
                    "tooltip": "Total number of scenes (H3 Scene Pick's `count` or a writer's `scene_count`).",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("status", "remaining")
    FUNCTION = "tally"
    OUTPUT_NODE = True
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Shows scene-render progress as a big 'X of N — M remaining' readout on the "
        "node. Connect H3 Scene Pick's `index` and `count` outputs; updates every run. "
        "Also outputs the status text and the remaining count."
    )

    def tally(self, index, count):
        index = int(index)
        count = max(0, int(count))
        done = min(index + 1, count) if count else 0
        remaining = max(0, count - done)
        status = (
            f"Scene {done} of {count} — {remaining} remaining" if count else "No scenes"
        )
        print(f"🎬 H3 Scene Counter | {status}")
        return {
            "ui": {"counter": [{"index": index, "count": count, "remaining": remaining, "status": status}]},
            "result": (status, remaining),
        }
