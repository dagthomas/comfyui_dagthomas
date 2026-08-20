# APNext H3 Scene Brief
#
# Manual scene authoring for the multi-scene writers (Music Video, Crossover,
# Scenes): one node = one scene the USER plans - what happens, where, who is
# on screen and which reference pictures apply. Chain several through
# `brief_in` (like H3 Characters chains through cast_in) and wire the final
# `briefs` output into a writer's `scene_briefs` socket. The writer then
# treats each brief as the binding plan for that scene - it still writes the
# full production-ready H3 envelope (grammar, timestamps, wardrobe locks,
# lyric sync), but the content follows the brief.
#
# Numbering: `scene_number` pins a brief to a specific scene/piece; 0 lets the
# briefs fill scenes in chaining order (skipping pinned ones). Scenes without
# a brief stay the model's to invent within the concept.

from ...utils.constants import CUSTOM_CATEGORY


class H3SceneBrief:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "What happens in this scene, in your words - the action, the beat "
                        "of the story, the image you want. The writer stages exactly this "
                        "(adapted to the scene's duration, lyrics and energy)."
                    ),
                }),
            },
            "optional": {
                "scene_number": ("INT", {
                    "default": 0, "min": 0, "max": 99,
                    "tooltip": (
                        "Pin this brief to a specific scene/piece number (1 = first). "
                        "0 = unpinned: unpinned briefs fill scenes in chaining order, "
                        "skipping pinned ones."
                    ),
                }),
                "location": ("STRING", {
                    "default": "",
                    "tooltip": "Where the scene is set (matches or extends the location lock).",
                }),
                "cast": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Who is on screen, comma separated, using the names from your cast "
                        "(e.g. `Lena, Sheldon`). Empty = the writer decides."
                    ),
                }),
                "pictures": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Which reference pictures matter here, e.g. `1` or "
                        "`Picture 2 = the rooftop, use as the location`."
                    ),
                }),
                "camera": ("STRING", {
                    "default": "",
                    "tooltip": "Optional camera / framing wish (e.g. `one slow push-in, no cuts`).",
                }),
                "brief_in": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Chain from another H3 Scene Brief to build a list of scenes.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("briefs",)
    FUNCTION = "build"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "One manually planned scene for the multi-scene writers: what happens, where, "
        "which cast members and reference pictures. Chain nodes through brief_in and wire "
        "`briefs` into a writer's `scene_briefs` input - each brief becomes the binding "
        "plan for its scene while the writer still produces the full H3 envelope."
    )

    def build(self, description, scene_number=0, location="", cast="", pictures="", camera="", brief_in=""):
        description = (description or "").strip()
        fields = [
            ("location", (location or "").strip()),
            ("cast", (cast or "").strip()),
            ("pictures", (pictures or "").strip()),
            ("camera", (camera or "").strip()),
        ]
        prev = (brief_in or "").strip()
        if not description and not any(v for _, v in fields):
            return (prev,)
        head = f"SCENE {scene_number:02d}" if scene_number else "SCENE (next in order)"
        lines = [f"{head}: {description}" if description else f"{head}:"]
        lines += [f"    {k}: {v}" for k, v in fields if v]
        block = "\n".join(lines)
        return ((prev + "\n\n" + block) if prev else block,)
