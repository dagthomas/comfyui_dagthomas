# APNext H3 Scenes → Contex Loop Plan
#
# Turns the `scenes` / `durations` lists from the H3 Crossover Writer or the H3
# Scenes Writer into the plan JSON that ComfyUI-MiniMaxH3-Contex-Loop's
# `MiniMax H3 Contex Loop Plan` node accepts on its `plan_json_input` socket:
#
#   {"prompt_prefix": "...", "defaults": {...}, "shots": [{"id", "prompt",
#    "duration_seconds", "seed"}, ...]}
#
# That pack then renders every scene in sequence, carrying the last frames
# (and optionally audio) of each scene into the next one.

import json

from ...utils.constants import CUSTOM_CATEGORY


class H3ScenesToChainPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenes": ("STRING", {
                    "forceInput": True,
                    "tooltip": "The `scenes` list from an H3 Crossover / Scenes Writer.",
                }),
                "id_prefix": ("STRING", {
                    "default": "scene",
                    "tooltip": "Shot ids become `<id_prefix>_01`, `_02`, ... (checkpoint names).",
                }),
                "base_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Seed for shot 1; each later shot uses base_seed + index. -1 in seed_mode omits seeds.",
                }),
                "seed_mode": (["base_seed + index", "omit (plan node derives seeds)"], {
                    "default": "base_seed + index",
                }),
                "steps": ("INT", {
                    "default": 0, "min": 0, "max": 10000,
                    "tooltip": "Sampler steps written into defaults. 0 = leave to the Plan node.",
                }),
            },
            "optional": {
                "durations": ("FLOAT", {
                    "forceInput": True,
                    "tooltip": "The matching `durations` list; written as duration_seconds per shot.",
                }),
                "prompt_prefix": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Global continuity text prepended to every scene by the Plan node.",
                }),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("plan_json", "shot_count")
    FUNCTION = "build"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Converts the scenes/durations lists into MiniMax H3 Contex Loop plan JSON "
        "(wire into MiniMax H3 Contex Loop Plan → plan_json_input) so the whole run "
        "renders scene after scene with motion/audio carried across each join."
    )

    @staticmethod
    def _first(value, default):
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value if value is not None else default

    def build(self, scenes, id_prefix, base_seed, seed_mode, steps, durations=None, prompt_prefix=None):
        scenes = [s for s in (scenes or []) if isinstance(s, str) and s.strip()]
        durations = list(durations or [])
        id_prefix = (self._first(id_prefix, "scene") or "scene").strip() or "scene"
        base_seed = int(self._first(base_seed, 0))
        seed_mode = self._first(seed_mode, "base_seed + index")
        steps = int(self._first(steps, 0))
        prompt_prefix = (self._first(prompt_prefix, "") or "").strip()

        shots = []
        for i, prompt in enumerate(scenes):
            shot = {"id": f"{id_prefix}_{i + 1:02d}", "prompt": prompt.splitlines()}
            if i < len(durations):
                shot["duration_seconds"] = float(durations[i])
            if seed_mode.startswith("base_seed"):
                shot["seed"] = (base_seed + i) & 0xffffffffffffffff
            shots.append(shot)

        plan = {}
        if prompt_prefix:
            plan["prompt_prefix"] = prompt_prefix
        defaults = {}
        if steps > 0:
            defaults["steps"] = steps
        if durations:
            defaults["duration_seconds"] = float(durations[0])
        if defaults:
            plan["defaults"] = defaults
        plan["shots"] = shots

        return (json.dumps(plan, indent=2, ensure_ascii=False), len(shots))
