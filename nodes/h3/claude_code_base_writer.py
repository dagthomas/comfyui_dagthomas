# APNext H3 Claude Code Writer
#
# The base-format H3 writer (T2VA / I2VA / FL2VA / L2VA) driven by the local
# Claude Code CLI instead of an API key. It inherits every prompt-building rule
# from H3BasePromptWriter, so the guide, camera vocabulary and wildness bands
# stay in one place; only the transport and the widgets differ.

import random

from ...utils.claude_code import is_interrupt
from ...utils.constants import CUSTOM_CATEGORY
from ...utils.image_utils import tensor2pil
from .base_prompt_writer import TASK_TYPES, H3BasePromptWriter, _FIELDS
from .claude_code_support import (
    claude_code_inputs,
    claude_code_optional_inputs,
    directions_with_research,
    run_h3_claude_code,
)
from .common import (
    AUTO,
    CAMERA_AMPLITUDES,
    CAMERA_MOTIONS,
    CAMERA_SPEEDS,
    DIALOGUE_LANGUAGES,
    SHOT_PLANS,
    VISUAL_STYLES,
    extract_section,
    resolve_dialogue_language,
    strip_code_fence,
)


class H3ClaudeCodeBaseWriter(H3BasePromptWriter):
    """
    APNext H3 Claude Code Writer

    Writes a MiniMax-H3 base-format prompt with the local Claude Code CLI, using
    its own login rather than an API key. Can research real references first, and
    returns a session_id so the refiner can revise the result in context.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "idea": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Your short prompt or image description. This is what gets expanded into a full H3 prompt.",
            }),
            "task_type": (TASK_TYPES, {
                "default": "T2VA (text only)",
                "tooltip": "Which H3 task the prompt targets. Anything other than T2VA emits the matching reference-alignment instruction line.",
            }),
            "duration_seconds": ("FLOAT", {
                "default": 6.0, "min": 1.0, "max": 60.0, "step": 0.5,
                "tooltip": "Effective video duration. Drives the cut times and the S.SS value in the alignment instruction.",
            }),
            "shot_plan": (SHOT_PLANS, {"default": AUTO}),
            "visual_style": (VISUAL_STYLES, {
                "default": AUTO,
                "tooltip": "Style stated at the start of [Shot 1]. Auto derives it from the idea or the attached image.",
            }),
            "wildness": ("INT", {
                "default": 25, "min": 0, "max": 100, "step": 1,
                "tooltip": "0 = literal and conservative, 100 = fully unhinged. Above 40 the node also injects concrete surreal elements picked from the seed.",
            }),
            "camera_motion": (CAMERA_MOTIONS, {"default": AUTO}),
            "camera_amplitude": (CAMERA_AMPLITUDES, {"default": AUTO}),
            "camera_speed": (CAMERA_SPEEDS, {"default": AUTO}),
            "include_dialogue": ("BOOLEAN", {"default": True}),
            "dialogue_language": (DIALOGUE_LANGUAGES, {
                "default": "English",
                "tooltip": (
                    "The language the characters actually speak, and the tag written inside "
                    "<d>[...]</d>. Auto lets the model pick one that fits the setting. Pick "
                    "Custom (or just fill in custom_dialogue_language) for anything not listed."
                ),
            }),
            "include_on_screen_text": ("BOOLEAN", {"default": False}),
            "include_soundscape": ("BOOLEAN", {"default": True}),
            "include_non_diegetic_music": ("BOOLEAN", {"default": True}),
        }
        required.update(claude_code_inputs())
        required["seed"] = ("INT", {
            "default": -1, "min": -1, "max": 0xffffffffffffffff,
            "tooltip": (
                "Picks the surreal elements at high wildness, and controls ComfyUI caching. "
                "-1 re-rolls and re-runs every queue."
            ),
        })

        optional = {
            "image": ("IMAGE", {
                "tooltip": "Reference frame(s), sent inline to the CLI as vision input.",
            }),
            "extra_instructions": ("STRING", {"multiline": True, "default": ""}),
            "custom_dialogue_language": ("STRING", {
                "default": "",
                "tooltip": (
                    "Any language or dialect not in the dropdown, e.g. 'Norwegian (Bergen "
                    "dialect)' or 'Latin'. Overrides the dropdown when filled in."
                ),
            }),
        }
        optional.update(claude_code_optional_inputs())

        return {"required": required, "optional": optional}

    RETURN_TYPES = ("STRING",) * 6
    RETURN_NAMES = (
        "h3_prompt",
        "integrated_multimodal_description",
        "overall_soundscape",
        "non_diegetic_music",
        "session_id",
        "info",
    )
    FUNCTION = "write_with_claude_code"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Writes a MiniMax-H3 base-format prompt (T2VA / I2VA / FL2VA / L2VA) using the "
        "local Claude Code CLI and its own login. Optionally researches real references first."
    )

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        return float("nan") if seed == -1 else seed

    def write_with_claude_code(
        self,
        idea,
        task_type,
        duration_seconds,
        shot_plan,
        visual_style,
        wildness,
        camera_motion,
        camera_amplitude,
        camera_speed,
        include_dialogue,
        dialogue_language,
        include_on_screen_text,
        include_soundscape,
        include_non_diegetic_music,
        model,
        research,
        use_subscription,
        timeout_seconds,
        seed,
        image=None,
        extra_instructions="",
        custom_dialogue_language="",
        resume_session_id="",
        working_dir="",
    ):
        try:
            if not idea.strip() and image is None:
                raise ValueError("Provide an idea, an image, or both.")

            dialogue_language = resolve_dialogue_language(
                dialogue_language, custom_dialogue_language
            )

            current_seed = seed if seed != -1 else random.randint(0, 0xffffffffffffffff)
            rng = random.Random(current_seed)

            images = [tensor2pil(frame) for frame in image] if image is not None else None

            user_prompt, wild_label = self._build_user_prompt(
                idea,
                task_type,
                duration_seconds,
                shot_plan,
                visual_style,
                wildness,
                camera_motion,
                camera_amplitude,
                camera_speed,
                include_dialogue,
                dialogue_language,
                include_on_screen_text,
                include_soundscape,
                include_non_diegetic_music,
                directions_with_research(extra_instructions, research),
                images is not None,
                rng,
            )

            print(
                f"🎬 H3 Claude Code Writer | {task_type} | {duration_seconds:.2f}s | "
                f"wildness {wildness} ({wild_label}) | research "
                f"{'on' if research else 'off'} | seed {current_seed}"
            )

            text, session_id, info = run_h3_claude_code(
                self._build_system_prompt(),
                user_prompt,
                images,
                model,
                research,
                use_subscription,
                timeout_seconds,
                resume_session_id,
                working_dir,
            )

            prompt = strip_code_fence(text)

            return (
                prompt,
                extract_section(prompt, "integrated_multimodal_description", _FIELDS),
                extract_section(prompt, "overall_soundscape", _FIELDS),
                extract_section(prompt, "non_diegetic_music", _FIELDS),
                session_id,
                info,
            )

        except Exception as exc:
            # A cancelled queue must stop the run, not become an error string.
            if is_interrupt(exc):
                raise
            print(f"❌ H3 Claude Code Writer error: {exc}")
            import traceback

            print(traceback.format_exc())
            message = f"Error occurred while writing the H3 prompt: {exc}"
            return (message, message, "", "", "", "error")
