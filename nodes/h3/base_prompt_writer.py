# APNext H3 Prompt Writer
#
# Turns a short idea (and optionally a reference image) into a complete
# MiniMax-H3 video prompt in the T2VA / I2VA / FL2VA / L2VA format, using the
# official writing guide as the system prompt.

import random

from ...utils.constants import CUSTOM_CATEGORY
from ...utils.image_utils import tensor2pil
from ...utils.llm_router import AUTO_DETECT, call_llm, list_all_models
from .common import (
    AUTO,
    CAMERA_AMPLITUDES,
    CAMERA_MOTIONS,
    CAMERA_SPEEDS,
    SHOT_PLANS,
    VISUAL_STYLES,
    camera_directive,
    extract_section,
    load_guide,
    shot_directive,
    strip_code_fence,
    toggle_directives,
    wildness_directive,
)

TASK_TYPES = [
    "T2VA (text only)",
    "I2VA (first frame)",
    "FL2VA (first + last frame)",
    "L2VA (last frame)",
]

_FIELDS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)


class H3BasePromptWriter:
    """
    APNext H3 Prompt Writer

    Writes a MiniMax-H3 video prompt from a short idea or an image description,
    following the official Video Prompt Writing Guide (T2VA / I2VA / FL2VA / L2VA).
    Attach images to describe them directly instead of writing the description
    yourself; leave them unconnected for pure text-to-video.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
                "camera_motion": (CAMERA_MOTIONS, {
                    "default": AUTO,
                    "tooltip": "Primary camera movement, using the guide's vocabulary.",
                }),
                "camera_amplitude": (CAMERA_AMPLITUDES, {"default": AUTO}),
                "camera_speed": (CAMERA_SPEEDS, {"default": AUTO}),
                "include_dialogue": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off means no (Sx) speaker IDs and no <d> blocks at all.",
                }),
                "dialogue_language": ("STRING", {
                    "default": "English",
                    "tooltip": "Language tag written inside <d>[...]</d>.",
                }),
                "include_on_screen_text": ("BOOLEAN", {"default": False}),
                "include_soundscape": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off writes N/A into overall_soundscape.",
                }),
                "include_non_diegetic_music": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off writes N/A into non_diegetic_music.",
                }),
                "model": (list_all_models(), {
                    "default": AUTO_DETECT,
                    "tooltip": "Which LLM writes the prompt. auto-detect picks the first provider with an API key set.",
                }),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Reference frame(s). Sent to the model as vision input so it can describe them itself.",
                }),
                "extra_instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Free-form extra direction appended to the request.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "h3_prompt",
        "integrated_multimodal_description",
        "overall_soundscape",
        "non_diegetic_music",
        "model_used",
    )
    FUNCTION = "write"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Expands a short idea or image description into a full MiniMax-H3 video prompt "
        "(T2VA / I2VA / FL2VA / L2VA) using the official writing guide."
    )

    def _instruction_rule(self, task_type, duration_seconds):
        duration = f"{duration_seconds:.2f}"

        if task_type.startswith("T2VA"):
            return (
                "Task: T2VA. There is no image-alignment instruction. Begin the output "
                "directly with `integrated_multimodal_description:`."
            )

        if task_type.startswith("I2VA"):
            return (
                "Task: I2VA. The first line of the output must be exactly:\n"
                "For the target video, at 0.00 seconds into the target video, <Picture 1> "
                "(from [Shot 1]) is fully referenced.\n"
                "Follow it with one blank line, then the core fields. <Picture 1> is the "
                "real first frame at 0.00s and belongs to [Shot 1]: anchor its style, "
                "subjects, composition and scene, then develop forward "
                "(anchor -> action onset -> continuous development -> result)."
            )

        if task_type.startswith("FL2VA"):
            return (
                "Task: FL2VA. The first line of the output must be exactly:\n"
                f"How the reference pictures align with the target video - Picture 1 (from "
                f"Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 "
                f"(from Shot N) aligns with the {duration}-second mark of the target video.\n"
                "Replace N with the index of the actual final shot. Follow it with one blank "
                "line, then the core fields. Favour a single shot so the model can interpolate, "
                "and supply the motion path between the two frames rather than two static "
                "descriptions (first-frame state -> intermediate changes -> narrowing "
                "differences -> last-frame state)."
            )

        return (
            "Task: L2VA. The first line of the output must be exactly:\n"
            f"How the reference pictures align with the target video - <Picture 1> (from "
            f"[Shot N]) aligns with the {duration}-second mark of the target video.\n"
            "Replace N with the index of the actual final shot. Follow it with one blank "
            "line, then the core fields. <Picture 1> is the final frame and belongs to the "
            "last shot, not Shot 1: infer a plausible earlier state and converge onto the "
            "image (preceding state -> transition path -> gradual convergence -> landing)."
        )

    def _build_system_prompt(self):
        guide = load_guide("guide_base_en.md")
        return (
            "You are a MiniMax-H3 video prompt engineer. You rewrite a user's short idea "
            "into a complete, production-ready H3 prompt.\n\n"
            "The authoritative specification follows. Obey it exactly - field names, shot "
            "labels, timestamp formats, camera vocabulary, speaker IDs and <d> blocks all "
            "follow this document.\n\n"
            "=== BEGIN MINIMAX-H3 VIDEO PROMPT WRITING GUIDE ===\n"
            f"{guide}\n"
            "=== END MINIMAX-H3 VIDEO PROMPT WRITING GUIDE ===\n\n"
            "Output rules:\n"
            "- Emit only the finished prompt. No preamble, no commentary, no markdown "
            "fences, no headings of your own.\n"
            "- Keep the exact field names `integrated_multimodal_description:`, "
            "`overall_soundscape:` and `non_diegetic_music:`, each separated by one blank line.\n"
            "- Write everything in English except dialogue, lyrics and on-screen text, which "
            "keep their original language.\n"
            "- Never invent reference labels for a task type that does not use them."
        )

    def _build_user_prompt(
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
        extra_instructions,
        has_image,
        rng,
    ):
        directives = [self._instruction_rule(task_type, duration_seconds)]
        directives.append(shot_directive(shot_plan, duration_seconds))

        if visual_style == AUTO:
            directives.append(
                "Visual style: choose one that fits the idea"
                + (" and the attached image" if has_image else "")
                + ", and state it at the start of [Shot 1]."
            )
        else:
            directives.append(
                f"Visual style: open [Shot 1] with `{visual_style}` as the stated style."
            )

        directives.append(camera_directive(camera_motion, camera_amplitude, camera_speed))
        directives.extend(
            toggle_directives(
                include_dialogue,
                include_on_screen_text,
                include_soundscape,
                include_non_diegetic_music,
                dialogue_language,
            )
        )

        wild_lines, wild_label = wildness_directive(wildness, rng)
        directives.extend(wild_lines)

        if extra_instructions.strip():
            directives.append(f"Additional direction from the user: {extra_instructions.strip()}")

        numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(directives, 1))

        if has_image:
            source = (
                "The attached image(s) are the reference frames. Read them directly: derive "
                "style, subjects, clothing, colours, key objects and spatial relationships "
                "from what you actually see, and keep them consistent."
            )
            if idea.strip():
                source += f"\n\nThe user also wrote:\n{idea.strip()}"
        else:
            source = f"The user's idea:\n{idea.strip()}"

        return (
            f"{source}\n\n"
            f"Write the H3 prompt under these constraints:\n{numbered}\n\n"
            "Return only the finished prompt."
        ), wild_label

    def write(
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
        temperature,
        seed,
        image=None,
        extra_instructions="",
    ):
        try:
            if not idea.strip() and image is None:
                raise ValueError("Provide an idea, an image, or both.")

            current_seed = seed if seed != -1 else random.randint(0, 0xffffffffffffffff)
            rng = random.Random(current_seed)

            images = None
            if image is not None:
                images = [tensor2pil(frame) for frame in image]

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
                extra_instructions,
                images is not None,
                rng,
            )

            print(
                f"🎬 H3 Prompt Writer | {task_type} | {duration_seconds:.2f}s | "
                f"wildness {wildness} ({wild_label}) | seed {current_seed}"
            )

            text, resolved_model = call_llm(
                model,
                user_prompt,
                system_prompt=self._build_system_prompt(),
                images=images,
                temperature=temperature,
                seed=current_seed,
                max_tokens=4000,
            )

            prompt = strip_code_fence(text)

            return (
                prompt,
                extract_section(prompt, "integrated_multimodal_description", _FIELDS),
                extract_section(prompt, "overall_soundscape", _FIELDS),
                extract_section(prompt, "non_diegetic_music", _FIELDS),
                resolved_model,
            )

        except Exception as exc:
            print(f"❌ H3 Prompt Writer error: {exc}")
            import traceback

            print(traceback.format_exc())
            error_message = f"Error occurred while writing the H3 prompt: {exc}"
            return (error_message, error_message, "", "", "error")
