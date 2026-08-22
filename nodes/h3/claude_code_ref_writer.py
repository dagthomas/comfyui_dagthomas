# APNext H3 Claude Code Reference Writer
#
# The full-reference H3 writer (six sections) driven by the local Claude Code
# CLI. Inherits every prompt-building rule from H3RefPromptWriter so the two
# guides, reference labelling and wildness bands stay in one place.

import random

from ...utils.claude_code import is_interrupt
from ...utils.apnext_context import (
    build_context,
    context_hidden_inputs,
    context_inputs,
    context_summary,
    with_context,
)
from ...utils.constants import CUSTOM_CATEGORY
from ...utils.image_utils import tensor2pil
from .claude_code_support import (
    REF_SKILLS,
    claude_code_inputs,
    claude_code_optional_inputs,
    local_llm_inputs,
    local_llm_options,
    directions_with_research,
    run_h3_claude_code,
)
from .template_vars import collect_template_vars, expand_all, log_template_vars
from .common import (
    scale_reference_passthrough,
    AUTO,
    CAMERA_AMPLITUDES,
    CAMERA_MOTIONS,
    CAMERA_SPEEDS,
    DIALOGUE_LANGUAGES,
    SHOT_PLANS,
    VISUAL_STYLES,
    resolve_visual_style,
    collect_reference_images,
    extract_section,
    reference_image_inputs,
    reference_image_outputs,
    resolve_dialogue_language,
    strip_code_fence,
)
from .ref_prompt_writer import (
    REFERENCE_ROLES,
    TASK_TYPES,
    H3RefPromptWriter,
    _FIELDS,
)


class H3ClaudeCodeRefWriter(H3RefPromptWriter):
    """
    APNext H3 Claude Code Reference Writer

    Writes the six-section MiniMax-H3 full-reference rewrite with the local
    Claude Code CLI, using its own login rather than an API key.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "idea": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Your short prompt: what should happen in the target video.",
            }),
            "task_type": (TASK_TYPES, {
                "default": "Auto (decide from the references)",
                "tooltip": "Square-bracketed prefix of the summary section. Auto lets the model combine types with ' + '.",
            }),
            "reference_role": (REFERENCE_ROLES, {
                "default": "Auto (decide per image)",
                "tooltip": "How the attached images should be labelled in subject_definitions.",
            }),
            "duration_seconds": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 60.0, "step": 0.5}),
            "shot_plan": (SHOT_PLANS, {"default": AUTO}),
            "visual_style": (VISUAL_STYLES, {
                "default": AUTO,
                "tooltip": (
                    "In full-reference mode the style is stated in one or two sentences BEFORE [Shot 1]. "
                    "The list is the guide's styles plus the APNext Cinematic vocabulary (film stock, grading, aesthetics); pick Custom and fill in custom_visual_style to write your own."
                ),
            }),
            "wildness": ("INT", {
                "default": 25, "min": 0, "max": 100, "step": 1,
                "tooltip": "0 = literal and conservative, 100 = fully unhinged. Above 40 the node also injects concrete surreal elements picked from the seed.",
            }),
            "word_target": ("INT", {
                "default": 425, "min": 150, "max": 1200, "step": 25,
                "tooltip": "Target length of detailed_description. The guide recommends 350-500 words for generation tasks.",
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
            "reference_notes": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Optional per-reference notes, one per line, e.g. 'Image 1: the woman, keep her cardigan'. Also use this to describe video or audio references you cannot attach.",
            }),
            "extra_instructions": ("STRING", {"multiline": True, "default": ""}),
            "custom_dialogue_language": ("STRING", {
                "default": "",
                "tooltip": (
                    "Any language or dialect not in the dropdown, e.g. 'Norwegian (Bergen "
                    "dialect)' or 'Latin'. Overrides the dropdown when filled in."
                ),
            }),
            "custom_visual_style": ("STRING", {
                "default": "",
                "tooltip": (
                    "Any visual style not in the dropdown, e.g. 'hand-painted cel animation' or "
                    "'Kodak Vision3 500T, anamorphic'. Overrides the dropdown when filled in."
                ),
            }),
        }
        optional.update(claude_code_optional_inputs())
        optional.update(local_llm_inputs())
        # Image sockets last, so the front-end can grow and trim them at the tail.
        optional.update(context_inputs())
        optional.update(reference_image_inputs())

        return {"required": required, "optional": optional, "hidden": context_hidden_inputs()}

    _IMAGE_OUTPUT_TYPES, _IMAGE_OUTPUT_NAMES = reference_image_outputs()
    RETURN_TYPES = ("STRING",) * 9 + _IMAGE_OUTPUT_TYPES
    RETURN_NAMES = (
        "h3_prompt",
        "subject_definitions",
        "summary",
        "retention_analysis",
        "detailed_description",
        "overall_soundscape",
        "non_diegetic_music",
        "session_id",
        "info",
    ) + _IMAGE_OUTPUT_NAMES
    FUNCTION = "write_with_claude_code"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Writes the six-section MiniMax-H3 full-reference rewrite using the local Claude "
        "Code CLI and its own login. Optionally researches real references first."
    )

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        return float("nan") if seed == -1 else seed

    def write_with_claude_code(
        self,
        idea,
        task_type,
        reference_role,
        duration_seconds,
        shot_plan,
        visual_style,
        wildness,
        word_target,
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
        director,
        use_subscription,
        timeout_seconds,
        seed,
        reference_notes="",
        extra_instructions="",
        custom_dialogue_language="",
        custom_visual_style="",
        resume_session_id="",
        working_dir="",
        llm=None,
        **image_slots,
    ):
        # The same tensors go back out on image_1..image_9 so the video node
        # can be wired from this node and numbering can never drift.
        template_vars, template_summary = collect_template_vars(image_slots)
        idea, reference_notes, extra_instructions, custom_dialogue_language, custom_visual_style = expand_all(
            template_vars, idea, reference_notes, extra_instructions, custom_dialogue_language, custom_visual_style
        )
        log_template_vars(template_vars, template_summary, idea, reference_notes, extra_instructions, custom_dialogue_language, custom_visual_style)
        context_text, context_entries = build_context(image_slots)
        passthrough = scale_reference_passthrough(image_slots, self._IMAGE_OUTPUT_NAMES)
        try:
            dialogue_language = resolve_dialogue_language(
                dialogue_language, custom_dialogue_language
            )
            visual_style = resolve_visual_style(visual_style, custom_visual_style)

            images = [pil for _, pil in collect_reference_images(passthrough, tensor2pil)]

            if not idea.strip() and not images and not reference_notes.strip():
                raise ValueError("Provide an idea, at least one reference image, or reference notes.")

            current_seed = seed if seed != -1 else random.randint(0, 0xffffffffffffffff)
            rng = random.Random(current_seed)

            user_prompt, wild_label = self._build_user_prompt(
                idea,
                task_type,
                reference_role,
                duration_seconds,
                shot_plan,
                visual_style,
                wildness,
                word_target,
                camera_motion,
                camera_amplitude,
                camera_speed,
                include_dialogue,
                dialogue_language,
                include_on_screen_text,
                include_non_diegetic_music,
                include_soundscape,
                reference_notes,
                directions_with_research(extra_instructions, research),
                len(images),
                rng,
            )

            user_prompt = with_context(user_prompt, context_text)

            print(
                f"🎬 H3 Claude Code Reference Writer | {len(images)} image(s) | "
                f"context: {context_summary(context_entries)} | "
                f"{duration_seconds:.2f}s | wildness {wildness} ({wild_label}) | "
                f"research {'on' if research else 'off'} | director "
                f"{'on' if director else 'off'} | seed {current_seed}"
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
                director,
                skills=REF_SKILLS,
                local=local_llm_options(llm),
            )

            prompt = strip_code_fence(text)

            return (
                prompt,
                extract_section(prompt, "subject_definitions", _FIELDS),
                extract_section(prompt, "summary", _FIELDS),
                extract_section(prompt, "retention_analysis", _FIELDS),
                extract_section(prompt, "detailed_description", _FIELDS),
                extract_section(prompt, "overall_soundscape", _FIELDS),
                extract_section(prompt, "non_diegetic_music", _FIELDS),
                session_id,
                info,
            ) + passthrough

        except Exception as exc:
            # A cancelled queue must stop the run, not become an error string.
            if is_interrupt(exc):
                raise
            print(f"❌ H3 Claude Code Reference Writer error: {exc}")
            import traceback

            print(traceback.format_exc())
            message = f"Error occurred while writing the H3 reference prompt: {exc}"
            return (message, message, "", "", "", "", "", "", "error") + passthrough
