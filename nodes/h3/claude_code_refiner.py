# APNext H3 Claude Code Refiner
#
# Revises an existing H3 prompt in plain language - "make shot 2 wilder", "cut
# the dialogue", "keep everything but move it to night".
#
# Given a session_id from one of the Claude Code writers it resumes that
# conversation, so the model still has the guide, the reference images and its
# own reasoning in context and only the change needs describing. Without one it
# still works: the prompt is re-sent alongside the matching guide.

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
from .base_prompt_writer import H3BasePromptWriter
from .claude_code_support import (
    BASE_SKILLS,
    REF_SKILLS,
    claude_code_inputs,
    directions_with_research,
    local_llm_inputs,
    local_llm_options,
    run_h3_claude_code,
)
from .template_vars import collect_template_vars, expand_all, log_template_vars
from .common import extract_section, strip_code_fence
from .ref_prompt_writer import H3RefPromptWriter

# Every field either format can produce. Absent labels simply extract as "".
_ALL_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)


class H3ClaudeCodeRefiner:
    """
    APNext H3 Claude Code Refiner

    Rewrites an existing H3 prompt from a plain-language instruction, resuming
    the writer's Claude Code session when one is supplied so nothing has to be
    re-explained.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "h3_prompt": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "The prompt to revise. Wire the h3_prompt output of an H3 writer in here.",
            }),
            "instruction": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": (
                    "What to change, in plain language: 'make shot 2 wilder', 'cut the dialogue', "
                    "'same scene at night', 'shorten it to one shot'."
                ),
            }),
        }
        required.update(claude_code_inputs())
        required["seed"] = ("INT", {
            "default": -1, "min": -1, "max": 0xffffffffffffffff,
            "tooltip": "Controls ComfyUI caching only. -1 re-runs every queue.",
        })

        optional = {
            "session_id": ("STRING", {
                "default": "",
                "tooltip": (
                    "The writer's session_id. With it, the model still has the guide, the images "
                    "and its own reasoning in context - far cheaper and more consistent. Without "
                    "it the prompt is re-sent with the matching guide."
                ),
            }),
            "image": ("IMAGE", {
                "tooltip": "Extra reference frame(s) to introduce with this revision.",
            }),
            "working_dir": ("STRING", {"default": ""}),
        }
        optional.update(local_llm_inputs())
        optional.update(context_inputs())

        return {"required": required, "optional": optional, "hidden": context_hidden_inputs()}

    RETURN_TYPES = ("STRING",) * 10
    RETURN_NAMES = (
        "h3_prompt",
        "subject_definitions",
        "summary",
        "retention_analysis",
        "detailed_description",
        "integrated_multimodal_description",
        "overall_soundscape",
        "non_diegetic_music",
        "session_id",
        "info",
    )
    FUNCTION = "refine"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Revises an existing MiniMax-H3 prompt from a plain-language instruction, resuming "
        "the writer's Claude Code session when given its session_id."
    )

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        return float("nan") if seed == -1 else seed

    def _is_reference_format(self, prompt):
        """Full-reference rewrites open with subject_definitions; base ones do not."""
        return "subject_definitions:" in prompt

    def _system_prompt(self, prompt):
        writer = H3RefPromptWriter() if self._is_reference_format(prompt) else H3BasePromptWriter()
        return writer._build_system_prompt()

    def refine(
        self,
        h3_prompt,
        instruction,
        model,
        research,
        director,
        use_subscription,
        timeout_seconds,
        seed,
        session_id="",
        image=None,
        working_dir="",
        llm=None,
        **context_slots,
    ):
        template_vars, template_summary = collect_template_vars(context_slots)
        instruction, = expand_all(
            template_vars, instruction,
        )
        log_template_vars(template_vars, template_summary, instruction)
        context_text, context_entries = build_context(context_slots, target="the revised prompt")
        try:
            if not instruction.strip():
                raise ValueError("Describe what to change in `instruction`.")

            resuming = bool(session_id.strip())
            if not h3_prompt.strip() and not resuming:
                raise ValueError(
                    "Provide the prompt to revise, or a session_id to resume the run that wrote it."
                )

            images = [tensor2pil(frame) for frame in image] if image is not None else None
            change = directions_with_research(instruction.strip(), research)

            if resuming:
                user_prompt = (
                    "Revise the H3 prompt you just wrote.\n\n"
                    f"Change requested:\n{change}\n\n"
                    "Return the complete revised prompt, using exactly the same field labels, "
                    "shot labels and formatting rules as before. Change only what the request "
                    "implies and leave the rest intact. Output only the prompt."
                )
            else:
                user_prompt = (
                    "Revise this MiniMax-H3 prompt.\n\n"
                    "=== CURRENT PROMPT ===\n"
                    f"{h3_prompt.strip()}\n"
                    "=== END CURRENT PROMPT ===\n\n"
                    f"Change requested:\n{change}\n\n"
                    "Return the complete revised prompt, using exactly the same field labels, "
                    "shot labels and formatting rules as the current one. Change only what the "
                    "request implies and leave the rest intact. Output only the prompt."
                )

            user_prompt = with_context(user_prompt, context_text)

            print(
                f"🎬 H3 Claude Code Refiner | context: {context_summary(context_entries)} | "
                f"{'resuming ' + session_id.strip()[:8] if resuming else 'fresh session'} | "
                f"research {'on' if research else 'off'}"
            )

            text, new_session, info = run_h3_claude_code(
                None if resuming else self._system_prompt(h3_prompt),
                user_prompt,
                images,
                model,
                research,
                use_subscription,
                timeout_seconds,
                session_id,
                working_dir,
                director,
                skills=REF_SKILLS if self._is_reference_format(h3_prompt) else BASE_SKILLS,
                local=local_llm_options(llm),
            )

            prompt = strip_code_fence(text)

            return (
                prompt,
                extract_section(prompt, "subject_definitions", _ALL_FIELDS),
                extract_section(prompt, "summary", _ALL_FIELDS),
                extract_section(prompt, "retention_analysis", _ALL_FIELDS),
                extract_section(prompt, "detailed_description", _ALL_FIELDS),
                extract_section(prompt, "integrated_multimodal_description", _ALL_FIELDS),
                extract_section(prompt, "overall_soundscape", _ALL_FIELDS),
                extract_section(prompt, "non_diegetic_music", _ALL_FIELDS),
                new_session,
                info,
            )

        except Exception as exc:
            # A cancelled queue must stop the run, not become an error string.
            if is_interrupt(exc):
                raise
            print(f"❌ H3 Claude Code Refiner error: {exc}")
            import traceback

            print(traceback.format_exc())
            message = f"Error occurred while refining the H3 prompt: {exc}"
            return (message, "", "", "", "", "", "", "", "", "error")
