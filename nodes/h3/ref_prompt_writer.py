# APNext H3 Reference Prompt Writer
#
# Writes a MiniMax-H3 full-reference rewrite (six sections) from a short idea
# plus up to nine reference images, using the official full-reference guide.

import random

from ...utils.apnext_context import (
    build_context,
    context_hidden_inputs,
    context_inputs,
    context_summary,
    with_context,
)
from ...utils.constants import CUSTOM_CATEGORY
from ...utils.image_utils import tensor2pil
from ...utils.llm_router import AUTO_DETECT, call_llm, is_interrupt, list_all_models
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
    camera_directive,
    collect_reference_images,
    resolve_dialogue_language,
    extract_section,
    load_guide,
    reference_image_inputs,
    reference_image_outputs,
    shot_directive,
    strip_code_fence,
    toggle_directives,
    wildness_directive,
)
from .template_vars import collect_template_vars, expand_all, log_template_vars

TASK_TYPES = [
    "Auto (decide from the references)",
    "keyframe completion",
    "reference generation",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
]

# How each attached image should be labelled in subject_definitions.
REFERENCE_ROLES = [
    "Auto (decide per image)",
    "Subject (character / object / scene to reuse)",
    "Picture (concrete frame anchor)",
    "Style reference only",
    "Storyboard / shot-planning reference",
]

_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)


class H3RefPromptWriter:
    """
    APNext H3 Reference Prompt Writer

    Writes a MiniMax-H3 full-reference rewrite - subject_definitions, summary,
    retention_analysis, detailed_description, overall_soundscape and
    non_diegetic_music - from a short idea plus reference images, following the
    official Full-Reference Mode Rewrite Output Format Guide.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
                "duration_seconds": ("FLOAT", {
                    "default": 8.0, "min": 1.0, "max": 60.0, "step": 0.5,
                }),
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
                "model": (list_all_models(), {
                    "default": AUTO_DETECT,
                    "tooltip": (
                        "Which LLM writes the rewrite. auto-detect picks the first provider with an "
                        "API key set, then the Claude Code CLI, then a running local server. "
                        "claudecode: entries use your Claude Code login instead of an API key; "
                        "ollama:/lmstudio:/local: entries are whatever your local servers were "
                        "serving when the page loaded."
                    ),
                }),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
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
                "model_override": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Exact provider:model string, used instead of the dropdown when filled in. "
                        "Handy for a local model the dropdown has not discovered, e.g. "
                        "'ollama:qwen3:8b', 'lmstudio:qwen/qwen3-8b' or 'local:my-model'."
                    ),
                }),
                "local_base_url": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Where to reach the local server, e.g. 'http://192.168.1.10:11434'. Empty "
                        "uses the default for the chosen prefix: ollama 11434, lmstudio 1234, "
                        "local 8000. Ignored by the cloud providers."
                    ),
                }),
                **context_inputs(),
                # Image sockets last, so the front-end can grow and trim them at the tail.
                **reference_image_inputs(),
            },
            "hidden": context_hidden_inputs(),
        }

    _IMAGE_OUTPUT_TYPES, _IMAGE_OUTPUT_NAMES = reference_image_outputs()
    RETURN_TYPES = ("STRING",) * 8 + _IMAGE_OUTPUT_TYPES
    RETURN_NAMES = (
        "h3_prompt",
        "subject_definitions",
        "summary",
        "retention_analysis",
        "detailed_description",
        "overall_soundscape",
        "non_diegetic_music",
        "model_used",
    ) + _IMAGE_OUTPUT_NAMES
    FUNCTION = "write"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Writes a MiniMax-H3 full-reference rewrite (six sections) from a short idea plus "
        "reference images, following the official full-reference format guide."
    )

    def _build_system_prompt(self):
        guide = load_guide("guide_ref_en.md")
        base_guide = load_guide("guide_base_en.md")

        return (
            "You are a MiniMax-H3 video prompt engineer working in full-reference mode. You "
            "rewrite a user's short idea plus their reference assets into a complete "
            "six-section H3 rewrite.\n\n"
            "Two authoritative specifications follow. The full-reference guide governs the "
            "output structure and reference labels; the base guide governs shots, camera "
            "vocabulary, speakers, dialogue and sound. Obey both exactly.\n\n"
            "=== BEGIN FULL-REFERENCE MODE GUIDE ===\n"
            f"{guide}\n"
            "=== END FULL-REFERENCE MODE GUIDE ===\n\n"
            "=== BEGIN BASE VIDEO PROMPT WRITING GUIDE ===\n"
            f"{base_guide}\n"
            "=== END BASE VIDEO PROMPT WRITING GUIDE ===\n\n"
            "Output rules:\n"
            "- Emit only the finished rewrite. No preamble, no commentary, no markdown "
            "fences, no headings of your own.\n"
            "- Emit all six sections, in this order, with these exact labels: "
            "`subject_definitions:`, `summary:`, `retention_analysis:`, "
            "`detailed_description:`, `overall_soundscape:`, `non_diegetic_music:`.\n"
            "- A reference label keeps one meaning across every section. Do not introduce "
            "new labels in `summary`.\n"
            "- Do not write (Sx) speaker IDs in `retention_analysis`.\n"
            "- Write everything in English except dialogue, lyrics and on-screen text."
        )

    def _reference_directive(self, reference_role, image_count, reference_notes):
        if image_count == 0:
            base = (
                "No images are attached. Build the reference labels from the user's notes "
                "below; if there are no usable references either, define the minimum set of "
                "<Subject N> entries the idea implies and say so honestly in the summary."
            )
        else:
            listing = ", ".join(f"image {i}" for i in range(1, image_count + 1))
            base = (
                f"{image_count} reference image(s) are attached in order ({listing}). Read them "
                "directly and define them in subject_definitions. Attached image k IS "
                "<Picture k>: the same numbering the ComfyUI H3 video node uses for its "
                "reference images in connection order. Whenever a definition draws on an "
                "attached image, cite it as <Picture k> with that exact number; never renumber, "
                "skip, or merge pictures."
            )

            if reference_role.startswith("Auto"):
                base += (
                    " Decide per image whether it is a <Subject N> (reusable visible content), "
                    "a standalone <Picture N> (a concrete frame or composition anchor), or only "
                    "a source cited inside another item's definition."
                )
            elif reference_role.startswith("Subject"):
                base += (
                    " Treat each one as reusable visible content: define a <Subject N> per image "
                    "and cite the image inside that definition rather than creating standalone "
                    "<Picture N> entries."
                )
            elif reference_role.startswith("Picture"):
                base += (
                    " Treat each one as a concrete frame anchor: define a standalone <Picture N> "
                    "per image and state which shot and which position (first frame, keyframe, "
                    "last frame) it anchors."
                )
            elif reference_role.startswith("Style"):
                base += (
                    " Treat them as style references only: do not create standalone <Picture N> "
                    "entries, and fold the style provenance into the relevant <Subject N> "
                    "definitions and the style sentence before [Shot 1]."
                )
            else:
                base += (
                    " Treat them as storyboard / shot-planning references: define standalone "
                    "<Picture N> entries stating which shots they map to and what planning "
                    "information they provide."
                )

        if reference_notes.strip():
            base += f"\n   User notes on the references:\n   {reference_notes.strip()}"

        return base

    def _build_user_prompt(
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
        include_non_diegetic_music,
        include_soundscape,
        reference_notes,
        extra_instructions,
        image_count,
        rng,
    ):
        directives = [self._reference_directive(reference_role, image_count, reference_notes)]

        if task_type.startswith("Auto"):
            directives.append(
                "Task type: choose the prefix that matches the actual role each reference "
                "plays, combining several with ' + ' when they apply. Do not add a type just "
                "because an asset exists."
            )
        else:
            directives.append(
                f"Task type: the summary begins with `[{task_type}]`, extended with ' + ' only "
                "if another relationship genuinely applies."
            )

        directives.append(shot_directive(shot_plan, duration_seconds))

        if visual_style == AUTO:
            directives.append(
                "Visual style: choose one that fits the idea and the references, and state it "
                "in one or two English sentences before [Shot 1]."
            )
        else:
            directives.append(
                f"Visual style: state `{visual_style}` in one or two English sentences before "
                "[Shot 1], not inside the shot label."
            )

        directives.append(camera_directive(camera_motion, camera_amplitude, camera_speed))
        directives.append(
            f"Length: aim for roughly {word_target} English words in detailed_description, "
            "distributing detail across the shots by information load. Fitting a complete "
            "spoken timeline matters more than hitting the number exactly."
        )
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

        idea_text = idea.strip() or "(no written idea - build the target video from the references)"

        return (
            f"The user's idea:\n{idea_text}\n\n"
            f"Write the full-reference H3 rewrite under these constraints:\n{numbered}\n\n"
            "Return only the six finished sections."
        ), wild_label

    def write(
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
        temperature,
        seed,
        reference_notes="",
        extra_instructions="",
        custom_dialogue_language="",
        custom_visual_style="",
        model_override="",
        local_base_url="",
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

            if not idea.strip() and not images and not reference_notes.strip() and not context_entries:
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
                extra_instructions,
                len(images),
                rng,
            )

            user_prompt = with_context(user_prompt, context_text)

            print(
                f"🎬 H3 Reference Prompt Writer | {len(images)} image(s) | "
                f"context: {context_summary(context_entries)} | "
                f"{duration_seconds:.2f}s | wildness {wildness} ({wild_label}) | seed {current_seed}"
            )

            text, resolved_model = call_llm(
                model_override.strip() or model,
                user_prompt,
                system_prompt=self._build_system_prompt(),
                images=images or None,
                temperature=temperature,
                seed=current_seed,
                max_tokens=6000,
                base_url=local_base_url.strip() or None,
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
                resolved_model,
            ) + passthrough

        except Exception as exc:
            # A cancelled queue must stop the run, not become an error string.
            if is_interrupt(exc):
                raise
            print(f"❌ H3 Reference Prompt Writer error: {exc}")
            import traceback

            print(traceback.format_exc())
            error_message = f"Error occurred while writing the H3 reference prompt: {exc}"
            return (error_message, error_message, "", "", "", "", "", "error") + passthrough
