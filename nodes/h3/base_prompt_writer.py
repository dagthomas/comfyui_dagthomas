# APNext H3 Prompt Writer
#
# Turns a short idea (and optionally a reference image) into a complete
# MiniMax-H3 video prompt in the T2VA / I2VA / FL2VA / L2VA format, using the
# official writing guide as the system prompt.

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
    AUTO,
    CAMERA_AMPLITUDES,
    CAMERA_MOTIONS,
    CAMERA_SPEEDS,
    DIALOGUE_LANGUAGES,
    SHOT_PLANS,
    VISUAL_STYLES,
    resolve_visual_style,
    camera_directive,
    resolve_dialogue_language,
    extract_section,
    load_guide,
    shot_directive,
    strip_code_fence,
    toggle_directives,
    wildness_directive,
)
from .template_vars import collect_template_vars, expand_all, log_template_vars

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


def warn_if_images_unused(task_type, image_count, reference_count=0):
    """
    Say out loud when the images will not become <Picture N>: T2VA never labels a
    picture, and typed references never do regardless of task type.
    """
    if image_count and task_type.startswith("T2VA"):
        print(
            f"⚠️  H3 task_type is T2VA, so the {image_count} image(s) on `image` are context "
            "only and no <Picture 1> will appear. Pick I2VA / L2VA / FL2VA to use one as a "
            "keyframe, or move it to a subject/scenery/object socket to have it described."
        )
    if reference_count and not image_count and not task_type.startswith("T2VA"):
        print(
            f"ℹ️  Only typed references are attached ({reference_count}); nothing is on `image` "
            f"to align, so the prompt is written as T2VA even though task_type is {task_type}."
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
                    "tooltip": (
                        "Style stated at the start of [Shot 1]. Auto derives it from the idea or the attached image. "
                        "The list is the guide's styles plus the APNext Cinematic vocabulary (film stock, grading, aesthetics); pick Custom and fill in custom_visual_style to write your own."
                    ),
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
                "dialogue_language": (DIALOGUE_LANGUAGES, {
                    "default": "English",
                    "tooltip": (
                        "The language the characters actually speak, and the tag written inside "
                        "<d>[...]</d>. Auto lets the model pick one that fits the setting. Pick "
                        "Custom (or just fill in custom_dialogue_language) for anything not listed."
                    ),
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
                    "tooltip": (
                        "Which LLM writes the prompt. auto-detect picks the first provider with an "
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
                "image": ("IMAGE", {
                    "tooltip": (
                        "Reference frame(s), sent to the model as vision input. I2VA: the first "
                        "frame. L2VA: the last frame. FL2VA: batch both (frame 0 = first, last = "
                        "last). The first_frame / last_frame outputs hand them back for the H3 "
                        "video node."
                    ),
                }),
                "extra_instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Free-form extra direction appended to the request.",
                }),
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
            },
            "hidden": context_hidden_inputs(),
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "h3_prompt",
        "integrated_multimodal_description",
        "overall_soundscape",
        "non_diegetic_music",
        "model_used",
        "first_frame",
        "last_frame",
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

    def _reference_rule(self, image_count, references):
        """
        The typed references (subject / scenery / object) and how to use each.
        `references` is the ordered list of labels, e.g. ["Subject 1", "Scenery 1"].
        They are attached AFTER the `image` batch, so their attachment numbers
        start at image_count + 1. Returns "" when there are none.
        """
        if not references:
            return ""

        how = {
            "Subject": (
                "identity only. Describe the person/creature/character precisely and "
                "repeatably in [Shot 1] - species/gender/age impression, face, hair, skin, "
                "build, wardrobe with colours and materials, accessories, distinctive marks - "
                "and keep it identical in every shot. IGNORE the photo's backdrop, location, "
                "lighting, weather, camera angle and framing; the scene comes from the idea "
                "and the other references."
            ),
            "Scenery": (
                "setting only. Describe the place - architecture or terrain, materials, light "
                "direction and quality, weather, time of day, palette, mood, spatial layout - "
                "as the world the video happens in. IGNORE any people in it."
            ),
            "Object": (
                "a prop to depict faithfully - shape, colour, material, markings, scale. "
                "IGNORE where it sits in the photo; place it where the idea needs it."
            ),
        }
        mapping = ", ".join(
            f"attached image {offset} = {label}"
            for offset, label in enumerate(references, image_count + 1)
        )
        kinds_present = []
        for label in references:
            kind = label.split()[0]
            if kind not in kinds_present:
                kinds_present.append(kind)
        rules = "\n".join(f"   - {kind}: {how[kind]}" for kind in kinds_present)

        return (
            "Typed references. The video model NEVER sees these pictures - your words are all "
            "it gets - so put what matters into text, and never write an alignment line or a "
            f"<Picture N> label for them. {mapping}.\n{rules}"
        )

    def _image_mapping_rule(self, task_type, image_count):
        """
        Which attached image plays which role. The base format knows at most two
        keyframes (FL2VA); every further image is visual context, never a
        <Picture N>. Returns "" when nothing is attached.
        """
        if image_count <= 0:
            return ""

        if task_type.startswith("T2VA"):
            keyframes = (
                "In T2VA no image is a keyframe: treat every attached image as visual "
                "context to describe from (characters, wardrobe, location, palette, mood), "
                "with no alignment line and no <Picture N> labels."
            )
        elif task_type.startswith("FL2VA"):
            keyframes = (
                "Image 1 is <Picture 1>, the exact opening frame; image 2 is <Picture 2>, the "
                "exact final frame."
            )
        elif task_type.startswith("L2VA"):
            keyframes = "Image 1 is <Picture 1>, the exact final frame."
        else:
            keyframes = "Image 1 is <Picture 1>, the exact opening frame."

        extras = ""
        keyframe_count = 0 if task_type.startswith("T2VA") else (2 if task_type.startswith("FL2VA") else 1)
        if image_count > keyframe_count and keyframe_count:
            span = (
                f"Image {image_count} is a supporting reference"
                if image_count == keyframe_count + 1
                else f"Images {keyframe_count + 1}-{image_count} are supporting references"
            )
            extras = (
                f" {span} only "
                "(a character sheet, a location, a lookbook, a prop): use them to keep "
                "identity, wardrobe, colours and setting consistent, but never label them "
                "<Picture N> and never treat them as frames of the video."
            )

        return (
            f"Attached images ({image_count}, numbered in the order given): {keyframes}{extras}"
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
        image_count,
        rng,
        references=(),
    ):
        # `image_count` is how many keyframe/context images sit on `image`
        # (older callers pass a bool; True means one). `references` lists the
        # typed subject/scenery/object labels attached after them.
        image_count = int(image_count)
        references = list(references or [])
        has_image = image_count > 0
        has_refs = bool(references)
        # Nothing on `image` means nothing to align: references alone are T2VA.
        effective_task = task_type if has_image or not has_refs else TASK_TYPES[0]

        directives = [self._instruction_rule(effective_task, duration_seconds)]
        mapping = self._image_mapping_rule(task_type, image_count)
        if mapping:
            directives.append(mapping)
        reference_rule = self._reference_rule(image_count, references)
        if reference_rule:
            directives.append(reference_rule)
        directives.append(shot_directive(shot_plan, duration_seconds))

        if visual_style == AUTO:
            if has_image:
                style_hint = " and the attached image"
            elif has_refs:
                style_hint = (
                    " (a subject reference does not dictate style - only the subject carries "
                    "over; a scenery reference may)"
                )
            else:
                style_hint = ""
            directives.append(
                f"Visual style: choose one that fits the idea{style_hint}, and state it at "
                "the start of [Shot 1]."
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
                "The attached image(s) are the reference frames, in order. Read them "
                "directly: derive style, subjects, clothing, colours, key objects and spatial "
                "relationships from what you actually see, and keep them consistent."
            )
            if has_refs:
                source += (
                    f" The last {len(references)} attached image(s) are typed references "
                    "(see the constraints), not frames."
                )
            if idea.strip():
                source += f"\n\nThe user also wrote:\n{idea.strip()}"
        elif has_refs:
            source = (
                "The attached images are typed references - who is in the video, where it "
                "happens, what props appear - and nothing more. Look closely and put each one "
                "into words so the video model can reproduce it from text alone; the story, "
                "action and everything not covered by a reference come from the user's idea."
            )
            if idea.strip():
                source += f"\n\nThe user's idea:\n{idea.strip()}"
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
        custom_dialogue_language="",
        custom_visual_style="",
        model_override="",
        local_base_url="",
        **context_slots,
    ):
        # Frame 0 and the final frame of the batch go back out, so the H3 video
        # node's first_frame / last_frame can be wired straight from this node.
        frames = (image[0:1], image[-1:]) if image is not None else (None, None)
        template_vars, template_summary = collect_template_vars(context_slots)
        idea, extra_instructions, custom_dialogue_language, custom_visual_style = expand_all(
            template_vars, idea, extra_instructions, custom_dialogue_language, custom_visual_style
        )
        log_template_vars(template_vars, template_summary, idea, extra_instructions, custom_dialogue_language, custom_visual_style)
        context_text, context_entries = build_context(context_slots)
        try:
            if not idea.strip() and image is None and not context_entries:
                raise ValueError("Provide an idea, an image, or both.")

            dialogue_language = resolve_dialogue_language(
                dialogue_language, custom_dialogue_language
            )

            visual_style = resolve_visual_style(visual_style, custom_visual_style)

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
                len(images) if images else 0,
                rng,
            )
            user_prompt = with_context(user_prompt, context_text)
            warn_if_images_unused(task_type, len(images) if images else 0)

            print(
                f"🎬 H3 Prompt Writer | {task_type} | {duration_seconds:.2f}s | "
                f"context: {context_summary(context_entries)} | "
                f"wildness {wildness} ({wild_label}) | seed {current_seed}"
            )

            text, resolved_model = call_llm(
                model_override.strip() or model,
                user_prompt,
                system_prompt=self._build_system_prompt(),
                images=images,
                temperature=temperature,
                seed=current_seed,
                max_tokens=4000,
                base_url=local_base_url.strip() or None,
            )

            prompt = strip_code_fence(text)

            return (
                prompt,
                extract_section(prompt, "integrated_multimodal_description", _FIELDS),
                extract_section(prompt, "overall_soundscape", _FIELDS),
                extract_section(prompt, "non_diegetic_music", _FIELDS),
                resolved_model,
            ) + frames

        except Exception as exc:
            # A cancelled queue must stop the run, not become an error string.
            if is_interrupt(exc):
                raise
            print(f"❌ H3 Prompt Writer error: {exc}")
            import traceback

            print(traceback.format_exc())
            error_message = f"Error occurred while writing the H3 prompt: {exc}"
            return (error_message, error_message, "", "", "error") + frames
