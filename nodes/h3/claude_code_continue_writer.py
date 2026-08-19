# APNext H3 Claude Code Continue Writer
#
# Writes the NEXT clip. Feed it the frames of a clip that H3 already generated
# (a decoded video batch), and it keeps the last N of them, shows them to Claude
# Code together with the prompt that made the clip, and asks for a fresh H3
# prompt that carries the scene on for another X seconds.
#
# The very last frame is the new clip's first frame, so the output is an I2VA
# prompt where <Picture 1> is that frame - and the frame itself comes back out
# on `first_frame` for the video node. The earlier frames are context only: they
# tell the model which way things were moving, so the continuation does not
# reverse a pan or freeze an action mid-swing.
#
# Chain it: clip 1 -> Continue Writer -> clip 2 -> Continue Writer -> clip 3...
# Give it the previous node's session_id and Claude Code still remembers the
# whole story so far.

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
from .base_prompt_writer import H3BasePromptWriter, _FIELDS
from .claude_code_support import (
    BASE_SKILLS,
    claude_code_inputs,
    directions_with_research,
    local_llm_inputs,
    local_llm_options,
    run_h3_claude_code,
)
from .template_vars import collect_template_vars, expand_all, log_template_vars
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
    extract_section,
    resolve_dialogue_language,
    shot_directive,
    strip_code_fence,
    toggle_directives,
    wildness_directive,
)

CONTINUATION_MODES = [
    "I2VA (last frame becomes the first frame)",
    "T2VA (new scene, frames are context only)",
]

# The whole clip is rarely needed. Sending every frame of a 6 s clip is 140+
# near-identical images; the CLI has a hard cap on attachments anyway.
MAX_CONTEXT_FRAMES = 20


def select_last_frames(image, frame_count, frame_stride):
    """
    Indices of the frames to send: the last `frame_count` frames of the batch,
    `frame_stride` apart, always ending on the very last frame. Returned in
    chronological order.
    """
    total = int(image.shape[0])
    stride = max(1, int(frame_stride))
    wanted = max(1, min(int(frame_count), MAX_CONTEXT_FRAMES))
    last = total - 1
    picks = [last - k * stride for k in range(wanted)]
    picks = [index for index in picks if index >= 0]
    return sorted(set(picks))


class H3ClaudeCodeContinueWriter(H3BasePromptWriter):
    """
    APNext H3 Claude Code Continue Writer

    Takes the last frames of a generated clip plus the prompt that made it and
    writes the H3 prompt for the next clip with the local Claude Code CLI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "frames": ("IMAGE", {
                "tooltip": (
                    "The frames of the clip to continue - wire a decoded video batch in here. "
                    "Only the last frames are sent to Claude Code; the very last one is the "
                    "next clip's first frame."
                ),
            }),
            "frame_count": ("INT", {
                "default": 4, "min": 1, "max": MAX_CONTEXT_FRAMES, "step": 1,
                "tooltip": (
                    "How many of the last frames to show Claude Code. The final frame is always "
                    "included; the ones before it only show which way things were moving."
                ),
            }),
            "frame_stride": ("INT", {
                "default": 6, "min": 1, "max": 120, "step": 1,
                "tooltip": (
                    "Gap between the sampled frames. 1 = strictly consecutive frames (nearly "
                    "identical at 24 fps); 6 = one every quarter second. Counting back from "
                    "the last frame."
                ),
            }),
            "idea": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": (
                    "What should happen next. Leave empty and Claude Code decides how the "
                    "scene naturally carries on."
                ),
            }),
            "continuation_mode": (CONTINUATION_MODES, {
                "default": CONTINUATION_MODES[0],
                "tooltip": (
                    "I2VA: the last frame is <Picture 1> at 0.00s of the new clip - wire "
                    "first_frame into the video node. T2VA: cut to a fresh scene that follows "
                    "on; the frames only inform look and story."
                ),
            }),
            "duration_seconds": ("FLOAT", {
                "default": 6.0, "min": 1.0, "max": 60.0, "step": 0.5,
                "tooltip": "Length of the NEW clip.",
            }),
            "shot_plan": (SHOT_PLANS, {"default": AUTO}),
            "visual_style": (VISUAL_STYLES, {
                "default": AUTO,
                "tooltip": (
                    "Auto keeps the style of the previous clip, which is almost always what you want. "
                    "The list is the guide's styles plus the APNext Cinematic vocabulary (film stock, grading, aesthetics); pick Custom and fill in custom_visual_style to write your own."
                ),
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
            "previous_prompt": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": (
                    "The H3 prompt that generated the clip you are continuing - wire the "
                    "writer's h3_prompt output in here. For a longer chain, paste every "
                    "prompt so far, oldest first. Not needed when resume_session_id is set."
                ),
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
            "resume_session_id": ("STRING", {
                "default": "",
                "tooltip": (
                    "The session_id of the writer (or the previous Continue Writer). Claude "
                    "Code then still has the guide, the earlier frames and every prompt so far "
                    "in context, so the story stays consistent across many clips."
                ),
            }),
            "working_dir": ("STRING", {
                "default": "",
                "tooltip": (
                    "A folder Claude Code may read while writing - a script, a shot list, lookbook "
                    "notes. Empty uses a throwaway scratch folder, which is the safe default."
                ),
            }),
        }

        optional.update(local_llm_inputs())
        optional.update(context_inputs())

        return {"required": required, "optional": optional, "hidden": context_hidden_inputs()}

    RETURN_TYPES = ("STRING",) * 6 + ("IMAGE", "IMAGE")
    RETURN_NAMES = (
        "h3_prompt",
        "integrated_multimodal_description",
        "overall_soundscape",
        "non_diegetic_music",
        "session_id",
        "info",
        "first_frame",
        "context_frames",
    )
    FUNCTION = "continue_with_claude_code"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Continues a generated clip: shows Claude Code the last frames and the previous "
        "prompt, and writes the MiniMax-H3 prompt for the next clip. first_frame is the last "
        "frame of the input, ready for the H3 video node."
    )

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        return float("nan") if seed == -1 else seed

    def _continuation_rule(self, continuation_mode, frame_count):
        """The task line: how the attached frames relate to the new clip."""
        if frame_count == 1:
            frames_line = (
                "The single attached image is the final frame of the previous clip."
            )
        else:
            frames_line = (
                f"The {frame_count} attached images are the last {frame_count} frames of the "
                "previous clip, in chronological order and evenly spaced; the LAST image is the "
                "clip's final frame. Read the sequence for motion: where the camera was heading, "
                "what each subject was mid-way through doing, how the light was changing. The "
                "earlier frames are context only - never label them <Picture N> and never "
                "describe them as part of the new video."
            )

        if continuation_mode.startswith("I2VA"):
            task = (
                "Task: I2VA. The final frame of the previous clip is <Picture 1> and is the "
                "exact opening frame of the new clip. The first line of the output must be "
                "exactly:\n"
                "For the target video, at 0.00 seconds into the target video, <Picture 1> "
                "(from [Shot 1]) is fully referenced.\n"
                "Follow it with one blank line, then the core fields. Anchor [Shot 1] on that "
                "frame - same subjects, wardrobe, framing, light and scene - and carry the "
                "motion forward without a jump: whatever was in progress keeps going "
                "(anchor -> action onset -> continuous development -> result)."
            )
        else:
            task = (
                "Task: T2VA. There is no image-alignment instruction and no <Picture N> label. "
                "Begin the output directly with `integrated_multimodal_description:`. The new "
                "clip is the next scene of the same story: it cuts to a new moment or place "
                "that follows on from where the previous clip ended, keeping the same "
                "characters, look and world."
            )

        return f"{frames_line}\n{task}"

    def _build_continue_prompt(
        self,
        idea,
        previous_prompt,
        resuming,
        continuation_mode,
        frame_count,
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
        rng,
    ):
        directives = [self._continuation_rule(continuation_mode, frame_count)]
        directives.append(
            "Continuity: this is a continuation, not a restart. Keep every character's "
            "identity, wardrobe, props and the geography of the space exactly as established. "
            "Reuse the same (S1)/(S2) speaker IDs for the same people. Keep the soundscape "
            "and the music consistent with the previous clip unless the story turns. Do not "
            "re-describe or replay anything that already happened; move the story forward "
            "with a new beat and end somewhere a further clip could pick up from."
        )
        directives.append(shot_directive(shot_plan, duration_seconds))

        if visual_style == AUTO:
            directives.append(
                "Visual style: keep the exact style of the previous clip as seen in the frames, "
                "and state it at the start of [Shot 1]."
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

        if resuming:
            opening = (
                "The clip from the prompt you wrote has been generated. Attached are its last "
                "frames. Now write the H3 prompt for the NEXT clip."
            )
        else:
            opening = (
                "A MiniMax-H3 clip has been generated and its last frames are attached. Write "
                "the H3 prompt for the NEXT clip, which continues it."
            )

        parts = [opening]
        if previous_prompt.strip():
            parts.append(
                "=== PROMPT(S) THAT MADE THE PREVIOUS CLIP(S), OLDEST FIRST ===\n"
                f"{previous_prompt.strip()}\n"
                "=== END PREVIOUS PROMPT(S) ==="
            )
        elif not resuming:
            parts.append(
                "No previous prompt is available: infer the story, style and characters from "
                "the frames alone."
            )

        if idea.strip():
            parts.append(f"What should happen next, from the user:\n{idea.strip()}")
        else:
            parts.append(
                "The user gave no direction for what happens next: choose the most natural "
                "and interesting next beat for this scene."
            )

        parts.append(
            f"Write the H3 prompt for the new clip under these constraints:\n{numbered}\n\n"
            "Return only the finished prompt."
        )
        return "\n\n".join(parts), wild_label

    def continue_with_claude_code(
        self,
        frames,
        frame_count,
        frame_stride,
        idea,
        continuation_mode,
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
        director,
        use_subscription,
        timeout_seconds,
        seed,
        previous_prompt="",
        extra_instructions="",
        custom_dialogue_language="",
        custom_visual_style="",
        resume_session_id="",
        working_dir="",
        llm=None,
        **context_slots,
    ):
        template_vars, template_summary = collect_template_vars(context_slots)
        idea, extra_instructions, custom_dialogue_language, custom_visual_style = expand_all(
            template_vars, idea, extra_instructions, custom_dialogue_language, custom_visual_style
        )
        log_template_vars(template_vars, template_summary, idea, extra_instructions, custom_dialogue_language, custom_visual_style)
        context_text, context_entries = build_context(context_slots, target="the next clip")
        indices = select_last_frames(frames, frame_count, frame_stride)
        context = frames[indices]
        first_frame = frames[-1:]
        try:
            dialogue_language = resolve_dialogue_language(
                dialogue_language, custom_dialogue_language
            )
            visual_style = resolve_visual_style(visual_style, custom_visual_style)

            resuming = bool(resume_session_id.strip())
            current_seed = seed if seed != -1 else random.randint(0, 0xffffffffffffffff)
            rng = random.Random(current_seed)

            images = [tensor2pil(frame) for frame in context]

            user_prompt, wild_label = self._build_continue_prompt(
                idea,
                previous_prompt,
                resuming,
                continuation_mode,
                len(images),
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
                rng,
            )

            user_prompt = with_context(user_prompt, context_text)

            print(
                f"🎬 H3 Claude Code Continue Writer | {continuation_mode.split(' ')[0]} | "
                f"frames {indices} of {int(frames.shape[0])} | {duration_seconds:.2f}s | "
                f"wildness {wildness} ({wild_label}) | "
                f"{'resuming ' + resume_session_id.strip()[:8] if resuming else 'fresh session'} | "
                f"research {'on' if research else 'off'} | director "
                f"{'on' if director else 'off'} | seed {current_seed}"
            )

            text, session_id, info = run_h3_claude_code(
                None if resuming else self._build_system_prompt(),
                user_prompt,
                images,
                model,
                research,
                use_subscription,
                timeout_seconds,
                resume_session_id,
                working_dir,
                director,
                skills=BASE_SKILLS,
                local=local_llm_options(llm),
            )

            prompt = strip_code_fence(text)

            return (
                prompt,
                extract_section(prompt, "integrated_multimodal_description", _FIELDS),
                extract_section(prompt, "overall_soundscape", _FIELDS),
                extract_section(prompt, "non_diegetic_music", _FIELDS),
                session_id,
                info,
                first_frame,
                context,
            )

        except Exception as exc:
            # A cancelled queue must stop the run, not become an error string.
            if is_interrupt(exc):
                raise
            print(f"❌ H3 Claude Code Continue Writer error: {exc}")
            import traceback

            print(traceback.format_exc())
            message = f"Error occurred while writing the H3 continuation prompt: {exc}"
            return (message, message, "", "", "", "error", first_frame, context)
