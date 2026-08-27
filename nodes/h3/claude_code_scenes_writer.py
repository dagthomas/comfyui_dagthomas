# APNext H3 Claude Code Scenes Writer
#
# The base-format H3 writer, but for a run of scenes: one idea in, 1-10
# complete T2VA prompts out as a ComfyUI list, each in the three-field base
# format (integrated_multimodal_description / overall_soundscape /
# non_diegetic_music) and each with its own duration. Same director skills,
# camera vocabulary and wildness bands as the single-prompt writer; the scenes
# form one continuous story with hand-offs between them.

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
from .base_prompt_writer import _FIELDS
from .claude_code_support import (
    BASE_SKILLS,
    claude_code_inputs,
    claude_code_optional_inputs,
    local_llm_inputs,
    local_llm_options,
    directions_with_research,
    project_name_input,
    project_name_prefix,
    resolve_project_name,
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
    collect_typed_references,
    load_guide,
    resolve_dialogue_language,
    toggle_directives,
    typed_reference_inputs,
    wildness_directive,
)
from .scenes_support import (
    is_chain_mode,
    resolve_transition,
    transition_directive,
    transition_input,
    interpretation_directive,
    interpretation_input,
    resolve_interpretation,
    ENFORCE_WARDROBE_TOOLTIP,
    WARDROBE_TOOLTIP,
    LOCATIONS_TOOLTIP,
    enforce_continuity,
    locations_directive,
    wardrobe_directive,
    CONTINUITY_MODES,
    DURATION_MODES,
    MAX_SCENES,
    chain_system_block,
    continuity_directive,
    duration_directive,
    envelope_contract,
    parse_scenes,
    scenes_to_text,
)

_SHOT_COUNTS = {"Single shot": 1, "Two shots": 2, "Three shots": 3, "Four shots": 4}


class H3ClaudeCodeScenesWriter:
    """
    APNext H3 Claude Code Scenes Writer

    Expands one idea into 1-10 consecutive MiniMax-H3 T2VA scenes with the local
    Claude Code CLI. Outputs the scenes and their durations as lists.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "idea": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": (
                    "The story, sequence or idea to expand into several scenes. Say what "
                    "happens; the node handles the H3 grammar."
                ),
            }),
            "scene_count": ("INT", {
                "default": 3, "min": 1, "max": MAX_SCENES,
                "tooltip": "How many scenes to write. Each becomes one element of the scenes list.",
            }),
            "duration_mode": (DURATION_MODES, {"default": DURATION_MODES[0]}),
            "continuity_mode": (CONTINUITY_MODES, {
                "default": CONTINUITY_MODES[0],
                "tooltip": (
                    "Independent clips: each scene is its own T2V clip with hard cuts. "
                    "Continuous chain: scenes are written for C2V / motion-context chaining "
                    "(Contex Loop, Add Guide, Motion Context) - scene N+1 opens on scene N's "
                    "last frame, one continuous take, speaker hand-off beats, rotating closers."
                ),
            }),
            "scene_duration": ("FLOAT", {
                "default": 10.0, "min": 1.0, "max": 20.0, "step": 0.5,
                "tooltip": (
                    "Seconds per scene in Fixed mode, and the fallback when Claude omits a "
                    "duration in Vary mode."
                ),
            }),
            "shot_plan": (SHOT_PLANS, {"default": AUTO, "tooltip": "Shots per scene."}),
            "visual_style": (VISUAL_STYLES, {
                "default": AUTO,
                "tooltip": (
                    "Style stated at the start of every [Shot 1]; kept identical across the run. "
                    "The list is the guide's styles plus the APNext Cinematic vocabulary (film stock, grading, aesthetics); pick Custom and fill in custom_visual_style to write your own."
                ),
            }),
            "wildness": ("INT", {
                "default": 25, "min": 0, "max": 100, "step": 1,
                "tooltip": "0 = literal and conservative, 100 = fully unhinged.",
            }),
            "camera_motion": (CAMERA_MOTIONS, {"default": AUTO}),
            "camera_amplitude": (CAMERA_AMPLITUDES, {"default": AUTO}),
            "camera_speed": (CAMERA_SPEEDS, {"default": AUTO}),
            "include_dialogue": ("BOOLEAN", {"default": True}),
            "dialogue_language": (DIALOGUE_LANGUAGES, {"default": "English"}),
            "include_on_screen_text": ("BOOLEAN", {"default": False}),
            "include_soundscape": ("BOOLEAN", {"default": True}),
            "include_non_diegetic_music": ("BOOLEAN", {"default": True}),
        }
        required.update(claude_code_inputs())
        required["seed"] = ("INT", {
            "default": -1, "min": -1, "max": 0xffffffffffffffff,
            "tooltip": "Seeds the surreal picks and controls caching. -1 re-runs every queue.",
        })

        optional = {
            "image": ("IMAGE", {
                "tooltip": (
                    "Context only: pictures to describe from (a look, a place, a person). "
                    "The scenes are T2VA, so nothing becomes <Picture N>."
                ),
            }),
            "extra_instructions": ("STRING", {"multiline": True, "default": ""}),
            "custom_dialogue_language": ("STRING", {"default": ""}),
            "custom_visual_style": ("STRING", {
                "default": "",
                "tooltip": (
                    "Any visual style not in the dropdown, e.g. 'hand-painted cel animation' or "
                    "'Kodak Vision3 500T, anamorphic'. Overrides the dropdown when filled in."
                ),
            }),
            "wardrobe": ("STRING", {"multiline": True, "default": "", "tooltip": WARDROBE_TOOLTIP}),
            "enforce_wardrobe": ("BOOLEAN", {"default": True, "tooltip": ENFORCE_WARDROBE_TOOLTIP}),
        }
        optional["scene_briefs"] = ("STRING", {
            "forceInput": True,
            "tooltip": (
                "Manually planned scenes from chained H3 Scene Brief nodes: each brief "
                "(what happens, where, which cast members and pictures) becomes the "
                "binding plan for its scene. Pinned numbers take that scene; unpinned "
                "briefs fill in order; scenes without a brief stay the model's to invent."
            ),
        })
        optional.update(claude_code_optional_inputs())
        optional["locations"] = ("STRING", {"multiline": True, "default": "", "tooltip": LOCATIONS_TOOLTIP})
        optional.update(local_llm_inputs())
        optional.update(typed_reference_inputs())
        optional.update(context_inputs())
        # appended LAST so saved workflows keep their widget positions
        optional.update(project_name_input())

        # appended last so saved workflows keep their widget positions
        optional["interpretation"] = interpretation_input('the idea')

        optional["transition_style"] = transition_input()

        return {"required": required, "optional": optional, "hidden": context_hidden_inputs()}

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "STRING", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "scenes",
        "durations",
        "scenes_text",
        "synopsis",
        "scene_count",
        "session_id",
        "info",
        "project_name",
    )
    OUTPUT_IS_LIST = (True, True, False, False, False, False, False, False)
    FUNCTION = "write_scenes"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Expands one idea into 1-10 consecutive MiniMax-H3 T2VA scenes (base format) with "
        "the local Claude Code CLI. `scenes` and `durations` are lists so a video node "
        "downstream renders them all; use H3 Scene Pick to grab one."
    )

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        return float("nan") if seed == -1 else seed

    # ------------------------------------------------------------------
    # Wardrobe verification (one repair turn in the same session)
    # ------------------------------------------------------------------

    def _enforce_wardrobe(
        self, enabled, synopsis, parsed, scene_count, scene_duration, session_id,
        info, model, use_subscription, timeout_seconds, working_dir, director,
        continuity_mode, local=None,
    ):
        """Wardrobe + location lock check with one shared repair turn (scenes_support)."""
        def repair(prompt):
            text, _, repair_info = run_h3_claude_code(
                self._build_system_prompt(continuity_mode), prompt,
                None, model, False, use_subscription, timeout_seconds,
                session_id, working_dir, director, local=local,
            )
            return text, repair_info

        return enforce_continuity(
            enabled, synopsis, parsed, scene_count, scene_duration, session_id, info, repair,
        )

    # ------------------------------------------------------------------

    def _build_system_prompt(self, continuity_mode=CONTINUITY_MODES[0]):
        guide = load_guide("guide_base_en.md")
        return (
            "You are a MiniMax-H3 video prompt engineer and sequence director. You expand a "
            "user's idea into a run of consecutive scenes, each a complete, production-ready "
            "H3 T2VA prompt in the base format, that together tell one continuous story.\n\n"
            "The authoritative specification follows. Obey it exactly - field names, shot "
            "labels, timestamp formats, camera vocabulary, speaker IDs and <d> blocks all "
            "follow this document.\n\n"
            "=== BEGIN MINIMAX-H3 VIDEO PROMPT WRITING GUIDE ===\n"
            f"{guide}\n"
            "=== END MINIMAX-H3 VIDEO PROMPT WRITING GUIDE ==="
            + chain_system_block(continuity_mode) + "\n\n"
            + envelope_contract(_FIELDS) + "\n"
            "- Each scene is a T2VA prompt: no alignment lines and no <Picture N> labels. "
            "Continuity comes from restating the same style, lighting, wardrobe and setting "
            "words in every scene, from ending one scene on a beat the next one picks up, and "
            "- in continuous-chain mode - from opening each scene on the previous ending as "
            "the chain rules describe.\n"
            "- Keep every recurring subject visually anchored the same way in every scene "
            "(hair, wardrobe, colours, distinguishing marks); give each vocal source a stable "
            "(S1)/(S2) ID across the run.\n"
            "- Fill each scene's whole duration with action; no dead air, no static stares "
            "at the end. If dialogue ends early, close on concrete movement.\n"
            "- Write everything in English except dialogue, lyrics and on-screen text, which "
            "keep their own language."
        )

    def _build_user_prompt(
        self,
        idea,
        scene_count,
        duration_mode,
        continuity_mode,
        scene_duration,
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
        references,
        rng,
        wardrobe="",
        locations="",
        scene_briefs="",
    ):
        directives = [
            f"Write exactly {scene_count} scene{'s' if scene_count != 1 else ''}, numbered "
            f"01 to {scene_count:02d}, forming one continuous story with a hand-off between "
            "adjacent scenes (a look, a movement, an object, a question the next scene answers).",
            duration_directive(duration_mode, scene_duration),
            continuity_directive(continuity_mode),
        ]
        if getattr(self, "_interpretation", None):
            directives.append(interpretation_directive(self._interpretation, 'the idea'))
        if is_chain_mode(continuity_mode) and getattr(self, "_transition", None):
            directives.append(transition_directive(self._transition))
        if shot_plan == AUTO:
            directives.append(
                "Choose the shot count per scene that fits the action (usually 1-3), varying "
                "the pattern between scenes. [Shot 1] carries no timestamp; every later shot "
                "opens with a strictly increasing `[Shot N] At MM:SS.mmm, ...` cut time inside "
                "that scene's duration."
            )
        else:
            count = _SHOT_COUNTS.get(shot_plan, 1)
            directives.append(
                f"Use exactly {count} shot{'s' if count > 1 else ''} per scene. [Shot 1] "
                "carries no timestamp; every later shot opens with a strictly increasing "
                "`[Shot N] At MM:SS.mmm, ...` cut time inside that scene's duration."
            )
        if visual_style == AUTO:
            hint = " and the attached image(s)" if image_count else ""
            directives.append(
                f"Visual style: choose one that fits the idea{hint}, state it at the start of "
                "every [Shot 1], and keep it identical across all scenes."
            )
        else:
            directives.append(
                f"Visual style: open every [Shot 1] with `{visual_style}` as the stated style."
            )
        directives.append(wardrobe_directive(wardrobe))
        directives.append(locations_directive(locations))
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
        briefs = (scene_briefs or "").strip()
        if briefs:
            directives.append(
                "SCENE BRIEFS ARE BINDING: a numbered brief in the source section is the "
                "plan for that scene - set it where the brief says, with the cast members "
                "it names and the reference pictures it points at, honour its camera wish, "
                "and stage what it describes within the scene's duration. `SCENE (next in "
                "order)` briefs fill scenes in order from 01, skipping numbered ones. "
                "Scenes without a brief are yours to write - but never contradict a brief."
            )
        if extra_instructions.strip():
            directives.append(f"Additional direction from the user: {extra_instructions.strip()}")

        numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(directives, 1))

        source_parts = []
        if image_count:
            source_parts.append(
                f"The first {image_count} attached image(s) are visual context: derive look, "
                "subjects, wardrobe, colours and setting from what you actually see and keep "
                "them consistent across every scene. They are not keyframes."
            )
        if references:
            source_parts.append(
                "The remaining attached images are typed references (" + ", ".join(references) +
                "): describe each in words so the video model can reproduce it from text - a "
                "subject carries over its identity and wardrobe, a scenery its place and light, "
                "an object its shape and material - and use them consistently in every scene."
            )
        source_parts.append(f"The user's idea:\n{idea.strip() or '(none - use the images)'}")
        if briefs:
            source_parts.append(f"Scene briefs from the user - the plan for those scenes:\n{briefs}")
        source = "\n\n".join(source_parts)

        return (
            f"{source}\n\n"
            f"Write the scenes under these constraints:\n{numbered}\n\n"
            "Return only the synopsis block and the scene envelopes."
        ), wild_label

    # ------------------------------------------------------------------

    def write_scenes(
        self,
        idea,
        scene_count,
        duration_mode,
        continuity_mode,
        scene_duration,
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
        image=None,
        extra_instructions="",
        custom_dialogue_language="",
        custom_visual_style="",
        wardrobe="",
        enforce_wardrobe=True,
        resume_session_id="",
        working_dir="",
        locations="",
        llm=None,
        scene_briefs="",
        project_name="",
        interpretation=None,
        transition_style=None,
        **reference_slots,
    ):
        project_name = resolve_project_name(project_name, seed)
        print(f"🎬 H3 Scenes Writer | project: {project_name}")
        template_vars, template_summary = collect_template_vars(reference_slots)
        idea, extra_instructions, wardrobe, custom_dialogue_language, custom_visual_style, locations, scene_briefs = expand_all(
            template_vars, idea, extra_instructions, wardrobe, custom_dialogue_language, custom_visual_style, locations, scene_briefs
        )
        log_template_vars(template_vars, template_summary, idea, extra_instructions, wardrobe, custom_dialogue_language, custom_visual_style)
        context_text, context_entries = build_context(reference_slots, target="the scenes")
        references = collect_typed_references(reference_slots, tensor2pil)
        try:
            if not idea.strip() and image is None and not references and not context_entries:
                raise ValueError("Provide an idea, an image, a reference, or some of each.")

            dialogue_language = resolve_dialogue_language(
                dialogue_language, custom_dialogue_language
            )

            visual_style = resolve_visual_style(visual_style, custom_visual_style)
            current_seed = seed if seed != -1 else random.randint(0, 0xffffffffffffffff)
            self._interpretation = resolve_interpretation(interpretation, current_seed)
            self._transition = resolve_transition(transition_style)
            if self._interpretation:
                picked = " (picked by seed)" if str(interpretation or "").startswith("Surprise") else ""
                print(f"🎭 interpretation: {self._interpretation}{picked}")
            rng = random.Random(current_seed)

            context = [tensor2pil(frame) for frame in image] if image is not None else []
            images = (context + [pil for _, pil in references]) or None

            user_prompt, wild_label = self._build_user_prompt(
                idea,
                scene_count,
                duration_mode,
                continuity_mode,
                scene_duration,
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
                len(context),
                [label for label, _ in references],
                rng,
                wardrobe=wardrobe,
                locations=locations,
                scene_briefs=scene_briefs,
            )
            user_prompt = with_context(user_prompt, context_text)

            print(
                f"🎬 H3 Scenes Writer | {scene_count} scene(s) | "
                f"context: {context_summary(context_entries)} | "
                f"{duration_mode.split(' ')[0].lower()} {scene_duration:.1f}s | "
                f"{'chain' if continuity_mode == CONTINUITY_MODES[1] else 'independent'} | "
                f"{len(context)} context image(s) | wildness {wildness} ({wild_label}) | "
                f"research {'on' if research else 'off'} | director {'on' if director else 'off'} | "
                f"seed {current_seed}"
            )

            local = local_llm_options(llm)
            text, session_id, info = run_h3_claude_code(
                self._build_system_prompt(continuity_mode),
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
                local=local,
            )

            synopsis, parsed = parse_scenes(text, scene_duration)
            if not parsed:
                raise ValueError("Claude Code returned no scenes.")
            if len(parsed) != scene_count:
                print(f"⚠️ H3 Scenes Writer: asked for {scene_count} scene(s), parsed {len(parsed)}.")

            synopsis, parsed, info = self._enforce_wardrobe(
                enforce_wardrobe, synopsis, parsed, scene_count, scene_duration,
                session_id, info, model, use_subscription, timeout_seconds,
                working_dir, director, continuity_mode, local=local,
            )

            scenes = [p for _, _, p in parsed]
            durations = [d for _, d, _ in parsed]

            return (
                scenes,
                durations,
                scenes_to_text(synopsis, parsed),
                synopsis,
                len(scenes),
                session_id,
                info,
                project_name_prefix(project_name),
            )

        except Exception as exc:
            if is_interrupt(exc):
                raise
            print(f"❌ H3 Scenes Writer error: {exc}")
            import traceback

            print(traceback.format_exc())
            message = f"Error occurred while writing the H3 scenes: {exc}"
            return ([message], [float(scene_duration)], message, "", 0, "", "error", project_name_prefix(project_name))
