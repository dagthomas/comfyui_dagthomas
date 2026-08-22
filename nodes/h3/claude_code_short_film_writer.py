# APNext H3 Short Film Writer
#
# Turns a MANUSCRIPT (a story, a treatment, a script, a synopsis - any text)
# into a whole short film: Claude / Codex adapts it into consecutive
# MiniMax-H3 scenes exactly like the Music Video Writer does for songs -
# chunked writing in one session, wardrobe/location locks, reference images,
# REF/FL prompt modes - but paced by STORY instead of by audio. The user
# either sets the scene count by hand or types a target film length and the
# node works out the scenes. Everything comes out as the same lists the other
# writers emit (`scenes`, `durations`, `lengths`) so the render side is
# identical: one clip per scene, saved directly, with the model's generated
# dialogue / soundscape / score as the film's sound.

import re
from concurrent.futures import ThreadPoolExecutor

from ...utils.claude_code import is_interrupt
from ...utils.image_utils import tensor2pil
from ...utils.apnext_context import (
    build_context,
    context_hidden_inputs,
    context_inputs,
    context_summary,
    with_context,
)
from ...utils.constants import CUSTOM_CATEGORY
from .claude_code_support import (
    BASE_SKILLS,
    CORE_SKILL,
    claude_code_inputs,
    claude_code_optional_inputs,
    draft_model_input,
    local_llm_inputs,
    local_llm_options,
    directions_with_research,
    resolve_draft_model,
    run_h3_claude_code,
)
from .claude_code_crossover_writer import CAST_SOCKETS, cast_wardrobe, merge_wardrobe, parse_cast
from .claude_code_music_video_writer import _scene_gist
from .claude_code_presentation_writer import _extract_script, _scene_table
from .music_support import fmt_time, frames_for_seconds
from .scenes_store import save_scene_bundle
from .template_vars import collect_template_vars, expand_all, log_template_vars
from .common import (
    AUTO,
    LITERAL_CAMERA_DIRECTIVE,
    VISUAL_STYLES,
    scale_reference_passthrough,
    resolve_visual_style,
    DIALOGUE_LANGUAGES,
    REFERENCE_IMAGE_USE,
    REFERENCE_IMAGE_USE_TOOLTIP,
    characters_only_directive,
    characters_only_refs,
    collect_reference_images,
    downscale_for_vision,
    load_guide,
    reference_image_inputs,
    reference_image_outputs,
    resolve_dialogue_language,
    toggle_directives,
    wildness_directive,
)
from .scenes_support import (
    ENFORCE_WARDROBE_TOOLTIP,
    WARDROBE_TOOLTIP,
    LOCATIONS_TOOLTIP,
    CONTINUITY_MODES,
    chain_system_block,
    continuity_directive,
    enforce_continuity,
    enforce_continuity_chunked,
    envelope_contract,
    locations_directive,
    parse_location_lock,
    parse_scenes,
    parse_wardrobe_lock,
    scenes_to_text,
    wardrobe_directive,
)

FILM_SKILLS = (CORE_SKILL, "h3-ref2va", "h3-style-craft")

PROMPT_MODES = [
    "Auto (Ref with images, FL without)",
    "Ref2VA (bind reference images)",
    "FL / T2VA (from scratch - pictures ignored)",
]

LENGTH_MODES = [
    "Scene count (use scene_count)",
    "Target length (use target_minutes)",
]

_SCENE_FIELDS = (
    "subject_definitions",
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)

MAX_FILM_SCENES = 80
SCENES_PER_CALL = 4
# a film scene averages ~11 s inside H3's 5-15 s clip range
_AVG_SCENE_SECONDS = 11.0
_DEFAULT_SCENE_SECONDS = 11.0


def scenes_for_minutes(minutes):
    return max(1, min(MAX_FILM_SCENES, round(minutes * 60.0 / _AVG_SCENE_SECONDS)))


class H3ClaudeCodeShortFilmWriter:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "manuscript": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": (
                    "The film's source: a story, a treatment, a script, a synopsis - any "
                    "length. The model adapts it faithfully: named characters, events and "
                    "written dialogue survive into the scenes; it invents connective "
                    "tissue only where the manuscript is silent."
                ),
            }),
            "length_mode": (LENGTH_MODES, {
                "default": LENGTH_MODES[0],
                "tooltip": (
                    "How the film is sized. Scene count: exactly `scene_count` scenes. "
                    "Target length: the node derives the scene count from "
                    "`target_minutes` (~11 s per scene) and the model paces each "
                    "scene's duration so the finished film lands close to the target."
                ),
            }),
            "scene_count": ("INT", {
                "default": 8, "min": 1, "max": MAX_FILM_SCENES,
                "tooltip": "How many scenes/clips, used in Scene count mode.",
            }),
            "target_minutes": ("FLOAT", {
                "default": 2.0, "min": 0.5, "max": 30.0, "step": 0.5,
                "tooltip": "How long the film should be, used in Target length mode.",
            }),
            "continuity_mode": (CONTINUITY_MODES, {
                "default": CONTINUITY_MODES[0],
                "tooltip": (
                    "Independent clips: each scene is its own clip with hard cuts. "
                    "Continuous chain: written for C2V / motion-context chaining."
                ),
            }),
            "visual_style": (VISUAL_STYLES, {
                "default": "Live-action, 35mm cinematic film aesthetic",
                "tooltip": "Opens every [Shot 1]; kept identical across the whole film.",
            }),
            "dialogue_language": (DIALOGUE_LANGUAGES, {"default": "English"}),
            "wildness": ("INT", {
                "default": 25, "min": 0, "max": 100, "step": 1,
                "tooltip": "0 = literal adaptation, 100 = fully unhinged staging.",
            }),
        }
        required.update(claude_code_inputs())
        required["seed"] = ("INT", {
            "default": -1, "min": -1, "max": 0xffffffffffffffff,
            "tooltip": "Seeds the surreal picks and controls caching. -1 re-runs every queue.",
        })

        optional = {
            name: ("STRING", {
                "forceInput": True,
                "tooltip": "A cast line or block from an H3 Characters node (its `cast` output).",
            })
            for name in CAST_SOCKETS
        }
        optional["extra_cast"] = ("STRING", {
            "multiline": True,
            "default": "",
            "tooltip": (
                "Characters typed by hand, one per line. Merged with the cast sockets "
                "and with whoever the manuscript names."
            ),
        })
        optional["custom_dialogue_language"] = ("STRING", {"default": ""})
        optional["custom_visual_style"] = ("STRING", {"default": ""})
        optional["wardrobe"] = ("STRING", {"multiline": True, "default": "", "tooltip": WARDROBE_TOOLTIP})
        optional["locations"] = ("STRING", {"multiline": True, "default": "", "tooltip": LOCATIONS_TOOLTIP})
        optional["enforce_wardrobe"] = ("BOOLEAN", {"default": True, "tooltip": ENFORCE_WARDROBE_TOOLTIP})
        optional["extra_instructions"] = ("STRING", {"multiline": True, "default": ""})
        optional["image_notes"] = ("STRING", {
            "multiline": True,
            "default": "",
            "tooltip": "Per-picture notes, one per line: `Image 1: the lead`, `Image 2: the farmhouse, use as the location`.",
        })
        optional["include_on_screen_text"] = ("BOOLEAN", {"default": False})
        optional["include_soundscape"] = ("BOOLEAN", {"default": True})
        optional["include_non_diegetic_music"] = ("BOOLEAN", {
            "default": True,
            "tooltip": "The film's score, described per scene and kept in one musical voice.",
        })
        optional.update(claude_code_optional_inputs())
        optional.update(local_llm_inputs())
        optional.update(context_inputs())
        optional.update(reference_image_inputs())
        optional["reference_image_use"] = (REFERENCE_IMAGE_USE, {
            "default": REFERENCE_IMAGE_USE[0],
            "tooltip": REFERENCE_IMAGE_USE_TOOLTIP,
        })
        optional["scene_briefs"] = ("STRING", {
            "forceInput": True,
            "tooltip": "Manually planned scenes from chained H3 Scene Brief nodes.",
        })
        optional["save_scenes"] = ("BOOLEAN", {"default": True})
        optional["scenes_per_call"] = ("INT", {
            "default": SCENES_PER_CALL, "min": 1, "max": 8,
            "tooltip": "Scenes per model call; long films are written in chunks in one session.",
        })
        # appended LAST so saved workflows keep their widget positions
        optional["prompt_mode"] = (PROMPT_MODES, {
            "default": PROMPT_MODES[1],  # Ref2VA
            "tooltip": (
                "Ref (guide_ref_en.md) binds attached pictures as <Picture N>; FL / "
                "T2VA (guide_base_en.md) creates everything from scratch in words. "
                "Auto picks Ref when images are connected."
            ),
        })
        optional.update(draft_model_input())
        optional["parallel_chunks"] = ("BOOLEAN", {
            "default": True,
            "tooltip": (
                "Write the scene chunks concurrently instead of one after another: one "
                "planning call (by `model`) fixes the synopsis, the Beats plan and the "
                "locks, then up to 4 chunks at a time are drafted from that plan (by "
                "`draft_model`), and one continuity pass repairs any drift. Much faster "
                "for long films. Off = the classic serial run where every chunk continues "
                "one session. Ignored when resume_session_id is set or the whole film "
                "fits in one call."
            ),
        })

        return {"required": required, "optional": optional, "hidden": context_hidden_inputs()}

    _IMAGE_OUTPUT_TYPES, _IMAGE_OUTPUT_NAMES = reference_image_outputs()
    RETURN_TYPES = (
        ("STRING", "FLOAT", "INT", "STRING", "STRING", "STRING", "STRING", "INT", "FLOAT", "STRING", "STRING")
        + _IMAGE_OUTPUT_TYPES
    )
    RETURN_NAMES = (
        "scenes",
        "durations",
        "lengths",
        "scenes_text",
        "synopsis",
        "script",
        "cast",
        "scene_count",
        "total_seconds",
        "session_id",
        "info",
    ) + _IMAGE_OUTPUT_NAMES
    OUTPUT_IS_LIST = (True, True, True) + (False,) * (8 + len(_IMAGE_OUTPUT_NAMES))
    FUNCTION = "write_film"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Adapts a manuscript into a whole short film: pick a scene count or a target "
        "length and Claude/Codex writes one MiniMax-H3 scene per clip - chunked in one "
        "session like the Music Video Writer, with locks, reference images and REF/FL "
        "prompt modes. `scenes` / `durations` / `lengths` are matching lists for the "
        "render side; the film's sound is the model's generated dialogue, ambience and "
        "score."
    )

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        return float("nan") if seed == -1 else seed

    @classmethod
    def VALIDATE_INPUTS(cls, prompt_mode=None):
        return True

    # ------------------------------------------------------------------

    def _build_system_prompt(self, characters_only=True, continuity_mode=CONTINUITY_MODES[0],
                             ref_mode=True):
        base_guide = load_guide("guide_base_en.md")
        ref_guide = load_guide("guide_ref_en.md") if ref_mode else ""
        return (
            "You are a film director, screenwriter and MiniMax-H3 prompt engineer. You "
            "are given a MANUSCRIPT - a story, treatment, script or synopsis - and you "
            "adapt it into a run of consecutive scenes, each a complete production-ready "
            "H3 prompt of 5-15 seconds, so that the rendered clips cut together into one "
            "coherent short film.\n\n"
            + (
                "Two documents follow. The official guide is authoritative for grammar, camera "
                "vocabulary, timestamps, speaker IDs and <d> blocks. The reference guide is "
                "authoritative for the <Picture N> labels and the relationship markers.\n\n"
                if ref_mode else
                "The official guide follows and is authoritative for grammar, camera "
                "vocabulary, timestamps, speaker IDs and <d> blocks.\n\n"
            )
            + "=== BEGIN MINIMAX-H3 VIDEO PROMPT WRITING GUIDE ===\n"
            f"{base_guide}\n"
            "=== END MINIMAX-H3 VIDEO PROMPT WRITING GUIDE ==="
            + (
                "\n\n=== BEGIN MINIMAX-H3 REFERENCE (REF2VA) GUIDE ===\n"
                f"{ref_guide}\n"
                "=== END MINIMAX-H3 REFERENCE (REF2VA) GUIDE ==="
                if ref_mode else ""
            )
            + chain_system_block(continuity_mode) + "\n\n"
            + envelope_contract(_SCENE_FIELDS) + "\n"
            "- The synopsis block also carries `Cast:` (one line per character), "
            "`Wardrobe:` and `Locations:` locks, and `Beats:` - one line per scene, "
            "`NN: the story beat that scene covers` - the whole film planned before "
            "scene 01, with a beginning, escalating middle and an ending that RESOLVES.\n"
            "- THE MANUSCRIPT IS THE SOURCE: adapt it faithfully. Every named character, "
            "event, place and object in it survives into the film; dialogue written in "
            "the manuscript is spoken VERBATIM in <d> blocks (trim lines only to fit a "
            "scene's duration, never paraphrase); the tone and genre of the manuscript "
            "set the tone of the film. Invent connective tissue - establishing moments, "
            "transitions, reaction beats - only where the manuscript is silent, and "
            "never contradict it.\n"
            "- CINEMATIC CRAFT: establish the geography of every new place before "
            "working inside it; vary coverage (wide establishing, mediums for action, "
            "close-ups for decisions and feelings); motivate every light source; end "
            "scenes on beats the next scene picks up (a look, a sound, an object, a "
            "question). Each scene's `duration:` (5.0-15.0) matches its dramatic "
            "weight - lingering where the story breathes, cutting fast where it runs.\n"
            "- SOUND IS THE FILM'S SOUND: dialogue with stable (S1)/(S2) speaker IDs, a "
            "concrete diegetic soundscape, and a score (non_diegetic_music) that keeps "
            "ONE musical identity across the whole film, developing scene by scene.\n"
            "- Write everything in English except dialogue and on-screen text, which "
            "keep their own language.\n"
            + (
                "- When reference pictures are attached the scenes are rendered with the "
                "same pictures as <Picture 1>..<Picture N>. Bind pictured people in "
                "subject_definitions as `<Subject N> ..., appearance from <Picture k>`"
                + (
                    ", and refer to the pictures again inside the shots where those people "
                    "appear. Only a picture the user's notes explicitly declare a location "
                    "or prop may be treated as one; a pictured person's backdrop is NEVER "
                    "the scene."
                    if characters_only else
                    ", treat a location or prop picture as `<Picture k> is ...` on its own "
                    "line, and refer to the pictures again inside the shots where their "
                    "content appears."
                )
                if ref_mode else
                "- This run uses NO reference pictures: never write a <Picture N> label. "
                "Every character is created from scratch in words - define each fully in "
                "subject_definitions and keep that written identity anchored "
                "word-for-word in every scene."
            )
        )

    def _build_user_prompt(
        self, cast, manuscript, n, target_seconds, continuity_mode, visual_style,
        dialogue_language, wildness, include_on_screen_text, include_soundscape,
        include_non_diegetic_music, extra_instructions, rng,
        wardrobe="", locations="", image_labels=(), image_notes="",
        first=1, last=None, characters_only=True, scene_briefs="", prior_scenes=(),
        plan_only=False, plan_text="",
    ):
        last = last or n
        briefs = (scene_briefs or "").strip()
        lines = ["CAST (use these strings verbatim in subject_definitions; the manuscript may add more):"]
        lines += [f"- {c}" for c in cast] if cast else ["- (no cast given - take the characters from the manuscript)"]
        lines.append("")
        lines.append("THE MANUSCRIPT (the source - adapt it faithfully):")
        lines.append(manuscript.strip())
        lines.append("")
        if briefs:
            lines.append("SCENE BRIEFS FROM THE USER - the plan for those scenes:")
            lines.append(briefs)
            lines.append("")
        if plan_text:
            lines.append(
                "THE SYNOPSIS AND BEATS PLAN - already fixed for this film in an earlier "
                "turn. Its cast, wardrobe locks, location locks and per-scene Beats are "
                "BINDING:"
            )
            lines.append(plan_text.strip())
            lines.append("")
        if first > 1 and prior_scenes:
            lines.append("THE FILM SO FAR - what the scenes you already wrote show:")
            for no, gist in prior_scenes:
                lines.append(f"  {no:02d}: {gist}")
            lines.append("")
        lines.append("DIRECTIVES:")
        if plan_only:
            lines.append(
                "- Write ONLY the planning document for the whole film - no scene "
                "envelopes. It is the synopsis block exactly as the contract defines it: "
                "Synopsis, `Cast:`, the `Wardrobe:` and `Locations:` locks, and `Beats:` "
                f"- one line per scene, all {n} of them, `NN: the story beat that scene "
                "covers`. The scenes are written later from this plan, several at a time "
                "by different writers who see only this document - so it must carry ALL "
                "the continuity: exact wardrobe anchors, exact location anchors, each "
                "scene's beat, and the full arc from opening to a resolution that answers "
                "the opening."
            )
        elif plan_text:
            lines.append(
                f"- Write scenes {first:02d} to {last:02d} (of {n}) ONLY, following the "
                "synopsis and Beats plan above exactly: stage each scene's beat, copy the "
                "wardrobe and location anchors character-for-character, and keep the "
                "story moving. Other writers handle the other scenes from the same plan - "
                "never write, restate or borrow a scene outside your range. Start "
                "directly with the scene envelopes; do not write a synopsis block."
            )
        elif first == 1:
            lines.append(
                f"- Write the synopsis block (with the full {n}-scene Beats plan), then scenes "
                f"{first:02d} to {last:02d} (of {n}). The remaining scenes are requested in "
                "follow-up turns; plan the WHOLE film now."
                if last < n else
                f"- Write the synopsis block (with the full {n}-scene Beats plan) and all "
                f"{n} scenes, numbered 01 to {n:02d}."
            )
        else:
            lines.append(
                f"- Continue the SAME film: write scenes {first:02d} to {last:02d} (of {n}) "
                "only, following the synopsis Beats, cast, wardrobe and location locks you "
                f"already fixed. Pick up exactly where scene {first - 1:02d} in the "
                "film-so-far list leaves off and advance the story; never restage what an "
                "earlier scene already showed. Start directly with the scene envelopes; do "
                "not repeat the synopsis."
            )
        lines.append(
            f"- STRUCTURE: across {n} scenes the film has a real shape - the opening "
            "establishes who wants what, the middle escalates through complications, and "
            "the final scenes deliver a climax and a resolution that answers the opening. "
            "Place the act turns where the manuscript places them; if it has no act "
            f"structure, turn near scene {max(1, round(n * 0.25)):02d} and scene "
            f"{max(2, round(n * 0.75)):02d}."
        )
        if target_seconds:
            lines.append(
                f"- LENGTH TARGET: the finished film must land close to "
                f"{fmt_time(target_seconds)} total. Across {n} scenes that averages "
                f"{target_seconds / n:.1f}s - pace each scene's `duration:` (5.0-15.0) to "
                "its dramatic weight while keeping the total on target."
            )
        else:
            lines.append(
                "- Choose each scene's `duration:` yourself between 5.0 and 15.0 seconds "
                "to match its dramatic weight, and fill every second with story."
            )
        lines.append(f"- {continuity_directive(continuity_mode)}")
        if visual_style == AUTO:
            lines.append(
                "- Visual style: choose one concrete style that suits the manuscript and "
                "keep it identical in every scene, opening every [Shot 1] with it."
            )
        else:
            lines.append(
                f"- Visual style: open every [Shot 1] with `{visual_style.strip()}` followed "
                "by lighting, time of day and setting; keep it identical across the film."
            )
        lines.extend(
            f"- {d}" for d in toggle_directives(
                True,  # films speak
                include_on_screen_text,
                include_soundscape,
                include_non_diegetic_music,
                dialogue_language,
            )
        )
        if image_labels:
            labels_s = ", ".join(f"<Picture {i}>" for i in image_labels)
            lines.append(
                f"- Reference pictures: {len(image_labels)} attached, in order {labels_s}; "
                "the video model receives the same pictures under the same labels in every "
                "scene. "
                + (characters_only_directive() + " Bind pictured people to their <Picture k> "
                   "in subject_definitions."
                   if characters_only else
                   "Decide what each shows (use the notes below); bind pictured people to "
                   "their <Picture k> in subject_definitions.")
            )
            notes = (image_notes or "").strip()
            if notes:
                lines.append("- Picture notes from the user:")
                lines.extend(f"    {n_.strip()}" for n_ in notes.splitlines() if n_.strip())
        lines.append(f"- {wardrobe_directive(wardrobe)}")
        lines.append(f"- {locations_directive(locations)}")
        lines.append(f"- {LITERAL_CAMERA_DIRECTIVE}")
        if briefs:
            lines.append(
                "- SCENE BRIEFS ARE BINDING: a numbered brief is the plan for that scene; "
                "`SCENE (next in order)` briefs fill scenes in order from 01. Scenes "
                "without a brief are yours - but never contradict a brief or the manuscript."
            )
        wild_lines, _wild_label = wildness_directive(wildness, rng)
        lines += [f"- {w}" for w in wild_lines]
        extra = (extra_instructions or "").strip()
        if extra:
            lines.append("")
            lines.append("EXTRA INSTRUCTIONS:")
            lines.append(extra)
        lines.append("")
        lines.append(
            "Now write the synopsis block with the full Beats plan, and nothing else."
            if plan_only else
            "Now write the requested envelopes (and the synopsis block when asked), and nothing else."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def write_film(
        self, manuscript, length_mode, scene_count, target_minutes, continuity_mode,
        visual_style, dialogue_language, wildness, model, research, director,
        use_subscription, timeout_seconds, seed,
        extra_cast="", custom_dialogue_language="", custom_visual_style="",
        wardrobe="", locations="", enforce_wardrobe=True, extra_instructions="",
        image_notes="", include_on_screen_text=False, include_soundscape=True,
        include_non_diegetic_music=True, resume_session_id="", working_dir="",
        llm=None, reference_image_use=None, scene_briefs="",
        save_scenes=True, scenes_per_call=SCENES_PER_CALL, prompt_mode=None,
        draft_model="haiku", parallel_chunks=True,
        **cast_slots,
    ):
        import random
        passthrough = scale_reference_passthrough(cast_slots, self._IMAGE_OUTPUT_NAMES)
        references = collect_reference_images(passthrough, tensor2pil)
        mode = str(prompt_mode or PROMPT_MODES[1])
        ref_mode = bool(references) if mode.startswith("Auto") else mode.startswith("Ref")
        if not ref_mode and references:
            print(f"ℹ️ H3 Short Film Writer: FL prompt mode - {len(references)} picture(s) ignored for writing.")
            references = []
        images = [downscale_for_vision(pil) for _, pil in references] or None
        image_labels = tuple(range(1, len(references) + 1))

        template_vars, template_summary = collect_template_vars(cast_slots)
        (manuscript, extra_cast, wardrobe, locations, extra_instructions, image_notes,
         custom_dialogue_language, custom_visual_style, scene_briefs) = expand_all(
            template_vars, manuscript, extra_cast, wardrobe, locations, extra_instructions,
            image_notes, custom_dialogue_language, custom_visual_style, scene_briefs,
        )
        log_template_vars(template_vars, template_summary, manuscript, extra_cast, wardrobe, locations, extra_instructions, image_notes)
        context_text, context_entries = build_context(cast_slots, target="the film")
        cast_blocks = [cast_slots.get(name) for name in CAST_SOCKETS] + [extra_cast]
        cast = parse_cast(*cast_blocks)
        cast_text = "\n".join(cast)
        wardrobe = merge_wardrobe(wardrobe, cast_wardrobe(*cast_blocks))

        target_mode = str(length_mode or "").startswith("Target")
        n = scenes_for_minutes(float(target_minutes)) if target_mode else int(scene_count)
        target_seconds = float(target_minutes) * 60.0 if target_mode else 0.0
        fallback_durations = [_DEFAULT_SCENE_SECONDS] * n
        fallback_lengths = [frames_for_seconds(_DEFAULT_SCENE_SECONDS)] * n

        try:
            if not (manuscript or "").strip():
                raise ValueError("The manuscript is empty - there is nothing to film.")
            dialogue_language = resolve_dialogue_language(dialogue_language, custom_dialogue_language)
            visual_style = resolve_visual_style(visual_style, custom_visual_style)
            current_seed = seed if seed != -1 else random.randint(0, 0xffffffffffffffff)
            rng = random.Random(current_seed)
            local = local_llm_options(llm)
            chars_only = characters_only_refs(reference_image_use)
            skills = FILM_SKILLS if ref_mode else BASE_SKILLS
            system_prompt = self._build_system_prompt(
                characters_only=chars_only, continuity_mode=continuity_mode, ref_mode=ref_mode,
            )

            print(
                f"🎥 H3 Short Film Writer | {'ref' if ref_mode else 'fl'} prompts | "
                f"{len(cast)} cast | {n} scene(s)"
                + (f" (~{fmt_time(target_seconds)} target)" if target_mode else "")
                + f" | context: {context_summary(context_entries)} | "
                f"{len(references)} reference image(s) | wildness {wildness} | "
                f"research {'on' if research else 'off'} | director {'on' if director else 'off'}"
            )

            synopsis = ""
            parsed = []
            session_id = (resume_session_id or "").strip()
            infos = []
            per_call = max(1, min(8, int(scenes_per_call or 0) or SCENES_PER_CALL))
            # draft/synthesis split: `model` plans and repairs, chunk_model drafts
            chunk_model = resolve_draft_model(draft_model, model, local)
            parallel = bool(parallel_chunks) and n > per_call and not session_id
            if bool(parallel_chunks) and n > per_call and session_id:
                print(
                    "ℹ️ H3 Short Film Writer: resume_session_id is set - writing "
                    "serially in that session (parallel_chunks needs a fresh run)."
                )

            def build_prompt(lo, hi, rng_, plan_only=False, plan_text="", prior=()):
                # reads wardrobe/locations at call time, so the locks merged from
                # the synopsis reach every later prompt
                return self._build_user_prompt(
                    cast, manuscript, n, target_seconds, continuity_mode, visual_style,
                    dialogue_language, wildness, include_on_screen_text, include_soundscape,
                    include_non_diegetic_music,
                    directions_with_research(extra_instructions, research), rng_,
                    wardrobe=wardrobe, locations=locations, image_labels=image_labels,
                    image_notes=image_notes, first=lo, last=hi, characters_only=chars_only,
                    scene_briefs=scene_briefs, prior_scenes=prior,
                    plan_only=plan_only, plan_text=plan_text,
                )

            def merge_locks(text):
                nonlocal wardrobe, locations
                w_locks = parse_wardrobe_lock(text)
                if w_locks:
                    wardrobe = merge_wardrobe(wardrobe, [f"{k}: {', '.join(v)}" for k, v in w_locks.items()])
                l_locks = parse_location_lock(text)
                if l_locks:
                    locations = merge_wardrobe(locations, [f"{k}: {', '.join(v)}" for k, v in l_locks.items()])

            if parallel:
                # one plan call (strong model) -> chunks drafted concurrently
                # (draft model, fresh sessions) -> one continuity repair below
                ranges = [(lo, min(n, lo + per_call - 1)) for lo in range(1, n + 1, per_call)]
                workers = min(4, len(ranges))
                print(
                    f"⚡ H3 Short Film Writer: parallel run - plan with '{model}', then "
                    f"{len(ranges)} chunk(s) drafted with '{chunk_model}', up to {workers} at once."
                )
                plan_prompt = with_context(build_prompt(1, n, rng, plan_only=True), context_text)
                text, session_id, info = run_h3_claude_code(
                    system_prompt, plan_prompt, images, model, research, use_subscription,
                    timeout_seconds, "", working_dir, director, skills=skills, local=local,
                )
                infos.append(f"plan: {info}")
                synopsis = (text or "").strip()
                if not synopsis:
                    raise ValueError("the planning call returned no synopsis / Beats plan.")
                merge_locks(synopsis)

                def write_range(lo, hi, depth=0):
                    rng_ = random.Random(f"{current_seed}:{lo}:{hi}")
                    user_prompt = build_prompt(lo, hi, rng_, plan_text=synopsis)
                    try:
                        text, _sid, info = run_h3_claude_code(
                            system_prompt, user_prompt, images, chunk_model, research,
                            use_subscription, timeout_seconds, "", working_dir, director,
                            skills=skills, local=local,
                        )
                    except Exception as exc:
                        if (is_interrupt(exc) or "did not finish within" not in str(exc)
                                or hi <= lo or depth >= 2):
                            raise
                        mid = lo + (hi - lo) // 2
                        print(
                            f"⏱️ H3 Short Film Writer: scenes {lo:02d}-{hi:02d} timed out - "
                            f"splitting into {lo:02d}-{mid:02d} and {mid + 1:02d}-{hi:02d}."
                        )
                        a_scenes, a_infos = write_range(lo, mid, depth + 1)
                        b_scenes, b_infos = write_range(mid + 1, hi, depth + 1)
                        return a_scenes + b_scenes, a_infos + b_infos
                    _syn, chunk = parse_scenes(text, _DEFAULT_SCENE_SECONDS)
                    if not chunk:
                        raise ValueError(f"the model returned no scenes for {lo:02d}-{hi:02d}.")
                    got = chunk[: hi - lo + 1]
                    if len(got) < hi - lo + 1:
                        print(f"⚠️ H3 Short Film Writer: asked for scenes {lo:02d}-{hi:02d}, got {len(got)}.")
                    scenes_out = [(lo + k, float(d), p) for k, (_no, d, p) in enumerate(got)]
                    return scenes_out, [info]

                errors = []
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(write_range, lo, hi) for lo, hi in ranges]
                    for future in futures:
                        try:
                            scenes_out, chunk_infos = future.result()
                        except Exception as exc:
                            errors.append(exc)
                            continue
                        parsed.extend(scenes_out)
                        infos.extend(chunk_infos)
                if errors:
                    raise errors[0]
                parsed.sort(key=lambda item: item[0])
            else:
                first = 1
                film_so_far = []

                def ask(lo, hi):
                    user_prompt = build_prompt(lo, hi, rng, prior=tuple(film_so_far))
                    if lo == 1:
                        user_prompt = with_context(user_prompt, context_text)
                    return run_h3_claude_code(
                        None if (lo > 1 and session_id) else system_prompt,
                        user_prompt,
                        images if lo == 1 else None,
                        model if lo == 1 else chunk_model,
                        research, use_subscription, timeout_seconds,
                        session_id, working_dir, director, skills=skills, local=local,
                    )

                while first <= n:
                    last = min(n, first + per_call - 1)
                    try:
                        text, session_id, info = ask(first, last)
                    except Exception as exc:
                        if is_interrupt(exc) or "did not finish within" not in str(exc) or last <= first:
                            raise
                        half = first + (last - first) // 2
                        print(
                            f"⏱️ H3 Short Film Writer: scenes {first:02d}-{last:02d} timed out - "
                            f"retrying as {first:02d}-{half:02d}."
                        )
                        last = half
                        text, session_id, info = ask(first, last)
                    infos.append(info)
                    chunk_synopsis, chunk = parse_scenes(text, _DEFAULT_SCENE_SECONDS)
                    if first == 1:
                        synopsis = chunk_synopsis
                        merge_locks(synopsis)
                    if not chunk:
                        raise ValueError(f"the model returned no scenes for {first:02d}-{last:02d}.")
                    got = [(no, d, p) for no, d, p in chunk]
                    for k, (_no, d, p) in enumerate(got[: last - first + 1]):
                        idx = first + k
                        parsed.append((idx, float(d), p))
                        film_so_far.append((idx, _scene_gist(p)))
                    if len(got) < last - first + 1:
                        print(f"⚠️ H3 Short Film Writer: asked for scenes {first:02d}-{last:02d}, got {len(got)}.")
                    first = first + max(1, min(len(got), last - first + 1))

            if len(parsed) != n:
                print(f"⚠️ H3 Short Film Writer: asked for {n} scene(s), parsed {len(parsed)}.")

            def repair(prompt):
                if parallel:
                    # the plan session never saw the drafted scenes, so the
                    # repair turn carries them along
                    prompt = (
                        "The film's scenes as currently written (drafted from the plan "
                        "you fixed earlier in this session):\n\n"
                        + scenes_to_text("", parsed)
                        + "\n\n" + prompt
                    )
                text, _, repair_info = run_h3_claude_code(
                    system_prompt, prompt, None, model, False, use_subscription, timeout_seconds,
                    session_id, working_dir, director, skills=skills, local=local,
                )
                return text, repair_info

            info = " || ".join(infos)
            if len(parsed) == n and n <= per_call:
                synopsis, parsed, info = enforce_continuity(
                    enforce_wardrobe, synopsis, parsed, n, _DEFAULT_SCENE_SECONDS, session_id, info, repair,
                )
            else:
                parsed, info = enforce_continuity_chunked(
                    enforce_wardrobe, synopsis, parsed, session_id, info, repair,
                )

            scenes = [p for _, _, p in parsed][:n]
            while len(scenes) < n:
                scenes.append(scenes[-1] if scenes else "")
            raw_durations = [d for _, d, _ in parsed][:n]
            while len(raw_durations) < n:
                raw_durations.append(_DEFAULT_SCENE_SECONDS)
            lengths = [frames_for_seconds(d) for d in raw_durations]
            durations = [fr / 24.0 for fr in lengths]
            total_seconds = float(sum(durations))
            scenes_text = scenes_to_text(synopsis, [(i + 1, durations[i], s) for i, s in enumerate(scenes)])
            script = _extract_script(scenes)
            table = _scene_table(durations, lengths)
            print(f"🎥 H3 Short Film Writer | {fmt_time(total_seconds)} film in {n} scene(s)"
                  + (f" (target {fmt_time(target_seconds)})" if target_mode else "") + f"\n{table}")

            if save_scenes:
                starts, acc = [], 0.0
                for d in durations:
                    starts.append(acc)
                    acc += d
                segments = [(s, s + d) for s, d in zip(starts, durations)]
                save_scene_bundle(
                    "H3ClaudeCodeShortFilmWriter", synopsis, scenes, segments, durations,
                    lengths, starts, cast_text, total_seconds, scenes_text, table, info,
                )

            return (
                scenes, durations, lengths, scenes_text, synopsis, script,
                cast_text, n, total_seconds, session_id, info,
            ) + passthrough

        except Exception as exc:
            if is_interrupt(exc):
                raise
            print(f"❌ H3 Short Film Writer error: {exc}")
            import traceback
            print(traceback.format_exc())
            message = f"Error occurred while writing the film: {exc}"
            return (
                [message] * n, fallback_durations, fallback_lengths, message, "", "",
                cast_text, n, float(sum(fallback_durations)), "", "error",
            ) + passthrough
