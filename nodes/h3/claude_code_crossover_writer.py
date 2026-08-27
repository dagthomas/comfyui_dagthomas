# APNext H3 Crossover Writer
#
# Takes a cast of characters (from H3 Characters nodes or typed by hand) plus
# the user's steer, and has the local Claude Code CLI write 1-10 crossover
# scenes in the four-section MiniMax-H3 T2VA layout (subject_definitions /
# integrated_multimodal_description / overall_soundscape / non_diegetic_music).
#
# The scenes come back as a ComfyUI list, one prompt per element, with a
# matching list of durations, so a video node downstream renders every scene in
# one queue. H3ScenePick pulls a single scene out of that list.

import random
import re

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
    CORE_SKILL,
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
from .scenes_store import save_scene_bundle
from .template_vars import collect_template_vars, expand_all, log_template_vars
from .characters import cast_line_name, split_cast_line
from .common import (
    scale_reference_passthrough,
    AUTO,
    LITERAL_CAMERA_DIRECTIVE,
    VISUAL_STYLES,
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
    wildness_directive,
)
from .scenes_support import (
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

CROSSOVER_SKILLS = (CORE_SKILL, "h3-crossover", "h3-style-craft")

MAX_CAST_SOCKETS = 4
CAST_SOCKETS = tuple(f"cast_{i}" for i in range(1, MAX_CAST_SOCKETS + 1))

SHOTS_PER_SCENE = [AUTO, "1", "2", "3", "4"]

_SCENE_FIELDS = (
    "subject_definitions",
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)


def parse_cast(*blocks):
    """
    Merge cast blocks into unique, non-empty lines in first-seen order. A
    `| wardrobe: ...` suffix (H3 Characters with a wardrobe) is stripped here;
    cast_wardrobe() collects those suffixes as lock lines.
    """
    seen = set()
    cast = []
    for block in blocks:
        for line in (block or "").splitlines():
            line = line.strip().lstrip("-*• ").strip()
            if not line:
                continue
            # Tolerate lines already carrying a <Subject N> tag.
            line = re.sub(r"^<Subject\s+\d+>\s*", "", line, flags=re.IGNORECASE)
            line, _ = split_cast_line(line)
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            cast.append(line)
    return cast


def cast_wardrobe(*blocks):
    """`Name: anchors` lock lines carried on cast lines (H3 Characters `wardrobe`)."""
    locks = []
    seen = set()
    for block in blocks:
        for line in (block or "").splitlines():
            line = line.strip().lstrip("-*• ").strip()
            head, anchors = split_cast_line(line)
            if not anchors:
                continue
            name = cast_line_name(head)
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            locks.append(f"{name}: {anchors}")
    return locks


def log_wardrobe_locks(node_label, wardrobe):
    """Print the final wardrobe lock block, so a mis-wired cast socket (e.g.
    the `character` name output instead of `cast`, which carries the wardrobe)
    is caught at a glance."""
    if (wardrobe or "").strip():
        print(f"👔 {node_label} | wardrobe locks:\n"
              + "\n".join(f"    {l}" for l in wardrobe.splitlines() if l.strip()))
    else:
        print(f"👔 {node_label} | no wardrobe locks reached the writer - the model invents "
              "outfits. If you set wardrobe on H3 Characters, wire its `cast` output "
              "(not `character`) into cast_N.")


def log_cast(node_label, cast):
    """Print the parsed cast lines, so a phantom member (a chained cast_in,
    a second cast socket, leftover extra_cast text) is caught at a glance."""
    if cast:
        print(f"🎭 {node_label} | cast ({len(cast)}):\n"
              + "\n".join(f"    {c}" for c in cast))


def merge_wardrobe(user_wardrobe, cast_locks):
    """User-typed wardrobe lines win over cast-carried ones for the same name."""
    lines = [l.strip() for l in (user_wardrobe or "").splitlines() if l.strip()]
    have = {l.split(":", 1)[0].strip().lower() for l in lines if ":" in l}
    for lock in cast_locks:
        name = lock.split(":", 1)[0].strip().lower()
        if name not in have:
            lines.append(lock)
            have.add(name)
    return "\n".join(lines)


class H3ClaudeCodeCrossoverWriter:
    """
    APNext H3 Crossover Writer

    Writes 1-10 crossover scenes for a given cast with the local Claude Code CLI,
    following the production rules distilled from rendered crossover movies.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "direction": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": (
                    "Your steer for the AI: premise, tone, genre, where it happens, what "
                    "must happen, running gags, who should clash. Free text - this is the "
                    "creative brief the scenes are built from."
                ),
            }),
            "extra_cast": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": (
                    "Extra cast typed by hand, one per line, ideally as "
                    "`Character (played by Actor) from Show`. Merged with the cast sockets."
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
                "default": 15.0, "min": 5.0, "max": 20.0, "step": 0.5,
                "tooltip": (
                    "Seconds per scene in Fixed mode. Also the fallback if Claude omits a "
                    "duration in Vary mode. Cut timecodes are spread across this length."
                ),
            }),
            "shots_per_scene": (SHOTS_PER_SCENE, {
                "default": AUTO,
                "tooltip": "Shots per scene. Auto lets Claude choose 2-3 to fit the duration.",
            }),
            "visual_style": (VISUAL_STYLES, {
                "default": "Live-action, 35mm cinematic film aesthetic",
                "tooltip": (
                    "Opens every [Shot 1]; kept identical across the run for continuity. The "
                    "list is the guide's styles plus the APNext Cinematic vocabulary (film "
                    "stock, grading, aesthetics); Custom uses custom_visual_style, Auto lets "
                    "Claude pick one that suits the cast."
                ),
            }),
            "dialogue_language": (DIALOGUE_LANGUAGES, {"default": "English"}),
            "wildness": ("INT", {
                "default": 35, "min": 0, "max": 100, "step": 1,
                "tooltip": "0 = grounded and faithful to each show, 100 = fully unhinged. Above 40 seeds surreal events.",
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
                "tooltip": (
                    "A cast line or block from an H3 Characters node (its `cast` output). "
                    "Chain several Characters nodes or use several sockets."
                ),
            })
            for name in CAST_SOCKETS
        }
        optional["custom_dialogue_language"] = ("STRING", {"default": ""})
        optional["custom_visual_style"] = ("STRING", {
            "default": "",
            "tooltip": (
                "Any visual style not in the dropdown, e.g. 'hand-painted cel animation' or "
                "'Kodak Vision3 500T, anamorphic'. Overrides the dropdown when filled in."
            ),
        })
        optional["wardrobe"] = ("STRING", {"multiline": True, "default": "", "tooltip": WARDROBE_TOOLTIP})
        optional["enforce_wardrobe"] = ("BOOLEAN", {"default": True, "tooltip": ENFORCE_WARDROBE_TOOLTIP})
        optional["extra_instructions"] = ("STRING", {"multiline": True, "default": ""})
        optional["image_notes"] = ("STRING", {
            "multiline": True,
            "default": "",
            "tooltip": (
                "What each reference image is, one per line: `Image 1: Sheldon`, "
                "`Image 3: the diner, use as the location`. With reference_image_use = "
                "Characters only (the default), such a note is the ONLY way a picture may be "
                "read as a location or prop - otherwise every picture is a character and its "
                "backdrop is ignored. In Auto mode Claude works it out from the cast and the "
                "pictures."
            ),
        })
        optional.update(claude_code_optional_inputs())
        optional["locations"] = ("STRING", {"multiline": True, "default": "", "tooltip": LOCATIONS_TOOLTIP})
        optional.update(local_llm_inputs())
        optional.update(context_inputs())
        # Image sockets last, so the front-end can grow and trim them at the tail.
        optional.update(reference_image_inputs())
        # appended LAST so saved workflows keep their widget positions
        optional["reference_image_use"] = (REFERENCE_IMAGE_USE, {
            "default": REFERENCE_IMAGE_USE[0],
            "tooltip": REFERENCE_IMAGE_USE_TOOLTIP,
        })
        optional["scene_briefs"] = ("STRING", {
            "forceInput": True,
            "tooltip": (
                "Manually planned scenes from chained H3 Scene Brief nodes: each brief "
                "(what happens, where, which cast members and pictures) becomes the "
                "binding plan for its scene. Pinned numbers take that scene; unpinned "
                "briefs fill in order; scenes without a brief stay the model's to invent."
            ),
        })
        # appended LAST so saved workflows keep their widget positions
        optional.update(project_name_input())
        optional["avoid_previous"] = ("INT", {
            "default": 0, "min": 0, "max": 20,
            "tooltip": (
                "NO LONGER USED - every run is a clean slate. This once fed the synopses "
                "of previous saved runs back to the model as concepts that were USED UP, "
                "but quoting an old logline under a 'do not reuse' header primes the idea "
                "far more reliably than it forbids it: the run reproduced what it was told "
                "to avoid, got saved, and fed itself back. The slot stays so saved "
                "workflows keep their widget positions; the value is ignored."
            ),
        })

        return {"required": required, "optional": optional, "hidden": context_hidden_inputs()}

    _IMAGE_OUTPUT_TYPES, _IMAGE_OUTPUT_NAMES = reference_image_outputs()
    RETURN_TYPES = (
        ("STRING", "FLOAT", "STRING", "STRING", "STRING", "INT", "STRING", "STRING")
        + _IMAGE_OUTPUT_TYPES
        + ("STRING",)
    )
    RETURN_NAMES = (
        "scenes",
        "durations",
        "scenes_text",
        "synopsis",
        "cast",
        "scene_count",
        "session_id",
        "info",
    ) + _IMAGE_OUTPUT_NAMES + ("project_name",)
    OUTPUT_IS_LIST = (True, True) + (False,) * (6 + len(_IMAGE_OUTPUT_NAMES)) + (False,)
    FUNCTION = "write_scenes"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Writes 1-10 MiniMax-H3 crossover scenes for a cast of characters with the local "
        "Claude Code CLI. `scenes` and `durations` are lists (one element per scene) so a "
        "video node downstream renders them all; use H3 Scene Pick to grab one."
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
                session_id, working_dir, director, skills=CROSSOVER_SKILLS, local=local,
            )
            return text, repair_info

        return enforce_continuity(
            enabled, synopsis, parsed, scene_count, scene_duration, session_id, info, repair,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self, continuity_mode=CONTINUITY_MODES[0], characters_only=True):
        base_guide = load_guide("guide_base_en.md")
        crossover_guide = load_guide("guide_crossover_en.md")
        return (
            "You are a MiniMax-H3 crossover screenwriter and prompt engineer. You are "
            "given a cast of characters from different shows and films and a creative brief, "
            "and you write a short run of self-contained scenes in which those characters "
            "meet, each scene as a complete production-ready H3 T2VA prompt.\n\n"
            "Two documents follow. The official guide is authoritative for grammar, camera "
            "vocabulary, timestamps, speaker IDs and <d> blocks. The crossover rules are "
            "authoritative for the four-section scene layout, subject binding, silence "
            "mandates, pacing and story flow across scenes.\n\n"
            "=== BEGIN MINIMAX-H3 VIDEO PROMPT WRITING GUIDE ===\n"
            f"{base_guide}\n"
            "=== END MINIMAX-H3 VIDEO PROMPT WRITING GUIDE ===\n\n"
            "=== BEGIN MINIMAX-H3 CROSSOVER SCENE RULES ===\n"
            f"{crossover_guide}\n"
            "=== END MINIMAX-H3 CROSSOVER SCENE RULES ==="
            + chain_system_block(continuity_mode) + "\n\n"
            + envelope_contract(_SCENE_FIELDS) + "\n"
            "- The synopsis block also carries `Cast:` with one line per character and why "
            "they are there.\n"
            "- Use the character, actor and show strings exactly as given.\n"
            "- Write everything in English except dialogue, which uses the requested language "
            "inside the <d>[...] tag.\n"
            "- When reference pictures are attached the scenes are rendered with the same "
            "pictures as <Picture 1>..<Picture N> (ref2va). Keep the four-section layout, but "
            "bind each pictured character in subject_definitions as "
            "`<Subject N> Character (played by Actor) from Show, appearance from <Picture k>`"
            + (
                "; refer to the pictures again inside the shots where the characters appear. "
                "Only a picture the user's notes explicitly declare a location or prop may "
                "be treated as one (`<Picture k> is ...` on its own line); a pictured "
                "person's backdrop is NEVER the scene."
                if characters_only else
                " and treat a location or prop picture as `<Picture k> is ...` on its own "
                "line; refer to the pictures again inside the shots where their content "
                "appears."
            )
            + " A label keeps one meaning across the whole run and is never renumbered."
        )

    def _build_user_prompt(
        self,
        cast,
        direction,
        scene_count,
        duration_mode,
        continuity_mode,
        scene_duration,
        shots_per_scene,
        visual_style,
        dialogue_language,
        wildness,
        extra_instructions,
        rng,
        wardrobe="",
        image_labels=(),
        image_notes="",
        locations="",
        characters_only=True,
        scene_briefs="",
    ):
        lines = ["CAST (use these strings verbatim in subject_definitions):"]
        lines += [f"- {c}" for c in cast]
        lines.append("")
        lines.append("CREATIVE BRIEF FROM THE USER:")
        lines.append(direction.strip() or "(none - invent a premise that fits this cast)")
        lines.append("")
        briefs = (scene_briefs or "").strip()
        if briefs:
            lines.append("SCENE BRIEFS FROM THE USER - the plan for those scenes:")
            lines.append(briefs)
            lines.append("")
        lines.append("DIRECTIVES:")
        if briefs:
            lines.append(
                "- SCENE BRIEFS ARE BINDING: a numbered brief is the plan for that scene - "
                "set it where the brief says, put exactly the cast members it names on "
                "screen, use the reference pictures it points at, honour its camera wish, "
                "and stage what it describes within the scene's duration. `SCENE (next in "
                "order)` briefs fill scenes in order from 01, skipping numbered ones. "
                "Scenes without a brief are yours to write - but never contradict a brief."
            )
        lines.append(
            f"- Write exactly {scene_count} scene{'s' if scene_count != 1 else ''}, "
            f"numbered 01 to {scene_count:02d}, forming one continuous story with "
            "hand-offs between adjacent scenes."
        )
        lines.append(f"- {duration_directive(duration_mode, scene_duration)}")
        lines.append(f"- {continuity_directive(continuity_mode)}")
        if shots_per_scene == AUTO:
            lines.append("- Use 2-3 shots per scene, varying the pattern between scenes.")
        else:
            lines.append(f"- Use exactly {shots_per_scene} shot(s) per scene.")
        lines.append(f"- {LITERAL_CAMERA_DIRECTIVE}")
        if visual_style == AUTO:
            lines.append(
                "- Visual style: choose one concrete style that suits this cast (e.g. "
                "`Live-action, 35mm cinematic film aesthetic`), open every [Shot 1] with it "
                "followed by lighting, time of day and setting, and keep it identical across "
                "the run."
            )
        else:
            lines.append(
                f"- Visual style: open every [Shot 1] with `{visual_style.strip()}` followed by "
                "lighting, time of day and setting; keep it identical across the run."
            )
        if dialogue_language:
            lines.append(
                f"- Dialogue language: {dialogue_language}. Every <d> tag reads "
                f"`<d>[{dialogue_language} in Character's voice from Show] ...</d>`."
            )
        else:
            lines.append(
                "- Dialogue language: pick one that fits the setting and use it consistently "
                "in every <d> tag."
            )
        if image_labels:
            labels = ", ".join(f"<Picture {i}>" for i in image_labels)
            wardrobe_clause = (
                "take their wardrobe lock from what the picture shows, and restate it "
                "in the shots."
                if not (wardrobe or "").strip() else
                "the picture fixes their face, hair and build, while the written "
                "wardrobe lock below fixes the clothes - where they differ, the written "
                "lock wins."
            )
            lines.append(
                f"- Reference pictures: {len(image_labels)} attached, in order {labels}; the "
                "video model receives the same pictures under the same labels in every "
                "scene. "
                + (
                    characters_only_directive() + " Match faces and costumes to the cast, "
                    "bind pictured characters to their <Picture k> in subject_definitions, "
                    + wardrobe_clause
                    if characters_only else
                    "Decide what each one shows (use the notes below, else match faces "
                    "and costumes to the cast); bind pictured characters to their <Picture k> in "
                    "subject_definitions, " + wardrobe_clause +
                    " Unmatched pictures are locations or props: "
                    "define them as `<Picture k> is ...` and use them where the story needs them."
                )
            )
            notes = (image_notes or "").strip()
            if notes:
                lines.append("- Picture notes from the user:")
                lines.extend(f"    {n.strip()}" for n in notes.splitlines() if n.strip())
        lines.append(f"- {wardrobe_directive(wardrobe)}")
        lines.append(f"- {locations_directive(locations)}")
        lines.append(
            "- Every listed character appears at least once across the run (unless the brief "
            "says otherwise); introduce them with an on-screen reason to be there and stagger "
            "arrivals rather than crowding scene 01. Max 3 people visible in one shot."
        )
        lines.append(
            "- The character with the most dialogue in a scene is <Subject 1> in that scene."
        )
        wild_lines, wild_label = wildness_directive(wildness, rng)
        lines += [f"- {w}" for w in wild_lines]
        extra = (extra_instructions or "").strip()
        if extra:
            lines.append("")
            lines.append("EXTRA INSTRUCTIONS:")
            lines.append(extra)
        lines.append("")
        lines.append(
            "Now write the synopsis block and the scene envelopes, and nothing else."
        )
        return "\n".join(lines), wild_label

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def write_scenes(
        self,
        direction,
        extra_cast,
        scene_count,
        duration_mode,
        continuity_mode,
        scene_duration,
        shots_per_scene,
        visual_style,
        dialogue_language,
        wildness,
        model,
        research,
        director,
        use_subscription,
        timeout_seconds,
        seed,
        custom_dialogue_language="",
        custom_visual_style="",
        wardrobe="",
        enforce_wardrobe=True,
        extra_instructions="",
        image_notes="",
        resume_session_id="",
        working_dir="",
        locations="",
        llm=None,
        reference_image_use=None,
        scene_briefs="",
        project_name="",
        avoid_previous=0,
        **cast_slots,
    ):
        project_name = resolve_project_name(project_name, seed)
        print(f"🎬 H3 Crossover Writer | project: {project_name}")
        # The same tensors go back out on image_1..image_9 so the video node
        # can be wired from here; Claude gets downscaled copies.
        passthrough = scale_reference_passthrough(cast_slots, self._IMAGE_OUTPUT_NAMES)
        references = collect_reference_images(passthrough, tensor2pil)
        images = [downscale_for_vision(pil) for _, pil in references] or None
        image_labels = tuple(range(1, len(references) + 1))
        template_vars, template_summary = collect_template_vars(cast_slots)
        direction, extra_cast, wardrobe, extra_instructions, image_notes, custom_dialogue_language, custom_visual_style, locations, scene_briefs = expand_all(
            template_vars, direction, extra_cast, wardrobe, extra_instructions, image_notes, custom_dialogue_language, custom_visual_style, locations, scene_briefs
        )
        log_template_vars(template_vars, template_summary, direction, extra_cast, wardrobe, extra_instructions, image_notes, custom_dialogue_language, custom_visual_style)
        context_text, context_entries = build_context(cast_slots, target="the scenes")
        cast_blocks = [cast_slots.get(name) for name in CAST_SOCKETS] + [extra_cast]
        cast = parse_cast(*cast_blocks)
        cast_text = "\n".join(cast)
        log_cast("H3 Crossover Writer", cast)
        wardrobe = merge_wardrobe(wardrobe, cast_wardrobe(*cast_blocks))
        log_wardrobe_locks("H3 Crossover Writer", wardrobe)
        try:
            if not cast:
                raise ValueError(
                    "No cast. Wire an H3 Characters `cast` output into cast_1 or type "
                    "characters into extra_cast."
                )

            dialogue_language = resolve_dialogue_language(
                dialogue_language, custom_dialogue_language
            )
            visual_style = resolve_visual_style(visual_style, custom_visual_style)
            current_seed = seed if seed != -1 else random.randint(0, 0xffffffffffffffff)
            rng = random.Random(current_seed)

            user_prompt, wild_label = self._build_user_prompt(
                cast,
                direction,
                scene_count,
                duration_mode,
                continuity_mode,
                scene_duration,
                shots_per_scene,
                visual_style,
                dialogue_language,
                wildness,
                directions_with_research(extra_instructions, research),
                rng,
                wardrobe=wardrobe,
                image_labels=image_labels,
                image_notes=image_notes,
                locations=locations,
                characters_only=characters_only_refs(reference_image_use),
                scene_briefs=scene_briefs,
            )
            user_prompt = with_context(user_prompt, context_text)

            print(
                f"🎬 H3 Crossover Writer | {len(cast)} cast | {scene_count} scene(s) | "
                f"context: {context_summary(context_entries)} | "
                f"{duration_mode.split(' ')[0].lower()} {scene_duration:.1f}s | "
                f"{'chain' if continuity_mode == CONTINUITY_MODES[1] else 'independent'} | "
                f"{len(references)} reference image(s) | "
                f"wildness {wildness} ({wild_label}) | research {'on' if research else 'off'} | "
                f"director {'on' if director else 'off'} | seed {current_seed}"
            )

            local = local_llm_options(llm)
            text, session_id, info = run_h3_claude_code(
                self._build_system_prompt(continuity_mode, characters_only_refs(reference_image_use)),
                user_prompt,
                images,
                model,
                research,
                use_subscription,
                timeout_seconds,
                resume_session_id,
                working_dir,
                director,
                skills=CROSSOVER_SKILLS,
                local=local,
            )

            synopsis, parsed = parse_scenes(text, scene_duration)
            if not parsed:
                raise ValueError("Claude Code returned no scenes.")
            if len(parsed) != scene_count:
                print(
                    f"⚠️ H3 Crossover Writer: asked for {scene_count} scene(s), parsed {len(parsed)}."
                )

            synopsis, parsed, info = self._enforce_wardrobe(
                enforce_wardrobe, synopsis, parsed, scene_count, scene_duration,
                session_id, info, model, use_subscription, timeout_seconds,
                working_dir, director, continuity_mode, local=local,
            )

            scenes = [p for _, _, p in parsed]
            durations = [d for _, d, _ in parsed]
            scenes_text = scenes_to_text(synopsis, parsed)

            save_scene_bundle(
                "H3ClaudeCodeCrossoverWriter", synopsis, scenes, [], durations,
                [], [], cast_text, sum(durations), scenes_text, "", info,
                project_name=project_name,
            )

            return (
                scenes,
                durations,
                scenes_text,
                synopsis,
                cast_text,
                len(scenes),
                session_id,
                info,
            ) + passthrough + (project_name_prefix(project_name),)

        except Exception as exc:
            if is_interrupt(exc):
                raise
            print(f"❌ H3 Crossover Writer error: {exc}")
            import traceback

            print(traceback.format_exc())
            # A failed run is an ERROR, not a scene list: returning the message as
            # the scenes let a dead backend (Ollama down, CLI missing, timeout) go
            # on to a full render of the error text - 31 minutes on 27 Aug. Raising
            # marks this node red with the reason and stops the graph here.
            raise RuntimeError(f"H3 Crossover Writer: {exc}") from exc
