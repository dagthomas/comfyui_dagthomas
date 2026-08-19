# APNext H3 Music Video Writer
#
# Takes a song (AUDIO), an optional cast, lyrics and a concept, cuts the song
# into pieces no longer than MiniMax-H3 can render in one clip (5-15 s, cut on
# the music / before lyric lines, snapped to H3's frame grid), and has the model
# write ONE scene per piece - a complete four-section H3 prompt in which the
# piece is <Audio 1>, reused 1:1 as the clip's soundtrack, and the performer
# sings the piece's lyric lines on camera. Everything comes out as lists the
# H3 video node renders one element at a time: `scenes`, `durations`,
# `lengths` (frame counts) and `audio_segments` (the matching AUDIO slices for
# ref_audio_1). H3 Scenes Join (+ replace_audio = the original song) stitches
# the clips into the finished music video.

import random

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
    run_h3_claude_code,
)
from .claude_code_crossover_writer import CAST_SOCKETS, cast_wardrobe, merge_wardrobe, parse_cast
from .template_vars import collect_template_vars, expand_all, log_template_vars
from .common import (
    AUTO,
    VISUAL_STYLES,
    resolve_visual_style,
    DIALOGUE_LANGUAGES,
    collect_reference_images,
    downscale_for_vision,
    load_guide,
    reference_image_inputs,
    reference_image_outputs,
    resolve_dialogue_language,
    wildness_directive,
)
from .music_support import (
    SEGMENT_MODES,
    energy_labels,
    fmt_time,
    frames_for_seconds,
    lyrics_for_segment,
    parse_lyrics,
    place_untimed_lyrics,
    segment_song,
    segment_table,
    slice_audio,
)
from .scenes_support import (
    ENFORCE_WARDROBE_TOOLTIP,
    WARDROBE_TOOLTIP,
    LOCATIONS_TOOLTIP,
    enforce_continuity,
    locations_directive,
    wardrobe_directive,
    envelope_contract,
    parse_scenes,
    scenes_to_text,
)

MUSIC_SKILLS = (CORE_SKILL, "h3-ref2va", "h3-style-craft")

_SCENE_FIELDS = (
    "subject_definitions",
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)

PERFORMANCE_MODES = [
    "Performance (the singer lip-syncs the lyrics on camera)",
    "Narrative (story visuals, nobody sings on camera)",
    "Mixed (performance and story, alternate or blend)",
]

SHOTS_PER_SCENE = [AUTO, "1", "2", "3", "4"]

# How many scenes to ask for per model call. A 3-minute song is ~13-20 scenes;
# one answer that long gets sloppy, so the run is split into chunks that
# continue the same session (same synopsis, same locks).
SCENES_PER_CALL = 6


class H3ClaudeCodeMusicVideoWriter:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "audio": ("AUDIO", {
                "tooltip": "The song. It is cut into 5-15 s pieces and every piece becomes one scene / one clip.",
            }),
            "direction": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": (
                    "The music-video concept: who performs, where, the look, the story arc, "
                    "recurring motifs, what the chorus looks like vs the verses. Free text."
                ),
            }),
            "lyrics": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": (
                    "Lyrics, one line per line. Timestamps make the sync exact: `[0:15] line`, "
                    "`0:15 line` or LRC `[00:15.20] line`; section tags like [Chorus] are kept. "
                    "Untimed lines are spread evenly over the song (approximate). Empty = "
                    "instrumental video."
                ),
            }),
            "performance_mode": (PERFORMANCE_MODES, {"default": PERFORMANCE_MODES[0]}),
            "segment_mode": (SEGMENT_MODES, {
                "default": SEGMENT_MODES[0],
                "tooltip": (
                    "How the song is cut. Auto cuts on onsets / energy changes inside the "
                    "allowed length range (lyric-line starts are preferred when lyrics are "
                    "timed); Fixed takes the longest allowed piece every time; Lyric lines "
                    "tries hardest to cut right before a line."
                ),
            }),
            "max_segment_seconds": ("FLOAT", {
                "default": 15.0, "min": 5.2, "max": 15.1, "step": 0.1,
                "tooltip": "Longest piece (H3 renders up to ~15 s). Lengths snap to H3's frame grid.",
            }),
            "min_segment_seconds": ("FLOAT", {
                "default": 5.2, "min": 5.2, "max": 15.0, "step": 0.1,
                "tooltip": "Shortest piece. 124 frames (~5.2 s) is the shortest trained clip.",
            }),
            "shots_per_scene": (SHOTS_PER_SCENE, {
                "default": AUTO,
                "tooltip": "Shots per scene. Auto lets the model cut to the music (more shots in loud parts).",
            }),
            "visual_style": (VISUAL_STYLES, {
                "default": "Live-action, 35mm cinematic film aesthetic",
                "tooltip": "Opens every [Shot 1]; kept identical across the whole video.",
            }),
            "dialogue_language": (DIALOGUE_LANGUAGES, {
                "default": "English",
                "tooltip": "Language of the lyrics (the <d>[...] tag).",
            }),
            "wildness": ("INT", {
                "default": 45, "min": 0, "max": 100, "step": 1,
                "tooltip": "0 = grounded performance video, 100 = fully surreal. Above 40 seeds surreal events.",
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
                    "A cast line or block from an H3 Characters node (its `cast` output): the "
                    "performer(s) and anyone else in the video."
                ),
            })
            for name in CAST_SOCKETS
        }
        optional["extra_cast"] = ("STRING", {
            "multiline": True,
            "default": "",
            "tooltip": (
                "Performer(s) / characters typed by hand, one per line, e.g. `Lead singer: "
                "a woman in her 30s with a platinum pixie cut` or `Character (played by "
                "Actor) from Show`. Merged with the cast sockets."
            ),
        })
        optional["custom_dialogue_language"] = ("STRING", {"default": ""})
        optional["custom_visual_style"] = ("STRING", {
            "default": "",
            "tooltip": "Any visual style not in the dropdown; overrides the dropdown when filled in.",
        })
        optional["wardrobe"] = ("STRING", {"multiline": True, "default": "", "tooltip": WARDROBE_TOOLTIP})
        optional["locations"] = ("STRING", {"multiline": True, "default": "", "tooltip": LOCATIONS_TOOLTIP})
        optional["enforce_wardrobe"] = ("BOOLEAN", {"default": True, "tooltip": ENFORCE_WARDROBE_TOOLTIP})
        optional["extra_instructions"] = ("STRING", {"multiline": True, "default": ""})
        optional["image_notes"] = ("STRING", {
            "multiline": True,
            "default": "",
            "tooltip": "Per-picture notes, one per line: `Image 1: the singer`, `Image 2: the rooftop`.",
        })
        optional.update(claude_code_optional_inputs())
        optional.update(local_llm_inputs())
        optional.update(context_inputs())
        optional.update(reference_image_inputs())

        return {"required": required, "optional": optional, "hidden": context_hidden_inputs()}

    _IMAGE_OUTPUT_TYPES, _IMAGE_OUTPUT_NAMES = reference_image_outputs()
    RETURN_TYPES = (
        ("STRING", "FLOAT", "INT", "AUDIO", "STRING", "STRING", "STRING", "STRING", "INT", "FLOAT", "STRING", "STRING")
        + _IMAGE_OUTPUT_TYPES
    )
    RETURN_NAMES = (
        "scenes",
        "durations",
        "lengths",
        "audio_segments",
        "segment_table",
        "scenes_text",
        "synopsis",
        "cast",
        "scene_count",
        "song_seconds",
        "session_id",
        "info",
    ) + _IMAGE_OUTPUT_NAMES
    OUTPUT_IS_LIST = (True, True, True, True) + (False,) * (8 + len(_IMAGE_OUTPUT_NAMES))
    FUNCTION = "write_video"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Cuts a song into 5-15 s pieces on the music and writes one MiniMax-H3 scene per "
        "piece (the piece is <Audio 1>, reused 1:1; the performer sings its lyric lines), so "
        "a downstream H3 video node renders the whole music video clip by clip. `scenes`, "
        "`durations`, `lengths` and `audio_segments` are matching lists; join the clips with "
        "H3 Scenes Join (replace_audio = the song)."
    )

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        return float("nan") if seed == -1 else seed

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self):
        base_guide = load_guide("guide_base_en.md")
        ref_guide = load_guide("guide_ref_en.md")
        return (
            "You are a MiniMax-H3 music-video director and prompt engineer. You are given a "
            "song that has already been cut into consecutive pieces of 5-15 seconds, the "
            "lyric lines that fall inside each piece, the energy of each piece, the "
            "performer(s) and a creative concept. You write ONE scene per piece, in order, "
            "each a complete production-ready H3 prompt that is rendered with that piece of "
            "the song as its soundtrack, so that the clips cut together into one music video.\n\n"
            "Two documents follow. The official guide is authoritative for grammar, camera "
            "vocabulary, timestamps, speaker IDs and <d> blocks. The reference guide is "
            "authoritative for the <Picture N> / <Audio N> labels and the relationship "
            "markers.\n\n"
            "=== BEGIN MINIMAX-H3 VIDEO PROMPT WRITING GUIDE ===\n"
            f"{base_guide}\n"
            "=== END MINIMAX-H3 VIDEO PROMPT WRITING GUIDE ===\n\n"
            "=== BEGIN MINIMAX-H3 REFERENCE (REF2VA) GUIDE ===\n"
            f"{ref_guide}\n"
            "=== END MINIMAX-H3 REFERENCE (REF2VA) GUIDE ===\n\n"
            + envelope_contract(_SCENE_FIELDS) + "\n"
            "- The synopsis block also carries `Cast:` (one line per performer/character), "
            "`Concept:` (the visual idea in two or three sentences) and `Motifs:` (recurring "
            "images, colours, props and the chorus look, so every scene can reuse them).\n"
            "- Each scene envelope's `duration:` is the exact length of its song piece, given "
            "to you; copy it.\n"
            "- AUDIO: every scene is rendered with its own song piece attached as <Audio 1>. "
            "In subject_definitions define it as `<Audio 1> is the song, reused 1:1 as the "
            "complete final soundtrack of this clip` (fully_copy). Cite <Audio 1> in the "
            "description where the music drives the picture. overall_soundscape is "
            "`<Audio 1> is the complete soundtrack; no other sound.` and non_diegetic_music "
            "is `N/A (the song is <Audio 1>)`. Never invent other music, ambience or effects.\n"
            "- LYRICS: the lines listed for a piece are sung at the moments given. In "
            "Performance mode the singer visibly sings them on camera, lip-synced: write "
            "`<Subject 1> sings <d>[Language] exact lyric line</d> in sync with <Audio 1>` at "
            "the right timestamp, with the exact words, once per line, no paraphrase. In "
            "Narrative mode nobody sings on camera - the lyric is audible from <Audio 1> and "
            "the picture answers it (`as <Audio 1> reaches <d>[Language] line</d>, ...`). "
            "Mixed mode alternates. Instrumental pieces get pure visuals cut to the beat.\n"
            "- ENERGY: quiet pieces get long, slow, intimate shots; loud and peak pieces get "
            "more cuts, bigger moves, bolder light, the chorus look. The whole video keeps "
            "one style, one palette, one performer identity, the wardrobe and location locks.\n"
            "- Write everything in English except the lyric lines, which stay in the song's "
            "language inside the <d>[...] tag.\n"
            "- When reference pictures are attached the scenes are rendered with the same "
            "pictures as <Picture 1>..<Picture N>. Bind pictured performers in "
            "subject_definitions as `<Subject N> ..., appearance from <Picture k>`, treat a "
            "location or prop picture as `<Picture k> is ...` on its own line, and refer to "
            "the pictures again inside the shots where their content appears."
        )

    def _build_user_prompt(
        self,
        cast,
        direction,
        segments,
        labels,
        frames,
        placed_lyrics,
        performance_mode,
        shots_per_scene,
        visual_style,
        dialogue_language,
        wildness,
        extra_instructions,
        rng,
        wardrobe="",
        locations="",
        image_labels=(),
        image_notes="",
        first=1,
        last=None,
        total_seconds=0.0,
    ):
        last = last or len(segments)
        n = len(segments)
        lines = ["CAST (use these strings verbatim in subject_definitions):"]
        lines += [f"- {c}" for c in cast] if cast else ["- (no cast given - invent a performer that fits the concept and keep them identical in every scene)"]
        lines.append("")
        lines.append("CONCEPT FROM THE USER:")
        lines.append(direction.strip() or "(none - invent a concept that fits the song, its lyrics and its energy)")
        lines.append("")
        lines.append(
            f"THE SONG: {fmt_time(total_seconds)} long, cut into {n} consecutive pieces. One scene per piece, "
            "in this order. Each piece's lyric lines are given with the second inside the piece at which "
            "they start ([+s]); `exact` times come from the user's timestamps, `~` times are estimates."
        )
        for i, ((s, e), label, fr) in enumerate(zip(segments, labels, frames), 1):
            lyr = lyrics_for_segment(placed_lyrics, s, e)
            sung = [l for l in lyr if l[1] and not l[1].startswith("#")]
            tags = [l[1][1:] for l in lyr if l[1].startswith("#")]
            marker = "  <-- write this one" if first <= i <= last else ""
            lines.append(
                f"PIECE {i:02d}: {fmt_time(s)}-{fmt_time(e)} | duration {e - s:.2f} | energy {label}"
                + (f" | section {'/'.join(tags)}" if tags else "") + marker
            )
            if sung:
                for t, line, exact in sung:
                    lines.append(f"    [+{t - s:5.2f}s {'exact' if exact else '~'}] {line}")
            else:
                lines.append("    [instrumental]")
        lines.append("")
        lines.append("DIRECTIVES:")
        if first == 1:
            lines.append(
                f"- Write the synopsis block, then scenes {first:02d} to {last:02d} (of {n}). "
                "The remaining scenes are requested in follow-up turns; plan the whole video now."
                if last < n else
                f"- Write the synopsis block and all {n} scenes, numbered 01 to {n:02d}."
            )
        else:
            lines.append(
                f"- Continue the SAME video: write scenes {first:02d} to {last:02d} (of {n}) only, "
                "following the synopsis, cast, wardrobe and location locks you already fixed. "
                "Start directly with the scene envelopes; do not repeat the synopsis."
            )
        lines.append(f"- Performance mode: {performance_mode}.")
        if shots_per_scene != AUTO:
            lines.append(f"- Use exactly {shots_per_scene} shot(s) per scene.")
        else:
            lines.append(
                "- Shots per scene: your call, cut to the music - 1-2 in quiet pieces, 2-4 in "
                "loud and peak pieces, each cut on a beat or a lyric."
            )
        if visual_style == AUTO:
            lines.append(
                "- Visual style: choose one concrete style that suits the song and keep it "
                "identical in every scene, opening every [Shot 1] with it."
            )
        else:
            lines.append(
                f"- Visual style: open every [Shot 1] with `{visual_style.strip()}` followed by "
                "lighting, time of day and setting; keep it identical across the video."
            )
        lines.append(
            f"- Lyric language: {dialogue_language or 'the song language'}. Every sung line reads "
            f"`<d>[{dialogue_language or 'Language'}] exact words</d>`."
        )
        if image_labels:
            labels_s = ", ".join(f"<Picture {i}>" for i in image_labels)
            lines.append(
                f"- Reference pictures: {len(image_labels)} attached, in order {labels_s}; the "
                "video model receives the same pictures under the same labels in every scene. "
                "Decide what each shows (use the notes below); bind pictured performers to "
                "their <Picture k> in subject_definitions and take their wardrobe lock from "
                "the picture."
            )
            notes = (image_notes or "").strip()
            if notes:
                lines.append("- Picture notes from the user:")
                lines.extend(f"    {n_.strip()}" for n_ in notes.splitlines() if n_.strip())
        lines.append(f"- {wardrobe_directive(wardrobe)}")
        lines.append(f"- {locations_directive(locations)}")
        lines.append(
            "- Continuity: the same performer identity, wardrobe, palette and style in every "
            "scene; recurring locations restate their anchors; the chorus pieces share one "
            "signature look; the last scene resolves the concept."
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
            "Now write the requested envelopes (and the synopsis block when asked), and nothing else."
        )
        return "\n".join(lines), wild_label

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def write_video(
        self,
        audio,
        direction,
        lyrics,
        performance_mode,
        segment_mode,
        max_segment_seconds,
        min_segment_seconds,
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
        extra_cast="",
        custom_dialogue_language="",
        custom_visual_style="",
        wardrobe="",
        locations="",
        enforce_wardrobe=True,
        extra_instructions="",
        image_notes="",
        resume_session_id="",
        working_dir="",
        llm=None,
        **cast_slots,
    ):
        passthrough = tuple(cast_slots.get(name) for name in self._IMAGE_OUTPUT_NAMES)
        references = collect_reference_images(passthrough, tensor2pil)
        images = [downscale_for_vision(pil) for _, pil in references] or None
        image_labels = tuple(range(1, len(references) + 1))
        template_vars, template_summary = collect_template_vars(cast_slots)
        (direction, lyrics, extra_cast, wardrobe, locations, extra_instructions, image_notes,
         custom_dialogue_language, custom_visual_style) = expand_all(
            template_vars, direction, lyrics, extra_cast, wardrobe, locations, extra_instructions,
            image_notes, custom_dialogue_language, custom_visual_style,
        )
        log_template_vars(template_vars, template_summary, direction, extra_cast, wardrobe, locations, extra_instructions, image_notes)
        context_text, context_entries = build_context(cast_slots, target="the music video")
        cast_blocks = [cast_slots.get(name) for name in CAST_SOCKETS] + [extra_cast]
        cast = parse_cast(*cast_blocks)
        cast_text = "\n".join(cast)
        wardrobe = merge_wardrobe(wardrobe, cast_wardrobe(*cast_blocks))

        # --- cut the song -------------------------------------------------
        parsed_lyrics = parse_lyrics(lyrics)
        timed = [t for t, _ in parsed_lyrics if t is not None]
        segments, feats = segment_song(
            audio, max_segment_seconds, min_segment_seconds, segment_mode, lyric_times=timed,
        )
        total_seconds = feats["duration"]
        placed = place_untimed_lyrics(parsed_lyrics, total_seconds)
        labels = energy_labels(feats, segments)
        frames = [
            frames_for_seconds(e - s, round_up=(i == len(segments) - 1))
            for i, (s, e) in enumerate(segments)
        ]
        durations = [seconds_for(fr) for fr in frames]
        audio_segments = [slice_audio(audio, s, e, fr) for (s, e), fr in zip(segments, frames)]
        table = segment_table(segments, labels, frames, placed)
        n = len(segments)
        print(f"🎵 H3 Music Video Writer | {fmt_time(total_seconds)} song -> {n} piece(s)\n{table}")

        try:
            dialogue_language = resolve_dialogue_language(dialogue_language, custom_dialogue_language)
            visual_style = resolve_visual_style(visual_style, custom_visual_style)
            current_seed = seed if seed != -1 else random.randint(0, 0xffffffffffffffff)
            rng = random.Random(current_seed)
            local = local_llm_options(llm)
            system_prompt = self._build_system_prompt()

            print(
                f"🎬 H3 Music Video Writer | {len(cast)} cast | {n} scene(s) | "
                f"context: {context_summary(context_entries)} | {len(references)} reference image(s) | "
                f"{performance_mode.split(' ')[0].lower()} | wildness {wildness} | "
                f"research {'on' if research else 'off'} | director {'on' if director else 'off'} | seed {current_seed}"
            )

            # --- write, in chunks that continue one session -------------------
            synopsis = ""
            parsed = []
            session_id = (resume_session_id or "").strip()
            infos = []
            first = 1
            while first <= n:
                last = min(n, first + SCENES_PER_CALL - 1)
                user_prompt, wild_label = self._build_user_prompt(
                    cast, direction, segments, labels, frames, placed, performance_mode,
                    shots_per_scene, visual_style, dialogue_language, wildness,
                    directions_with_research(extra_instructions, research), rng,
                    wardrobe=wardrobe, locations=locations, image_labels=image_labels,
                    image_notes=image_notes, first=first, last=last, total_seconds=total_seconds,
                )
                if first == 1:
                    user_prompt = with_context(user_prompt, context_text)
                text, session_id, info = run_h3_claude_code(
                    None if (first > 1 and session_id) else system_prompt,
                    user_prompt,
                    images if first == 1 else None,
                    model, research, use_subscription, timeout_seconds,
                    session_id, working_dir, director, skills=MUSIC_SKILLS, local=local,
                )
                infos.append(info)
                chunk_synopsis, chunk = parse_scenes(text, durations[first - 1])
                if first == 1:
                    synopsis = chunk_synopsis
                if not chunk:
                    raise ValueError(f"the model returned no scenes for pieces {first:02d}-{last:02d}.")
                got = [(no, d, p) for no, d, p in chunk]
                # renumber defensively onto the requested range
                for k, (no, _d, p) in enumerate(got[: last - first + 1]):
                    idx = first + k
                    parsed.append((idx, durations[idx - 1], p))
                if len(got) < last - first + 1:
                    print(f"⚠️ H3 Music Video Writer: asked for scenes {first:02d}-{last:02d}, got {len(got)}.")
                first = first + max(1, min(len(got), last - first + 1))

            if len(parsed) != n:
                print(f"⚠️ H3 Music Video Writer: {n} piece(s) but {len(parsed)} scene(s) parsed.")

            # --- continuity check (wardrobe + locations), one repair turn --------
            def repair(prompt):
                text, _, repair_info = run_h3_claude_code(
                    system_prompt, prompt, None, model, False, use_subscription, timeout_seconds,
                    session_id, working_dir, director, skills=MUSIC_SKILLS, local=local,
                )
                return text, repair_info

            info = " || ".join(infos)
            if len(parsed) == n and n <= SCENES_PER_CALL:
                synopsis, parsed, info = enforce_continuity(
                    enforce_wardrobe, synopsis, parsed, n, durations[0], session_id, info, repair,
                )
            elif enforce_wardrobe:
                # multi-chunk runs: verify and report, but a full re-emit is too long to ask for
                from .scenes_support import (
                    location_summary, location_violations, parse_location_lock,
                    parse_wardrobe_lock, wardrobe_summary, wardrobe_violations,
                )
                sc = [p for _, _, p in parsed]
                wl, ll = parse_wardrobe_lock(synopsis), parse_location_lock(synopsis)
                wm, lm = wardrobe_violations(sc, wl), location_violations(sc, ll)
                info += f" | {wardrobe_summary(wl, wm)} | {location_summary(ll, lm)} (checked, not repaired: {n} scenes)"
                print(f"👔 {wardrobe_summary(wl, wm)} | {location_summary(ll, lm)}")

            # pad / trim to exactly one scene per piece so the lists stay aligned
            scenes = [p for _, _, p in parsed][:n]
            while len(scenes) < n:
                scenes.append(scenes[-1] if scenes else "")
            scenes_text = scenes_to_text(synopsis, [(i + 1, durations[i], s) for i, s in enumerate(scenes)])

            return (
                scenes, durations, frames, audio_segments, table, scenes_text, synopsis,
                cast_text, n, float(total_seconds), session_id, info,
            ) + passthrough

        except Exception as exc:
            if is_interrupt(exc):
                raise
            print(f"❌ H3 Music Video Writer error: {exc}")
            import traceback
            print(traceback.format_exc())
            message = f"Error occurred while writing the music video: {exc}"
            return (
                [message] * n, durations, frames, audio_segments, table, message, "", cast_text,
                n, float(total_seconds), "", "error",
            ) + passthrough


def seconds_for(frames):
    return frames / 24.0
