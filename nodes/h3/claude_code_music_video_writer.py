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
from .scenes_store import save_scene_bundle
from .template_vars import collect_template_vars, expand_all, log_template_vars
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
from .music_support import (
    SEGMENT_MODES,
    energy_labels,
    fmt_time,
    frames_for_seconds,
    lyrics_for_segment,
    parse_lyrics,
    place_untimed_lyrics,
    profile_line,
    segment_by_lyrics,
    segment_song,
    segment_table,
    slice_audio,
    song_profile,
)
from .scenes_support import (
    ENFORCE_WARDROBE_TOOLTIP,
    WARDROBE_TOOLTIP,
    LOCATIONS_TOOLTIP,
    enforce_continuity,
    enforce_continuity_chunked,
    locations_directive,
    parse_location_lock,
    parse_wardrobe_lock,
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

AUDIO_MODES = [
    "Reference audio (<Audio 1> = the song piece)",
    "Masked latent (song injected into the audio latent)",
]

# Which prompt guide the scenes are written against. Ref (guide_ref_en.md)
# binds attached pictures as <Picture N> - use it when reference images of
# the performer(s) are connected. FL (guide_base_en.md, the T2VA/FL2VA
# family) creates everything from scratch in words - no <Picture N> labels.
PROMPT_MODES = [
    "Auto (Ref with images, FL without)",
    "Ref2VA (bind reference images)",
    "FL / T2VA (from scratch - pictures ignored)",
]

SHOTS_PER_SCENE = [AUTO, "1", "2", "3", "4"]

# Identical example plots and ending advice in every run steer the model into
# its favourite clichés (cosmic spectacle mid-video, the performer walking away
# into the distance at the end). These pools are sampled PER SEED, so every run
# gets different suggestions - and the stock moves are banned outright below.
PLOT_ARCHETYPES = [
    "a heist that goes wrong at the hand-off",
    "two strangers swapping lives for one night",
    "a chase in which the pursuer and the prey trade places",
    "building something impossible by hand before sunrise",
    "a competition with real stakes - a dance-off, a race, a card game",
    "sneaking out (or breaking in), one room at a time",
    "a rescue: someone or something small gets saved",
    "one object passed hand to hand across a whole city",
    "a slow-burn revenge that lands in the final chorus",
    "an ordinary work shift that mutates into a spectacle",
    "a party assembling itself piece by piece from an empty room",
    "getting ready for someone who never shows - and showing up for yourself",
    "a bet between the performer and the band",
    "repairing something broken - a car, a friendship, a neon sign",
    "a wedding, funeral or festival crashed and transformed",
    "teaching someone to dance, sing or drive while the world reacts",
]

ENDING_MOVES = [
    "the opening image returns, transformed by everything that happened",
    "a hard cut on the biggest physical action, exactly on the last beat",
    "the performer finally gets - or loses for good - the thing the video chased",
    "a direct look into the lens, held through the final note",
    "the crowd is gone and one small prop from scene 01 remains in frame",
    "a reversal: whoever had the power in scene 01 has lost it",
    "the two storylines collide in one frame for the first time",
    "a diegetic punchline: something on screen answers the last lyric",
    "the lights cut out mid-move, leaving one practical light burning",
    "a door, a case or a curtain closes on the action",
    "the performer joins the crowd and disappears INTO it, not away from it",
    "the last note freezes on contact - a catch, a kiss, a handshake, a fist on a wall",
]

_ANTI_CLICHE_DIRECTIVE = (
    "BANNED STOCK MOVES at every wildness level (these are model cliches, not "
    "creativity - use one only if the user's concept explicitly asks for it): "
    "ANYTHING OUTER SPACE - asteroids, meteors, comets, planets, moons, "
    "galaxies, nebulae, star fields, spacesuits, zero-gravity floating, the "
    "Earth viewed from space, the sky cracking open to reveal the cosmos; the camera "
    "diving INTO the singer's mouth and down the throat to the stomach as a "
    "transition (lip-sync close-ups stay outside the singer; other body "
    "surrealism - holes through bodies, impossible anatomy - is fair game); "
    "the performer "
    "walking away from camera, into the distance or out of frame as the final "
    "image; slow-motion walking toward camera as the default chorus staging; "
    "dissolving into particles or light; 'the camera pulls back to reveal...' "
    "endings; it-was-all-a-dream wake-ups. Space is not a personality and a "
    "mouth is not a doorway. Invent imagery specific to THIS song and concept "
    "instead."
)

_WILD_FUN_DIRECTIVE = (
    "Wild means FUN, not cosmic: keep the unhinged energy, but make the "
    "weirdness physical, playful and shootable inside the song's own world - "
    "mischief with scale, materials, gravity of ORDINARY things, crowds moving "
    "wrong, animals with agendas, furniture with opinions, weather indoors, "
    "machines coming alive, the set itself misbehaving. Absurd and hilarious "
    "beats grand and cosmic every time, and the weirdness should escalate "
    "scene to scene like a joke building to its punchline."
)

# How many scenes to ask for per model call. A 3-minute song is ~13-20 scenes;
# one answer that long gets sloppy, so the run is split into chunks that
# continue the same session (same synopsis, same locks).
SCENES_PER_CALL = 6

_GIST_RE = re.compile(r"integrated_multimodal_description\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _scene_gist(prompt, limit=160):
    """One compressed line of what a written scene shows, for the story-so-far recap."""
    m = _GIST_RE.search(prompt or "")
    text = m.group(1) if m else (prompt or "")
    text = re.split(
        r"\n\s*(?:overall_soundscape|non_diegetic_music|subject_definitions)\s*:", text
    )[0]
    text = re.sub(r"\[Shot\s*\d+\]\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return text


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
            "tooltip": (
                "Per-picture notes, one per line: `Image 1: the singer`, `Image 2: the "
                "rooftop, use as the location`. With reference_image_use = Characters only "
                "(the default), a note like that is the ONLY way a picture may be read as "
                "a location or prop - otherwise every picture is a performer and its "
                "backdrop is ignored."
            ),
        })
        optional.update(claude_code_optional_inputs())
        optional.update(local_llm_inputs())
        optional.update(context_inputs())
        optional.update(reference_image_inputs())
        # appended LAST so saved workflows keep their widget positions
        optional["scenes_from_lyrics"] = ("BOOLEAN", {
            "default": False,
            "tooltip": (
                "Build the whole video from the lyrics: the song is cut where lyric phrases "
                "start (needs timestamped lyrics like `[0:15] line`; without timestamps it "
                "falls back to the segment mode above) and every scene's imagery is written "
                "from its lyric lines - the pictures stage what the words say, while the "
                "concept supplies style, palette and motifs. Instrumental stretches still "
                "cut on the music."
            ),
        })
        optional["reference_image_use"] = (REFERENCE_IMAGE_USE, {
            "default": REFERENCE_IMAGE_USE[0],
            "tooltip": REFERENCE_IMAGE_USE_TOOLTIP,
        })
        optional["scene_briefs"] = ("STRING", {
            "forceInput": True,
            "tooltip": (
                "Manually planned scenes from chained H3 Scene Brief nodes: each brief "
                "(what happens, where, which cast members and pictures) becomes the "
                "binding plan for its scene/piece. Pinned numbers take that piece; "
                "unpinned briefs fill in order; pieces without a brief stay the "
                "model's to invent."
            ),
        })
        optional["audio_mode"] = (AUDIO_MODES, {
            "default": AUDIO_MODES[0],
            "tooltip": (
                "How the song reaches the video model. Reference audio: each piece is "
                "attached as ref_audio_1 and the prompts define <Audio 1> (classic Ref2VA; "
                "lip-sync is a strong suggestion). Masked latent: for workflows that write "
                "the song slice straight into the H3 audio latent and protect it from "
                "denoising (e.g. `H3 Song Audio + Masked Video Context` fed by this node's "
                "`clip_starts`) - the prompts then reference the protected master-song audio "
                "and define no <Audio N>, and lip-sync is enforced by the model itself. "
                "Do not wire audio_segments to ref_audio in that setup."
            ),
        })
        optional["save_scenes"] = ("BOOLEAN", {
            "default": True,
            "tooltip": (
                "Store every successful run as a JSON bundle in output/apnext_scenes/ "
                "(scenes, synopsis, segment times, durations, clip starts, cast). Reload "
                "it any time with APNext H3 Scenes Load - re-render without paying for "
                "the LLM again."
            ),
        })
        optional["scenes_per_call"] = ("INT", {
            "default": 4, "min": 1, "max": 8,
            "tooltip": (
                "How many scenes to ask the model for per call. Smaller chunks finish well "
                "inside timeout_seconds and fail smaller (a timed-out chunk is retried at "
                "half size automatically); larger chunks are slightly cheaper per scene. "
                "1 = every scene written in its own call."
            ),
        })
        # appended LAST so saved workflows keep their widget positions
        optional["prompt_mode"] = (PROMPT_MODES, {
            "default": PROMPT_MODES[1],  # Ref2VA - the pre-switch behaviour
            "tooltip": (
                "Which official prompt guide the scenes follow. Ref (guide_ref_en.md) "
                "binds the attached pictures as <Picture N> - use it when reference "
                "images of the performer(s) are connected. FL / T2VA (guide_base_en.md) "
                "creates everything from scratch in words - pictures are ignored and no "
                "<Picture N> labels are written. Auto picks Ref when images are "
                "connected, FL otherwise."
            ),
        })
        # appended LAST so saved workflows keep their widget positions
        optional.update(draft_model_input())
        optional["parallel_chunks"] = ("BOOLEAN", {
            "default": True,
            "tooltip": (
                "Write the scene chunks concurrently instead of one after another: one "
                "planning call (by `model`) fixes the synopsis, the locks and a per-scene "
                "plan, then up to 4 chunks at a time are drafted from that plan (by "
                "`draft_model`), and one continuity pass repairs any drift. Much faster "
                "for long songs. Off = the classic serial run where every chunk continues "
                "one session. Ignored when resume_session_id is set or the whole song "
                "fits in one call."
            ),
        })

        return {"required": required, "optional": optional, "hidden": context_hidden_inputs()}

    _IMAGE_OUTPUT_TYPES, _IMAGE_OUTPUT_NAMES = reference_image_outputs()
    # clip_starts sits LAST (after the image passthroughs) so saved workflows
    # keep their link indices: each piece's start second in the master song,
    # for masked-audio music-video chains (clip_start_seconds on a song-audio
    # latent-context node) where the song slice is written into the H3 audio
    # latent instead of being passed as ref_audio.
    RETURN_TYPES = (
        ("STRING", "FLOAT", "INT", "AUDIO", "STRING", "STRING", "STRING", "STRING", "INT", "FLOAT", "STRING", "STRING")
        + _IMAGE_OUTPUT_TYPES
        + ("FLOAT",)
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
    ) + _IMAGE_OUTPUT_NAMES + ("clip_starts",)
    OUTPUT_IS_LIST = (True, True, True, True) + (False,) * (8 + len(_IMAGE_OUTPUT_NAMES)) + (True,)
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

    @classmethod
    def VALIDATE_INPUTS(cls, prompt_mode=None):
        # Workflows saved before this widget existed restore it as '' - accept
        # anything here; write_video coerces empty/unknown values to Auto.
        return True

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self, characters_only=True, masked_audio=False, ref_mode=True):
        base_guide = load_guide("guide_base_en.md")
        # The ref guide defines <Picture N>/<Audio N> grammar. FL mode drops it
        # unless the song pieces are attached as reference audio, which still
        # needs the <Audio 1> labels.
        include_ref_guide = ref_mode or not masked_audio
        ref_guide = load_guide("guide_ref_en.md") if include_ref_guide else ""
        audio_ref = "the protected master-song audio" if masked_audio else "<Audio 1>"
        return (
            "You are a MiniMax-H3 music-video director and prompt engineer. You are given a "
            "song that has already been cut into consecutive pieces of 5-15 seconds, the "
            "lyric lines that fall inside each piece, the energy of each piece, the "
            "performer(s) and a creative concept. You write ONE scene per piece, in order, "
            "each a complete production-ready H3 prompt that is rendered with that piece of "
            "the song as its soundtrack, so that the clips cut together into one music video.\n\n"
            + (
                "Two documents follow. The official guide is authoritative for grammar, camera "
                "vocabulary, timestamps, speaker IDs and <d> blocks. The reference guide is "
                "authoritative for the <Picture N> / <Audio N> labels and the relationship "
                "markers.\n\n"
                if include_ref_guide else
                "The official guide follows and is authoritative for grammar, camera "
                "vocabulary, timestamps, speaker IDs and <d> blocks.\n\n"
            )
            + "=== BEGIN MINIMAX-H3 VIDEO PROMPT WRITING GUIDE ===\n"
            f"{base_guide}\n"
            "=== END MINIMAX-H3 VIDEO PROMPT WRITING GUIDE ===\n\n"
            + (
                "=== BEGIN MINIMAX-H3 REFERENCE (REF2VA) GUIDE ===\n"
                f"{ref_guide}\n"
                "=== END MINIMAX-H3 REFERENCE (REF2VA) GUIDE ===\n\n"
                if include_ref_guide else ""
            )
            + envelope_contract(_SCENE_FIELDS) + "\n"
            "- The synopsis block also carries `Cast:` (one line per performer/character), "
            "`Concept:` (the visual idea in two or three sentences) and `Motifs:` (recurring "
            "images, colours, props and the chorus look, so every scene can reuse them).\n"
            "- Each scene envelope's `duration:` is the exact length of its song piece, given "
            "to you; copy it.\n"
            + (
                "- AUDIO: every scene's piece of the master song is written directly into the "
                "clip's audio latent and protected from denoising - the soundtrack already "
                "exists and drives the singing, breathing, body rhythm and emotional timing. "
                "Do NOT define any <Audio N> reference and never invent music, ambience or "
                "sound effects. overall_soundscape is `The protected master-song audio is the "
                "complete soundtrack; no other sound.` and non_diegetic_music is `The "
                "protected master-song audio drives the performance and the emotional "
                "timing.`\n"
                if masked_audio else
                "- AUDIO: every scene is rendered with its own song piece attached as <Audio 1>. "
                "In subject_definitions define it as `<Audio 1> is the song, reused 1:1 as the "
                "complete final soundtrack of this clip` (fully_copy). Cite <Audio 1> in the "
                "description where the music drives the picture. overall_soundscape is "
                "`<Audio 1> is the complete soundtrack; no other sound.` and non_diegetic_music "
                "is `N/A (the song is <Audio 1>)`. Never invent other music, ambience or effects.\n"
            ) +
            "- LYRICS: the lines listed for a piece are sung at the moments given. In "
            "Performance mode the singer visibly sings them on camera, lip-synced: write "
            "`<Subject 1> (S1) keeps visibly singing with readable mouth and jaw movement in "
            f"exact sync with {audio_ref}: <d>[Language] exact lyric line</d>` at the right "
            "timestamp, with the exact words, once per line, no paraphrase. Between lyric "
            "lines the performer keeps performing - breathing, phrasing, moving on the beat - "
            "never an idle closed mouth while the vocal is audible, and a sung line is never "
            "interrupted by a cut. In "
            f"Narrative mode nobody sings on camera - the lyric is audible from {audio_ref} and "
            f"the picture answers it (`as {audio_ref} reaches <d>[Language] line</d>, ...`). "
            "Mixed mode alternates. Instrumental pieces get pure visuals cut to the beat.\n"
            "- ENERGY: quiet pieces get long, slow, intimate shots; loud and peak pieces get "
            "more cuts, bigger moves, bolder light, the chorus look. The whole video keeps "
            "one style, one palette, one performer identity, the wardrobe and location locks.\n"
            "- Write everything in English except the lyric lines, which stay in the song's "
            "language inside the <d>[...] tag.\n"
            + (
                "- When reference pictures are attached the scenes are rendered with the same "
                "pictures as <Picture 1>..<Picture N>. Bind pictured performers in "
                "subject_definitions as `<Subject N> ..., appearance from <Picture k>`"
                + (
                    ", and refer to the pictures again inside the shots where the performers "
                    "appear. Only a picture the user's notes explicitly declare a location or "
                    "prop may be treated as one (`<Picture k> is ...` on its own line); a "
                    "pictured person's backdrop is NEVER the scene."
                    if characters_only else
                    ", treat a location or prop picture as `<Picture k> is ...` on its own "
                    "line, and refer to the pictures again inside the shots where their "
                    "content appears."
                )
                if ref_mode else
                "- This run uses NO reference pictures: never write a <Picture N> label. "
                "Every performer is created from scratch in words - define each one fully "
                "in subject_definitions (age, face, hair, build, wardrobe) and keep that "
                "written identity anchored word-for-word in every scene."
            )
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
        lyrics_driven=False,
        characters_only=True,
        masked_audio=False,
        scene_briefs="",
        prior_scenes=(),
        plot_picks=(),
        ending_picks=(),
        profile=None,
        plan_only=False,
        plan_text="",
    ):
        last = last or len(segments)
        n = len(segments)
        audio_ref = "the protected master-song audio" if masked_audio else "<Audio 1>"
        briefs = (scene_briefs or "").strip()
        lines = ["CAST (use these strings verbatim in subject_definitions):"]
        lines += [f"- {c}" for c in cast] if cast else ["- (no cast given - invent a performer that fits the concept and keep them identical in every scene)"]
        lines.append("")
        lines.append("CONCEPT FROM THE USER:")
        lines.append(direction.strip() or "(none - invent a concept that fits the song, its lyrics and its energy)")
        lines.append("")
        if briefs:
            lines.append("SCENE BRIEFS FROM THE USER - the plan for those pieces:")
            lines.append(briefs)
            lines.append("")
        lines.append(
            f"THE SONG: {fmt_time(total_seconds)} long, cut into {n} consecutive pieces. One scene per piece, "
            "in this order. Each piece's lyric lines are given with the second inside the piece at which "
            "they start ([+s]); `exact` times come from the user's timestamps, `~` times are estimates."
        )
        if profile:
            lines.append(f"THE SOUND (measured from the audio): {profile_line(profile)}")
        for i, ((s, e), label, fr) in enumerate(zip(segments, labels, frames), 1):
            lyr = lyrics_for_segment(placed_lyrics, s, e)
            sung = [l for l in lyr if l[1] and not l[1].startswith("#")]
            tags = [l[1][1:] for l in lyr if l[1].startswith("#")]
            marker = "" if plan_only else ("  <-- write this one" if first <= i <= last else "")
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
        if plan_text:
            lines.append(
                "THE SYNOPSIS AND SCENE PLAN - already fixed for this video in an earlier "
                "turn. Its cast, wardrobe locks, location locks and motifs are BINDING, and "
                "its per-scene plan is the story:"
            )
            lines.append(plan_text.strip())
            lines.append("")
        if first > 1 and prior_scenes:
            lines.append("THE STORY SO FAR - what the scenes you already wrote show:")
            for no, gist in prior_scenes:
                lines.append(f"  {no:02d}: {gist}")
            lines.append("")
        lines.append("DIRECTIVES:")
        if plan_only:
            lines.append(
                "- Write ONLY the planning document for the whole video - no scene "
                "envelopes. First the synopsis block exactly as the contract defines it "
                "(Synopsis, Cast, Concept, Motifs, and the wardrobe and location locks), "
                f"then a section `SCENE PLAN:` with one line per piece, all {n} of them: "
                "`NN: setting | what happens | how it advances the story`. The scenes are "
                "written later from this plan, several at a time by different writers who "
                "see only this document - so it must carry ALL the continuity: exact "
                "wardrobe anchors, exact location anchors, the motif wording, the chorus "
                "look, and the full arc from opening image to payoff."
            )
        elif plan_text:
            lines.append(
                f"- Write scenes {first:02d} to {last:02d} (of {n}) ONLY, following the "
                "synopsis and SCENE PLAN above exactly: stage each scene's plan line, copy "
                "the wardrobe and location anchors character-for-character, and keep the "
                "arc moving. Other writers handle the other scenes from the same plan - "
                "never write, restate or borrow a scene outside your range. Start directly "
                "with the scene envelopes; do not write a synopsis block."
            )
        elif first == 1:
            lines.append(
                f"- Write the synopsis block, then scenes {first:02d} to {last:02d} (of {n}). "
                "The remaining scenes are requested in follow-up turns; plan the whole video now."
                if last < n else
                f"- Write the synopsis block and all {n} scenes, numbered 01 to {n:02d}."
            )
        else:
            lines.append(
                f"- Continue the SAME video: write scenes {first:02d} to {last:02d} (of {n}) only, "
                "following the synopsis, cast, wardrobe and location locks you already fixed - "
                "and KEEP THE JOURNEY MOVING through the settings and story beats the synopsis "
                "planned; do not fall back into one place. Pick up exactly where scene "
                f"{first - 1:02d} in the story-so-far list leaves off and advance the arc; do "
                "not restage or repeat what an earlier scene already showed. "
                "Start directly with the scene envelopes; do not repeat the synopsis."
            )
        lines.append(f"- Performance mode: {performance_mode}.")
        if lyrics_driven:
            lines.append(
                "- LYRICS DRIVE THE VIDEO: the pieces are cut where lyric phrases start, and "
                "every scene's imagery is built from its lyric lines - stage, illustrate or "
                "answer what the words say at that moment, so someone watching without sound "
                "could still follow the words. The CONCEPT supplies the style, palette, "
                "world and recurring motifs, but the moment-to-moment content of each scene "
                "comes from its lines. Instrumental pieces bridge between the lyric images "
                "using the motifs."
            )
        if performance_mode.startswith("Performance"):
            lines.append(
                "- Lip-sync: in every scene with sung lines the singer is on camera facing "
                "the lens (or in a clear profile) with the mouth fully visible - no hands, "
                "microphones, hair, shadows or props covering it - and readable mouth and "
                f"jaw movement locked to {audio_ref} from the first frame of the line to the "
                "last. Frame sung lines MEDIUM CLOSE-UP or closer (the face large in frame) "
                "and keep the camera slow and smooth while a line lasts; save wide shots and "
                "fast moves for the instrumental moments. Never cut away from the singer in "
                "the middle of a sung line."
            )
        if shots_per_scene != AUTO:
            lines.append(f"- Use exactly {shots_per_scene} shot(s) per scene.")
        else:
            lines.append(
                "- Shots per scene: your call, cut to the music - 1-2 in quiet pieces, 2-4 in "
                "loud and peak pieces, each cut on a beat or a lyric. In Performance mode "
                "prefer ONE continuous shot for a piece whose lyric lines run through it: "
                "every cut risks breaking the lip-sync."
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
            wardrobe_clause = (
                " and take their wardrobe lock from the picture."
                if not (wardrobe or "").strip() else
                "; the picture fixes their face, hair and build, while the written "
                "wardrobe lock below fixes the clothes - where they differ, the written "
                "lock wins."
            )
            lines.append(
                f"- Reference pictures: {len(image_labels)} attached, in order {labels_s}; the "
                "video model receives the same pictures under the same labels in every scene. "
                + (characters_only_directive() + " Bind pictured performers to their "
                   "<Picture k> in subject_definitions" + wardrobe_clause
                   if characters_only else
                   "Decide what each shows (use the notes below); bind pictured performers to "
                   "their <Picture k> in subject_definitions" + wardrobe_clause)
            )
            notes = (image_notes or "").strip()
            if notes:
                lines.append("- Picture notes from the user:")
                lines.extend(f"    {n_.strip()}" for n_ in notes.splitlines() if n_.strip())
        lines.append(f"- {wardrobe_directive(wardrobe)}")
        lines.append(f"- {locations_directive(locations)}")
        lines.append(
            "- The look is part of the brief: every appearance detail in the CAST lines and "
            "the CONCEPT (hair, clothing, accessories, make-up, styling) is binding. Build "
            "each performer's wardrobe lock from those exact details FIRST, quoting them "
            "word-for-word as anchors, and invent anchors only for what the brief leaves "
            "open. Never restyle a described performer and never contradict the concept's "
            "look; where the concept and an invented idea differ, the concept wins."
        )
        lines.append(
            "- Continuity: the same performer identity, wardrobe, palette and style in every "
            "scene; recurring locations restate their anchors; the chorus pieces share one "
            "signature look; the last scene resolves the concept."
        )
        if briefs:
            lines.append(
                "- SCENE BRIEFS ARE BINDING: a numbered brief is the plan for that piece - "
                "set the scene where it says, put exactly the cast members it names on "
                "screen, use the reference pictures it points at, honour its camera wish, "
                "and stage what it describes, adapted to the piece's lyric lines, duration "
                "and energy. `SCENE (next in order)` briefs fill pieces in order from 01, "
                "skipping numbered ones. Pieces without a brief are yours to write within "
                "the concept - but never contradict a brief."
            )
        distinct = min(6, max(3, n // 4))
        lines.append(
            "- THE VIDEO IS A JOURNEY, NOT A ROOM: plan the whole visual arc in the synopsis "
            "before scene 01. Unless the user's concept explicitly pins one place, the video "
            f"MOVES: at least {distinct} clearly distinct settings across the {n} scenes, each "
            "verse pushing the story somewhere new (a new place, or a visible transformation "
            "of the last one), every chorus returning to ONE signature look that escalates "
            "each time - through light, choreography, crowd and camera, never through cosmic "
            "spectacle - and the bridge breaking the pattern completely. Lock only the places "
            "that actually recur; everything else travels."
        )
        plots = "; ".join(plot_picks) if plot_picks else (
            "a transformation, a chase, a heist, a ritual"
        )
        lines.append(
            "- FIND A PLOT: invent a concrete story told in images, with a visible setup, an "
            f"escalation, and a payoff in the final scene - for example: {plots} - or "
            "something better that this specific song demands. Never many variations of a "
            "performer standing in one room: something HAPPENS in this video, and every "
            "scene advances it."
        )
        if ending_picks:
            lines.append(
                "- THE LAST SCENE lands a REAL payoff - a concrete story beat, staged and "
                f"shot like it matters. Strong shapes for this song: {'; '.join(ending_picks)}. "
                "Pick one of these, or beat them."
            )
        lines.append(f"- {_ANTI_CLICHE_DIRECTIVE}")
        lines.append(f"- {LITERAL_CAMERA_DIRECTIVE}")
        if profile:
            beat = f" At ~{profile['bpm']:g} BPM a bar is ~{240.0 / profile['bpm']:.1f}s - time cuts, gestures and choreography to land ON the beat." if profile.get("bpm") else ""
            intensity = profile.get("intensity", 50)
            if intensity < 40:
                lines.append(
                    "- MATCH THE SOUND - this is a SOFT song: favour long unbroken takes, "
                    "slow tender camera moves, close intimacy, soft motivated light and "
                    "small precise gestures. Cuts breathe with the phrases; nothing slams." + beat
                )
            elif intensity < 65:
                lines.append(
                    "- MATCH THE SOUND - mid-energy song: let the quiet pieces breathe with "
                    "longer takes and gentler camera, and make the loud pieces visibly hit "
                    "harder - the contrast IS the video's rhythm." + beat
                )
            else:
                lines.append(
                    "- MATCH THE SOUND - this is an AGGRESSIVE song: punchy staging - hard "
                    "cuts on the beat, bold fast camera moves, physical committed "
                    "choreography, stark high-contrast light, real impact in every scene. "
                    "Keep sung lines stable for lip-sync, then let everything around them "
                    "hit." + beat
                )
        wild_lines, wild_label = wildness_directive(wildness, rng)
        lines += [f"- {w}" for w in wild_lines]
        if wildness > 40:
            lines.append(f"- {_WILD_FUN_DIRECTIVE}")
        extra = (extra_instructions or "").strip()
        if extra:
            lines.append("")
            lines.append("EXTRA INSTRUCTIONS:")
            lines.append(extra)
        lines.append("")
        lines.append(
            "Now write the synopsis block and the SCENE PLAN, and nothing else."
            if plan_only else
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
        scenes_from_lyrics=False,
        reference_image_use=None,
        audio_mode=None,
        scene_briefs="",
        scenes_per_call=4,
        save_scenes=True,
        prompt_mode=None,
        draft_model="haiku",
        parallel_chunks=True,
        **cast_slots,
    ):
        passthrough = scale_reference_passthrough(cast_slots, self._IMAGE_OUTPUT_NAMES)
        references = collect_reference_images(passthrough, tensor2pil)
        # REF binds pictures via guide_ref_en.md; FL writes from scratch against
        # guide_base_en.md; Auto follows whether pictures are connected. Empty /
        # missing (stale workflows) defaults to Ref - the pre-switch behaviour.
        mode = str(prompt_mode or PROMPT_MODES[1])
        ref_mode = bool(references) if mode.startswith("Auto") else mode.startswith("Ref")
        if not ref_mode and references:
            print(
                f"ℹ️ H3 Music Video Writer: FL prompt mode - the {len(references)} connected "
                "picture(s) are ignored for writing (they still pass through the image outputs)."
            )
            references = []
        images = [downscale_for_vision(pil) for _, pil in references] or None
        image_labels = tuple(range(1, len(references) + 1))
        template_vars, template_summary = collect_template_vars(cast_slots)
        (direction, lyrics, extra_cast, wardrobe, locations, extra_instructions, image_notes,
         custom_dialogue_language, custom_visual_style, scene_briefs) = expand_all(
            template_vars, direction, lyrics, extra_cast, wardrobe, locations, extra_instructions,
            image_notes, custom_dialogue_language, custom_visual_style, scene_briefs,
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
        lyrics_driven = bool(scenes_from_lyrics)
        if lyrics_driven and timed:
            segments, feats = segment_by_lyrics(
                audio, max_segment_seconds, min_segment_seconds, lyric_times=timed,
            )
        else:
            if lyrics_driven and not timed:
                print(
                    "⚠️ H3 Music Video Writer: scenes_from_lyrics is on but no lyric line "
                    "carries a timestamp (`[0:15] line`); cutting with the segment mode "
                    "instead. The scenes are still written from their lyric lines."
                )
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
        clip_starts = [float(s) for s, _ in segments]
        audio_segments = [slice_audio(audio, s, e, fr) for (s, e), fr in zip(segments, frames)]
        profile = song_profile(feats)
        table = f"SOUND: {profile_line(profile)}\n" + segment_table(segments, labels, frames, placed)
        n = len(segments)
        print(f"🎵 H3 Music Video Writer | {fmt_time(total_seconds)} song -> {n} piece(s)\n{table}")

        try:
            dialogue_language = resolve_dialogue_language(dialogue_language, custom_dialogue_language)
            visual_style = resolve_visual_style(visual_style, custom_visual_style)
            current_seed = seed if seed != -1 else random.randint(0, 0xffffffffffffffff)
            rng = random.Random(current_seed)
            # one sample per RUN (not per chunk), so every chunk of a long video
            # is steered toward the same story and the same ending
            plot_picks = tuple(rng.sample(PLOT_ARCHETYPES, 3))
            ending_picks = tuple(rng.sample(ENDING_MOVES, 2))
            local = local_llm_options(llm)
            chars_only = characters_only_refs(reference_image_use)
            masked_audio = str(audio_mode or "").startswith("Masked")
            skills = MUSIC_SKILLS if ref_mode else BASE_SKILLS
            system_prompt = self._build_system_prompt(
                characters_only=chars_only, masked_audio=masked_audio, ref_mode=ref_mode,
            )

            print(
                f"🎬 H3 Music Video Writer | {'ref' if ref_mode else 'fl'} prompts | "
                f"{len(cast)} cast | {n} scene(s) | "
                f"context: {context_summary(context_entries)} | {len(references)} reference image(s) | "
                f"{performance_mode.split(' ')[0].lower()}{' | lyrics-driven' if lyrics_driven else ''} | wildness {wildness} | "
                f"research {'on' if research else 'off'} | director {'on' if director else 'off'} | seed {current_seed}"
            )

            # --- write --------------------------------------------------------
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
                    "ℹ️ H3 Music Video Writer: resume_session_id is set - writing "
                    "serially in that session (parallel_chunks needs a fresh run)."
                )

            def build_prompt(lo, hi, rng_, plan_only=False, plan_text="", prior=()):
                # reads wardrobe/locations at call time, so the locks merged from
                # the synopsis reach every later prompt
                user_prompt, _wild = self._build_user_prompt(
                    cast, direction, segments, labels, frames, placed, performance_mode,
                    shots_per_scene, visual_style, dialogue_language, wildness,
                    directions_with_research(extra_instructions, research), rng_,
                    wardrobe=wardrobe, locations=locations, image_labels=image_labels,
                    image_notes=image_notes, first=lo, last=hi, total_seconds=total_seconds,
                    lyrics_driven=lyrics_driven, characters_only=chars_only,
                    masked_audio=masked_audio, scene_briefs=scene_briefs,
                    prior_scenes=prior, plot_picks=plot_picks, ending_picks=ending_picks,
                    profile=profile, plan_only=plan_only, plan_text=plan_text,
                )
                return user_prompt

            def merge_locks(text):
                # restate the locks the model just fixed in every follow-up
                # prompt, so later scenes copy the same anchors verbatim
                # instead of drifting from memory
                nonlocal wardrobe, locations
                w_locks = parse_wardrobe_lock(text)
                if w_locks:
                    wardrobe = merge_wardrobe(
                        wardrobe, [f"{k}: {', '.join(v)}" for k, v in w_locks.items()]
                    )
                l_locks = parse_location_lock(text)
                if l_locks:
                    locations = merge_wardrobe(
                        locations, [f"{k}: {', '.join(v)}" for k, v in l_locks.items()]
                    )

            if parallel:
                # one plan call (strong model) -> chunks drafted concurrently
                # (draft model, fresh sessions) -> one continuity repair below
                ranges = [(lo, min(n, lo + per_call - 1)) for lo in range(1, n + 1, per_call)]
                workers = min(4, len(ranges))
                print(
                    f"⚡ H3 Music Video Writer: parallel run - plan with '{model}', then "
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
                    raise ValueError("the planning call returned no synopsis / scene plan.")
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
                        # a timed-out multi-scene chunk is split in two and both
                        # halves rewritten; anything else is fatal
                        if (is_interrupt(exc) or "did not finish within" not in str(exc)
                                or hi <= lo or depth >= 2):
                            raise
                        mid = lo + (hi - lo) // 2
                        print(
                            f"⏱️ H3 Music Video Writer: scenes {lo:02d}-{hi:02d} timed out "
                            f"after {timeout_seconds}s - splitting into {lo:02d}-{mid:02d} "
                            f"and {mid + 1:02d}-{hi:02d}."
                        )
                        a_scenes, a_infos = write_range(lo, mid, depth + 1)
                        b_scenes, b_infos = write_range(mid + 1, hi, depth + 1)
                        return a_scenes + b_scenes, a_infos + b_infos
                    _syn, chunk = parse_scenes(text, durations[lo - 1])
                    if not chunk:
                        raise ValueError(f"the model returned no scenes for pieces {lo:02d}-{hi:02d}.")
                    got = chunk[: hi - lo + 1]
                    if len(got) < hi - lo + 1:
                        print(f"⚠️ H3 Music Video Writer: asked for scenes {lo:02d}-{hi:02d}, got {len(got)}.")
                    # renumber defensively onto the requested range
                    scenes_out = [
                        (lo + k, durations[lo + k - 1], p) for k, (_no, _d, p) in enumerate(got)
                    ]
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
                # the classic serial run: chunks continue one session; `model`
                # writes the first chunk (with the synopsis), chunk_model the rest
                first = 1
                story_so_far = []  # [(scene_no, one-line gist)] for the chunk 2+ recap

                def ask(lo, hi):
                    user_prompt = build_prompt(lo, hi, rng, prior=tuple(story_so_far))
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
                        # a timed-out multi-scene chunk gets ONE retry at half size;
                        # anything else (or a single scene timing out) is fatal
                        if is_interrupt(exc) or "did not finish within" not in str(exc) or last <= first:
                            raise
                        half = first + (last - first) // 2
                        print(
                            f"⏱️ H3 Music Video Writer: scenes {first:02d}-{last:02d} timed out "
                            f"after {timeout_seconds}s - retrying as {first:02d}-{half:02d}. "
                            "Raise timeout_seconds or lower scenes_per_call to avoid this."
                        )
                        last = half
                        text, session_id, info = ask(first, last)
                    infos.append(info)
                    chunk_synopsis, chunk = parse_scenes(text, durations[first - 1])
                    if first == 1:
                        synopsis = chunk_synopsis
                        merge_locks(synopsis)
                    if not chunk:
                        raise ValueError(f"the model returned no scenes for pieces {first:02d}-{last:02d}.")
                    got = [(no, d, p) for no, d, p in chunk]
                    # renumber defensively onto the requested range
                    for k, (no, _d, p) in enumerate(got[: last - first + 1]):
                        idx = first + k
                        parsed.append((idx, durations[idx - 1], p))
                        story_so_far.append((idx, _scene_gist(p)))
                    if len(got) < last - first + 1:
                        print(f"⚠️ H3 Music Video Writer: asked for scenes {first:02d}-{last:02d}, got {len(got)}.")
                    first = first + max(1, min(len(got), last - first + 1))

            if len(parsed) != n:
                print(f"⚠️ H3 Music Video Writer: {n} piece(s) but {len(parsed)} scene(s) parsed.")

            # --- continuity check (wardrobe + locations), one repair turn --------
            def repair(prompt):
                if parallel:
                    # the plan session never saw the drafted scenes, so the
                    # repair turn carries them along
                    prompt = (
                        "The video's scenes as currently written (drafted from the plan "
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
            if len(parsed) == n and n <= SCENES_PER_CALL:
                synopsis, parsed, info = enforce_continuity(
                    enforce_wardrobe, synopsis, parsed, n, durations[0], session_id, info, repair,
                )
            else:
                # multi-chunk runs: a full re-emit is too long, so one repair
                # turn re-emits only the violating scenes and splices them in
                parsed, info = enforce_continuity_chunked(
                    enforce_wardrobe, synopsis, parsed, session_id, info, repair,
                )

            # pad / trim to exactly one scene per piece so the lists stay aligned
            scenes = [p for _, _, p in parsed][:n]
            while len(scenes) < n:
                scenes.append(scenes[-1] if scenes else "")
            scenes_text = scenes_to_text(synopsis, [(i + 1, durations[i], s) for i, s in enumerate(scenes)])

            if save_scenes:
                save_scene_bundle(
                    "H3ClaudeCodeMusicVideoWriter", synopsis, scenes, segments, durations,
                    frames, clip_starts, cast_text, total_seconds, scenes_text, table, info,
                )

            return (
                scenes, durations, frames, audio_segments, table, scenes_text, synopsis,
                cast_text, n, float(total_seconds), session_id, info,
            ) + passthrough + (clip_starts,)

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
            ) + passthrough + (clip_starts,)


def seconds_for(frames):
    return frames / 24.0
