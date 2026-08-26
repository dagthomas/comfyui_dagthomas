# APNext H3 Presentation Writer
#
# Turns source material (scientific findings, benchmark data, a changelog, a
# code walkthrough, product numbers - any text) into a presented video: a
# presenter on camera walks through the material scene by scene, and every
# chart, graph, number and on-screen label shown comes VERBATIM from the
# source material - nothing is invented, rounded or "improved". No audio
# input; the presenter's spoken script, the visual aids and the pacing are all
# generated. Everything comes out as the same lists the other multi-scene
# writers emit (`scenes`, `durations`, `lengths`), so a downstream H3 video
# node renders the talk clip by clip and H3 Scenes Join stitches the clips
# into the finished presentation.

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
    resolve_backend_model,
    BASE_SKILLS,
    CORE_SKILL,
    claude_code_inputs,
    claude_code_optional_inputs,
    draft_model_input,
    resolve_draft_model,
    local_llm_inputs,
    local_llm_options,
    directions_with_research,
    project_name_input,
    project_name_prefix,
    resolve_project_name,
    run_h3_claude_code,
)
from .claude_code_crossover_writer import (
    CAST_SOCKETS, cast_wardrobe, log_cast, log_wardrobe_locks, merge_wardrobe, parse_cast,
)
from .claude_code_music_video_writer import _scene_gist
from .music_support import fmt_time, frames_for_seconds
from .scenes_store import save_scene_bundle
from .template_vars import collect_template_vars, expand_all, log_template_vars
from .common import (
    PROMPT_MODES,
    PROMPT_MODE_TOOLTIP,
    blind_reference_directive,
    resolve_prompt_mode,
    scale_reference_passthrough,
    AUTO,
    LITERAL_CAMERA_DIRECTIVE,
    VISUAL_STYLES,
    _WILDNESS_BANDS,
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
)
from .scenes_support import (
    ENFORCE_WARDROBE_TOOLTIP,
    WARDROBE_TOOLTIP,
    LOCATIONS_TOOLTIP,
    CONTINUITY_MODES,
    DURATION_MODES,
    chain_system_block,
    continuity_directive,
    duration_directive,
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

PRESENTATION_SKILLS = (CORE_SKILL, "h3-ref2va", "h3-style-craft")

# REF writes against the reference-image guide (<Picture N> binds the
# presenter photo); FL writes against the base guide, from scratch in words.
_SCENE_FIELDS = (
    "subject_definitions",
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)

MAX_PRESENTATION_SCENES = 24

# One well-planned chunk per call, same trade-off as the music video writer:
# small chunks finish inside the timeout and fail small.
SCENES_PER_CALL = 4

PRESENTATION_FORMATS = [
    AUTO,
    "Keynote stage (presenter + giant LED screen)",
    "Whiteboard explainer (drawing while talking)",
    "News studio (desk, lower-thirds, inset graphics)",
    "Lab / workshop demo (showing the real thing)",
    "Boardroom pitch (projector, small audience)",
    "Documentary (on location, cutaway inserts)",
    "Tech screencast (code/UI full screen, presenter inset)",
]

_FORMAT_STAGING = {
    PRESENTATION_FORMATS[1]: (
        "a keynote stage: the presenter stands and moves on a stage, a giant LED screen "
        "behind them carries the visual aids, stage lighting picks the presenter out, and "
        "they gesture at the screen as they cite it"
    ),
    PRESENTATION_FORMATS[2]: (
        "a whiteboard explainer: the presenter stands at a large whiteboard and DRAWS the "
        "visual aids by hand while talking - boxes, arrows, axes, bars, labels appearing "
        "marker stroke by marker stroke, timed to the words"
    ),
    PRESENTATION_FORMATS[3]: (
        "a news studio: the presenter sits or stands at a desk in a broadcast set, visual "
        "aids appear as inset graphics beside them or full-screen cutaways, and short "
        "lower-third caption bars carry names and headline numbers"
    ),
    PRESENTATION_FORMATS[4]: (
        "a lab / workshop demo: the presenter is at the bench with the real apparatus, "
        "prototype or equipment, demonstrates the thing itself, and visual aids appear on a "
        "nearby monitor or wall screen when numbers are cited"
    ),
    PRESENTATION_FORMATS[5]: (
        "a boardroom pitch: the presenter stands beside a projector screen or wall display "
        "in a meeting room, a small audience is visible listening at the table, and the "
        "slides on the screen carry the visual aids"
    ),
    PRESENTATION_FORMATS[6]: (
        "a documentary: the presenter speaks on location in places that relate to the "
        "material, with cutaway insert shots illustrating what they describe; visual aids "
        "appear as full-screen graphic cutaways with the presenter's voice continuing over "
        "them"
    ),
    PRESENTATION_FORMATS[7]: (
        "a tech screencast: the screen content (code editor, terminal, dashboard, UI) "
        "fills the frame, the presenter appears in a small picture-in-picture inset in a "
        "corner, and the camera pushes toward the exact lines or values being discussed"
    ),
}

VISUAL_AIDS_MODES = [
    "Auto (a graphic wherever it helps)",
    "Every scene (always a chart or graphic on screen)",
    "Key data moments only",
    "None (talk only, no graphics)",
]

# Words of dialogue an H3 clip carries comfortably per second of runtime.
_WORDS_PER_SECOND = 2.3

_D_RE = re.compile(r"<d>\s*(?:\[[^\]\n]*\]\s*)?(.*?)</d>", re.DOTALL | re.IGNORECASE)


def _wildness_scale_directive(wildness):
    """
    The 0-100 wildness slider as pure creative latitude: the shared band labels
    and directives, but never the seeded list of specific surreal elements -
    how wild the staging gets is stated, what the weirdness IS stays the
    model's to invent around the material.
    """
    wildness = max(0, min(100, int(wildness)))
    for upper, label, directive, _count in _WILDNESS_BANDS:
        if wildness <= upper:
            break
    return f"Creative latitude ({label}, wildness {wildness}/100): {directive}"


def _extract_script(scenes):
    """Teleprompter view: every spoken <d> line, per scene, in order."""
    parts = []
    for i, prompt in enumerate(scenes, 1):
        lines = [re.sub(r"\s+", " ", m).strip() for m in _D_RE.findall(prompt or "")]
        lines = [l for l in lines if l]
        if lines:
            parts.append(f"SCENE {i:02d}:\n" + "\n".join(f"  {l}" for l in lines))
    return "\n\n".join(parts)


def _scene_table(durations, lengths):
    """Human-readable list of the scenes' runtimes (mirrors the music writer's table)."""
    lines, start = [], 0.0
    for i, (d, fr) in enumerate(zip(durations, lengths), 1):
        lines.append(f"{i:02d}  {fmt_time(start)} – {fmt_time(start + d)}  ({d:5.2f}s, {fr} frames)")
        start += d
    return "\n".join(lines)


class H3ClaudeCodePresentationWriter:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "source_material": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": (
                    "The content to present - findings, benchmark numbers, a paper abstract, "
                    "a changelog, data tables, code, release notes. This is the GROUND TRUTH: "
                    "every number, name and claim spoken or shown on screen comes verbatim "
                    "from here, and nothing is invented."
                ),
            }),
            "direction": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": (
                    "The presentation concept: who presents, where, the tone (enthusiastic "
                    "keynote, calm lecture, playful explainer), the look, who the audience "
                    "is. Free text; empty = the model stages it to fit the material."
                ),
            }),
            "presentation_format": (PRESENTATION_FORMATS, {
                "default": AUTO,
                "tooltip": (
                    "How the talk is staged and where the charts live (stage screen, "
                    "whiteboard, studio insets, screencast). Auto picks what fits the "
                    "material; the direction text can override any of it."
                ),
            }),
            "scene_count": ("INT", {
                "default": 6, "min": 1, "max": MAX_PRESENTATION_SCENES,
                "tooltip": (
                    "How many scenes/clips the presentation is told in. Scene 01 hooks and "
                    "names the topic, the middle scenes cover one point each, the last scene "
                    "lands the takeaway."
                ),
            }),
            "duration_mode": (DURATION_MODES, {"default": DURATION_MODES[1]}),
            "scene_duration": ("FLOAT", {
                "default": 12.0, "min": 5.0, "max": 15.0, "step": 0.5,
                "tooltip": (
                    "Seconds per scene in Fixed mode, and the fallback when the model omits "
                    "a duration in Vary mode."
                ),
            }),
            "visual_aids": (VISUAL_AIDS_MODES, {
                "default": VISUAL_AIDS_MODES[0],
                "tooltip": (
                    "How often a chart, graph, diagram, table or code panel is on screen. "
                    "Every graphic shows real values from the source material with short "
                    "verbatim labels."
                ),
            }),
            "continuity_mode": (CONTINUITY_MODES, {
                "default": CONTINUITY_MODES[0],
                "tooltip": (
                    "Independent clips: each scene is its own clip with hard cuts. "
                    "Continuous chain: scenes are written for C2V / motion-context chaining - "
                    "scene N+1 opens on scene N's last frame, one continuous take."
                ),
            }),
            "visual_style": (VISUAL_STYLES, {
                "default": "Live-action, 35mm cinematic film aesthetic",
                "tooltip": "Opens every [Shot 1]; kept identical across the whole presentation.",
            }),
            "dialogue_language": (DIALOGUE_LANGUAGES, {
                "default": "English",
                "tooltip": "The language the presenter speaks (the <d>[...] tag).",
            }),
            "wildness": ("INT", {
                "default": 10, "min": 0, "max": 100, "step": 1,
                "tooltip": (
                    "How wild the STAGING may get: 0 = sober and literal, ~50 = weird and "
                    "imaginary elements creep in, 100 = totally unhinged. Only the level is "
                    "sent to the model - no specific surreal elements are injected - and the "
                    "facts, chart values and on-screen text stay verbatim from the source "
                    "material at every level."
                ),
            }),
        }
        required.update(claude_code_inputs())
        required["seed"] = ("INT", {
            "default": -1, "min": -1, "max": 0xffffffffffffffff,
            "tooltip": "Controls caching. -1 re-runs every queue; any fixed value reuses the cached result.",
        })

        optional = {
            name: ("STRING", {
                "forceInput": True,
                "tooltip": (
                    "A cast line or block from an H3 Characters node (its `cast` output): "
                    "the presenter, a co-host, an interviewee."
                ),
            })
            for name in CAST_SOCKETS
        }
        optional["extra_cast"] = ("STRING", {
            "multiline": True,
            "default": "",
            "tooltip": (
                "Presenter(s) typed by hand, one per line, e.g. `Presenter: a woman in her "
                "40s with grey-streaked hair and rectangular glasses`. Merged with the cast "
                "sockets. Empty = the model invents a presenter that fits the material."
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
                "Per-picture notes, one per line: `Image 1: the presenter`, `Image 2: the "
                "lecture hall, use as the location`. With reference_image_use = Characters "
                "only (the default), a note like that is the ONLY way a picture may be read "
                "as a location or prop."
            ),
        })
        optional["include_soundscape"] = ("BOOLEAN", {
            "default": True,
            "tooltip": "Room tone, audience reactions, marker squeaks, keyboard clicks. Off = `N/A`.",
        })
        optional["include_non_diegetic_music"] = ("BOOLEAN", {
            "default": False,
            "tooltip": (
                "A light music bed under the talk. Off (the default) keeps the voice clean, "
                "which reads as more credible for data-heavy material."
            ),
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
            "tooltip": (
                "Manually planned scenes from chained H3 Scene Brief nodes: each brief "
                "(what happens, where, which cast members and pictures) becomes the "
                "binding plan for its scene. Pinned numbers take that scene; unpinned "
                "briefs fill in order; scenes without a brief stay the model's to invent."
            ),
        })
        optional["save_scenes"] = ("BOOLEAN", {
            "default": True,
            "tooltip": (
                "Store every successful run as a JSON bundle in output/apnext_scenes/ "
                "(scenes, synopsis, durations, cast). Reload it any time with APNext H3 "
                "Scenes Load - re-render without paying for the LLM again."
            ),
        })
        optional["scenes_per_call"] = ("INT", {
            "default": SCENES_PER_CALL, "min": 1, "max": 8,
            "tooltip": (
                "How many scenes to ask the model for per call. Smaller chunks finish well "
                "inside timeout_seconds and fail smaller (a timed-out chunk is retried at "
                "half size automatically); larger chunks are slightly cheaper per scene."
            ),
        })
        # appended LAST so saved workflows keep their widget positions
        optional["prompt_mode"] = (PROMPT_MODES, {
            "default": PROMPT_MODES[1],  # Ref2VA - the pre-switch behaviour
            "tooltip": PROMPT_MODE_TOOLTIP,
        })
        optional.update(draft_model_input())
        optional["parallel_chunks"] = ("BOOLEAN", {
            "default": True,
            "tooltip": (
                "Write the scene chunks concurrently instead of one after another: one "
                "planning call (by `model`) fixes the synopsis, the Outline and the "
                "locks, then up to 4 chunks at a time are drafted from that plan (by "
                "`draft_model`), and one continuity pass repairs any drift. Much faster "
                "for long talks. Off = the classic serial run where every chunk continues "
                "one session. Ignored when resume_session_id is set or the whole talk "
                "fits in one call."
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
        ("STRING", "FLOAT", "INT", "STRING", "STRING", "STRING", "STRING", "INT", "FLOAT", "STRING", "STRING")
        + _IMAGE_OUTPUT_TYPES
        + ("STRING",)
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
    ) + _IMAGE_OUTPUT_NAMES + ("project_name",)
    OUTPUT_IS_LIST = (True, True, True) + (False,) * (8 + len(_IMAGE_OUTPUT_NAMES)) + (False,)
    FUNCTION = "write_presentation"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Turns source material (findings, data, code, a changelog) into a presented video: "
        "a presenter walks through it scene by scene, charts and on-screen numbers taken "
        "VERBATIM from the material. No audio input - the script, visuals and pacing are "
        "all generated. `scenes`, `durations` and `lengths` are matching lists for a "
        "downstream H3 video node; join the clips with H3 Scenes Join."
    )

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        return float("nan") if seed == -1 else seed

    @classmethod
    def VALIDATE_INPUTS(cls, prompt_mode=None, draft_model=None):
        # Workflows saved before these widgets existed restore them as '' -
        # accept anything here; write_presentation coerces empty prompt_mode
        # values to Auto and resolve_draft_model falls back to haiku.
        return True

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self, characters_only=True, continuity_mode=CONTINUITY_MODES[0],
                             ref_mode=True):
        base_guide = load_guide("guide_base_en.md")
        ref_guide = load_guide("guide_ref_en.md") if ref_mode else ""
        return (
            "You are a MiniMax-H3 presentation director and prompt engineer. You are given "
            "SOURCE MATERIAL (scientific findings, data, code, release notes - any content) "
            "and a presenter, and you stage a presented video: the presenter explains the "
            "material to camera scene by scene, with visual aids (charts, graphs, diagrams, "
            "tables, code panels) that display the material's real values. You write ONE "
            "scene per outline point, in order, each a complete production-ready H3 prompt, "
            "so that the clips cut together into one coherent talk.\n\n"
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
            "- The synopsis block also carries `Topic:` (what is being presented, one "
            "line), `Cast:` (one line per person on camera), and `Outline:` - one line per "
            "scene, `NN: the single point that scene covers + its visual aid (or none)` - "
            "covering ALL the key points of the source material in a logical teaching "
            "order. Every scene you then write follows its outline line.\n"
            "- FACTS ARE SACRED: every number, unit, percentage, date, proper noun, metric "
            "name, code identifier and claim that is spoken or shown on screen comes "
            "VERBATIM from the SOURCE MATERIAL. Never invent, extrapolate, round, convert "
            "or 'improve' a value; never add a statistic the material does not contain. If "
            "the material gives no number for something, the presenter speaks about it "
            "qualitatively instead of inventing one. Getting a fact wrong ruins the video.\n"
            "- VISUAL AIDS: when a scene shows a graphic, stage it as a concrete object in "
            "the scene (per the presentation format: the stage screen, the whiteboard, an "
            "inset panel, a monitor). Name the graphic type (bar chart, line graph, pie "
            "chart, scatter plot, table, flow diagram, code panel), give its title and its "
            "axis or column labels in double quotation marks verbatim, and state the 2-5 "
            "labeled values it displays, each drawn from the source material - e.g. `a bar "
            "chart titled \"Model accuracy\" with two bars labeled \"v1  78%\" and \"v2  "
            "91%\", the \"v2  91%\" bar clearly taller`. ALWAYS describe the visual "
            "relationship the data implies (which bar is taller, where the line rises, "
            "which slice dominates) so the picture reads correctly even where small text "
            "renders imperfectly. Never put more than ~5 labels on one graphic and never "
            "put a full sentence on screen: titles and labels are a few words, numbers "
            "keep their units, and the presenter speaks the detail.\n"
            "- THE SCRIPT CARRIES THE CONTENT: the presenter speaks in every scene, in "
            "natural spoken language - their own words for the explanations, but every "
            "cited figure and name exactly as the source material has it. Size each "
            f"scene's spoken lines to its duration (about {_WORDS_PER_SECOND:.1f} words per "
            "second - a 12-second scene holds roughly 25-28 words); never cram. Use "
            "spoken-language signposting (`So here's the surprising part...`, `Let's look "
            "at the numbers.`) and give the presenter a stable (S1) ID across the whole "
            "run. While a graphic is on screen the presenter refers to it and points, "
            "taps or gestures at the exact value as it is named - hands, eyes and words "
            "in sync.\n"
            "- PERFORMANCE: the presenter addresses the camera (or the visible audience) "
            "with open, energetic body language - never reading from notes, never an idle "
            "closed mouth mid-scene. Frame spoken passages MEDIUM CLOSE-UP or MEDIUM so "
            "the delivery reads; cut away to the graphic or a detail insert while the "
            "voice continues where that serves the point, and never cut mid-sentence in a "
            "way that breaks the delivery.\n"
            "- Write everything in English except the spoken lines and any on-screen "
            "text, which keep their own language inside the <d>[...] tag / the quotation "
            "marks.\n"
            + (
                "- When reference pictures are attached the scenes are rendered with the same "
                "pictures as <Picture 1>..<Picture N>. Bind pictured people in "
                "subject_definitions as `<Subject N> ..., appearance from <Picture k>`"
                + (
                    ", and refer to the pictures again inside the shots where those people "
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
                "The presenter is created from scratch in words - define them fully in "
                "subject_definitions (age, face, hair, build, wardrobe) and keep that "
                "written identity anchored word-for-word in every scene."
            )
        )

    def _build_user_prompt(
        self,
        cast,
        source_material,
        direction,
        presentation_format,
        scene_count,
        duration_mode,
        scene_duration,
        continuity_mode,
        visual_aids,
        visual_style,
        dialogue_language,
        wildness,
        include_soundscape,
        include_non_diegetic_music,
        extra_instructions,
        wardrobe="",
        locations="",
        image_labels=(),
        blind_refs=False,
        image_notes="",
        first=1,
        last=None,
        characters_only=True,
        scene_briefs="",
        prior_scenes=(),
        plan_only=False,
        plan_text="",
    ):
        n = scene_count
        last = last or n
        briefs = (scene_briefs or "").strip()
        lines = ["CAST (use these strings verbatim in subject_definitions):"]
        lines += [f"- {c}" for c in cast] if cast else [
            "- (no cast given - invent ONE presenter who fits the material and keep them "
            "identical in every scene)"
        ]
        lines.append("")
        lines.append("SOURCE MATERIAL TO PRESENT (the ground truth - every fact comes from here):")
        lines.append(source_material.strip())
        lines.append("")
        lines.append("CONCEPT FROM THE USER:")
        lines.append(direction.strip() or "(none - stage a presentation that fits the material)")
        lines.append("")
        if briefs:
            lines.append("SCENE BRIEFS FROM THE USER - the plan for those scenes:")
            lines.append(briefs)
            lines.append("")
        if plan_text:
            lines.append(
                "THE SYNOPSIS AND OUTLINE - already fixed for this talk in an earlier "
                "turn. Its cast, wardrobe locks, location locks and per-scene Outline are "
                "BINDING:"
            )
            lines.append(plan_text.strip())
            lines.append("")
        if first > 1 and prior_scenes:
            lines.append("THE TALK SO FAR - what the scenes you already wrote show:")
            for no, gist in prior_scenes:
                lines.append(f"  {no:02d}: {gist}")
            lines.append("")
        lines.append("DIRECTIVES:")
        if plan_only:
            lines.append(
                "- Write ONLY the planning document for the whole talk - no scene "
                "envelopes. It is the synopsis block exactly as the contract defines it: "
                "`Topic:`, `Cast:`, the wardrobe and location locks, and `Outline:` - one "
                f"line per scene, all {n} of them, `NN: the single point that scene "
                "covers + its visual aid (or none)`, covering ALL the key points of the "
                "source material in teaching order. The scenes are written later from "
                "this plan, several at a time by different writers who see only this "
                "document - so it must carry ALL the continuity: exact wardrobe anchors, "
                "exact location/staging anchors, each scene's point and graphic, and the "
                "arc from hook to takeaway."
            )
        elif plan_text:
            lines.append(
                f"- Write scenes {first:02d} to {last:02d} (of {n}) ONLY, following the "
                "synopsis and Outline above exactly: cover each scene's outline point "
                "with its visual aid, copy the wardrobe and staging anchors "
                "character-for-character, and keep the talk building. Other writers "
                "handle the other scenes from the same plan - never write, restate or "
                "borrow a scene outside your range. Start directly with the scene "
                "envelopes; do not write a synopsis block."
            )
        elif first == 1:
            lines.append(
                f"- Write the synopsis block (with the full {n}-scene Outline), then scenes "
                f"{first:02d} to {last:02d} (of {n}). The remaining scenes are requested in "
                "follow-up turns; plan the whole talk now."
                if last < n else
                f"- Write the synopsis block (with the full {n}-scene Outline) and all {n} "
                f"scenes, numbered 01 to {n:02d}."
            )
        else:
            lines.append(
                f"- Continue the SAME presentation: write scenes {first:02d} to {last:02d} "
                f"(of {n}) only, following the synopsis Outline, cast, wardrobe and location "
                "locks you already fixed. Pick up exactly where scene "
                f"{first - 1:02d} in the talk-so-far list leaves off; do not repeat a point "
                "an earlier scene already covered. Start directly with the scene envelopes; "
                "do not repeat the synopsis."
            )
        lines.append(
            "- STRUCTURE: scene 01 hooks the viewer and names the topic in one breath - no "
            "long throat-clearing. Each middle scene covers exactly ONE point from the "
            "outline, in an order that builds understanding (context before result, problem "
            f"before solution). Scene {n:02d} lands the takeaway: what the material means and "
            "what to remember, echoing the strongest number or finding."
        )
        lines.append(
            "- COVERAGE: the outline distributes ALL the key points of the source material "
            f"across the {n} scenes. If the material holds more points than scenes, keep the "
            "most important and fold minor ones into a related scene's spoken line - but "
            "never invent a point the material does not make, and never pad a scene with "
            "filler."
        )
        if presentation_format == AUTO:
            lines.append(
                "- Presentation format: choose ONE staging that fits the material (a stage "
                "with a screen, a whiteboard, a studio, a lab bench, a screencast...), state "
                "it in the synopsis, and keep it for the whole video."
            )
        else:
            lines.append(f"- Presentation format: {_FORMAT_STAGING[presentation_format]}. Keep it for the whole video.")
        aids = str(visual_aids or "")
        if aids.startswith("None"):
            lines.append(
                "- Visual aids: none. The presenter alone carries the talk; keep the frame "
                "free of readable charts, slides and captions."
            )
        elif aids.startswith("Every"):
            lines.append(
                "- Visual aids: EVERY scene has a chart, graph, diagram, table or code panel "
                "on screen, staged per the presentation format and populated with verbatim "
                "values from the source material."
            )
        elif aids.startswith("Key"):
            lines.append(
                "- Visual aids: only the scenes that cite hard numbers or comparisons get a "
                "graphic; the rest are pure presenter. When one appears it shows verbatim "
                "values from the source material."
            )
        else:
            lines.append(
                "- Visual aids: put a graphic on screen wherever it genuinely helps the "
                "point (data, comparisons, structure); let the presenter carry the rest. "
                "Every graphic shows verbatim values from the source material."
            )
        lines.append(f"- {duration_directive(duration_mode, scene_duration)}")
        lines.append(
            f"- Pace the script to the clock: about {_WORDS_PER_SECOND:.1f} spoken words per "
            "second of scene duration. A scene whose point needs more words gets a longer "
            "duration (in Vary mode), never a faster read."
        )
        lines.append(f"- {continuity_directive(continuity_mode)}")
        if visual_style == AUTO:
            lines.append(
                "- Visual style: choose one concrete style that suits the material and keep "
                "it identical in every scene, opening every [Shot 1] with it."
            )
        else:
            lines.append(
                f"- Visual style: open every [Shot 1] with `{visual_style.strip()}` followed "
                "by lighting, time of day and setting; keep it identical across the video."
            )
        lines.extend(
            f"- {d}" for d in toggle_directives(
                True,  # the presenter always speaks
                not aids.startswith("None"),
                include_soundscape,
                include_non_diegetic_music,
                dialogue_language,
            )
        )
        if image_labels:
            labels_s = ", ".join(f"<Picture {i}>" for i in image_labels)
            if blind_refs:
                lines.append(
                    f"- Reference pictures: {len(image_labels)} attached to the VIDEO model, in "
                    f"order {labels_s}; it receives the same pictures under the same labels in "
                    "every scene. " + blind_reference_directive()
                )
            else:
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
                    + (characters_only_directive() + " Bind pictured people to their "
                       "<Picture k> in subject_definitions" + wardrobe_clause
                       if characters_only else
                       "Decide what each shows (use the notes below); bind pictured people to "
                       "their <Picture k> in subject_definitions" + wardrobe_clause)
                )
            notes = (image_notes or "").strip()
            if notes:
                lines.append("- Picture notes from the user:")
                lines.extend(f"    {n_.strip()}" for n_ in notes.splitlines() if n_.strip())
        lines.append(f"- {wardrobe_directive(wardrobe)}")
        lines.append(f"- {locations_directive(locations)}")
        lines.append(
            "- Continuity: the same presenter identity, wardrobe, palette, staging and "
            "style in every scene; the graphics share one visual language (same screen, "
            "same board, same inset style, same title styling) so the talk reads as one "
            "production."
        )
        if briefs:
            lines.append(
                "- SCENE BRIEFS ARE BINDING: a numbered brief is the plan for that scene - "
                "set it where the brief says, with the cast members it names and the "
                "reference pictures it points at, honour its camera wish, and stage what it "
                "describes. `SCENE (next in order)` briefs fill scenes in order from 01, "
                "skipping numbered ones. Scenes without a brief are yours - but never "
                "contradict a brief."
            )
        lines.append(f"- {LITERAL_CAMERA_DIRECTIVE}")
        lines.append(f"- {_wildness_scale_directive(wildness)}")
        if wildness > 40:
            lines.append(
                "- However wild the staging gets, the spoken facts, the chart values and "
                "the on-screen text stay verbatim from the source material - the weirdness "
                "lives in the setting, the camera and the visual metaphors, never in the "
                "data."
            )
        extra = (extra_instructions or "").strip()
        if extra:
            lines.append("")
            lines.append("EXTRA INSTRUCTIONS:")
            lines.append(extra)
        lines.append("")
        lines.append(
            "Now write the synopsis block with the full Outline, and nothing else."
            if plan_only else
            "Now write the requested envelopes (and the synopsis block when asked), and nothing else."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def write_presentation(
        self,
        source_material,
        direction,
        presentation_format,
        scene_count,
        duration_mode,
        scene_duration,
        visual_aids,
        continuity_mode,
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
        include_soundscape=True,
        include_non_diegetic_music=False,
        resume_session_id="",
        working_dir="",
        llm=None,
        reference_image_use=None,
        scene_briefs="",
        scenes_per_call=SCENES_PER_CALL,
        save_scenes=True,
        prompt_mode=None,
        draft_model="haiku",
        parallel_chunks=True,
        project_name="",
        avoid_previous=0,
        **cast_slots,
    ):
        project_name = resolve_project_name(project_name, seed)
        print(f"📊 H3 Presentation Writer | project: {project_name}")
        passthrough = scale_reference_passthrough(cast_slots, self._IMAGE_OUTPUT_NAMES)
        references = collect_reference_images(passthrough, tensor2pil)
        # REF binds pictures via guide_ref_en.md; FL writes from scratch against
        # guide_base_en.md; Auto follows whether pictures are connected. Empty /
        # missing (stale workflows) defaults to Ref - the pre-switch behaviour.
        ref_mode, show_pictures = resolve_prompt_mode(prompt_mode, bool(references))
        if not ref_mode and references:
            print(
                f"ℹ️ H3 Presentation Writer: FL prompt mode - the {len(references)} connected "
                "picture(s) are ignored for writing (they still pass through the image outputs)."
            )
            references = []
        elif references and not show_pictures:
            print(
                f"ℹ️ H3 Presentation Writer: blind Ref2VA - {len(references)} picture(s) are bound "
                "as <Picture N> for the video model, but not shown to the writer; it describes "
                "them from the cast lines and image_notes."
            )
        images = ([downscale_for_vision(pil) for _, pil in references] or None) if show_pictures else None
        image_labels = tuple(range(1, len(references) + 1))
        template_vars, template_summary = collect_template_vars(cast_slots)
        (source_material, direction, extra_cast, wardrobe, locations, extra_instructions,
         image_notes, custom_dialogue_language, custom_visual_style, scene_briefs) = expand_all(
            template_vars, source_material, direction, extra_cast, wardrobe, locations,
            extra_instructions, image_notes, custom_dialogue_language, custom_visual_style,
            scene_briefs,
        )
        log_template_vars(template_vars, template_summary, direction, extra_cast, wardrobe, locations, extra_instructions, image_notes)
        context_text, context_entries = build_context(cast_slots, target="the presentation")
        cast_blocks = [cast_slots.get(name) for name in CAST_SOCKETS] + [extra_cast]
        cast = parse_cast(*cast_blocks)
        cast_text = "\n".join(cast)
        log_cast("H3 Presentation Writer", cast)
        wardrobe = merge_wardrobe(wardrobe, cast_wardrobe(*cast_blocks))
        log_wardrobe_locks("H3 Presentation Writer", wardrobe)

        n = int(scene_count)
        fallback_durations = [float(scene_duration)] * n
        fallback_lengths = [frames_for_seconds(scene_duration)] * n

        try:
            if not (source_material or "").strip():
                raise ValueError("source_material is empty - there is nothing to present.")

            dialogue_language = resolve_dialogue_language(dialogue_language, custom_dialogue_language)
            visual_style = resolve_visual_style(visual_style, custom_visual_style)
            local = local_llm_options(llm)
            chars_only = characters_only_refs(reference_image_use)
            skills = PRESENTATION_SKILLS if ref_mode else BASE_SKILLS
            system_prompt = self._build_system_prompt(
                characters_only=chars_only, continuity_mode=continuity_mode,
                ref_mode=ref_mode,
            )

            print(
                f"📊 H3 Presentation Writer | {len(cast)} cast | {n} scene(s) | "
                f"context: {context_summary(context_entries)} | {len(references)} reference image(s) | "
                f"{(presentation_format if presentation_format != AUTO else 'auto format').split(' (')[0].lower()} | "
                f"wildness {wildness} | research {'on' if research else 'off'} | "
                f"director {'on' if director else 'off'}"
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
                    "ℹ️ H3 Presentation Writer: resume_session_id is set - writing "
                    "serially in that session (parallel_chunks needs a fresh run)."
                )

            def build_prompt(lo, hi, plan_only=False, plan_text="", prior=()):
                # reads wardrobe/locations at call time, so the locks merged from
                # the synopsis reach every later prompt
                return self._build_user_prompt(
                    cast, source_material, direction, presentation_format, n,
                    duration_mode, scene_duration, continuity_mode, visual_aids,
                    visual_style, dialogue_language, wildness, include_soundscape,
                    include_non_diegetic_music,
                    directions_with_research(extra_instructions, research),
                    wardrobe=wardrobe, locations=locations, image_labels=image_labels,
                    blind_refs=not show_pictures,
                    image_notes=image_notes, first=lo, last=hi,
                    characters_only=chars_only, scene_briefs=scene_briefs,
                    prior_scenes=prior, plan_only=plan_only, plan_text=plan_text,
                )

            def merge_locks(text):
                # restate the locks the model just fixed in every follow-up
                # prompt, so later scenes copy the same anchors verbatim
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
                effective_model = resolve_backend_model(model, local.get("model_override", ""))
                effective_chunk = resolve_backend_model(chunk_model, local.get("model_override", ""))
                backend_note = " (the connected LLM Backend)" if (local.get("model_override") or "").strip() else ""
                print(
                    f"⚡ H3 Presentation Writer: parallel run - plan with '{effective_model}'{backend_note}, then "
                    f"{len(ranges)} chunk(s) drafted with '{effective_chunk}', up to {workers} at once."
                )
                plan_prompt = with_context(build_prompt(1, n, plan_only=True), context_text)
                text, session_id, info = run_h3_claude_code(
                    system_prompt, plan_prompt, images, model, research, use_subscription,
                    timeout_seconds, "", working_dir, director, skills=skills, local=local,
                )
                infos.append(f"plan: {info}")
                synopsis = (text or "").strip()
                if not synopsis:
                    raise ValueError("the planning call returned no synopsis / Outline.")
                merge_locks(synopsis)

                def write_range(lo, hi, depth=0):
                    user_prompt = build_prompt(lo, hi, plan_text=synopsis)
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
                            f"⏱️ H3 Presentation Writer: scenes {lo:02d}-{hi:02d} timed out - "
                            f"splitting into {lo:02d}-{mid:02d} and {mid + 1:02d}-{hi:02d}."
                        )
                        a_scenes, a_infos = write_range(lo, mid, depth + 1)
                        b_scenes, b_infos = write_range(mid + 1, hi, depth + 1)
                        return a_scenes + b_scenes, a_infos + b_infos
                    _syn, chunk = parse_scenes(text, scene_duration)
                    if not chunk:
                        raise ValueError(f"the model returned no scenes for {lo:02d}-{hi:02d}.")
                    got = chunk[: hi - lo + 1]
                    if len(got) < hi - lo + 1:
                        print(f"⚠️ H3 Presentation Writer: asked for scenes {lo:02d}-{hi:02d}, got {len(got)}.")
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
                talk_so_far = []  # [(scene_no, one-line gist)] for the chunk 2+ recap

                def ask(lo, hi):
                    user_prompt = build_prompt(lo, hi, prior=tuple(talk_so_far))
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
                            f"⏱️ H3 Presentation Writer: scenes {first:02d}-{last:02d} timed out "
                            f"after {timeout_seconds}s - retrying as {first:02d}-{half:02d}. "
                            "Raise timeout_seconds or lower scenes_per_call to avoid this."
                        )
                        last = half
                        text, session_id, info = ask(first, last)
                    infos.append(info)
                    chunk_synopsis, chunk = parse_scenes(text, scene_duration)
                    if first == 1:
                        synopsis = chunk_synopsis
                        merge_locks(synopsis)
                    if not chunk:
                        raise ValueError(f"the model returned no scenes for {first:02d}-{last:02d}.")
                    got = [(no, d, p) for no, d, p in chunk]
                    # renumber defensively onto the requested range
                    for k, (_no, d, p) in enumerate(got[: last - first + 1]):
                        idx = first + k
                        parsed.append((idx, float(d), p))
                        talk_so_far.append((idx, _scene_gist(p)))
                    if len(got) < last - first + 1:
                        print(f"⚠️ H3 Presentation Writer: asked for scenes {first:02d}-{last:02d}, got {len(got)}.")
                    first = first + max(1, min(len(got), last - first + 1))

            if len(parsed) != n:
                print(f"⚠️ H3 Presentation Writer: asked for {n} scene(s), parsed {len(parsed)}.")

            # --- continuity check (wardrobe + locations), one repair turn --------
            def repair(prompt):
                if parallel:
                    # the plan session never saw the drafted scenes, so the
                    # repair turn carries them along
                    prompt = (
                        "The talk's scenes as currently written (drafted from the plan "
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
                    enforce_wardrobe, synopsis, parsed, n, scene_duration, session_id, info, repair,
                )
            else:
                # multi-chunk runs: a full re-emit is too long, so one repair
                # turn re-emits only the violating scenes and splices them in
                parsed, info = enforce_continuity_chunked(
                    enforce_wardrobe, synopsis, parsed, session_id, info, repair,
                )

            # pad / trim to exactly scene_count scenes so the lists stay aligned,
            # and snap every duration to H3's frame grid so `lengths` matches
            scenes = [p for _, _, p in parsed][:n]
            while len(scenes) < n:
                scenes.append(scenes[-1] if scenes else "")
            raw_durations = [d for _, d, _ in parsed][:n]
            while len(raw_durations) < n:
                raw_durations.append(float(scene_duration))
            lengths = [frames_for_seconds(d) for d in raw_durations]
            durations = [fr / 24.0 for fr in lengths]
            total_seconds = float(sum(durations))
            scenes_text = scenes_to_text(synopsis, [(i + 1, durations[i], s) for i, s in enumerate(scenes)])
            script = _extract_script(scenes)
            table = _scene_table(durations, lengths)
            print(f"📊 H3 Presentation Writer | {fmt_time(total_seconds)} talk in {n} scene(s)\n{table}")

            if save_scenes:
                starts, acc = [], 0.0
                for d in durations:
                    starts.append(acc)
                    acc += d
                segments = [(s, s + d) for s, d in zip(starts, durations)]
                save_scene_bundle(
                    "H3ClaudeCodePresentationWriter", synopsis, scenes, segments, durations,
                    lengths, starts, cast_text, total_seconds, scenes_text, table, info,
                    project_name=project_name,
                )

            return (
                scenes, durations, lengths, scenes_text, synopsis, script,
                cast_text, n, total_seconds, session_id, info,
            ) + passthrough + (project_name_prefix(project_name),)

        except Exception as exc:
            if is_interrupt(exc):
                raise
            print(f"❌ H3 Presentation Writer error: {exc}")
            import traceback
            print(traceback.format_exc())
            message = f"Error occurred while writing the presentation: {exc}"
            return (
                [message] * n, fallback_durations, fallback_lengths, message, "", "",
                cast_text, n, float(sum(fallback_durations)), "", "error",
            ) + passthrough + (project_name_prefix(project_name),)
