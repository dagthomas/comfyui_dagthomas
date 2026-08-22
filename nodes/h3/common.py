# Shared helpers for the MiniMax-H3 prompt writer nodes
#
# The system prompts are the official MiniMax guides shipped verbatim under
# data/h3/, so the model is steered by the real spec rather than a paraphrase:
#   data/h3/guide_base_en.md  - VIDEO_PROMPT_WRITING_GUIDE_base_en.md (T2VA/I2VA/FL2VA/L2VA)
#   data/h3/guide_ref_en.md   - VIDEO_PROMPT_WRITING_GUIDE_ref_en.md  (full-reference mode)
#
# Source: https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/docs

import json
import os
import re

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "h3",
)

_guide_cache = {}


def load_guide(name):
    """
    Read (and cache) one of the shipped H3 guides from data/h3/.

    Only a bare `.md` filename is accepted, so the lookup can never escape the
    guide directory even if a caller later wires this to a node widget.
    """
    safe_name = os.path.basename(name)
    if safe_name != name or not safe_name.endswith(".md"):
        raise ValueError(
            f"H3 guide name must be a bare .md filename inside data/h3, got: {name!r}"
        )

    if safe_name in _guide_cache:
        return _guide_cache[safe_name]

    path = os.path.join(_DATA_DIR, safe_name)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not read the H3 prompt guide at {path}: {exc}"
        )

    _guide_cache[safe_name] = text
    return text


# ----------------------------------------------------------------------
# Vocabulary lifted from section 4.3 of the base guide
# ----------------------------------------------------------------------

AUTO = "Auto"

CAMERA_MOTIONS = [
    AUTO,
    "Static Shot",
    "Zoom In",
    "Zoom Out",
    "Push In",
    "Pull Out",
    "Pan Left",
    "Pan Right",
    "Truck Left",
    "Truck Right",
    "Tilt Up",
    "Tilt Down",
    "Pedestal Up",
    "Pedestal Down",
    "Arc Shot",
    "Tracking Shot",
    "Shake Slightly",
    "Shake Strongly",
    "POV",
    "Roll Clockwise",
    "Roll Counterclockwise",
]

# "medium amplitude and normal speed are usually omitted" - guide 4.3
CAMERA_AMPLITUDES = [AUTO, "with small amplitude", "medium (omit)", "with large amplitude"]
CAMERA_SPEEDS = [AUTO, "at slow speed", "normal (omit)", "at fast speed"]

VISUAL_STYLE_CUSTOM = "Custom (use custom_visual_style)"

# Curated cinematic looks first: each is a complete one-line style statement
# (medium, camera, lenses, movement, colour) written to open a [Shot 1]
# verbatim, the way "Live-action, 35mm cinematic film aesthetic" does - but
# with the photography actually spelled out so every scene renders the same
# look.
_CINEMATIC_LOOKS = [
    "Live-action, 35mm cinematic film aesthetic",
    "Live-action, modern large-format digital cinema: Alexa 65 clarity, creamy shallow depth of field, natural HDR skin tones, smooth floating gimbal moves, neutral filmic grade",
    "Live-action, 65mm epic scale: IMAX-format deep-focus vistas, slow deliberate crane and dolly moves, natural available light, cool desaturated grade with warm protected skin tones",
    "Live-action, Wes Anderson style: perfectly symmetrical planimetric framing, deadpan centered portraits and 90-degree whip pans, pastel candy-box palette, meticulous dollhouse production design, flat 40mm perspective",
    "Live-action, David Fincher style: locked-down surgically precise camera, low-key tungsten and sodium-vapour practicals, cold teal-green shadow grade, crisp clinical digital sharpness",
    "Live-action, Roger Deakins naturalism: motivated single-source lighting, silhouettes against windows and fire, clean geometric widescreen compositions, gentle drifting camera, restrained filmic grade",
    "Live-action, Denis Villeneuve scale: monolithic minimalist compositions, tiny figures in vast spaces, atmospheric dust and haze diffusion, slow creeping push-ins, muted near-monochrome grade",
    "Live-action, Stanley Kubrick style: one-point-perspective symmetry, slow menacing zooms, ultra-wide 18mm interiors, cold practical-lit spaces, immaculate composed stillness",
    "Live-action, Terrence Malick golden hour: wide-angle handheld lyricism, backlit magic-hour sun flare, natural bounce light only, wandering steadicam through grass and doorways",
    "Live-action, Wong Kar-wai style: step-printed motion smear, saturated neon greens and reds, handheld intimacy in cramped interiors, rain-streaked glass and mirrors, romantic halation glow",
    "Live-action, Quentin Tarantino grindhouse: punchy 35mm Kodak saturation, crash zooms and low trunk-shot angles, long unbroken dialogue two-shots, warm 70s amber cast",
    "Live-action, 1970s paranoid thriller: 2.39 anamorphic Panavision, zoom-heavy long-lens surveillance framing, grimy urban browns and greens, grainy push-processed stock",
    "Live-action, A24 indie realism: 35mm grain, soft natural window light, muted pastel grade with lifted blacks, static tableaux broken by intimate handheld close-ups",
    "Live-action, neon noir: rain-slick night streets, cyan and magenta neon reflections, volumetric smoke and searchlights, anamorphic blue-streak flares, deep crushed blacks",
    "Live-action, 1940s film noir: hard black-and-white chiaroscuro, venetian-blind shadows, single low-key key light, canted dutch angles, cigarette smoke drifting through light beams",
    "Live-action, golden-age Technicolor: three-strip saturated primaries, glamour key lighting with crisp eye-lights, stately dolly moves, painted-backdrop soundstage depth",
    "Live-action, 16mm vérité documentary: handheld reportage framing, visible film grain and gate weave, available light only, quick reactive zooms and refocuses",
    "Live-action, Super 8 home movie: heavy grain and light leaks, warm faded Kodachrome colours, jittery handheld framing, soft vignetted frame edges",
    "Live-action, late-80s VHS camcorder: smeared lo-fi video texture, auto-exposure pumping, bleeding oversaturated colour, abrupt handheld pans and snap zooms",
    "Live-action, BBC nature documentary: long-lens telephoto compression, patient locked-off observation, golden dawn haze, pristine 8K clarity, sweeping aerial establishing shots",
    "Live-action, glossy blockbuster: sweeping 360-degree arc shots circling the hero at chest height, teal-and-orange grade, horizontal lens flares, low-angle wide shots, crisp high-shutter action",
    "Live-action, high-fashion editorial: glossy beauty lighting, bold saturated gel colours, slow-motion hair and fabric, macro texture inserts, high-contrast punchy grade",
    "Hand-painted 2D animation, Studio Ghibli style: lush watercolour backgrounds, soft natural light, gentle wind moving grass and hair, warm pastoral palette",
    "2D anime, 1990s theatrical cel style: hand-inked linework over painted backgrounds, dramatic key-frame lighting, subtle film grain over the cels",
    "3D CG animation, Pixar style: rounded appealing character design, soft global illumination, shallow cinematic depth of field, warm saturated storybook palette",
    "Stop-motion animation, Laika style: tactile handcrafted puppets, visible fabric and clay texture, miniature-set depth of field, slightly staccato hand-animated motion",
]

# ... then the guide's own examples ...
_GUIDE_VISUAL_STYLES = [
    "Cinematic",
    "live-action",
    "2D-animated",
    "3D CG",
    "claymation",
    "watercolor",
    "vintage film",
]

# ... then the APNext Cinematic vocabulary (data/next/cinematic/*.json), so the
# same looks the APNext Cinematic node emits can be picked straight from the
# dropdown. Film stock / format, grading and the big aesthetics list are the
# ones that read as a *style*; genre, directors, movies and shot types steer
# better through the context sockets.
_APNEXT_CINEMATIC_DIR = os.path.join(os.path.dirname(_DATA_DIR), "next", "cinematic")
_APNEXT_VISUAL_STYLE_FILES = ("film_type", "color_grading", "aesthetics")


def _apnext_cinematic_items(name):
    path = os.path.join(_APNEXT_CINEMATIC_DIR, f"{name}.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            items = json.load(fh).get("items", [])
    except (OSError, ValueError):
        return []
    return [str(item).strip() for item in items if str(item).strip()]


def _build_visual_styles():
    styles = [AUTO, VISUAL_STYLE_CUSTOM]
    seen = {s.lower() for s in styles}
    for source in [_CINEMATIC_LOOKS, _GUIDE_VISUAL_STYLES] + [
        _apnext_cinematic_items(name) for name in _APNEXT_VISUAL_STYLE_FILES
    ]:
        for item in source:
            if item.lower() not in seen:
                seen.add(item.lower())
                styles.append(item)
    return styles


VISUAL_STYLES = _build_visual_styles()


def resolve_visual_style(choice, custom=""):
    """
    Turn the visual_style dropdown (plus its free-text companion) into the
    style string the prompt builders expect.

    A non-empty custom value always wins, so anything can be typed in. "Custom"
    with an empty text box, like an empty choice, falls back to AUTO so the
    model derives the style itself.
    """
    custom = (custom or "").strip()
    if custom:
        return custom
    if not choice or choice == VISUAL_STYLE_CUSTOM:
        return AUTO
    return choice

SHOT_PLANS = [AUTO, "Single shot", "Two shots", "Three shots", "Four shots"]

_SHOT_COUNTS = {
    "Single shot": 1,
    "Two shots": 2,
    "Three shots": 3,
    "Four shots": 4,
}

# ----------------------------------------------------------------------
# Dialogue language
# ----------------------------------------------------------------------

DIALOGUE_AUTO = "Auto (match the setting)"
DIALOGUE_CUSTOM = "Custom (use custom_dialogue_language)"

# The language the characters actually speak. The name picked here is also what
# goes inside the <d>[...]</d> tag, so it is written the way the guide writes it.
DIALOGUE_LANGUAGES = [
    DIALOGUE_AUTO,
    "English",
    "Norwegian",
    "Swedish",
    "Danish",
    "Finnish",
    "Icelandic",
    "German",
    "Dutch",
    "French",
    "Spanish",
    "Portuguese",
    "Italian",
    "Polish",
    "Czech",
    "Romanian",
    "Hungarian",
    "Greek",
    "Russian",
    "Ukrainian",
    "Turkish",
    "Arabic",
    "Hebrew",
    "Persian",
    "Hindi",
    "Bengali",
    "Urdu",
    "Chinese",
    "Cantonese",
    "Japanese",
    "Korean",
    "Thai",
    "Vietnamese",
    "Indonesian",
    "Malay",
    "Filipino",
    "Swahili",
    DIALOGUE_CUSTOM,
]


def resolve_dialogue_language(choice, custom=""):
    """
    Turn the dropdown (plus its free-text companion) into a language name.

    Returns None for "let the model decide", which the dialogue directive turns
    into an instruction to pick a language that fits the scene. A non-empty
    custom value always wins, so dialects and unlisted languages still work.
    """
    custom = (custom or "").strip()
    if custom:
        return custom
    if not choice or choice in (DIALOGUE_AUTO, DIALOGUE_CUSTOM):
        return None
    return choice


CUT_STYLES = [
    AUTO,
    "the camera cuts to",
    "the shot cuts to",
    "the shot transitions to",
    "the shot changes to",
    "the shot switches to",
]


# The video model takes every word at face value: `the camera orbits her`
# can render outer space, not an arc shot. Every writer appends this so the
# scene text spells out what is physically meant.
LITERAL_CAMERA_DIRECTIVE = (
    "WORDS ARE READ LITERALLY by the video model, so spell out what is "
    "physically meant: write `an orbital camera move circling the performer at "
    "eye level` - never just `orbit`, which reads as outer space; `a high "
    "top-down camera angle`, not `bird's-eye` or `satellite view`; a "
    "figurative phrase that could be read as a physical event (`explodes into "
    "the chorus`, `rockets upward`, `melts into the next shot`) becomes the "
    "picture actually intended (`the chorus hits with a burst of light and "
    "movement`, `rises fast`, `a soft dissolve`). This goes for everything - "
    "camera moves, transitions, actions, metaphors: always describe the "
    "literal thing that should be on screen."
)


def camera_directive(motion, amplitude, speed):
    """
    Turn the three camera widgets into one instruction line, honouring the
    guide's rule that medium amplitude and normal speed are left unwritten.
    """
    if motion == AUTO:
        return (
            "Camera motion: choose motion types that suit the action, and write them as "
            "natural English inside the shot (motion type, plus amplitude and speed only "
            "when meaningful). Do not stack them as labels at the end of a sentence. "
            + LITERAL_CAMERA_DIRECTIVE
        )

    parts = [motion]
    if amplitude != AUTO and not amplitude.startswith("medium"):
        parts.append(amplitude)
    if speed != AUTO and not speed.startswith("normal"):
        parts.append(speed)

    phrase = " ".join(parts)
    return (
        f"Camera motion: the primary camera movement is `{phrase}`. Express it as natural "
        "English action inside the shot rather than as a trailing label. Additional shots "
        "may use other motion types when the action calls for it. "
        + LITERAL_CAMERA_DIRECTIVE
    )


def shot_directive(shot_plan, duration_seconds):
    """Instruction covering shot count and cut-time formatting."""
    duration = f"{duration_seconds:.2f}"

    if shot_plan == AUTO:
        count_rule = (
            "Choose the shot count that fits the action. Prefer a single shot unless a cut "
            "genuinely introduces new information about the subject, space, state, viewpoint or time."
        )
    else:
        count = _SHOT_COUNTS[shot_plan]
        count_rule = (
            f"Use exactly {count} shot{'s' if count > 1 else ''}."
            if count > 1
            else "Use exactly 1 shot."
        )

    return (
        f"{count_rule} The effective video duration is {duration} seconds. [Shot 1] carries no "
        f"timestamp; every later shot opens with a strictly increasing cut time in `[Shot N] At "
        f"MM:SS.mmm, ...` form that falls inside {duration} seconds."
    )


def toggle_directives(
    include_dialogue,
    include_on_screen_text,
    include_soundscape,
    include_non_diegetic_music,
    dialogue_language,
):
    """Feature switches shared by both writer nodes."""
    lines = []

    if include_dialogue:
        if dialogue_language:
            # Naming the language twice is deliberate: the tag alone leaves weaker
            # models writing English lines under a foreign label.
            spoken = (
                f"Dialogue: include spoken lines, and write the spoken words themselves in "
                f"{dialogue_language}. Do not write them in English and label them "
                f"{dialogue_language}, and do not append a translation. Tag each block as "
                f"<d>[{dialogue_language}] ...</d>."
            )
        else:
            spoken = (
                "Dialogue: include spoken lines. Choose the language that genuinely fits the "
                "setting and characters, write every spoken line in it, and name that same "
                "language inside the tag: <d>[Language] ...</d>. Do not append a translation."
            )

        lines.append(
            f"{spoken} Give each vocal source a stable (S1)/(S2) ID, and keep the identifying "
            "phrase, action and delivery outside the <d> block and in English."
        )
    else:
        lines.append(
            "Dialogue: no character speaks, sings, or delivers a voiceover. Do not emit any "
            "(Sx) speaker IDs or <d> blocks."
        )

    if include_on_screen_text:
        lines.append(
            'On-screen text: signs, banners, labels or subtitles that are actually visible go in '
            'English double quotation marks, verbatim and untranslated.'
        )
    else:
        lines.append("On-screen text: keep the frame free of readable signs, banners, labels or subtitles.")

    if include_soundscape:
        lines.append(
            "overall_soundscape: 1-4 English sentences in one paragraph covering ambience, "
            "physical action sounds and non-verbal human sounds. Do not repeat dialogue or "
            "singing here."
        )
    else:
        lines.append("overall_soundscape: output exactly `N/A`.")

    if include_non_diegetic_music:
        lines.append(
            "non_diegetic_music: 1-3 English sentences on instrumentation, tempo, rhythm and "
            "dynamic change. No abstract mood words and no explanation of emotional function."
        )
    else:
        lines.append("non_diegetic_music: output exactly `N/A`.")

    return lines


# ----------------------------------------------------------------------
# Wildness
# ----------------------------------------------------------------------

# Bands are (upper_bound_inclusive, label, directive, number_of_random_elements).
_WILDNESS_BANDS = (
    (
        15,
        "Conservative",
        "Stay literal. Render only what the input actually implies, adding just enough "
        "concrete detail to make the timeline filmable. No invented events, no surreal "
        "flourishes, no unmotivated camera tricks.",
        0,
    ),
    (
        40,
        "Grounded",
        "Stay believable, but direct it properly. Choose expressive framing, motivated "
        "lighting and small human behaviour that enrich the input without changing what "
        "it is about.",
        0,
    ),
    (
        65,
        "Bold",
        "Make strong authorial choices. Heightened lighting, striking compositions, "
        "expressive camera work and one memorable visual idea are welcome, as long as the "
        "scene still obeys ordinary physics.",
        1,
    ),
    (
        85,
        "Wild",
        "Break realism on purpose. Surreal juxtapositions, impossible transitions and "
        "dreamlike logic are encouraged. The result must still be a coherent, shootable "
        "timeline rather than a list of random images.",
        2,
    ),
    (
        100,
        "Unhinged",
        "Go fully unhinged. Reality is negotiable: scale, gravity, continuity and material "
        "behaviour can all misbehave. Commit hard to the strangeness, and still deliver a "
        "timeline a video model can actually follow, shot by shot.",
        3,
    ),
)

# Concrete, filmable weirdness. Each is a visual event, not a mood word.
UNHINGED_ELEMENTS = [
    "gravity reverses for a single object while everything else stays put",
    "the subject's shadow moves a beat out of sync with the subject",
    "a mirror or reflective surface shows something that is not in the room",
    "one material behaves like another - stone flows, cloth turns molten, water holds an edge",
    "an impossible scale shift: something small becomes enormous, or the reverse",
    "the environment breathes, expanding and contracting like a slow lung",
    "weather that belongs outdoors happens indoors",
    "a doorway or window opens onto a completely different biome",
    "the subject multiplies into synchronized copies that share one motion",
    "everyone in the background freezes while the subject keeps moving",
    "the floor turns to water and nobody reacts to it",
    "practical lights pulse in time with a rhythm no one on screen can hear",
    "the camera passes straight through a solid surface",
    "time stutters: one action repeats half a beat before continuing",
    "the scene briefly rewinds, then resumes forward",
    "ordinary objects swarm and move as a single organism",
    "the horizon tilts past vertical while the subject stays upright",
    "an out-of-place animal crosses frame with complete confidence",
    "the set reveals itself as a miniature, then becomes full scale again",
    "colour drains from everything except one object",
    "the colour palette inverts for a single beat",
    "a texture spreads across the frame like frost, converting whatever it touches",
    "the subject walks and the background scrolls the wrong way",
    "objects rearrange themselves the instant the camera looks away",
    "a second, older version of the scene bleeds through as a double exposure",
    "the light source is physically present and can be picked up",
    "rain falls upward into the sky",
    "the subject's clothing changes between one cut and the next without comment",
    "a hallway extends further the longer the camera pushes down it",
    "sound arrives visibly, distorting the air before it is heard",
    "the frame edge becomes a physical wall the subject can lean on",
    "one object stays perfectly sharp while everything else smears into motion",
    "the ground tessellates into moving tiles",
    "a crowd moves in perfect unison as if choreographed by accident",
    "the subject steps out of frame and immediately re-enters from the opposite side",
    "smoke or steam holds a solid shape long after it should disperse",
    "the scene is briefly lit as if from underwater",
    "an object falls upward off the table and settles on the ceiling",
    "the subject's reflection stays behind when they walk away",
    "a season changes across a single continuous shot",
]


def wildness_directive(wildness, rng):
    """
    Map the 0-100 wildness slider onto a creative-latitude instruction, plus a
    seeded selection of concrete unhinged elements once the slider is high enough.
    """
    wildness = max(0, min(100, int(wildness)))

    for upper, label, directive, element_count in _WILDNESS_BANDS:
        if wildness <= upper:
            break

    lines = [f"Creative latitude ({label}, wildness {wildness}/100): {directive}"]

    if element_count and UNHINGED_ELEMENTS:
        picks = rng.sample(UNHINGED_ELEMENTS, min(element_count, len(UNHINGED_ELEMENTS)))
        joined = "; ".join(picks)
        lines.append(
            f"Weave in {'this element' if len(picks) == 1 else 'these elements'} and make "
            f"{'it' if len(picks) == 1 else 'them'} land as real, visible events on the "
            f"timeline: {joined}."
        )

    return lines, label


# ----------------------------------------------------------------------
# Output parsing
# ----------------------------------------------------------------------


def strip_code_fence(text):
    """Unwrap a ```...``` block if the model wrapped its whole answer in one."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped

    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip()


def extract_section(text, field, all_fields):
    """
    Pull one `field:` section out of an H3 prompt, stopping at the next known
    field label or at the end of the text. Returns "" when the field is absent.

    Handles both layouts the guides use: a value on the same line as the label
    (base mode) and a value starting on the next line (full-reference mode).
    """
    others = [re.escape(f) for f in all_fields if f != field]
    # The end-of-text alternative matters for the final section, which has no
    # following label to stop at.
    stop = (
        r"(?=^[ \t]*(?:" + "|".join(others) + r")[ \t]*:|\Z)"
        if others
        else r"(?=\Z)"
    )

    pattern = re.compile(
        r"^[ \t]*" + re.escape(field) + r"[ \t]*:(.*?)" + stop,
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(text)

    return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Reference images
# ---------------------------------------------------------------------------

# ComfyUI's "MiniMax H3 Reference to Video" node accepts up to nine reference
# images and numbers them in connection order. The writers mirror that: attached
# image k is <Picture k> in the prompt, and is handed back out on the matching
# image_k output so the same tensors can feed the video node in the same order.
MAX_REFERENCE_IMAGES = 9

REFERENCE_IMAGE_NAMES = tuple(f"image_{i}" for i in range(1, MAX_REFERENCE_IMAGES + 1))


def reference_image_inputs():
    """Optional IMAGE sockets image_1..image_9. Keep them last so the UI can autogrow."""
    return {
        name: ("IMAGE", {
            "tooltip": (
                f"Reference image {i}: <Picture {i}> in the prompt. Connect the same image "
                f"to image_{i} on the MiniMax H3 Reference to Video node, or use this node's "
                f"image_{i} output."
            ),
        })
        for i, name in enumerate(REFERENCE_IMAGE_NAMES, 1)
    }


def reference_image_outputs():
    """Pass-through IMAGE outputs, same names and order as the inputs."""
    return ("IMAGE",) * MAX_REFERENCE_IMAGES, REFERENCE_IMAGE_NAMES


# How the writers may interpret the reference pictures. "Characters only"
# stops the model from reading a scene or location out of a character photo's
# backdrop (a picture note like `Image 2: the diner, use as the location`
# still declares a picture a location explicitly).
REFERENCE_IMAGE_USE = [
    "Characters only (ignore picture backgrounds)",
    "Auto (decide what each picture shows)",
]

REFERENCE_IMAGE_USE_TOOLTIP = (
    "How the reference pictures may be read. Characters only: every picture is a "
    "character/performer reference - identity (and wardrobe) carries over, the photo's "
    "background, setting and lighting are ignored, and no scene or location is ever "
    "derived from them; a picture note can still declare a specific image a location or "
    "prop. Auto: the model decides what each picture shows (a backdrop can become the "
    "scene)."
)


def characters_only_refs(use):
    return not str(use or "").strip().startswith("Auto")


def characters_only_directive():
    return (
        "Every reference picture is a CHARACTER reference unless a picture note below "
        "explicitly says it shows a location or a prop: take only the person - face, "
        "hair, build, distinctive marks - and IGNORE the photo's background, setting, "
        "lighting, weather and mood completely. Never derive a scene, a location, a "
        "location lock or a lock anchor from what is behind or around a pictured "
        "person; every setting comes from the brief and the location lock alone."
    )


# ----------------------------------------------------------------------
# Typed references for the base writers
# ----------------------------------------------------------------------
#
# The base format has no <Picture N> for anything but keyframes, so a photo of
# the hero, the location or a prop can only reach the video model as words.
# These sockets say what each picture is FOR, so the writer knows what to
# describe and what to ignore.

TYPED_REFERENCE_KINDS = (
    ("subject", 3, (
        "Subject {i}: a person, creature or character. Only WHO they are carries over - face, "
        "hair, build, wardrobe, distinctive marks. The photo's backdrop, light and framing are "
        "ignored; the scene comes from your idea. The video model never sees this image."
    )),
    ("scenery", 3, (
        "Scenery {i}: a location or environment. Its architecture, terrain, light, weather, "
        "palette and mood become the setting, described in words. People in it are ignored."
    )),
    ("object", 3, (
        "Object {i}: a prop, product, vehicle or costume piece to depict faithfully - shape, "
        "colour, material, markings. Where it is in the photo is ignored."
    )),
)

TYPED_REFERENCE_NAMES = tuple(
    f"{kind}_{i}" for kind, count, _ in TYPED_REFERENCE_KINDS for i in range(1, count + 1)
)


def typed_reference_inputs():
    """Optional IMAGE sockets subject_1..3, scenery_1..3, object_1..3."""
    inputs = {}
    for kind, count, tooltip in TYPED_REFERENCE_KINDS:
        for i in range(1, count + 1):
            inputs[f"{kind}_{i}"] = ("IMAGE", {"tooltip": tooltip.format(i=i)})
    return inputs


def collect_typed_references(slots, to_pil):
    """
    Connected typed references as (label, PIL image), in socket order, e.g.
    ("Subject 1", img). Only the first frame of each input counts. Numbering is
    per kind and follows connection order, so a gap (subject_1 + subject_3)
    still yields Subject 1 and Subject 2.
    """
    picked = []
    for kind, count, _ in TYPED_REFERENCE_KINDS:
        n = 0
        for i in range(1, count + 1):
            tensor = slots.get(f"{kind}_{i}")
            if tensor is None:
                continue
            n += 1
            picked.append((f"{kind.capitalize()} {n}", to_pil(tensor[0])))
    return picked


# ----------------------------------------------------------------------
# Reference image normalisation - the released reference pipeline's rule
# (diffusers modular_pipelines/minimax_h3/before_encoder.py):
#
#     scale = reference_image_short_edge / min(width, height)   # 2048
#
# rounded to /32 - INCLUDING upscaling. ComfyUI's own node clamps this with
# min(1.0, ...) and never upscales, so a small reference (say 512px) reaches
# the DiT with 16x fewer reference latent rows than the model was built to
# see, and identity fidelity suffers. We do the resize on the writers' image
# pass-throughs instead: by the time the stock node sees the picture its own
# scale is min(1.0, 2048/2048) = 1.0 and its resize is a no-op.
#
# Upscaling is not free: reference rows ride through every sampling step, so
# quadrupling the short edge multiplies those rows by 16. That is the
# released pipeline's cost; the console line makes the final size visible.

REFERENCE_SHORT_EDGE = 2048
REFERENCE_MULTIPLE = 32
# the reference pipeline rejects references outside 1:4 .. 4:1
REFERENCE_MAX_ASPECT = 4.0


def scale_reference_image(tensor):
    """
    An IMAGE tensor [B,H,W,C] scaled so its SHORT edge is
    REFERENCE_SHORT_EDGE, both sides rounded to REFERENCE_MULTIPLE - up as
    well as down, exactly like the released reference pipeline (ComfyUI's
    node only ever downscales). None passes through; a conforming image is
    untouched. Warns on aspect ratios outside 1:4..4:1, which the reference
    pipeline rejects outright.
    """
    if tensor is None:
        return None
    import torch
    _b, h, w, _c = tensor.shape
    aspect = w / h if w >= h else h / w
    if aspect > REFERENCE_MAX_ASPECT:
        print(
            f"⚠️ H3 reference image {w}x{h} is outside the 1:4..4:1 aspect range "
            "the reference pipeline accepts - expect degraded identity transfer."
        )
    s = REFERENCE_SHORT_EDGE / float(min(h, w))
    nw = max(REFERENCE_MULTIPLE, round(w * s / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    nh = max(REFERENCE_MULTIPLE, round(h * s / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    if (nh, nw) == (h, w):
        return tensor
    x = tensor.movedim(-1, 1)  # BHWC -> BCHW
    if nw * nh < w * h:
        x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="area")
    else:
        x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bicubic", antialias=True)
    return x.movedim(1, -1).clamp(0.0, 1.0)


def scale_reference_passthrough(slots, names):
    """The image_N pass-through tuple, every connected tensor normalised."""
    scaled = tuple(scale_reference_image(slots.get(name)) for name in names)
    n = sum(1 for t in scaled if t is not None)
    if n:
        first = next(t for t in scaled if t is not None)
        print(
            f"🖼️ H3 reference images: {n} fitted to short edge {REFERENCE_SHORT_EDGE} "
            f"(/{REFERENCE_MULTIPLE}, reference-pipeline rule, upscaling included) - "
            f"e.g. {first.shape[2]}x{first.shape[1]}"
        )
    return scaled


VISION_MAX_SIDE = 1024


def downscale_for_vision(image, max_side=VISION_MAX_SIDE):
    """
    A copy of a PIL image small enough to send to the writer as context.

    The video node gets the original tensors; Claude only needs enough pixels to
    recognise a face, an outfit or a place, and a 4K still as base64 is a
    megabyte of prompt for nothing.
    """
    w, h = image.size
    scale = max_side / float(max(w, h))
    if scale >= 1.0:
        return image
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))))


def collect_reference_images(tensors, to_pil):
    """
    Attached images in slot order as (slot_number, PIL image).

    Only the first frame of each input acts as that reference. Empty slots
    between connected ones are reported, since the video node numbers by
    connection order and would count differently.
    """
    picked = [(i, to_pil(t[0])) for i, t in enumerate(tensors, 1) if t is not None]
    if picked and picked[-1][0] != len(picked):
        used = ", ".join(f"image_{i}" for i, _ in picked)
        print(
            f"⚠️  H3 reference images have a gap ({used}). The prompt numbers them "
            f"<Picture 1>..<Picture {len(picked)}> in that order; wire the video node "
            "the same way, or fill the slots without gaps."
        )
    return picked


# ----------------------------------------------------------------------
# "Show advanced inputs": ComfyUI hides inputs whose options carry
# `advanced: True` behind a per-node toggle. The H3 nodes keep a short
# "simple" form and put everything else behind that toggle.

_NEVER_ADVANCED_PREFIXES = ("context_", "image_", "cast_")


def mark_advanced(input_types, simple, *, also_advanced=()):
    """
    Return a copy of an INPUT_TYPES dict where every required/optional entry
    that is not in `simple` (and not an autogrow socket) carries
    `advanced: True`. Entries named in `also_advanced` are forced advanced even
    when they are sockets.
    """
    out = {}
    for group, entries in input_types.items():
        if group not in ("required", "optional") or not isinstance(entries, dict):
            out[group] = entries
            continue
        new = {}
        for name, spec in entries.items():
            if not isinstance(spec, tuple):
                spec = (spec,)
            typ = spec[0]
            opts = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], dict) else {}
            is_socket = not isinstance(typ, list) and typ not in ("STRING", "INT", "FLOAT", "BOOLEAN") or opts.get("forceInput")
            forced = name in also_advanced
            if (
                (name not in simple and not name.startswith(_NEVER_ADVANCED_PREFIXES) and name != "llm"
                 and (not is_socket or forced))
                or forced
            ):
                opts["advanced"] = True
            new[name] = (typ, opts) if (opts or len(spec) > 1) else (typ,)
        out[group] = new
    return out


def with_advanced_inputs(cls, simple, also_advanced=()):
    """Wrap cls.INPUT_TYPES so every non-simple input is marked advanced."""
    orig = cls.INPUT_TYPES.__func__ if hasattr(cls.INPUT_TYPES, "__func__") else cls.INPUT_TYPES

    @classmethod
    def INPUT_TYPES(kls):
        return mark_advanced(orig(kls), set(simple), also_advanced=set(also_advanced))

    cls.INPUT_TYPES = INPUT_TYPES
    return cls
