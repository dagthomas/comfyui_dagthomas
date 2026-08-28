# Shared pieces for the H3 nodes that write several scenes in one call.
#
# Both multi-scene writers (the crossover writer and the plain scenes writer)
# ask Claude Code for the same envelope layout, split it the same way, and hand
# the result out as a ComfyUI list. This keeps the contract in one place.

import json
import re

from .common import strip_code_fence

MAX_SCENES = 10

DURATION_MODES = [
    "Fixed (every scene = scene_duration)",
    "Vary 5-15s (let Claude pace each scene)",
]

_SYNOPSIS_RE = re.compile(
    r"===\s*SYNOPSIS\s*===\s*(.*?)\s*===\s*END SYNOPSIS\s*===", re.DOTALL | re.IGNORECASE
)
# A scene starts at its header. The `=== END SCENE NN ===` closer is part of
# the contract, but local models drop it often enough that requiring it lost
# whole chunks: without a match the fallback made one giant "scene" out of a
# four-scene reply and the rest were re-requested. So the body runs to the
# END marker when there is one, otherwise to the next header (or the end).
_SCENE_HEAD_RE = re.compile(
    r"^[ \t]*(?:\*\*|#+\s*)?===\s*SCENE\s+(\d+)\s*(?:\|\s*duration:\s*([0-9.]+)\s*(?:s|sec|seconds)?\s*)?===[ \t]*(?:\*\*)?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_SCENE_END_RE = re.compile(
    r"^[ \t]*(?:\*\*)?===\s*END\s+SCENE(?:\s*\d+)?\s*===(?:\*\*)?[ \t]*$", re.IGNORECASE | re.MULTILINE
)
_SYNOPSIS_HEAD_RE = re.compile(r"^[ \t]*===\s*(?:END\s+)?SYNOPSIS\s*===", re.IGNORECASE | re.MULTILINE)


# The four H3 sections, in order. Used as the stop-list when pulling one
# section back out of a finished scene.
_SCENE_FIELDS = (
    "subject_definitions",
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)


def envelope_contract(section_labels):
    """The output-contract paragraph both writers append to their system prompt."""
    labels = ", ".join(f"`{label}:`" for label in section_labels)
    first = section_labels[0]
    return (
        "Output contract - the node parses this mechanically, so it must be exact:\n"
        "- First a synopsis block:\n"
        "  === SYNOPSIS ===\n"
        "  Title: ...\n  Logline: one or two sentences\n"
        "  Wardrobe: one line per recurring character - `Name: anchor, anchor, anchor`\n"
        "  Locations: one line per recurring place - `Name: anchor, anchor, anchor`\n"
        "  === END SYNOPSIS ===\n"
        "- Then one envelope per scene, numbered from 01:\n"
        "  === SCENE 01 | duration: 15.0 ===\n"
        f"  {first}: ...\n  (remaining sections)\n"
        "  === END SCENE 01 ===\n"
        f"- Inside an envelope: exactly the section labels {labels} in that order, plain "
        "text, one blank line between sections and between shots. No markdown, no bold, no "
        "fences, no bullets, no commentary anywhere in the output.\n"
        "- `duration:` is the length in seconds that scene will be rendered at; write it as a "
        "plain number with one decimal.\n"
        "- Never write a title card, credit card, logo card or any shot without people unless "
        "explicitly asked. Every scene is story."
    )


WARDROBE_TOOLTIP = (
    "Wardrobe lock, one line per CAST member, e.g. `Sheldon: brown corduroy jacket, green "
    "Flash T-shirt, khaki trousers, small silver ring in the left nostril`. Used word-for-word "
    "in subject_definitions and at the character's first appearance in each scene; later "
    "shots carry it on the `<Subject N>` label, as H3's guide specifies. Empty = Claude "
    "fixes one outfit per cast member itself (in the synopsis) and reuses it. Only the "
    "cast is locked - extras and background people are described where they appear and "
    "never carry an anchor set."
)

ENFORCE_WARDROBE_TOOLTIP = (
    "After writing, check that each character's FIRST appearance in a scene states all of "
    "that character's wardrobe anchors verbatim, and that every scene set in a locked "
    "location restates that location's anchors. If anything is dropped or changed, the "
    "model gets one repair turn in the same session. Off = trust the first answer."
)

LOCATIONS_TOOLTIP = (
    "Location lock, one line per recurring place, e.g. `Sheldon's living room: beige "
    "three-seat sofa facing a wall-mounted TV on the LEFT, tall bookshelf of comics behind "
    "it, bay window with white blinds on the RIGHT, warm tungsten floor lamp in the far "
    "corner`. Used word-for-word in every scene set there, so the room looks the same in "
    "every scene. Empty = the model fixes each recurring place itself (in the synopsis) "
    "and repeats it."
)

_LOCATION_ANCHOR_RULES = (
    "Location anchor rules: every anchor is one exact phrase naming a fixed feature of the "
    "place with its colour/material and its position - the big furniture or landmarks and "
    "where they stand (`beige three-seat sofa facing a wall-mounted TV on the LEFT`), the "
    "openings (`bay window with white blinds on the RIGHT`, `door in the back wall`), the "
    "surfaces (`dark hardwood floor`, `exposed red-brick wall`) and the practical light "
    "(`warm tungsten floor lamp in the far corner`, `grey overcast daylight through the "
    "window`). Fix a side once (LEFT/RIGHT as seen from the main camera position) and never "
    "swap it; keep the camera on the same side of the room so screen direction holds. No "
    "synonyms or paraphrases later (`sofa` never becomes `couch`, `beige` never becomes "
    "`cream`): copy the phrase character-for-character. Nothing in the lock is ever dropped, "
    "moved or recoloured, and no new major furniture or openings appear. A place only "
    "changes when the story changes it on screen."
)


def locations_directive(locations=""):
    """
    Rooms and places drift between scenes unless each is fixed once and copied
    into every scene set there. With user text: that text is the lock. Without:
    the model writes the lock into the synopsis first, then reuses it.
    """
    locations = (locations or "").strip()
    if locations:
        lines = [line.strip() for line in locations.splitlines() if line.strip()]
        return (
            "Location lock (from the user) - copy these anchors word-for-word as the "
            "synopsis `Locations:` lines and, in EVERY scene set in that place, into the "
            "first shot of the scene right after the place is named (`... in Sheldon's "
            "living room - <anchors> - ...`). Later shots in the same place do NOT "
            "re-inventory the room: name only the one or two anchors actually in that "
            "frame, the way the guide asks for concrete frame anchors within what is "
            "visible. " + _LOCATION_ANCHOR_RULES +
            "\n" + "\n".join(f"    {line}" for line in lines)
        )
    return (
        "Location lock: before scene 01, name every place that appears in more than one "
        "scene and fix it ONCE - 3-6 anchors - as the synopsis `Locations:` lines in exactly "
        "this form, one place per line, anchors separated by commas:\n"
        "    Locations:\n"
        "    Sheldon's living room: beige three-seat sofa facing a wall-mounted TV on the "
        "LEFT, tall bookshelf of comics behind it, bay window with white blinds on the RIGHT, "
        "warm tungsten floor lamp in the far corner\n"
        "Then, in EVERY scene set in that place, name the place with exactly that name and "
        "copy its anchors word-for-word into the first shot of the scene right after the "
        "place is named (`... in Sheldon's living room - <anchors> - ...`); later shots in "
        "the same scene restate the anchors that are in frame. A place used in only one "
        "scene needs no lock. " + _LOCATION_ANCHOR_RULES
    )

# The official guide, 5.3: "At the first clear appearance of an important
# <Subject N>, describe its referenced characteristics, position in the frame,
# and current action within what is actually visible in the shot. Continue
# using the same label in later shots without redefining what the label
# represents." Restating the outfit every shot is not extra safety - it is
# 150-250 words a scene the description does not get to spend on the picture.
_CAST_ONLY = (
    "Lock ONLY the cast. Everyone else the story needs - crowds, extras, a "
    "shopkeeper, a neighbour, a passer-by, a friend in the background - gets NO "
    "`Wardrobe:` line and no anchor set. Describe them once, briefly, where they "
    "appear, in whatever detail that frame actually shows, and let them go. A "
    "locked outfit is a promise to restate it in full every time that person is on "
    "screen; spending that on someone standing at the back of one shot is a large "
    "part of the scene's word budget bought for nothing."
)

_LABEL_CARRIES = (
    "In the REST of that scene use the bare label (`<Subject 1>` or the character's "
    "name) and never repeat the outfit: the label already carries it. Describe only "
    "what changes - what they are doing, where they are in frame, what the light is "
    "doing to them - plus any anchor the action puts in close-up."
)

_ANCHOR_RULES = (
    "Anchor rules: every anchor is one exact phrase - a precise colour word plus material "
    "or pattern plus garment (`dark-brown corduroy jacket`, `forest-green cotton T-shirt`), "
    "hair as colour plus style, and every accessory or mark with its exact position "
    "(`small silver ring in the LEFT nostril`, `thin gold hoop in the RIGHT ear`, `scar "
    "over the LEFT eyebrow`). Fix a side once and never swap it. No synonyms or "
    "paraphrases later (`burgundy` never becomes `dark red`; `jacket` never becomes "
    "`coat`): copy the phrase character-for-character. Nothing extra appears that is not "
    "in the lock - no additional jewellery, glasses, hats or bags - and nothing in the lock "
    "is ever dropped, recoloured or moved. Clothes, hair, accessories and props stay "
    "identical across every scene unless the brief explicitly calls for a costume change, "
    "and then the change is stated on screen."
)


def wardrobe_directive(wardrobe=""):
    """
    Wardrobe drifts between scenes unless each outfit is fixed once and restated
    where H3 actually binds it: the character's FIRST appearance in the scene.

    It used to be restated in every shot, which is what the official guide tells
    you not to do - "continue using the same label in later shots without
    redefining what the label represents". H3 binds the description to the
    `<Subject N>` label at its first clear appearance and carries it through the
    clip, so the repeats bought no fidelity and cost 150-250 words a scene out
    of a description the guide budgets at 350-500.

    With user text: that text is the lock. Without: Claude writes the lock into
    the synopsis first, then reuses it.
    """
    wardrobe = (wardrobe or "").strip()
    if wardrobe:
        lines = [line.strip() for line in wardrobe.splitlines() if line.strip()]
        return (
            "Wardrobe lock (from the user) - copy these anchors word-for-word as the "
            "synopsis `Wardrobe:` lines, into that character's `<Subject N>` line in "
            "subject_definitions, and into the FIRST shot of each scene in which the "
            "character is on screen, right after the first mention of them in that shot "
            "(`... <Subject 1> Sheldon (S1), wearing <anchors>, ...`). " + _LABEL_CARRIES
            + " " + _CAST_ONLY + " " + _ANCHOR_RULES +
            "\n" + "\n".join(f"    {line}" for line in lines)
        )
    return (
        "Wardrobe lock: before scene 01, DESIGN one outfit per CAST member the way a "
        "costume department would - it fits the concept's genre, era, season and "
        "setting and the character's role, age and status in the story; its colours "
        "read on camera against the locked locations (contrast, not camouflage); and "
        "it is distinctive enough to identify the character in a wide shot. Never "
        "default to generic modern streetwear when the world suggests otherwise. "
        "Fix it as 3-5 anchors and write them as the synopsis `Wardrobe:` lines in "
        "exactly this form, one character per line, anchors separated by commas:\n"
        "    Wardrobe:\n"
        "    Sheldon Cooper: dark-brown corduroy jacket, forest-green cotton T-shirt, "
        "khaki chino trousers, short side-parted brown hair\n"
        "Then copy that character's anchors word-for-word into their `<Subject N>` line "
        "in subject_definitions and into the FIRST shot of each scene in which the "
        "character is on screen, right after the first mention of them in that shot "
        "(`... <Subject 1> Sheldon Cooper (S1), wearing <anchors>, ...`). " + _LABEL_CARRIES
        + " " + _CAST_ONLY + " " + _ANCHOR_RULES
    )


# ----------------------------------------------------------------------
# Wardrobe verification
# ----------------------------------------------------------------------

_SYNOPSIS_HEADERS = ("title:", "logline:", "lyric map:", "cast:", "concept:", "motifs:",
                     "wardrobe:", "locations:", "location:")
_SHOT_SPLIT_RE = re.compile(r"(\[Shot\s*\d+\])", re.IGNORECASE)
_OFFSCREEN_RE = re.compile(
    r"[^.\n]*\b(?:not in frame|out of frame|off[- ]screen|not visible|not yet in frame|"
    r"has not entered|remains? outside|is heard)\b[^.\n]*[.\n]?",
    re.IGNORECASE,
)


def parse_wardrobe_lock(synopsis):
    """
    {character: [anchor, ...]} from the synopsis `Wardrobe:` lines.

    Accepts both layouts: a `Wardrobe:` header followed by `Name: a, b, c`
    lines, or `Wardrobe: Name: a, b, c` on one line. Stops at the next synopsis
    header or a blank line.
    """
    return _parse_lock_block(synopsis, ("wardrobe:",))


def parse_location_lock(synopsis):
    """{place: [anchor, ...]} from the synopsis `Locations:` (or `Location:`) lines."""
    return _parse_lock_block(synopsis, ("locations:", "location:"))


def _parse_lock_block(synopsis, headers):
    locks = {}
    lines = (synopsis or "").splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip().lower().startswith(headers):
        i += 1
    if i == len(lines):
        return locks
    stripped = lines[i].strip()
    header = next(h for h in headers if stripped.lower().startswith(h))
    first = stripped[len(header):].strip()
    block = ([first] if first else []) + lines[i + 1:]
    for raw in block:
        line = raw.strip().lstrip("-*• ").strip()
        if not line:
            if locks:
                break
            continue
        if line.lower().startswith(_SYNOPSIS_HEADERS):
            break
        name, sep, rest = line.partition(":")
        if not sep or not rest.strip():
            continue
        name = re.sub(r"\s*\(.*?\)\s*", " ", name).strip()
        anchors = [a.strip(" .;") for a in re.split(r"[,;]", rest) if a.strip(" .;")]
        anchors = [a for a in anchors if len(a) >= 4]
        if name and anchors:
            locks[name] = anchors
    return locks


def cast_names(cast):
    """
    The bare names in a cast block - the `Name` of each `Name: description` line.

    Lines with no colon are free-form descriptions rather than named parts and
    have no name to match on, so they are skipped.
    """
    names = []
    for line in (cast if isinstance(cast, (list, tuple)) else (cast or "").splitlines()):
        head, sep, _ = str(line).strip().lstrip("-*• ").partition(":")
        if not sep:
            continue
        head = re.sub(r"\s*\(.*?\)\s*", " ", head).strip()
        if head:
            names.append(head)
    return names


def restrict_to_cast(locks, names):
    """
    Drop wardrobe locks for anyone who is not in the cast.

    A lock is a standing promise to restate a full anchor set every time that
    person appears. That is worth it for the people the video is about, and it
    is most of a scene's word budget when the model has quietly locked the
    neighbour, the shopkeeper and two kids in the background as well.

    With no named cast (the model invents the performer) there is nothing to
    filter against, so every lock stands - failing open, because dropping every
    lock would silently disable the continuity check.
    """
    if not names or not locks:
        return locks, []
    wanted = {n.strip().lower() for n in names}
    firsts = {n.split()[0] for n in wanted if n.split()}
    kept, dropped = {}, []
    for name, anchors in locks.items():
        low = name.strip().lower()
        first = low.split()[0] if low.split() else low
        if low in wanted or first in firsts or any(w.split()[0] == first for w in wanted if w.split()):
            kept[name] = anchors
        else:
            dropped.append(name)
    return (kept or locks), ([] if not kept else dropped)


def _name_patterns(name):
    """Regexes that mean `this character is mentioned`: full name, then first name."""
    full = re.escape(name.strip())
    pats = [re.compile(rf"\b{full}\b", re.IGNORECASE)]
    first = name.strip().split()[0]
    if len(first) >= 3 and first.lower() != name.strip().lower():
        pats.append(re.compile(rf"\b{re.escape(first)}\b", re.IGNORECASE))
    return pats


def wardrobe_violations(scenes, locks):
    """
    [(scene_no, shot_no, name, anchor), ...] for the shot in which a locked
    character FIRST appears in a scene without an anchor stated verbatim
    (case-insensitive).

    Once per scene, not once per shot. That is where H3 binds the outfit to the
    `<Subject N>` label; later shots reuse the label, and demanding the full
    parenthetical again in each of them is what pushed these descriptions past
    the guide's 350-500 word budget. Checking the first appearance still catches
    the failure that matters - a character arriving with no outfit fixed at all.

    Sentences that say the character is NOT in frame do not count as presence.
    """
    found = []
    if not locks:
        return found
    patterns = {name: _name_patterns(name) for name in locks}
    for scene_no, prompt in enumerate(scenes, 1):
        parts = _SHOT_SPLIT_RE.split(prompt)
        # parts: [pre, "[Shot 1]", body1, "[Shot 2]", body2, ...]
        pending = dict(locks)
        shot_no = 0
        for k in range(1, len(parts), 2):
            shot_no += 1
            if not pending:
                break
            body = parts[k + 1] if k + 1 < len(parts) else ""
            visible = _OFFSCREEN_RE.sub(" ", body)
            low = body.lower()
            for name in [n for n in pending if any(p.search(visible) for p in patterns[n])]:
                for anchor in pending.pop(name):
                    if anchor.lower() not in low:
                        found.append((scene_no, shot_no, name, anchor))
    return found


def wardrobe_repair_prompt(locks, violations, scene_count):
    """The one follow-up turn that asks for the full output again, anchors restored."""
    lock_lines = "\n".join(f"    {n}: {', '.join(a)}" for n, a in locks.items())
    by_shot = {}
    for scene_no, shot_no, name, anchor in violations:
        by_shot.setdefault((scene_no, shot_no, name), []).append(anchor)
    issue_lines = "\n".join(
        f"    Scene {s:02d} [Shot {sh}]: {n} is on screen but missing: "
        + ", ".join(f"`{a}`" for a in anchors)
        for (s, sh, n), anchors in sorted(by_shot.items())
    )
    return (
        "WARDROBE CHECK FAILED. The locked wardrobe from your synopsis is:\n"
        f"{lock_lines}\n"
        "These are the shots where the character FIRST appears in their scene "
        "without every anchor stated verbatim:\n"
        f"{issue_lines}\n"
        "Fix them: at each character's FIRST appearance in a scene, state ALL of that "
        "character's anchors character-for-character (same words, same colours, same "
        "side), right after the first mention of them in that shot. Leave the later "
        "shots alone - they use the bare label and must NOT repeat the outfit. Do not "
        "change the story, dialogue, timecodes, durations, shot count or anything else. "
        f"Return the COMPLETE output again in the exact same contract: the synopsis block "
        f"and all {scene_count} scene envelopes."
    )


def wardrobe_summary(locks, violations):
    if not locks:
        return "wardrobe: no lock found in synopsis"
    if not violations:
        return f"wardrobe: ok ({len(locks)} locked)"
    return f"wardrobe: {len(violations)} anchor miss(es)"


# ----------------------------------------------------------------------
# Location verification
def _place_patterns(name):
    """Regexes that mean `this scene is set in that place`: the full locked name."""
    full = re.escape(name.strip())
    pats = [re.compile(rf"\b{full}\b", re.IGNORECASE)]
    # "Sheldon's living room" is also referred to as "the living room"
    words = name.strip().split()
    if len(words) >= 2 and words[0].endswith(("'s", "’s")):
        tail = " ".join(words[1:])
        if len(tail) >= 5:
            pats.append(re.compile(rf"\b{re.escape(tail)}\b", re.IGNORECASE))
    return pats


def location_violations(scenes, locks):
    """
    [(scene_no, place, anchor), ...] for every scene that names a locked place
    but does not restate one of its anchors verbatim (case-insensitive) anywhere
    in the scene.
    """
    found = []
    if not locks:
        return found
    patterns = {name: _place_patterns(name) for name in locks}
    for scene_no, prompt in enumerate(scenes, 1):
        low = prompt.lower()
        for name, anchors in locks.items():
            if not any(p.search(prompt) for p in patterns[name]):
                continue
            for anchor in anchors:
                if anchor.lower() not in low:
                    found.append((scene_no, name, anchor))
    return found


# ----------------------------------------------------------------------
# Description length
# ----------------------------------------------------------------------

# The reference guide budgets `detailed_description` at 350-500 English words
# for a generation task. Measured against prompts that are known to render
# well (ostris/minimax_h3_1k: 1,000 prompts + their renders, 5 s clips), that
# is two to three times too long: median 152 words, p90 180, never above 250.
# So the budget scales with the clip: ~30 words per second, sub-linear at the
# top because a 15 s scene is not three 5 s scenes' worth of new information.
#   5 s: 120-190   8 s: 144-226   10 s: 160-250   15 s: 200-310
# The old constants remain as the ceiling for callers with no duration.
DESCRIPTION_MIN_WORDS = 350
DESCRIPTION_MAX_WORDS = 500


def description_budget(seconds):
    """(lo, hi) words for a scene of `seconds`; the guide's 350-500 when unknown."""
    try:
        sec = float(seconds or 0.0)
    except (TypeError, ValueError):
        sec = 0.0
    if sec <= 0:
        return DESCRIPTION_MIN_WORDS, DESCRIPTION_MAX_WORDS
    sec = max(3.0, min(15.5, sec))
    return int(round(80 + 8 * sec)), int(round(130 + 12 * sec))


def length_directive(seconds=None):
    """The LENGTH rule for a writer's prompt, with the numbers for its scene length."""
    if seconds:
        lo, hi = description_budget(seconds)
        target = f"for a {float(seconds):.0f}-second scene that is {lo}-{hi} words"
    else:
        target = "120-190 words for a 5-second scene, 160-250 for 10 seconds, 200-310 for 15"
    return (
        "LENGTH - a ceiling, not a target to fill: `integrated_multimodal_description` is "
        f"about 30 words per second of the scene, every shot together - {target}. Prompts "
        "measured to render well on H3 sit at 150 words for 5 seconds and never pass 250; "
        "past that the model stops reading and the picture gets vaguer, not richer. Spend "
        "the words on what the camera sees that is NEW in this scene - never on repeating "
        "an outfit a label already carries, re-listing a room an earlier shot fixed, or "
        "restating anything the scene has established. The budget is for the PICTURE - "
        "place, framing, who does what; the timed sync sentences a piece asks for (the "
        "`at 1.8 s ... lands on 2.1 s ... still settling` clauses for its beats, hits, "
        "drops) are written ON TOP of it and are never a reason to shorten the picture."
    )


def sound_fields_directive(music="auto"):
    """
    The two sound fields the way well-rendering prompts write them. `music`:
    "auto" (N/A unless the scene calls for score), "none" (always N/A), or a
    ready-made sentence describing the rule for the music field.
    """
    soundscape = (
        "`overall_soundscape` is ONE sentence: the concrete sounds the picture makes, in "
        "order, as a comma list - `A shin striking padded ribs with a heavy thump, ropes "
        "stretching under a falling body, boots pivoting and scuffing canvas, a stadium roar "
        "rising sharply.` No mood words, no dialogue, no music."
    )
    if music == "none":
        music_rule = "`non_diegetic_music` is `N/A`."
    elif music == "auto":
        music_rule = (
            "`non_diegetic_music` is `N/A` unless the scene genuinely calls for score - most "
            "scenes do not - and then ONE sentence: instrumentation, tempo, and one cue tied "
            "to a visible moment (`Pizzicato strings and solo bassoon at moderate tempo, "
            "staccato throughout, a single tuba note on the belly-flop.`)."
        )
    else:
        music_rule = str(music)
    return f"SOUND FIELDS: {soundscape} {music_rule}"


def shots_directive(seconds=None):
    """How many shots a scene should have - measured, not a matter of taste."""
    base = (
        "SHOTS: ONE shot for a scene up to about 8 seconds, two for a longer one, three at "
        "most and only past 10 seconds - never four. Half of the prompts that render well "
        "are a single shot, and a cut only earns its place when it brings new information "
        "(a new space, viewpoint or state); when only the distance changes, move the camera "
        "instead. `[Shot 2] At 00:0X.XXX, the shot cuts to ...` for every cut. A static "
        "shot is a normal choice, not a failure of imagination."
    )
    if seconds:
        sec = float(seconds)
        n = 1 if sec <= 8.5 else (2 if sec <= 11 else 3)
        base += f" For this scene's {sec:.0f} seconds that means {n} shot{'s' if n > 1 else ''}."
    return base


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TIMING_SENTENCE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*s\b")      # "at 1.8 s", "lands on 2.1 s"


def picture_words(body):
    """
    Words of picture in a description: the sentences that carry a beat / hit
    timing (`... at 1.8 s ... still settling at 2.45 s`) are additions the
    sound events asked for, on top of the scene's budget, so they are not
    counted against it.
    """
    kept = [s for s in _SENTENCE_SPLIT_RE.split(body or "") if s and not _TIMING_SENTENCE_RE.search(s)]
    return len(" ".join(kept).split())


def description_lengths(scenes, field="integrated_multimodal_description",
                        all_fields=_SCENE_FIELDS):
    """
    [(scene_no, words), ...] - the PICTURE word count of each scene's description.

    Counts only the description, and inside it only the picture: the timed
    sync sentences are excluded (see picture_words). The other three sections
    are a line or two by construction, so a scene that blew its budget blew it
    here.
    """
    from .common import extract_section
    out = []
    for scene_no, prompt in enumerate(scenes, 1):
        body = extract_section(prompt or "", field, all_fields)
        out.append((scene_no, picture_words(body)))
    return out


def description_summary(lengths, durations=None):
    """One line: how the scenes sit against their word budget (per scene when durations are known)."""
    counts = [w for _, w in lengths if w]
    if not counts:
        return "description: nothing to measure"
    counts.sort()
    median = counts[len(counts) // 2]
    durations = list(durations or [])

    def budget(n):
        sec = durations[n - 1] if n - 1 < len(durations) else None
        return description_budget(sec)

    over = [n for n, w in lengths if w > budget(n)[1]]
    under = [n for n, w in lengths if 0 < w < budget(n)[0]]
    if durations:
        los = sorted(budget(n)[0] for n, _ in lengths)
        his = sorted(budget(n)[1] for n, _ in lengths)
        label = f"~30 words/s: {los[0]}-{his[-1]}" if los[0] != los[-1] or his[0] != his[-1] else f"{los[0]}-{his[0]}"
    else:
        label = f"{DESCRIPTION_MIN_WORDS}-{DESCRIPTION_MAX_WORDS}"
    parts = [f"description: median {median} words (budget {label})"]
    if over:
        parts.append(f"{len(over)} over (worst {max(counts)})")
    if under:
        parts.append(f"{len(under)} thin")
    if not over and not under:
        parts.append("all in budget")
    return ", ".join(parts)


def report_description_lengths(scenes, field="integrated_multimodal_description",
                               all_fields=_SCENE_FIELDS, durations=None):
    """
    Print the length summary and name the scenes that overran. Returns the
    summary so callers can fold it into their info string.

    Reported, never repaired: asking for a rewrite to hit a word count trades a
    known-good scene for a shorter one, and the budget is guidance about where
    detail stops paying - not a contract the node should enforce behind the
    user's back.
    """
    lengths = description_lengths(scenes, field, all_fields)
    summary = description_summary(lengths, durations)
    print(f"📏 {summary}")
    over = sorted(((w, n) for n, w in lengths if w > DESCRIPTION_MAX_WORDS), reverse=True)
    for w, n in over[:5]:
        print(f"   ↳ scene {n:02d}: {w} words, {w - DESCRIPTION_MAX_WORDS} over")
    return summary


def location_summary(locks, violations):
    if not locks:
        return "locations: no lock found in synopsis"
    if not violations:
        return f"locations: ok ({len(locks)} locked)"
    return f"locations: {len(violations)} anchor miss(es)"


def _repair_issue_parts(wardrobe_locks, wardrobe_misses, location_locks, location_misses):
    """The failure descriptions shared by the full and the subset repair prompts."""
    parts = []
    if wardrobe_misses:
        lock_lines = "\n".join(f"    {n}: {', '.join(a)}" for n, a in wardrobe_locks.items())
        by_shot = {}
        for scene_no, shot_no, name, anchor in wardrobe_misses:
            by_shot.setdefault((scene_no, shot_no, name), []).append(anchor)
        issue_lines = "\n".join(
            f"    Scene {s:02d} [Shot {sh}]: {n} is on screen but missing: "
            + ", ".join(f"`{a}`" for a in anchors)
            for (s, sh, n), anchors in sorted(by_shot.items())
        )
        parts.append(
            "WARDROBE CHECK FAILED. The locked wardrobe from your synopsis is:\n"
            f"{lock_lines}\n"
            "These are the shots where the character FIRST appears in their scene "
            "without every anchor stated verbatim:\n"
            f"{issue_lines}\n"
            "Fix them: at each character's FIRST appearance in a scene, state ALL of that "
            "character's anchors character-for-character (same words, same colours, same "
            "side), right after the first mention of them in that shot. Leave the later "
            "shots alone - they use the bare label and must NOT repeat the outfit."
        )
    if location_misses:
        lock_lines = "\n".join(f"    {n}: {', '.join(a)}" for n, a in location_locks.items())
        by_scene = {}
        for scene_no, name, anchor in location_misses:
            by_scene.setdefault((scene_no, name), []).append(anchor)
        issue_lines = "\n".join(
            f"    Scene {s:02d}: set in {n} but missing: " + ", ".join(f"`{a}`" for a in anchors)
            for (s, n), anchors in sorted(by_scene.items())
        )
        parts.append(
            "LOCATION CHECK FAILED. The locked locations from your synopsis are:\n"
            f"{lock_lines}\n"
            "These scenes are set in a locked place without restating every anchor "
            "verbatim:\n"
            f"{issue_lines}\n"
            "Fix them: in every scene set in a locked place, name the place exactly as "
            "locked and restate ALL of its anchors character-for-character (same words, "
            "same colours, same LEFT/RIGHT) in the first shot of the scene, right after the "
            "place is named; later shots in that scene restate the anchors in frame."
        )
    return parts


def continuity_repair_prompt(wardrobe_locks, wardrobe_misses, location_locks,
                             location_misses, scene_count):
    """
    The one follow-up turn that asks for the full output again with wardrobe
    and/or location anchors restored. Either half may be empty.
    """
    parts = _repair_issue_parts(wardrobe_locks, wardrobe_misses, location_locks, location_misses)
    return (
        "\n\n".join(parts) + "\n"
        "Do not change the story, dialogue, timecodes, durations, shot count or anything "
        f"else. Return the COMPLETE output again in the exact same contract: the synopsis "
        f"block and all {scene_count} scene envelopes."
    )


def subset_repair_prompt(wardrobe_locks, wardrobe_misses, location_locks,
                         location_misses, scene_items):
    """
    The repair turn for chunked runs, where re-emitting everything is too long:
    ask for ONLY the violating scene envelopes again, same numbers and durations.
    `scene_items` is [(scene_no, duration), ...].
    """
    parts = _repair_issue_parts(wardrobe_locks, wardrobe_misses, location_locks, location_misses)
    nos = ", ".join(f"{no:02d} (duration {dur:.1f})" for no, dur in scene_items)
    return (
        "\n\n".join(parts) + "\n"
        "Do not change the story, dialogue, timecodes, durations, shot count or anything "
        f"else. Return ONLY the corrected envelopes for scenes {nos}, each in the exact "
        "same envelope contract (`=== SCENE NN | duration: S.S ===` ... `=== END SCENE NN "
        "===`) with its original number and duration. Do not repeat the synopsis and do "
        "not return any other scene."
    )


def enforce_continuity(enabled, synopsis, parsed, scene_count, scene_duration,
                       session_id, info, repair, cast=()):
    """
    Shared post-check for the multi-scene writers: verify the wardrobe lock per
    shot and the location lock per scene; if anything is missing and a session
    is open, ask for ONE repair turn via `repair(prompt) -> (text, repair_info)`
    and keep the repaired answer only if it still parses to the same scene count.
    Returns (synopsis, parsed, info).
    """
    w_locks, dropped = restrict_to_cast(parse_wardrobe_lock(synopsis), cast_names(cast))
    if dropped:
        print(f"👔 not cast, so not locked: {', '.join(dropped)}")
    l_locks = parse_location_lock(synopsis)
    scenes = [p for _, _, p in parsed]
    w_miss = wardrobe_violations(scenes, w_locks)
    l_miss = location_violations(scenes, l_locks)
    summary = f"{wardrobe_summary(w_locks, w_miss)} | {location_summary(l_locks, l_miss)}"
    if not enabled or (not w_miss and not l_miss) or not session_id:
        if enabled:
            print(f"👔 {summary}")
        return synopsis, parsed, f"{info} | {summary}"

    print(
        f"👔 continuity: {len(w_miss)} wardrobe miss(es) in {len(w_locks)} character(s), "
        f"{len(l_miss)} location miss(es) in {len(l_locks)} place(s) - asking for one repair pass"
    )
    for scene_no, shot_no, name, anchor in w_miss[:10]:
        print(f"   ↳ scene {scene_no:02d} [Shot {shot_no}] {name}: missing `{anchor}`")
    for scene_no, name, anchor in l_miss[:10]:
        print(f"   ↳ scene {scene_no:02d} {name}: missing `{anchor}`")
    try:
        text, repair_info = repair(
            continuity_repair_prompt(w_locks, w_miss, l_locks, l_miss, scene_count)
        )
        new_synopsis, new_parsed = parse_scenes(text, scene_duration)
    except Exception as exc:  # keep the first answer rather than fail the run
        print(f"⚠️ continuity repair failed: {exc}")
        return synopsis, parsed, f"{info} | {summary}, repair failed"
    if len(new_parsed) != len(parsed):
        print(f"⚠️ continuity repair returned {len(new_parsed)} scene(s); keeping the original {len(parsed)}")
        return synopsis, parsed, f"{info} | {summary}, repair discarded"
    new_scenes = [p for _, _, p in new_parsed]
    w_left = wardrobe_violations(new_scenes, parse_wardrobe_lock(new_synopsis) or w_locks)
    l_left = location_violations(new_scenes, parse_location_lock(new_synopsis) or l_locks)
    print(
        f"👔 after repair: wardrobe {len(w_miss)} -> {len(w_left)}, "
        f"locations {len(l_miss)} -> {len(l_left)} miss(es) | {repair_info}"
    )
    return (
        new_synopsis or synopsis,
        new_parsed,
        f"{info} | wardrobe: repaired {len(w_miss)} -> {len(w_left)} | "
        f"locations: repaired {len(l_miss)} -> {len(l_left)}",
    )


def enforce_continuity_chunked(enabled, synopsis, parsed, session_id, info, repair, cast=()):
    """
    Post-check for runs written in chunks (music videos: 13-20 scenes), where a
    full re-emit is too long to ask for: verify every scene against the synopsis
    locks, then use ONE repair turn to re-emit only the violating scenes and
    splice them back in by number. Returns (parsed, info).
    """
    w_locks, dropped = restrict_to_cast(parse_wardrobe_lock(synopsis), cast_names(cast))
    if dropped:
        print(f"👔 not cast, so not locked: {', '.join(dropped)}")
    l_locks = parse_location_lock(synopsis)
    scenes = [p for _, _, p in parsed]
    w_miss = wardrobe_violations(scenes, w_locks)
    l_miss = location_violations(scenes, l_locks)
    summary = f"{wardrobe_summary(w_locks, w_miss)} | {location_summary(l_locks, l_miss)}"
    if not enabled or (not w_miss and not l_miss) or not session_id:
        if enabled:
            print(f"👔 {summary}")
        return parsed, f"{info} | {summary}"

    by_no = {no: dur for no, dur, _ in parsed}
    bad = sorted({m[0] for m in w_miss} | {m[0] for m in l_miss})
    items = [(no, by_no[no]) for no in bad if no in by_no]
    if not items:
        return parsed, f"{info} | {summary}"
    print(
        f"👔 continuity: {len(w_miss)} wardrobe miss(es), {len(l_miss)} location miss(es) - "
        f"asking for one repair pass on scene(s) {', '.join(f'{no:02d}' for no, _ in items)}"
    )
    for scene_no, shot_no, name, anchor in w_miss[:10]:
        print(f"   ↳ scene {scene_no:02d} [Shot {shot_no}] {name}: missing `{anchor}`")
    for scene_no, name, anchor in l_miss[:10]:
        print(f"   ↳ scene {scene_no:02d} {name}: missing `{anchor}`")
    try:
        text, repair_info = repair(
            subset_repair_prompt(w_locks, w_miss, l_locks, l_miss, items)
        )
        _, fixed = parse_scenes(text, items[0][1])
    except Exception as exc:  # keep the first answer rather than fail the run
        print(f"⚠️ continuity repair failed: {exc}")
        return parsed, f"{info} | {summary}, repair failed"
    wanted = {no for no, _ in items}
    fixed_by_no = {no: p for no, _, p in fixed}
    # only accept an answer that returned exactly (a subset of) the scenes asked
    # for as real envelopes; the whole-text fallback of parse_scenes must not
    # silently replace scene 1
    if not fixed_by_no or not set(fixed_by_no) <= wanted or any(
        "[shot" not in p.lower() for p in fixed_by_no.values()
    ):
        print(f"⚠️ continuity repair returned scene(s) {sorted(fixed_by_no)}; expected {sorted(wanted)} - discarded")
        return parsed, f"{info} | {summary}, repair discarded"
    out = [
        (no, dur, fixed_by_no.get(no, p) if no in wanted else p)
        for no, dur, p in parsed
    ]
    new_scenes = [p for _, _, p in out]
    w_left = wardrobe_violations(new_scenes, w_locks)
    l_left = location_violations(new_scenes, l_locks)
    print(
        f"👔 after repair: wardrobe {len(w_miss)} -> {len(w_left)}, "
        f"locations {len(l_miss)} -> {len(l_left)} miss(es) "
        f"({len(fixed_by_no)} scene(s) re-emitted) | {repair_info}"
    )
    return out, (
        f"{info} | wardrobe: repaired {len(w_miss)} -> {len(w_left)} | "
        f"locations: repaired {len(l_miss)} -> {len(l_left)} "
        f"({len(fixed_by_no)} scene(s) re-emitted)"
    )


# ---- hand-off between chained scenes -----------------------------------
# In a continuous chain the last frame of scene N IS the first frame of scene
# N+1 - the render pins it there. The wardrobe and location locks keep the
# people and the place the same; what they do not catch is the STATE of the
# frame at the join: the glass that was at her lips, the door that had just
# opened, the lamp that was on, who stood where. That is what an LLM pass can
# tidy: read every ending against the opening that follows it and rewrite the
# openings that drift, nothing else.

def _description_of(prompt):
    m = re.search(r"integrated_multimodal_description\s*:\s*(.*?)(?:\n\s*(?:overall_soundscape|non_diegetic_music|subject_definitions|summary)\s*:|\Z)",
                  prompt or "", re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else (prompt or "")).strip()


def scene_ending(prompt, chars=420):
    """The closing sentences of a scene's description - the state the next scene inherits."""
    text = " ".join(_description_of(prompt).split())
    return text[-chars:].lstrip() if len(text) > chars else text


def scene_opening(prompt, chars=420):
    """The first sentences of a scene - [Shot 1] up to `chars`."""
    text = " ".join(_description_of(prompt).split())
    return text[:chars].rstrip()


def opens_on_previous(prompt):
    return bool(re.search(r"last frame of the previous scene", prompt or "", re.IGNORECASE))


def handoff_repair_prompt(parsed):
    """One repair turn: every join listed as ENDING -> OPENING, scenes to re-emit chosen by the model."""
    lines = [
        "HAND-OFF PASS. These scenes render as a CONTINUOUS CHAIN: the last frame of each "
        "scene is pinned as the first frame of the next. At every join below, compare the "
        "ENDING with the OPENING that follows it. The opening must show the SAME frame: the "
        "same people in the same screen positions and poses, the same objects and props in "
        "the same hands and places (a glass raised stays raised, an open door stays open, a "
        "lit lamp stays lit), the same light and the same framing. Where the opening drifts - "
        "a prop that vanished or changed, a person who moved or changed pose, a re-established "
        "wide shot, a different light - rewrite THAT scene's [Shot 1] so its first sentence "
        "is `Opens on the last frame of the previous scene: ...` naming what is actually on "
        "screen at the end of the previous scene, and then carry the story on. Change nothing "
        "else in the scene. Never rewrite an ending to fit an opening.",
        "",
    ]
    for k in range(1, len(parsed)):
        prev_no, _, prev_p = parsed[k - 1]
        no, _, p = parsed[k]
        flag = "" if opens_on_previous(p) else "   <-- does not open on the previous frame"
        lines.append(f"JOIN {prev_no:02d} -> {no:02d}{flag}")
        lines.append(f"  ENDING of {prev_no:02d}: {scene_ending(prev_p)}")
        lines.append(f"  OPENING of {no:02d}: {scene_opening(p)}")
        lines.append("")
    lines.append(
        "Re-emit ONLY the scenes whose opening you changed, each as a complete envelope with "
        "its original number and duration (`=== SCENE NN | duration: X.X ===` ... `=== END "
        "SCENE NN ===`), every field present, nothing else changed. If no opening needs a "
        "change, answer with the single word OK."
    )
    return "\n".join(lines)


def enforce_handoff(enabled, parsed, session_id, info, repair):
    """
    Hand-off pass for chained films: one repair turn that rewrites the scene
    openings that do not match the previous scene's ending. Only the scenes
    the model re-emits are spliced in (by number, subset only, real envelopes
    only); the rest stay as written. Returns (parsed, info).
    """
    if not enabled or len(parsed) < 2:
        return parsed, info
    drift = [no for k, (no, _, p) in enumerate(parsed) if k > 0 and not opens_on_previous(p)]
    note = f"hand-off: {len(drift)} scene(s) do not open on the previous frame" if drift else "hand-off: every scene opens on the previous frame"
    if not session_id:
        print(f"🔗 {note} - no session to repair in")
        return parsed, f"{info} | {note}"
    print(f"🔗 {note} - asking for one hand-off pass on {len(parsed) - 1} join(s)")
    try:
        text, repair_info = repair(handoff_repair_prompt(parsed))
    except Exception as exc:  # keep the first answer rather than fail the run
        print(f"⚠️ hand-off pass failed: {exc}")
        return parsed, f"{info} | {note}, hand-off pass failed"
    if (text or "").strip().upper().rstrip(".") == "OK":
        print("🔗 hand-off pass: nothing to change")
        return parsed, f"{info} | {note}, hand-off pass: unchanged"
    _, fixed = parse_scenes(text, parsed[0][1])
    allowed = {no for no, _, _ in parsed[1:]}
    fixed_by_no = {no: p for no, _, p in fixed}
    if not fixed_by_no or not set(fixed_by_no) <= allowed or any("[shot" not in p.lower() for p in fixed_by_no.values()):
        print(f"⚠️ hand-off pass returned scene(s) {sorted(fixed_by_no)}; expected a subset of {sorted(allowed)} - discarded")
        return parsed, f"{info} | {note}, hand-off pass discarded"
    out = [(no, dur, fixed_by_no.get(no, p)) for no, dur, p in parsed]
    left = [no for k, (no, _, p) in enumerate(out) if k > 0 and not opens_on_previous(p)]
    print(f"🔗 hand-off pass: {len(fixed_by_no)} scene opening(s) rewritten "
          f"({', '.join(f'{no:02d}' for no in sorted(fixed_by_no))}); "
          f"{len(drift)} -> {len(left)} not opening on the previous frame | {repair_info}")
    return out, f"{info} | hand-off: {len(fixed_by_no)} opening(s) rewritten, {len(drift)} -> {len(left)} drifting"


# ---- interpretation: how far the pictures may stray from the source ------
# The source (a manuscript, an idea, a direction, the material) is always
# honoured for what it states outright; this decides what the PICTURE does
# with it. Auto is the writer's own judgement; the rest are named readings a
# director picks on purpose. "Surprise me" picks a non-literal reading by
# seed, so re-running the same source gives a different film.
INTERPRETATIONS = [
    "Auto (the writer decides)",
    "Literal (the pictures show what the source says)",
    "Loose (the source is a starting point - stage the feeling, not the nouns)",
    "Metaphor (one image system stands in for the source)",
    "Counterpoint (the picture tells a different story that rhymes with the source)",
    "Reframe (the source retold in another world, era or genre)",
    "Surreal (dream logic - the source as images that should not be possible)",
    "Surprise me (a different non-literal reading every seed)",
]
_SURPRISE_POOL = ("Loose", "Metaphor", "Counterpoint", "Reframe", "Surreal")

_INTERPRETATION_TEXT = {
    "Literal": (
        "INTERPRETATION - LITERAL: the pictures show what {source} says. When it names a "
        "thing, a place or an action, that thing is on screen; the story, its people and its "
        "events are adapted faithfully. Imagination goes into HOW it is shown - framing, "
        "light, the detail chosen - never into replacing it."
    ),
    "Loose": (
        "INTERPRETATION - LOOSE: {source} is a starting point, not a shot list. Keep its "
        "people and what happens between them; stage the feeling and the situation behind "
        "each beat rather than its literal props and places - a scene written in a car may be "
        "two people not leaving a room, a storm may be a kitchen at 3 a.m. with nobody "
        "speaking. Never illustrate a line word-for-word; the audience gets the story, not "
        "a diagram of it."
    ),
    "Metaphor": (
        "INTERPRETATION - METAPHOR: choose ONE image system - a single concrete world of "
        "objects, materials and actions that is NOT what {source} literally describes (a house "
        "being dismantled room by room, a swimmer who never reaches the wall, a market closing "
        "for the night) - state it in the synopsis as the Concept, and let every scene be a "
        "stage of that one metaphor. Each beat of {source} maps onto a development of the "
        "image system, never onto its own literal content. The metaphor is never explained on "
        "screen; it is simply what we watch."
    ),
    "Counterpoint": (
        "INTERPRETATION - COUNTERPOINT: the picture tells a DIFFERENT story from {source} - "
        "different people, place and events - that rhymes with it emotionally and "
        "structurally: the same setup, the same turn, the same kind of ending, in another "
        "life. The gap between what {source} says and what is seen is the point; the audience "
        "closes it. Its dialogue, if any is written, may be kept as voice-over or given to "
        "the new people - but no literal imagery from {source} at all."
    ),
    "Reframe": (
        "INTERPRETATION - REFRAME: keep what {source} is ABOUT - the relationship, the loss, "
        "the boast, the escape, the argument - but retell it inside a world it never mentions: "
        "another era, another genre (heist, western, nature documentary, fairy tale, sports "
        "final, space station, kitchen brigade), another scale (insects, gods, a city as a "
        "body). Pick ONE such world in the synopsis and commit to it in every scene: its props, "
        "its costumes, its light. Nouns from {source} are translated into that world, never "
        "shown as themselves; written dialogue survives, adapted only where the new world "
        "makes a word impossible."
    ),
    "Surreal": (
        "INTERPRETATION - SURREAL: dream logic. Take {source} as images that should not be "
        "possible and show them as plainly as the weather: rooms that fill with the sea, a "
        "character who is also the audience, a road that loops back into the house, gravity "
        "that gives up on one object. Every scene contains at least one impossible thing "
        "treated as normal by everyone in it, and the impossibilities build on each other "
        "across the film instead of resetting - by the last scene the world has become "
        "something else entirely. Physical, literal descriptions of the impossible; never the "
        "words `surreal`, `dreamlike` or `as if`."
    ),
}


def resolve_interpretation(choice, seed=0):
    """The mode's key word ('Literal', ...), None for Auto; 'Surprise me' picks by seed."""
    key = str(choice or "").split(" ", 1)[0].strip()
    if not key or key.startswith("Auto"):
        return None
    if key.startswith("Surprise"):
        import random
        return random.Random(f"interpretation:{seed}").choice(_SURPRISE_POOL)
    return key if key in _INTERPRETATION_TEXT else None


def interpretation_directive(key, source="the manuscript", keep=""):
    """
    The directive for one reading. `source` names what is being interpreted
    ("the manuscript", "the idea", "the direction", "the material"); `keep`
    is an extra sentence for what must survive untouched (a presentation's
    facts, for instance).
    """
    text = _INTERPRETATION_TEXT.get(key or "")
    if not text:
        return ""
    text = text.format(source=source)
    tail = (
        " The user's explicit instructions still win on anything they state outright; this "
        "reading fills everything they leave open. Name the reading's central idea in the "
        "synopsis Concept so every chunk writer stages the same one."
    )
    return text + (" " + keep.strip() if keep else "") + tail


def interpretation_input(source_word="the manuscript"):
    """The widget spec, appended last in a writer's optional inputs."""
    return (INTERPRETATIONS, {
        "default": INTERPRETATIONS[0],
        "tooltip": (
            f"How far the pictures may stray from {source_word}. Auto leaves it to the writer. "
            "Literal adapts it faithfully. Loose keeps the people and the events but stages the "
            "feeling, never the literal props. Metaphor builds the whole film on one image system. "
            "Counterpoint tells a different story that rhymes with it. Reframe keeps the subject but "
            "moves it into another world, era or genre. Surreal is escalating dream logic. Surprise "
            "me picks one of the non-literal readings by seed, so every re-run is a different film."
        ),
    })


# ---- transitions: what a CONTINUING scene may do with its place ----------
# The chain carry hands H3 the previous scene's last frames - the actor's
# identity, pose, direction and speed continue exactly - but not the previous
# PLACE. So a change of place inside a continuing take has to be written:
# the actor carries the take through a door, or the camera moves and a new
# space is disclosed around the same figure. Left unsaid, the writers keep
# continuing scenes in the room they started in.
TRANSITION_STYLES = [
    "Auto (the writer chooses per scene)",
    "Stay (a continuing scene keeps its place)",
    "Walk-through (the actor carries the take into a new place)",
    "Reveal (a camera move discloses a new space around them)",
    "New place (every continuing scene moves somewhere new - walk-through or reveal)",
]

_TRANSITION_TEXT = {
    "Stay": (
        "TRANSITIONS - STAY: a continuing scene keeps its place - the same room, the same "
        "light, the same anchors; only what the story changes changes. A new place needs a "
        "hard cut, so it waits for a scene that opens on one."
    ),
    "Walk-through": (
        "TRANSITIONS - WALK-THROUGH: a continuing scene may CHANGE PLACE inside the take, and "
        "the actor carries it there: through a door, down a corridor, round a corner, up a "
        "stair, out of a room's lamplight into a street. The camera follows without cutting. "
        "The opening is still the previous scene's last frame - the same pose, direction and "
        "speed, nothing about the actor jumps - and the new place is established BEHIND and "
        "AROUND the moving actor within the first seconds: name its anchors as they come into "
        "frame (`the swing door gives onto the harbour boards, rain on the planks, the green "
        "lamp of the pilot boat`), lock the new place's light in the same sentence, and then "
        "carry the story on there. Use it whenever the plan moves the story to a new place; a "
        "continuing scene whose plan stays put stays put."
    ),
    "Reveal": (
        "TRANSITIONS - REVEAL: a continuing scene may CHANGE PLACE inside the take by "
        "revealing it: the actor stays roughly where the previous frame left them while the "
        "camera moves - pulls back, arcs, cranes, tilts, racks - and the space around them is "
        "not what we thought: the kitchen wall is a set flat, the alley opens into a market, "
        "the close-up was on a train. The reveal completes within the first half of the scene, "
        "the new place's anchors and light are named as they are disclosed, and nothing about "
        "the actor jumps. Use it whenever the plan moves the story to a new place; a scene "
        "whose plan stays put stays put."
    ),
    "Auto": (
        "TRANSITIONS - YOUR CALL: a continuing scene may change place inside the take, either "
        "by WALK-THROUGH (the actor carries it through a door, a corridor, a corner, the camera "
        "following without cutting) or by REVEAL (the actor holds, the camera moves, and a new "
        "space is disclosed around them). Choose per scene whatever serves the plan, alternate "
        "rather than repeat, keep the opening on the previous scene's last frame with nothing "
        "about the actor jumping, and establish the new place's anchors and light as they come "
        "into frame. A scene whose plan stays put stays put."
    ),
    "New": (
        "TRANSITIONS - NEW PLACE EVERY SCENE: every scene is its own place - no two consecutive "
        "scenes share a location, continuing ones included. The plan gives each scene a new, "
        "specific place that serves the story, and a continuing scene gets there INSIDE the take: "
        "by WALK-THROUGH (the actor carries it through a door, a corridor, a corner, out of one "
        "light into another, the camera following without cutting) or by REVEAL (the actor holds, "
        "the camera moves, and a new space is disclosed around them), alternating rather than "
        "repeating. The opening is still the previous scene's last frame - the same pose, "
        "direction and speed, nothing about the actor jumps - and the new place's anchors and "
        "light are named as they come into frame, in the first seconds. Returning looks (a "
        "chorus signature) may come back, but never in the scene right after they were left."
    ),
}


def resolve_transition(choice):
    key = str(choice or "").split(" ", 1)[0].strip()
    return key if key in _TRANSITION_TEXT else None


def transition_directive(key):
    return _TRANSITION_TEXT.get(key or "", "")


def transition_input():
    return (TRANSITION_STYLES, {
        "default": TRANSITION_STYLES[3],          # Reveal
        "tooltip": (
            "What a CONTINUING scene may do with its place. The chain carry keeps the actor "
            "exactly - pose, direction, speed - but not the previous place, so a change of place "
            "inside a take has to be written. Auto: the writer picks walk-through or reveal per "
            "scene where the plan moves. Stay: same place, always. Walk-through: the actor carries "
            "the take through a door / corridor / corner into the new place, camera following. "
            "Reveal: the actor holds, the camera moves, and a new space is disclosed around them. "
            "New place: every scene gets its own place - continuing scenes move there inside the "
            "take (walk-through or reveal, alternating), so no two consecutive scenes share a "
            "location. Hard cuts are unaffected."
        ),
    })


def duration_directive(duration_mode, scene_duration):
    if duration_mode == DURATION_MODES[0]:
        return (
            f"Every scene is exactly {scene_duration:.1f} seconds; write "
            f"`duration: {scene_duration:.1f}` in every envelope and spread the cut "
            "timecodes across that length. Fill the whole duration with action (and dialogue "
            "where allowed); no dead air at the end."
        )
    return (
        "Choose each scene's duration yourself between 5.0 and 15.0 seconds (prefer 12-15 "
        "for dialogue or a developing action, 5-8 for a transition or punch) and write it as "
        "`duration: S.S` in that scene's envelope. Fill the whole duration."
    )


# ---- structured transport ---------------------------------------------------
# The scene text stays what it is (the labelled fields the video node reads);
# only the envelope changes: a JSON object instead of === SCENE === markers.
# On Ollama the schema is enforced by the sampler, so the structure cannot
# break; elsewhere it is a request the parser checks, falling back to the
# text envelopes when a model ignores it.
SCENES_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "synopsis": {"type": "string"},
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene": {"type": "integer"},
                    "duration": {"type": "number"},
                    "prompt": {"type": "string"},
                },
                "required": ["scene", "prompt"],
            },
        },
    },
    "required": ["scenes"],
}


def scenes_json_instruction():
    return (
        "\n\nOUTPUT FORMAT - JSON, not envelopes: return ONLY one JSON object, nothing before "
        "or after it, shaped {\"synopsis\": \"<the synopsis block's text when one is asked for, "
        "otherwise an empty string>\", \"scenes\": [{\"scene\": <number>, \"duration\": "
        "<seconds>, \"prompt\": \"<that scene's complete text: the labelled fields exactly as "
        "specified, newlines included>\"}]}. One array item per requested scene, in order. Do "
        "not add === SCENE === or === END SCENE === lines inside the prompt strings - the JSON "
        "is the envelope."
    )


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_envelope_lines(prompt):
    """A model that writes JSON *and* envelopes: drop the marker lines, keep the fields."""
    kept = [
        line for line in prompt.splitlines()
        if not re.match(r"^\s*(?:\*\*)?===\s*(?:END\s+)?(?:SCENE|SYNOPSIS)\b.*===", line, re.IGNORECASE)
    ]
    return "\n".join(kept).strip()


_SCENE_ITEM_RE = re.compile(r'\{\s*"scene"\s*:\s*(\d+)')
_PROMPT_KEY_RE = re.compile(r'"prompt"\s*:\s*')
_DURATION_KEY_RE = re.compile(r'"duration"\s*:\s*([0-9.]+)')


def _salvage_truncated_scenes(raw):
    """
    A JSON scene envelope that stops mid-string (the model ran out of tokens)
    is not JSON - but every scene whose prompt string closed is intact. Pull
    those out so the caller gets the real scenes it did receive and can say
    how many are missing, instead of one bogus scene made of raw JSON.
    """
    if '"scenes"' not in raw or not raw.lstrip().startswith(("{", "[", "`")):
        return None
    decoder = json.JSONDecoder()
    items = []
    for m in _SCENE_ITEM_RE.finditer(raw):
        pk = _PROMPT_KEY_RE.search(raw, m.end())
        if not pk or raw[pk.end():pk.end() + 1] != '"':
            continue
        try:
            prompt, _end = decoder.raw_decode(raw, pk.end())
        except ValueError:
            continue                    # this is the cut-off one
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        item = {"scene": int(m.group(1)), "prompt": prompt}
        dk = _DURATION_KEY_RE.search(raw, m.end(), pk.start())
        if dk:
            item["duration"] = float(dk.group(1))
        items.append(item)
    if not items:
        return None
    print(f"\u26a0\ufe0f H3: the JSON reply was cut off - salvaged {len(items)} complete scene(s) from it.")
    return {"scenes": items}


def parse_scenes_json(text, fallback_duration):
    """(synopsis, [(number, duration, prompt), ...]) from a JSON reply, or None if it is not one."""
    raw = (text or "").strip()
    if not raw:
        return None
    candidates = [m.group(1) for m in _JSON_FENCE_RE.finditer(raw)] + [raw]
    a, b = raw.find("{"), raw.rfind("}")
    if a >= 0 and b > a:
        candidates.append(raw[a:b + 1])
    a, b = raw.find("["), raw.rfind("]")
    if a >= 0 and b > a:
        candidates.append(raw[a:b + 1])
    data = None
    for cand in candidates:
        try:
            data = json.loads(cand)
            break
        except ValueError:
            continue
    if data is None:
        data = _salvage_truncated_scenes(raw)
        if data is None:
            return None
    synopsis = ""
    items = data
    if isinstance(data, dict):
        synopsis = _strip_envelope_lines(str(data.get("synopsis") or ""))
        items = data.get("scenes")
    if not isinstance(items, list):
        return None
    scenes = []
    for k, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt") or item.get("text") or item.get("body") or ""
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        number = item.get("scene", item.get("number", item.get("no", k)))
        try:
            number = int(number)
        except (TypeError, ValueError):
            number = k
        duration = item.get("duration")
        try:
            duration = float(duration) if duration is not None else float(fallback_duration)
        except (TypeError, ValueError):
            duration = float(fallback_duration)
        prompt = _strip_envelope_lines(strip_code_fence(prompt))
        prompt = re.sub(r"\*\*(\w+):\*\*", r"\1:", prompt)
        scenes.append((number, duration, prompt))
    if not scenes:
        return None
    scenes.sort(key=lambda s: s[0])
    return synopsis, scenes


def parse_scenes(text, fallback_duration):
    """
    Split Claude's answer into (synopsis, [(number, duration, prompt), ...]).

    Falls back to treating the whole text as one scene if no envelopes are
    found, so a slightly off-format answer still yields something usable.
    """
    structured = parse_scenes_json(text, fallback_duration)
    if structured is not None:
        return structured

    synopsis_match = _SYNOPSIS_RE.search(text)
    synopsis = synopsis_match.group(1).strip() if synopsis_match else ""

    scenes = []
    heads = list(_SCENE_HEAD_RE.finditer(text))
    for k, match in enumerate(heads):
        number = int(match.group(1))
        duration = float(match.group(2)) if match.group(2) else float(fallback_duration)
        body_end = heads[k + 1].start() if k + 1 < len(heads) else len(text)
        body = text[match.end():body_end]
        # stop at an END marker or a synopsis block inside the span, if any
        stop = _SCENE_END_RE.search(body)
        if stop:
            body = body[: stop.start()]
        syn = _SYNOPSIS_HEAD_RE.search(body)
        if syn:
            body = body[: syn.start()]
        prompt = strip_code_fence(body.strip())
        prompt = re.sub(r"\*\*(\w+):\*\*", r"\1:", prompt)  # de-bold stray headers
        if prompt:
            scenes.append((number, duration, prompt))

    if not scenes:
        body = text
        if synopsis_match:
            body = text[: synopsis_match.start()] + text[synopsis_match.end():]
        body = strip_code_fence(body)
        if body.strip():
            scenes.append((1, float(fallback_duration), body.strip()))

    scenes.sort(key=lambda s: s[0])
    return synopsis, scenes


def scenes_to_text(synopsis, parsed):
    """Re-serialise parsed scenes into one readable, re-parseable string."""
    text = "\n\n".join(
        f"=== SCENE {n:02d} | duration: {d:.1f} ===\n{p}\n=== END SCENE {n:02d} ==="
        for n, d, p in parsed
    )
    if synopsis:
        text = f"=== SYNOPSIS ===\n{synopsis}\n=== END SYNOPSIS ===\n\n{text}"
    return text


# ----------------------------------------------------------------------
# Continuity: independent clips vs a continuous C2V chain
# ----------------------------------------------------------------------

CONTINUITY_MODES = [
    "Independent clips (hard cuts, T2V openers)",
    "Continuous chain (each scene continues the last frames)",
]


def is_chain_mode(continuity_mode):
    return continuity_mode == CONTINUITY_MODES[1]


def chain_system_block(continuity_mode):
    """Chain rules for the system prompt; "" when writing independent clips."""
    if not is_chain_mode(continuity_mode):
        return ""
    from .common import load_guide

    return (
        "\n\n=== BEGIN MINIMAX-H3 CONTINUOUS CHAIN RULES ===\n"
        f"{load_guide('guide_chain_en.md')}\n"
        "=== END MINIMAX-H3 CONTINUOUS CHAIN RULES ===\n\n"
        "These scenes will be rendered as a continuous chain: the last frames and audio of "
        "each scene are pinned to the head of the next. The chain rules override the "
        "independent-clip habits (hard cuts, re-establishing shots, 'already speaking' "
        "openers on a new speaker) for every scene after the first."
    )


def continuity_directive(continuity_mode):
    """One user-prompt directive line describing the continuity contract."""
    if not is_chain_mode(continuity_mode):
        return (
            "Continuity mode: independent clips. Each scene is a self-contained T2V clip "
            "with its own establishing [Shot 1] and hard cuts between shots; adjacent scenes "
            "hand off through story (a look, a prop, a question), not through pixels."
        )
    return (
        "Continuity mode: CONTINUOUS CHAIN. Scene 01 is a normal opener. From scene 02 on, "
        "[Shot 1] opens on the last frame of the previous scene (name who is where and the "
        "framing), the whole scene is one continuous take with no `the shot cuts to`, the "
        "closer is a moving shot of a different class than the scene's opening framing, and "
        "when the first speaker is not the person on screen at the previous ending, use the "
        "three-part hand-off (silent inherited beat to 00:02.000, named transition without "
        "cutting, first <d> at 00:02.000 or later, outgoing person declared as a subject). "
        "Keep the same lighting string, screen positions and wardrobe anchors in every scene."
    )
