# Shared pieces for the H3 nodes that write several scenes in one call.
#
# Both multi-scene writers (the crossover writer and the plain scenes writer)
# ask Claude Code for the same envelope layout, split it the same way, and hand
# the result out as a ComfyUI list. This keeps the contract in one place.

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
_SCENE_RE = re.compile(
    r"===\s*SCENE\s+(\d+)\s*(?:\|\s*duration:\s*([0-9.]+)\s*(?:s|sec|seconds)?\s*)?===\s*"
    r"(.*?)\s*===\s*END SCENE\s*\1\s*===",
    re.DOTALL | re.IGNORECASE,
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
    "Wardrobe lock, one line per character, e.g. `Sheldon: brown corduroy jacket, green "
    "Flash T-shirt, khaki trousers, small silver ring in the left nostril`. Used word-for-word "
    "in every shot. Empty = Claude fixes one outfit per character itself (in the synopsis) "
    "and repeats it in every shot."
)

ENFORCE_WARDROBE_TOOLTIP = (
    "After writing, check that every shot a character is in restates all of that "
    "character's wardrobe anchors verbatim, and that every scene set in a locked location "
    "restates that location's anchors. If anything is dropped or changed, the model gets "
    "one repair turn in the same session. Off = trust the first answer."
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
            "living room - <anchors> - ...`); later shots in the same scene restate the "
            "anchors that are in frame. " + _LOCATION_ANCHOR_RULES +
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
    Wardrobe drifts between scenes unless each outfit is fixed once and copied
    into EVERY shot. With user text: that text is the lock. Without: Claude
    writes the lock into the synopsis first, then reuses it.
    """
    wardrobe = (wardrobe or "").strip()
    if wardrobe:
        lines = [line.strip() for line in wardrobe.splitlines() if line.strip()]
        return (
            "Wardrobe lock (from the user) - copy these anchors word-for-word as the "
            "synopsis `Wardrobe:` lines and into EVERY shot in which that character is on "
            "screen, right after the first mention of the character in that shot "
            "(`... <Subject 1> Sheldon (S1), wearing <anchors>, ...`). " + _ANCHOR_RULES +
            "\n" + "\n".join(f"    {line}" for line in lines)
        )
    return (
        "Wardrobe lock: before scene 01, DESIGN one outfit per character the way a "
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
        "Then copy that character's anchors word-for-word into EVERY shot in which the "
        "character is on screen, right after the first mention of the character in that "
        "shot (`... <Subject 1> Sheldon Cooper (S1), wearing <anchors>, ...`). "
        + _ANCHOR_RULES
    )


# ----------------------------------------------------------------------
# Wardrobe verification
# ----------------------------------------------------------------------

_SYNOPSIS_HEADERS = ("title:", "logline:", "cast:", "wardrobe:", "locations:", "location:")
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
    [(scene_no, shot_no, name, anchor), ...] for every shot in which a locked
    character is on screen but an anchor is missing verbatim (case-insensitive).
    Sentences that say the character is NOT in frame do not count as presence.
    """
    found = []
    if not locks:
        return found
    patterns = {name: _name_patterns(name) for name in locks}
    for scene_no, prompt in enumerate(scenes, 1):
        parts = _SHOT_SPLIT_RE.split(prompt)
        # parts: [pre, "[Shot 1]", body1, "[Shot 2]", body2, ...]
        shot_no = 0
        for k in range(1, len(parts), 2):
            shot_no += 1
            body = parts[k + 1] if k + 1 < len(parts) else ""
            visible = _OFFSCREEN_RE.sub(" ", body)
            low = body.lower()
            for name, anchors in locks.items():
                if not any(p.search(visible) for p in patterns[name]):
                    continue
                for anchor in anchors:
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
        "These shots have the character on screen without restating every anchor "
        "verbatim:\n"
        f"{issue_lines}\n"
        "Fix them: in every shot in which a character is on screen, restate ALL of that "
        "character's anchors character-for-character (same words, same colours, same "
        "side), right after the first mention of the character in that shot. Do not "
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
            "These shots have the character on screen without restating every anchor "
            "verbatim:\n"
            f"{issue_lines}\n"
            "Fix them: in every shot in which a character is on screen, restate ALL of that "
            "character's anchors character-for-character (same words, same colours, same "
            "side), right after the first mention of the character in that shot."
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
                       session_id, info, repair):
    """
    Shared post-check for the multi-scene writers: verify the wardrobe lock per
    shot and the location lock per scene; if anything is missing and a session
    is open, ask for ONE repair turn via `repair(prompt) -> (text, repair_info)`
    and keep the repaired answer only if it still parses to the same scene count.
    Returns (synopsis, parsed, info).
    """
    w_locks = parse_wardrobe_lock(synopsis)
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


def enforce_continuity_chunked(enabled, synopsis, parsed, session_id, info, repair):
    """
    Post-check for runs written in chunks (music videos: 13-20 scenes), where a
    full re-emit is too long to ask for: verify every scene against the synopsis
    locks, then use ONE repair turn to re-emit only the violating scenes and
    splice them back in by number. Returns (parsed, info).
    """
    w_locks = parse_wardrobe_lock(synopsis)
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


def parse_scenes(text, fallback_duration):
    """
    Split Claude's answer into (synopsis, [(number, duration, prompt), ...]).

    Falls back to treating the whole text as one scene if no envelopes are
    found, so a slightly off-format answer still yields something usable.
    """
    synopsis_match = _SYNOPSIS_RE.search(text)
    synopsis = synopsis_match.group(1).strip() if synopsis_match else ""

    scenes = []
    for match in _SCENE_RE.finditer(text):
        number = int(match.group(1))
        duration = float(match.group(2)) if match.group(2) else float(fallback_duration)
        prompt = strip_code_fence(match.group(3))
        prompt = re.sub(r"\*\*(\w+):\*\*", r"\1:", prompt)  # de-bold stray headers
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
