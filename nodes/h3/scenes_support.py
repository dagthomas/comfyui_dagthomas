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
    "Wardrobe lock, one line per character, e.g. `Sheldon: brown corduroy jacket over a "
    "green Flash T-shirt, khakis`. Used word-for-word in every scene. Empty = Claude fixes "
    "one outfit per character itself (in the synopsis) and repeats it in every scene."
)


def wardrobe_directive(wardrobe=""):
    """
    Wardrobe drifts between scenes unless each outfit is fixed once and copied.
    With user text: that text is the lock. Without: Claude writes the lock into
    the synopsis first, then reuses it.
    """
    wardrobe = (wardrobe or "").strip()
    if wardrobe:
        lines = [line.strip() for line in wardrobe.splitlines() if line.strip()]
        return (
            "Wardrobe lock (from the user) - copy these anchors word-for-word into the "
            "synopsis `Wardrobe:` lines, into every scene's subject_definitions and into "
            "the first shot each character appears in; never add, remove or recolour a "
            "garment between scenes:\n" + "\n".join(f"    {line}" for line in lines)
        )
    return (
        "Wardrobe lock: before scene 01, fix ONE outfit per character - 2-4 concrete "
        "anchors (garment, colour, material or pattern, one distinctive item) - write "
        "them as the synopsis `Wardrobe:` lines, then reuse that exact wording in every "
        "scene's subject_definitions and in the first shot each character appears in. "
        "Clothes, hair and props never change between scenes unless the brief explicitly "
        "calls for a costume change, and then the change is stated on screen."
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
