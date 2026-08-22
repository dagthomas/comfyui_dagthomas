# APNext H3 Characters
#
# Picks a character from data/h3/characters.tsv (Relative File Path,
# Character / Subject Name, Real Actor / Actress, Franchise / Show) and exposes
# the character name, actor and franchise as separate STRING outputs, plus the
# reference clip path. Supports a fixed dropdown pick or a seeded random pick,
# optionally filtered by franchise.

import csv
import os
import random

from ...utils.constants import CUSTOM_CATEGORY
from .common import scale_reference_image

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "h3",
    "characters.tsv",
)

_RANDOM = "🎲 random"
_CUSTOM = "✏️ custom (type in custom_character)"
_ALL_FRANCHISES = "(all)"
WARDROBE_SEP = " | wardrobe: "


def custom_cast_line(text):
    """A hand-written character as a single cast line (whitespace collapsed)."""
    return " ".join((text or "").split())


def cast_line_with_wardrobe(line, wardrobe):
    """`Name (played by Actor) from Show | wardrobe: a, b, c` (one line)."""
    anchors = ""
    if wardrobe and wardrobe.strip():
        parts = [a.strip(" .;") for a in ", ".join(wardrobe.splitlines()).split(",")]
        anchors = ", ".join(a for a in parts if a)
    return f"{line}{WARDROBE_SEP}{anchors}" if anchors else line


def split_cast_line(line):
    """(cast_line_without_wardrobe, wardrobe_anchors_or_"")."""
    if WARDROBE_SEP.strip() in line:
        head, _, tail = line.partition(WARDROBE_SEP.strip())
        return head.rstrip(" |"), tail.strip()
    return line, ""


def cast_line_name(line):
    """The character name a cast line refers to (for wardrobe locks / {characterN})."""
    head, _ = split_cast_line(line)
    head = head.strip()
    if " (played by " in head:
        return head.split(" (played by ", 1)[0].strip()
    if " from " in head and not head.lower().startswith("from "):
        return head.split(" from ", 1)[0].strip()
    if ":" in head:
        return head.split(":", 1)[0].strip()
    return head


def _load_rows():
    """Read the TSV and return de-duplicated rows as dicts, in file order."""
    rows = []
    seen = set()
    if not os.path.isfile(_DATA_PATH):
        return rows
    with open(_DATA_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for raw in reader:
            if len(raw) < 4:
                continue
            path, character, actor, franchise = (c.strip() for c in raw[:4])
            if not character:
                continue
            key = (character, actor, franchise)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "path": path,
                    "character": character,
                    "actor": actor,
                    "franchise": franchise,
                    "label": _make_label(character, actor, franchise),
                }
            )
    return rows


def _make_label(character, actor, franchise):
    short_franchise = franchise.split(",")[0].strip()
    if actor and actor != character:
        return f"{character} — {actor} ({short_franchise})"
    return f"{character} ({short_franchise})"


def cast_line_for(row):
    """`Character (played by Actor) from Show` - the subject_definitions form."""
    show = row["franchise"].split(",")[0].strip()
    actor = row["actor"]
    if actor and actor != row["character"]:
        return f"{row['character']} (played by {actor}) from {show}"
    return f"{row['character']} from {show}"


_ROWS = _load_rows()
_BY_LABEL = {r["label"]: r for r in _ROWS}
_FRANCHISES = sorted({r["franchise"].split(",")[0].strip() for r in _ROWS}, key=str.lower)


class H3Characters:
    @classmethod
    def INPUT_TYPES(cls):
        labels = [_RANDOM, _CUSTOM] + [r["label"] for r in _ROWS]
        return {
            "required": {
                "character": (labels, {
                    "default": labels[2] if len(labels) > 2 else _RANDOM,
                    "tooltip": "Pick a character, 🎲 random (seeded), or ✏️ custom to describe your own in custom_character.",
                }),
                "franchise_filter": (
                    [_ALL_FRANCHISES] + _FRANCHISES,
                    {
                        "default": _ALL_FRANCHISES,
                        "tooltip": (
                            "Only used when character is random: limits the draw to one show. "
                            "Picking a named character snaps this to their show, so switching "
                            "to random afterwards draws a castmate."
                        ),
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "custom_character": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Your own character, used when character = ✏️ custom (or whenever this is "
                        "filled in). `Lena: a middle-aged woman with a limp and a silver bob` keeps "
                        "`Lena` as the name the writers and {characterN} use; `Name (played by "
                        "Actor) from Show` also works."
                    ),
                }),
                "wardrobe": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "This character's wardrobe lock: 3-5 exact anchors, comma separated "
                        "(`dark-brown corduroy jacket, forest-green cotton T-shirt, khaki chinos, "
                        "small silver ring in the LEFT nostril`). Travels with the cast line into "
                        "the Crossover / Music Video writers, which copy it word-for-word into "
                        "every shot. Empty = the writer invents one."
                    ),
                }),
                "cast_in": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Cast lines from an upstream H3 Characters node. This node's cast line "
                        "is appended, so several can be chained into one cast list."
                    ),
                }),
                "image": ("IMAGE", {
                    "tooltip": (
                        "This character's reference picture (their face photo). Passed through "
                        "unchanged on the `image` output, so the character and their picture "
                        "travel together: wire `cast` to a writer's cast_N and `image` to the "
                        "matching image_N."
                    ),
                }),
            },
        }

    # `image` is appended LAST so saved workflows keep their output link indices
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("character", "actor", "franchise", "file_path", "cast", "wardrobe", "image")
    FUNCTION = "pick"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Character / actor / franchise lookup for MiniMax-H3 prompts. Pick a "
        "character from the list or choose random (seeded, optionally filtered "
        "by franchise). Outputs the character name, the real actor/actress, the "
        "franchise/show, the reference clip path, and a ready-made cast line "
        "(`Character (played by Actor) from Show`) that chains through cast_in "
        "into the H3 Crossover Writer."
    )

    def pick(self, character, franchise_filter, seed, custom_character="", wardrobe="", cast_in="", image=None):
        # normalised here already (~0.8 MP, /32), so the picture is at the video
        # model's working size even when wired straight to a video node; the
        # writers' own pass-through normalisation is a no-op on top of this
        image = scale_reference_image(image)
        custom = custom_cast_line(custom_character)
        wardrobe = (wardrobe or "").strip()
        if character == _CUSTOM and not custom:
            raise ValueError("H3 Characters: character is set to custom but custom_character is empty.")
        if custom and character != _RANDOM:
            # custom text wins over the dropdown (the dropdown still shows a name
            # when the user only filled the box)
            name = cast_line_name(custom)
            line = cast_line_with_wardrobe(custom, wardrobe)
            cast = (cast_in or "").strip()
            cast = f"{cast}\n{line}" if cast else line
            return (name, "", "", "", cast, wardrobe, image)
        if not _ROWS:
            return ("", "", "", "", cast_in or "", wardrobe, image)

        if character == _RANDOM:
            pool = _ROWS
            if franchise_filter != _ALL_FRANCHISES:
                pool = [r for r in _ROWS if r["franchise"].split(",")[0].strip() == franchise_filter] or _ROWS
            row = random.Random(seed).choice(pool)
        else:
            row = _BY_LABEL.get(character) or _ROWS[0]

        cast_line = cast_line_with_wardrobe(cast_line_for(row), wardrobe)
        cast = (cast_in or "").strip()
        cast = f"{cast}\n{cast_line}" if cast else cast_line
        return (row["character"], row["actor"], row["franchise"], row["path"], cast, wardrobe, image)
