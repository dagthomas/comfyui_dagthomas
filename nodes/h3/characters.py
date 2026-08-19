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

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "h3",
    "characters.tsv",
)

_RANDOM = "🎲 random"
_ALL_FRANCHISES = "(all)"


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
        labels = [_RANDOM] + [r["label"] for r in _ROWS]
        return {
            "required": {
                "character": (labels, {"default": labels[1] if len(labels) > 1 else _RANDOM}),
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
                "cast_in": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Cast lines from an upstream H3 Characters node. This node's cast line "
                        "is appended, so several can be chained into one cast list."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("character", "actor", "franchise", "file_path", "cast")
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

    def pick(self, character, franchise_filter, seed, cast_in=""):
        if not _ROWS:
            return ("", "", "", "", cast_in or "")

        if character == _RANDOM:
            pool = _ROWS
            if franchise_filter != _ALL_FRANCHISES:
                pool = [r for r in _ROWS if r["franchise"].split(",")[0].strip() == franchise_filter] or _ROWS
            row = random.Random(seed).choice(pool)
        else:
            row = _BY_LABEL.get(character) or _ROWS[0]

        cast_line = cast_line_for(row)
        cast = (cast_in or "").strip()
        cast = f"{cast}\n{cast_line}" if cast else cast_line
        return (row["character"], row["actor"], row["franchise"], row["path"], cast)
