# APNext H3 Scenes Review
#
# A gate between a writer and the render, so the scene text can be checked and
# edited BEFORE anything is committed to the (expensive) video render:
#
#   Review   - fills the node's editor with the incoming scenes and STOPS the
#              run right here (clean "interrupted", nothing renders). The mode
#              flips to Continue automatically in the editor.
#   Continue - renders the editor text: parses the === SCENE NN === envelopes
#              back into the scenes list. An empty editor passes the incoming
#              scenes through unchanged.
#   Bypass   - transparent pass-through (no stop, no edits).
#
# The editor (web/js/h3_scenes_review.js) shows the same colour-coded H3 tags
# as the Prompt Preview, can edit all scenes at once or one scene at a time,
# and has one-click Continue / Recreate buttons (Recreate bumps the writer's
# seed and reviews again).
#
# Works with list outputs (scenes from the multi-scene writers) and with a
# single h3_prompt string alike.

import hashlib
import re

from ...utils.constants import CUSTOM_CATEGORY

MODE_REVIEW = "Review (stop here and edit)"
MODE_CONTINUE = "Continue (render the editor text)"
MODE_BYPASS = "Bypass (pass scenes through)"
MODES = [MODE_REVIEW, MODE_CONTINUE, MODE_BYPASS]

_ENV_RE = re.compile(
    r"===\s*SCENE\s+(\d+)\s*===\s*(.*?)\s*===\s*END\s+SCENE\s*\1\s*===",
    re.DOTALL | re.IGNORECASE,
)

_HEADER = (
    "# Review the scenes, edit anything you like, then queue again - the mode\n"
    "# switched to Continue, so the next run renders exactly this text.\n"
    "# Keep the SCENE markers and the scene count (durations stay aligned).\n"
)

_FP_RE = re.compile(r"#\s*source-fingerprint:\s*([0-9a-f]+)", re.IGNORECASE)


def _fingerprint(scenes):
    return hashlib.sha1("\x1e".join(scenes).encode("utf-8", "replace")).hexdigest()[:12]


def serialize_scenes(scenes):
    parts = [
        _HEADER
        + f"# source-fingerprint: {_fingerprint(scenes)} (leave this line alone - it lets the\n"
        "# node detect when the writer's scenes changed and re-review automatically)\n"
    ]
    for i, s in enumerate(scenes, 1):
        parts.append(f"=== SCENE {i:02d} ===\n{(s or '').strip()}\n=== END SCENE {i:02d} ===\n")
    return "\n".join(parts)


def parse_scenes_text(text):
    found = _ENV_RE.findall(text or "")
    if found:
        return [body.strip() for _no, body in sorted(found, key=lambda p: int(p[0]))]
    # no envelopes: treat everything except # comment lines as one scene
    body = "\n".join(
        line for line in (text or "").splitlines() if not line.lstrip().startswith("#")
    ).strip()
    return [body] if body else []


class H3ScenesReview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenes": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "The writer's `scenes` list (or a single `h3_prompt`). In Review "
                        "mode these fill the editor; in Continue mode the editor text is "
                        "what gets rendered."
                    ),
                }),
                "mode": (MODES, {
                    "default": MODE_REVIEW,
                    "tooltip": (
                        "Review: fill the editor with the incoming scenes and stop the run "
                        "before the render (the editor flips this to Continue for you). "
                        "Continue: render the editor text. Bypass: pass through untouched. "
                        "Give the writer a fixed seed so Continue runs reuse its cached "
                        "answer instead of writing new scenes."
                    ),
                }),
                "edited": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "The reviewed scene text, one === SCENE NN === envelope per scene. "
                        "Filled automatically by a Review run; edit freely. Empty = pass "
                        "the incoming scenes through unchanged. The source-fingerprint line "
                        "in the header lets the node detect when the writer's scenes changed "
                        "upstream (new cast / direction / seed) - it then re-reviews with the "
                        "fresh scenes instead of rendering the stale editor text."
                    ),
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("scenes", "scene_count")
    OUTPUT_IS_LIST = (True, False)
    OUTPUT_NODE = True
    FUNCTION = "review"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Review gate between a writer and the render: Review runs fill the editable, "
        "colour-coded scene editor and stop cleanly before anything renders; edit the "
        "text (all scenes or one at a time) and queue again to render exactly what the "
        "editor says. Recreate bumps the writer's seed for a fresh draft."
    )

    @classmethod
    def IS_CHANGED(cls, mode="", **kwargs):
        m = mode[0] if isinstance(mode, (list, tuple)) and mode else mode
        # Review must always re-run (refill the editor and stop the queue)
        return float("nan") if str(m).startswith("Review") else 0.0

    def review(self, scenes, mode, edited, unique_id=None):
        mode = mode[0] if isinstance(mode, (list, tuple)) and mode else mode
        edited = edited[0] if isinstance(edited, (list, tuple)) and edited else (edited or "")
        uid = unique_id[0] if isinstance(unique_id, (list, tuple)) and unique_id else unique_id
        scenes = ["" if s is None else str(s) for s in (scenes or [])]

        if mode == MODE_BYPASS:
            return (scenes, len(scenes))

        # Continue with an editor that was filled from DIFFERENT incoming
        # scenes (the cast, direction, lyrics or seed changed upstream since
        # the review): the edits belong to stale data - render the FRESH
        # scenes and refill the editor, never the old text. A new run always
        # produces a new video; pin the writer's seed to edit before render.
        if mode == MODE_CONTINUE and edited.strip() and scenes:
            m = _FP_RE.search(edited)
            if m and m.group(1) != _fingerprint(scenes):
                print(
                    "♻️ H3 Scenes Review: the incoming scenes changed since the editor was "
                    "filled (new cast / direction / seed upstream) - rendering the fresh "
                    "scenes; the previous edits belonged to the old data. Pin the writer's "
                    "seed to a fixed value to edit before rendering."
                )
                edited = ""
                try:
                    from server import PromptServer
                    PromptServer.instance.send_sync(
                        "apnext.h3.scenes_review",
                        {"node": str(uid), "text": serialize_scenes(scenes),
                         "count": len(scenes), "stopped": False},
                    )
                except Exception:
                    pass

        if mode == MODE_REVIEW:
            text = serialize_scenes(scenes)
            try:
                from server import PromptServer
                PromptServer.instance.send_sync(
                    "apnext.h3.scenes_review",
                    {"node": str(uid), "text": text, "count": len(scenes)},
                )
            except Exception as exc:
                print(f"⚠️ H3 Scenes Review: could not push the editor text to the UI: {exc}")
            print(
                f"🛑 H3 Scenes Review: {len(scenes)} scene(s) are in the editor - nothing has "
                "been rendered. Edit the text in the node, then queue again (the mode is now "
                "Continue). The Recreate button asks the writer for a fresh draft instead."
            )
            # Block only what is DOWNSTREAM of this node (the render) instead of
            # interrupting the whole prompt: previews / Show Text nodes wired to
            # the writer's other outputs (scenes_text, synopsis, script...) still
            # execute, so a Review stop never blanks them.
            try:
                from comfy_execution.graph import ExecutionBlocker
                return ([ExecutionBlocker(None)], ExecutionBlocker(None))
            except ImportError:
                import comfy.model_management
                raise comfy.model_management.InterruptProcessingException()

        # Continue: the editor text is authoritative; empty editor = pass-through
        out = parse_scenes_text(edited) if edited.strip() else list(scenes)
        if not out:
            out = list(scenes)
        if scenes and len(out) != len(scenes):
            print(
                f"⚠️ H3 Scenes Review: the editor has {len(out)} scene(s) but {len(scenes)} "
                "came in. The writer's durations/audio lists stay aligned to the incoming "
                "count, so the edited list is padded/trimmed to match."
            )
            while len(out) < len(scenes):
                out.append(out[-1] if out else "")
            out = out[: len(scenes)]
        return (out, len(out))
