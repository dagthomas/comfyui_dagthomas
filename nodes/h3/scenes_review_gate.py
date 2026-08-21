# APNext H3 Dailies Gate (H3ScenesReviewGate)
#
# The screening-room stop between a writer and the render: the run HOLDS here
# (no interrupt, no re-queue) while the freshly written scenes - the "dailies"
# - sit on the desk in the browser, and it moves again the moment a button is
# pressed:
#
#   Print it  - render this same run with the (possibly hand-edited) takes.
#   Punch-up  - one more turn in the WRITER'S OWN session: the director's
#               notes go to the model, the selected takes (or all of them)
#               are rewritten, and the desk shows the new version for another
#               look. Works with Claude Code, Codex (`codex-` sessions) and
#               local-model (`local-`) sessions alike. Hand edits made in the
#               editor are folded in first, so "fix it by hand, then have the
#               model regenerate around it" is one round trip.
#   New take  - the same rewrite turn with no notes: the model is asked for a
#               noticeably different take on the selected scenes.
#   Undo      - server-side history of every punch-up, so a bad rewrite rolls
#               back instantly (and survives a browser reload).
#   Cut       - end the run cleanly; nothing renders.
#
# Mechanics: the node function is async. It publishes the scenes to the
# browser over the websocket, then awaits an asyncio.Future with a heartbeat
# (honouring ComfyUI's Stop button and an optional auto-print timeout). The
# buttons POST to /apnext/h3/review_gate, which resolves the future. A page
# reload re-attaches through GET /apnext/h3/review_gate/pending.
#
# Compare H3 Scenes Review (scenes_review.py): that gate stops the queue and
# needs a second queue run to continue; this one holds the run open, which
# also means a punch-up happens inline without re-running the writer.

import asyncio
import re
import time
import uuid

from ...utils.constants import CUSTOM_CATEGORY
from .claude_code_support import (
    BASE_SKILLS,
    _model_choices,
    local_llm_inputs,
    local_llm_options,
    run_h3_claude_code,
)
from .scenes_review import parse_scenes_text, serialize_scenes
from .scenes_support import parse_scenes

try:
    from server import PromptServer
except Exception:
    PromptServer = None
try:
    from aiohttp import web
except Exception:
    web = None

EVENT_SHOW = "apnext.h3.review_gate"
EVENT_RESOLVED = "apnext.h3.review_gate_resolved"

# token -> {"future": Future, "loop": event loop, "public": payload}
_PENDING = {}

_SELECT_RE = re.compile(r"(\d+)(?:\s*-\s*(\d+))?")


def _parse_selection(text, count):
    """`"2, 4-5"` -> [2, 4, 5], clamped to 1..count. Empty -> all scenes."""
    picked = set()
    for m in _SELECT_RE.finditer(text or ""):
        lo = int(m.group(1))
        hi = int(m.group(2) or lo)
        for n in range(min(lo, hi), max(lo, hi) + 1):
            if 1 <= n <= count:
                picked.add(n)
    return sorted(picked) or list(range(1, count + 1))


def _display_id(unique_id):
    uid = unique_id[0] if isinstance(unique_id, (list, tuple)) and unique_id else unique_id
    return str(uid)


def _send(event, payload):
    if PromptServer is None:
        return
    try:
        PromptServer.instance.send_sync(event, dict(payload))
    except Exception as exc:
        print(f"⚠️ H3 Review Gate: could not reach the browser: {exc}")


def _throw_if_interrupted():
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    mm.throw_exception_if_processing_interrupted()


async def _await_decision(future, timeout_seconds):
    """Wait for a browser decision with a heartbeat (Stop button + timeout)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds if timeout_seconds > 0 else None
    while True:
        _throw_if_interrupted()
        if future.done():
            return future.result()
        wait = 0.25
        if deadline is not None:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return {"action": "approve", "timed_out": True}
            wait = min(wait, remaining)
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=wait)
        except asyncio.TimeoutError:
            continue


def _reroll_prompt(feedback, scene_items, current_text):
    """One rewrite turn in the writer's session: feedback -> fresh envelopes."""
    nos = ", ".join(f"{no:02d} (duration {dur:.1f})" for no, dur in scene_items)
    return (
        "REVIEW FEEDBACK from the user on the scenes you wrote:\n"
        f"{feedback.strip() or '(none - write a fresh, noticeably different take)'}\n\n"
        "THE SCENES AS THEY CURRENTLY STAND (the user may have hand-edited them; "
        "this text is authoritative over what you wrote before):\n"
        f"{current_text}\n\n"
        f"Rewrite ONLY scene(s) {nos}, applying the feedback while keeping "
        "everything the feedback does not touch - the synopsis, the cast, the "
        "wardrobe and location locks, the visual style, each scene's duration, its "
        "lyric lines and their timing, and the story hand-offs to the neighbouring "
        "scenes. Return ONLY the rewritten envelopes, each in the exact same "
        "envelope contract (`=== SCENE NN | duration: S.S ===` ... `=== END SCENE "
        "NN ===`) with its original number and duration. Do not repeat the "
        "synopsis and do not return any other scene or commentary."
    )


class H3ScenesReviewGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenes": ("STRING", {
                    "forceInput": True,
                    "tooltip": "The writer's `scenes` list (or a single h3_prompt).",
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off = transparent pass-through (no pause).",
                }),
                "auto_approve_minutes": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1440.0, "step": 0.5,
                    "tooltip": (
                        "Print (approve) automatically after this many minutes, so "
                        "an unattended run still renders. 0 holds indefinitely."
                    ),
                }),
                "model": (_model_choices(), {
                    "default": "sonnet",
                    "tooltip": (
                        "Who handles a punch-up: the same aliases as the writers. "
                        "The rewrite continues the writer's session, so it must be "
                        "the same backend - the gate auto-switches to codex for a "
                        "`codex-` session id."
                    ),
                }),
                "use_subscription": ("BOOLEAN", {"default": True}),
                "reroll_timeout_seconds": ("INT", {
                    "default": 600, "min": 60, "max": 7200, "step": 30,
                    "tooltip": "Watchdog for one punch-up turn.",
                }),
                "chime": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Play a soft browser chime when takes land on the desk.",
                }),
            },
            "optional": {
                "durations": ("FLOAT", {
                    "forceInput": True,
                    "tooltip": (
                        "The writer's `durations` list; keeps every rewritten "
                        "envelope on its original duration."
                    ),
                }),
                "session_id": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "The writer's `session_id` output. Required for punch-ups - "
                        "the rewrite happens inside that session, so the synopsis, "
                        "locks and images are still in context."
                    ),
                }),
                **local_llm_inputs(),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("scenes", "scene_count", "status")
    OUTPUT_IS_LIST = (True, False, False)
    OUTPUT_NODE = True
    FUNCTION = "gate"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "The dailies desk: the run HOLDS here with the freshly written scenes on screen. "
        "Print it (render, hand edits included), send director's notes for a punch-up - "
        "selected takes rewritten inside the writer's own Claude/Codex session - or cut "
        "the run. It moves again the moment a button is pressed; no re-queueing."
    )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    async def gate(self, scenes, enabled, auto_approve_minutes, model,
                   use_subscription, reroll_timeout_seconds, chime=False,
                   durations=None, session_id=None, llm=None, unique_id=None):
        def scalar(v, default=None):
            return v[0] if isinstance(v, (list, tuple)) and v else (v if v is not None else default)

        enabled = bool(scalar(enabled, True))
        auto_approve_minutes = float(scalar(auto_approve_minutes, 0.0))
        model = str(scalar(model, "sonnet"))
        use_subscription = bool(scalar(use_subscription, True))
        reroll_timeout = int(scalar(reroll_timeout_seconds, 600))
        chime = bool(scalar(chime, False))
        session = str(scalar(session_id, "") or "").strip()
        llm = scalar(llm)
        scenes = ["" if s is None else str(s) for s in (scenes or [])]
        n = len(scenes)
        durations = [float(d) for d in (durations or [])]
        while len(durations) < n:
            durations.append(durations[-1] if durations else 10.0)

        if not enabled or not n:
            return {"ui": {"text": ["gate bypassed"]},
                    "result": (scenes, n, "gate bypassed")}
        if PromptServer is None or web is None:
            raise RuntimeError("H3 Scenes Review Gate needs ComfyUI's prompt server.")

        # A codex-* session can only be continued by the codex backend.
        if session.startswith("codex-") and not model.startswith("codex"):
            model = "codex"

        token = uuid.uuid4().hex
        node_id = _display_id(unique_id)
        timeout_seconds = min(1440.0, max(0.0, auto_approve_minutes)) * 60.0
        loop = asyncio.get_running_loop()
        status_note = ""
        revision = 0
        current = list(scenes)
        history = []  # previous versions of `current`, newest last (undo stack)

        def publish(note=""):
            payload = {
                "token": token,
                "node": node_id,
                "text": serialize_scenes(current),
                "count": len(current),
                "can_reroll": bool(session),
                "can_undo": bool(history),
                "chime": chime,
                "revision": revision,
                "status": note,
                "timeout_seconds": timeout_seconds,
                "deadline": (time.time() + timeout_seconds) if timeout_seconds > 0 else None,
                "server_now": time.time(),
            }
            _PENDING[token]["public"] = payload
            _send(EVENT_SHOW, payload)

        _PENDING[token] = {"future": None, "loop": loop, "public": {}}
        print(
            f"🎞️ H3 Dailies Gate: {n} take(s) on the desk in the browser - the run is "
            "holding (not stopped). Print it, send punch-up notes, or cut."
            + ("" if session else " (No session_id connected: punch-ups are disabled.)")
        )

        try:
            while True:
                future = loop.create_future()
                _PENDING[token]["future"] = future
                publish(status_note)
                try:
                    decision = await _await_decision(future, timeout_seconds)
                except BaseException:
                    _send(EVENT_RESOLVED, {"token": token, "node": node_id,
                                           "action": "interrupted",
                                           "status": "review interrupted"})
                    raise

                action = str(decision.get("action") or "")
                if action == "undo":
                    if history:
                        current = history.pop()
                        status_note = "restored the previous version of the takes"
                    else:
                        status_note = "nothing to undo yet"
                    revision += 1
                    continue

                # Hand edits from the browser are folded in first for every
                # other action, so a punch-up regenerates AROUND the edits.
                edited = str(decision.get("text") or "")
                if edited.strip():
                    parsed = parse_scenes_text(edited)
                    if parsed:
                        while len(parsed) < n:
                            parsed.append(parsed[-1])
                        current = parsed[:n]

                if action == "approve":
                    status = (
                        f"auto-printed after {auto_approve_minutes:g} min"
                        if decision.get("timed_out") else "printed"
                    ) + f" - rendering {len(current)} take(s)"
                    _send(EVENT_RESOLVED, {"token": token, "node": node_id,
                                           "action": "approve", "status": status})
                    print(f"✅ H3 Dailies Gate: {status}")
                    return {"ui": {"text": [status]}, "result": (current, len(current), status)}

                if action == "stop":
                    status = "cut at the dailies desk - nothing rendered"
                    _send(EVENT_RESOLVED, {"token": token, "node": node_id,
                                           "action": "stop", "status": status})
                    print(f"✋ H3 Dailies Gate: {status}")
                    import comfy.model_management
                    raise comfy.model_management.InterruptProcessingException()

                if action != "reroll":
                    status_note = f"unknown action '{action}' ignored"
                    continue

                if not session:
                    status_note = (
                        "a punch-up needs the writer's session_id wired into this "
                        "node - edit by hand or print instead"
                    )
                    revision += 1
                    continue

                # "New take" is the same rewrite turn with the notes dropped:
                # the model is asked for a noticeably different version.
                variant = bool(decision.get("variant"))
                feedback = "" if variant else str(decision.get("feedback") or "")
                picked = _parse_selection(str(decision.get("scenes") or ""), len(current))
                items = [(no, durations[no - 1]) for no in picked]
                current_text = serialize_scenes(current)
                print(
                    f"✍️ H3 Dailies Gate: {'new take for' if variant else 'punching up'} "
                    f"take(s) {', '.join(f'{no:02d}' for no in picked)} with {model} - "
                    f"notes: {feedback.strip()[:120] or '(none - fresh take)'}"
                )
                try:
                    text, session, info = run_h3_claude_code(
                        None, _reroll_prompt(feedback, items, current_text), None,
                        model, False, use_subscription, reroll_timeout,
                        session, "", False, skills=BASE_SKILLS,
                        local=local_llm_options(llm),
                    )
                    _, fresh = parse_scenes(text, items[0][1])
                    by_no = {no: p for no, _d, p in fresh
                             if no in set(picked) and p.strip()}
                    if not by_no:
                        status_note = "the rewrite returned no usable envelopes - kept the current text"
                    else:
                        history.append(list(current))
                        current = [
                            by_no.get(i + 1, s) for i, s in enumerate(current)
                        ]
                        status_note = (
                            f"{'new take on' if variant else 'punched up'} take(s) "
                            f"{', '.join(f'{no:02d}' for no in sorted(by_no))} - back on "
                            f"the desk, undo available ({info.split('|')[0].strip()})"
                        )
                except BaseException as exc:
                    import comfy.model_management
                    if isinstance(exc, comfy.model_management.InterruptProcessingException):
                        raise
                    status_note = f"punch-up failed: {exc}"
                    print(f"⚠️ H3 Dailies Gate: {status_note}")
                revision += 1
        finally:
            entry = _PENDING.pop(token, None)
            future = entry and entry.get("future")
            if future is not None and not future.done():
                future.cancel()


# ---------------------------------------------------------------------------
# HTTP: the browser's buttons land here and resolve the pending future
# ---------------------------------------------------------------------------

async def _submit_decision(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
    token = str(body.get("token") or "")
    entry = _PENDING.get(token)
    if entry is None or entry.get("future") is None:
        return web.json_response(
            {"ok": False, "error": "no pending review for this token"}, status=404)
    future, loop = entry["future"], entry["loop"]

    def resolve():
        if not future.done():
            future.set_result({
                "action": str(body.get("action") or ""),
                "text": body.get("text") or "",
                "feedback": body.get("feedback") or "",
                "scenes": body.get("scenes") or "",
                "variant": bool(body.get("variant")),
            })

    loop.call_soon_threadsafe(resolve)
    return web.json_response({"ok": True})


async def _list_pending(request):
    return web.json_response(
        {"reviews": [e["public"] for e in _PENDING.values() if e.get("public")]})


if PromptServer is not None and web is not None and getattr(PromptServer, "instance", None) is not None:
    PromptServer.instance.routes.post("/apnext/h3/review_gate")(_submit_decision)
    PromptServer.instance.routes.get("/apnext/h3/review_gate/pending")(_list_pending)
