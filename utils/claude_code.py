# Claude Code CLI bridge
#
# Runs the locally installed `claude` binary in headless mode and hands back its
# answer, so a ComfyUI node can borrow the user's existing Claude Code login
# instead of a separate API key.
#
# The transport is Claude Code's bidirectional streaming JSON protocol:
#
#   claude -p --input-format stream-json --output-format stream-json --verbose
#
# One JSON message goes in over stdin, a stream of JSON events comes back on
# stdout. That buys four things over passing a plain prompt:
#
#   * Images travel as inline base64 content blocks. No scratch files, no Read
#     tool, no permission grant, and one less turn per request.
#   * Progress is visible while the model works, instead of a silent 45s block.
#   * Rate-limit events are seen as they arrive, so quota problems are reported
#     as quota problems.
#   * The process can be killed the moment ComfyUI's queue is cancelled.
#
# Two constraints shape the rest. Windows caps a command line at 32,767
# characters and the H3 reference guide alone is ~40 KB, so a long system prompt
# is folded into the message rather than passed as an argument. And
# `--setting-sources ""` keeps the user's own CLAUDE.md, skills and hooks out of
# the request - a personal "always use Go best practices" rule has no business
# rewriting a video prompt.

import base64
import io
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time

# Present only when running inside ComfyUI; the module is importable standalone.
try:
    import comfy.model_management as _model_management
except Exception:
    _model_management = None

# Model aliases the CLI accepts. A full id (claude-sonnet-5) works too.
CLAUDE_CODE_MODELS = ["sonnet", "opus", "haiku", "fable", "default"]

# Read-only tools, granted only when research mode is on. Bash/Write/Edit are
# deliberately absent - a prompt writer has no business changing anything.
RESEARCH_TOOLS = ["Read", "Glob", "Grep", "WebSearch", "WebFetch"]

# A system prompt longer than this moves into the stdin message instead of being
# passed as an argument, well under the Windows command-line limit.
_ARGV_SAFE_LIMIT = 8000

_EMBEDDED_SYSTEM_FRAMING = (
    "You follow the specification in the user message exactly. Emit only what it "
    "asks for: no preamble, no commentary, no summary of what you did."
)

# How often the reader loop wakes up to check for cancellation and timeout.
_TICK_SECONDS = 0.2

# Grace period between asking the CLI to stop and killing it outright.
_KILL_GRACE_SECONDS = 3

_cli_cache = {"path": None, "checked": False}


class ClaudeCodeError(RuntimeError):
    """The CLI is missing, timed out, or reported a failure."""


def find_cli(refresh=False):
    """Absolute path to the claude binary, or None when it is not installed."""
    if _cli_cache["checked"] and not refresh:
        return _cli_cache["path"]

    candidates = []
    override = os.environ.get("CLAUDE_CODE_PATH")
    if override:
        candidates.append(override)

    found = shutil.which("claude")
    if found:
        candidates.append(found)

    home = os.path.expanduser("~")
    candidates.extend(
        [
            os.path.join(home, ".local", "bin", "claude.exe"),
            os.path.join(home, ".local", "bin", "claude"),
            os.path.join(os.environ.get("APPDATA", ""), "npm", "claude.cmd"),
            "/usr/local/bin/claude",
            "/opt/homebrew/bin/claude",
        ]
    )

    path = next((c for c in candidates if c and os.path.isfile(c)), None)
    _cli_cache["path"] = path
    _cli_cache["checked"] = True
    return path


def is_available():
    return find_cli() is not None


def is_interrupt(exc):
    """Did this exception come from the user cancelling the ComfyUI queue?"""
    if _model_management is None:
        return False
    return isinstance(exc, _model_management.InterruptProcessingException)


def _processing_interrupted():
    if _model_management is None:
        return False
    try:
        return _model_management.processing_interrupted()
    except Exception:
        return False


def _subprocess_flags():
    """Keep Windows from flashing a console window on every node run."""
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def _child_env(use_subscription):
    """
    Environment for the child process.

    ANTHROPIC_API_KEY is dropped by default: with it set, Claude Code bills the
    API account, which defeats the point of routing through a subscription seat.
    The other nodes in this pack want that key, hence the per-call switch.
    """
    env = os.environ.copy()
    if use_subscription:
        for name in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            env.pop(name, None)
    return env


def _image_block(image):
    """PIL image -> an Anthropic base64 image content block."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": encoded},
    }


def _user_message(text, images):
    """
    The single stream-json message sent to the CLI.

    Images come first: the model reads them before the instruction that refers
    to them, which is what Anthropic recommends for image-led prompts.
    """
    content = [_image_block(image) for image in images]
    content.append({"type": "text", "text": text})
    return {"type": "user", "message": {"role": "user", "content": content}}


def _pump(stream, sink, tag):
    """Drain a pipe on its own thread so the child never blocks on a full buffer."""
    try:
        for line in stream:
            sink.put((tag, line))
    except Exception:
        pass
    finally:
        sink.put((tag, None))
        try:
            stream.close()
        except Exception:
            pass


def _write_stdin(process, payload):
    """
    Feed the request in on a thread.

    A prompt carrying a few base64 images is far larger than the OS pipe buffer,
    so writing it inline would deadlock against a child that has not started
    reading yet.
    """
    try:
        process.stdin.write(payload)
        process.stdin.flush()
    except Exception:
        pass
    finally:
        try:
            process.stdin.close()
        except Exception:
            pass


def _stop(process):
    """Ask the CLI to exit, then insist."""
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=_KILL_GRACE_SECONDS)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _describe_event(event):
    """A short progress line for an event, or None when it is not worth showing."""
    kind = event.get("type")

    if kind == "system":
        # thinking_tokens arrives many times a second; init is the useful one.
        if event.get("subtype") == "init":
            return f"session {(event.get('session_id') or '')[:8]} ready"
        return None

    if kind == "assistant":
        parts = []
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "tool_use":
                parts.append(f"using {block.get('name', 'a tool')}")
            elif block.get("type") == "text":
                snippet = " ".join((block.get("text") or "").split())
                if snippet:
                    parts.append(snippet[:110] + ("..." if len(snippet) > 110 else ""))
        return " | ".join(parts) or None

    return None


def _rate_limit_note(info):
    """A warning for a rate-limit event, or None while everything is fine."""
    status = (info.get("status") or "").lower()
    if status in ("", "allowed", "ok"):
        return None

    window = info.get("rateLimitType") or "usage"
    note = f"Claude Code {window} limit: {status}"
    resets_at = info.get("resetsAt")
    if resets_at:
        try:
            when = time.strftime("%H:%M", time.localtime(int(resets_at)))
            note += f" (resets around {when})"
        except Exception:
            pass
    return note


def run_claude_code(
    prompt,
    system_prompt=None,
    images=None,
    model="sonnet",
    timeout=600,
    tools=None,
    working_dir=None,
    resume_session_id=None,
    use_subscription=True,
    on_progress=None,
    add_dirs=None,
):
    """
    Run one headless Claude Code turn and return its result plus telemetry.

    `images` is a list of PIL images, sent inline. `tools` is an explicit
    allow-list; None grants nothing. `add_dirs` are extra directories the
    granted tools may read (`--add-dir`), on top of the working directory.
    `on_progress` is called with short status strings while the turn runs.
    Raises ClaudeCodeError on failure, and lets a ComfyUI cancellation
    propagate as an interrupt.
    """
    cli = find_cli()
    if not cli:
        raise ClaudeCodeError(
            "The Claude Code CLI was not found. Install it from "
            "https://claude.com/claude-code, or set CLAUDE_CODE_PATH to the binary."
        )

    images = list(images or [])
    body = prompt

    args = [
        cli,
        "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
        "--setting-sources", "",
    ]

    if model and model != "default":
        args += ["--model", model]

    # Long system prompts cannot be arguments, so they ride along in the message.
    if system_prompt and len(system_prompt) > _ARGV_SAFE_LIMIT:
        args += ["--system-prompt", _EMBEDDED_SYSTEM_FRAMING]
        body = (
            "=== BEGIN SPECIFICATION - follow it exactly ===\n"
            f"{system_prompt}\n"
            "=== END SPECIFICATION ===\n\n"
            f"{prompt}"
        )
    elif system_prompt:
        args += ["--system-prompt", system_prompt]

    if tools:
        args += ["--allowedTools", ",".join(tools)]
        # A headless run cannot answer a permission prompt; without this the
        # tool call is denied and the model answers blind.
        args += ["--permission-mode", "dontAsk"]

    for extra in add_dirs or []:
        if extra and os.path.isdir(extra):
            args += ["--add-dir", extra]

    if resume_session_id:
        args += ["--resume", resume_session_id]

    # An isolated working directory keeps the CLI out of the ComfyUI tree.
    scratch = tempfile.mkdtemp(prefix="apnext_claude_code_")
    cwd = working_dir if working_dir and os.path.isdir(working_dir) else scratch

    payload = json.dumps(_user_message(body, images), ensure_ascii=False) + "\n"

    events = queue.Queue()
    result_payload = None
    rate_limit_info = {}
    stderr_lines = []
    warned_rate_limit = False
    open_streams = 2

    try:
        try:
            process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=cwd,
                env=_child_env(use_subscription),
                **_subprocess_flags(),
            )
        except OSError as exc:
            raise ClaudeCodeError(f"Could not start the Claude Code CLI at {cli}: {exc}")

        threading.Thread(target=_write_stdin, args=(process, payload), daemon=True).start()
        threading.Thread(target=_pump, args=(process.stdout, events, "out"), daemon=True).start()
        threading.Thread(target=_pump, args=(process.stderr, events, "err"), daemon=True).start()

        deadline = time.monotonic() + timeout

        while open_streams:
            try:
                tag, line = events.get(timeout=_TICK_SECONDS)
            except queue.Empty:
                if _processing_interrupted():
                    _stop(process)
                    # Raises ComfyUI's own interrupt so the queue stops cleanly.
                    _model_management.throw_exception_if_processing_interrupted()
                if time.monotonic() > deadline:
                    _stop(process)
                    raise ClaudeCodeError(
                        f"Claude Code did not finish within {timeout}s. Raise the timeout, "
                        "or use a faster model."
                    )
                continue

            if line is None:
                open_streams -= 1
                continue

            if tag == "err":
                text = line.strip()
                if text:
                    stderr_lines.append(text)
                continue

            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except ValueError:
                continue

            kind = event.get("type")

            if kind == "result":
                result_payload = event
                continue

            if kind == "rate_limit_event":
                rate_limit_info = event.get("rate_limit_info") or {}
                note = _rate_limit_note(rate_limit_info)
                if note and not warned_rate_limit:
                    warned_rate_limit = True
                    print(f"⚠️  {note}")
                continue

            if on_progress:
                description = _describe_event(event)
                if description:
                    on_progress(description)

        process.wait(timeout=_KILL_GRACE_SECONDS)

        if result_payload is None:
            detail = " ".join(stderr_lines) or f"exit code {process.returncode}"
            raise ClaudeCodeError(f"Claude Code returned no result: {detail}")

        text = (result_payload.get("result") or "").strip()

        if result_payload.get("is_error") or result_payload.get("subtype") not in (None, "success"):
            detail = text or " ".join(stderr_lines) or result_payload.get("subtype") or "unknown error"
            note = _rate_limit_note(rate_limit_info)
            if note:
                detail = f"{detail} ({note})"
            if "login" in detail.lower() or "authenticat" in detail.lower():
                detail += (
                    " - run `claude` once in a terminal to log in, or untick the "
                    "subscription switch to use ANTHROPIC_API_KEY."
                )
            raise ClaudeCodeError(f"Claude Code reported an error: {detail}")

        if not text:
            raise ClaudeCodeError("Claude Code returned an empty result.")

        return {
            "text": text,
            "session_id": result_payload.get("session_id") or "",
            "cost_usd": float(result_payload.get("total_cost_usd") or 0.0),
            "duration_ms": int(result_payload.get("duration_ms") or 0),
            "num_turns": int(result_payload.get("num_turns") or 0),
            "model": model,
            "rate_limit": rate_limit_info,
        }

    finally:
        shutil.rmtree(scratch, ignore_errors=True)
