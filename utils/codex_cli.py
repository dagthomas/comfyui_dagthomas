# OpenAI Codex CLI bridge
#
# Runs the locally installed `codex` binary headlessly (`codex exec --json`)
# and hands back its answer, so the H3 nodes can write with a ChatGPT / Codex
# subscription seat the same way they write with a Claude Code seat.
#
# Differences from the Claude Code bridge that shape this file:
#
#   * Codex has no system-prompt flag, so the specification is folded into the
#     user message under the same framing run_claude_code uses for oversized
#     system prompts. The prompt itself travels over stdin (`codex exec -`),
#     which also dodges the Windows 32,767-character argv limit.
#   * Images have no inline transport; they are written as PNGs to the scratch
#     folder and attached with `-i`.
#   * Sessions are Codex "threads": `codex exec resume <id>` continues one.
#     The ids handed back to the graph are prefixed `codex-` so a Claude Code
#     session id can never be fed to Codex by accident (and vice versa).
#   * The final answer is read from `--output-last-message` first - the event
#     stream has changed shape between Codex releases, so the file is the
#     stable contract and the parsed events are the fallback.

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import queue

from .claude_code import (
    _KILL_GRACE_SECONDS,
    _TICK_SECONDS,
    _model_management,
    _processing_interrupted,
    _pump,
    _stop,
    _subprocess_flags,
)

_EMBEDDED_SYSTEM_FRAMING = (
    "=== BEGIN SPECIFICATION - follow it exactly ===\n"
    "{system_prompt}\n"
    "=== END SPECIFICATION ===\n\n"
    "{prompt}"
)

_cli_cache = {"path": None, "checked": False}


class CodexCLIError(RuntimeError):
    """The Codex CLI is missing, timed out, or reported a failure."""


def find_cli(refresh=False):
    """Absolute path to the codex binary, or None when it is not installed."""
    if _cli_cache["checked"] and not refresh:
        return _cli_cache["path"]

    candidates = []
    override = os.environ.get("CODEX_CLI_PATH")
    if override:
        candidates.append(override)

    found = shutil.which("codex")
    if found:
        candidates.append(found)

    home = os.path.expanduser("~")
    candidates.extend(
        [
            os.path.join(home, ".local", "bin", "codex.exe"),
            os.path.join(home, ".local", "bin", "codex"),
            os.path.join(os.environ.get("APPDATA", ""), "npm", "codex.cmd"),
            os.path.join(home, ".bun", "bin", "codex.exe"),
            os.path.join(home, ".volta", "bin", "codex.exe"),
            "/usr/local/bin/codex",
            "/opt/homebrew/bin/codex",
        ]
    )

    path = next((c for c in candidates if c and os.path.isfile(c)), None)
    _cli_cache["path"] = path
    _cli_cache["checked"] = True
    return path


def is_available():
    return find_cli() is not None


def _child_env(use_subscription):
    """
    Environment for the child process.

    OPENAI_API_KEY is dropped by default so Codex uses the ChatGPT login in
    ~/.codex/auth.json (the subscription seat) instead of billing the API key.
    """
    env = os.environ.copy()
    if use_subscription:
        env.pop("OPENAI_API_KEY", None)
    return env


def _describe_item(item):
    """A short progress line for one event item, or None when not worth showing."""
    kind = item.get("type")
    if kind == "command_execution":
        command = " ".join((item.get("command") or "").split())
        return f"running {command[:90]}" if command else "running a command"
    if kind == "web_search":
        q = (item.get("query") or "").strip()
        return f"searching the web{f': {q[:70]}' if q else ''}"
    if kind == "mcp_tool_call":
        return f"using {item.get('tool') or 'a tool'}"
    if kind == "agent_message":
        snippet = " ".join((item.get("text") or "").split())
        if snippet:
            return snippet[:110] + ("..." if len(snippet) > 110 else "")
    return None


def run_codex(
    prompt,
    system_prompt=None,
    images=None,
    model=None,
    timeout=600,
    research=False,
    working_dir=None,
    resume_session_id=None,
    use_subscription=True,
    on_progress=None,
):
    """
    Run one headless Codex turn and return its result plus telemetry.

    Mirrors run_claude_code's contract: `images` is a list of PIL images,
    `model` is the Codex model id (None/"" = whatever ~/.codex/config.toml
    picks), `research` turns the web-search tool on, `resume_session_id` is a
    BARE Codex thread id (no `codex-` prefix). Returns a dict with `text`,
    `session_id` (bare thread id), `duration_ms`, `num_turns`, `model`,
    `cost_usd` (always 0.0 - the CLI does not report cost). Raises
    CodexCLIError on failure, and lets a ComfyUI cancellation propagate.
    """
    cli = find_cli()
    if not cli:
        raise CodexCLIError(
            "The Codex CLI was not found. Install it with `npm install -g @openai/codex` "
            "and log in once with `codex login`, or set CODEX_CLI_PATH to the binary."
        )

    body = prompt
    if system_prompt:
        body = _EMBEDDED_SYSTEM_FRAMING.format(system_prompt=system_prompt, prompt=prompt)

    scratch = tempfile.mkdtemp(prefix="apnext_codex_")
    cwd = working_dir if working_dir and os.path.isdir(working_dir) else scratch
    last_message_path = os.path.join(scratch, "last_message.txt")

    args = [cli, "exec"]
    if resume_session_id:
        args += ["resume", resume_session_id]
    args += [
        "--json",
        "--skip-git-repo-check",
        "--sandbox", "read-only",
        "--output-last-message", last_message_path,
    ]
    if model and model not in ("default", "codex"):
        args += ["--model", model]
    if research:
        # Web search is a config key rather than a stable flag across releases.
        args += ["-c", "tools.web_search=true"]
    for i, image in enumerate(images or [], 1):
        image_path = os.path.join(scratch, f"reference_{i}.png")
        image.convert("RGB").save(image_path, format="PNG")
        args += ["-i", image_path]
    args.append("-")  # read the prompt from stdin

    events = queue.Queue()
    thread_id = resume_session_id or ""
    agent_text = ""
    error_detail = ""
    num_turns = 0
    stderr_lines = []
    open_streams = 2
    started = time.monotonic()

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
            raise CodexCLIError(f"Could not start the Codex CLI at {cli}: {exc}")

        def _write_stdin():
            try:
                process.stdin.write(body)
                process.stdin.flush()
            except Exception:
                pass
            finally:
                try:
                    process.stdin.close()
                except Exception:
                    pass

        threading.Thread(target=_write_stdin, daemon=True).start()
        threading.Thread(target=_pump, args=(process.stdout, events, "out"), daemon=True).start()
        threading.Thread(target=_pump, args=(process.stderr, events, "err"), daemon=True).start()

        deadline = time.monotonic() + timeout

        while open_streams:
            try:
                tag, line = events.get(timeout=_TICK_SECONDS)
            except queue.Empty:
                if _processing_interrupted():
                    _stop(process)
                    _model_management.throw_exception_if_processing_interrupted()
                if time.monotonic() > deadline:
                    _stop(process)
                    raise CodexCLIError(
                        f"Codex did not finish within {timeout}s. Raise the timeout, "
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

            kind = event.get("type") or ""

            if kind == "thread.started":
                thread_id = event.get("thread_id") or thread_id
                if on_progress and thread_id:
                    on_progress(f"session {thread_id[:8]} ready")
                continue
            if kind == "turn.completed":
                num_turns += 1
                continue
            if kind in ("turn.failed", "error"):
                err = event.get("error") or {}
                error_detail = (
                    (err.get("message") if isinstance(err, dict) else None)
                    or event.get("message") or error_detail or kind
                )
                continue
            if kind.startswith("item."):
                item = event.get("item") or {}
                if item.get("type") == "agent_message":
                    agent_text = item.get("text") or agent_text
                if on_progress:
                    description = _describe_item(item)
                    if description:
                        on_progress(description)
                continue

            # Pre-0.44 event shape: {"id": ..., "msg": {"type": ..., ...}}
            msg = event.get("msg")
            if isinstance(msg, dict):
                msg_kind = msg.get("type") or ""
                if msg_kind == "session_configured":
                    thread_id = msg.get("session_id") or thread_id
                elif msg_kind == "agent_message":
                    agent_text = msg.get("message") or agent_text
                elif msg_kind in ("error", "turn_failed"):
                    error_detail = msg.get("message") or error_detail or msg_kind

        process.wait(timeout=_KILL_GRACE_SECONDS)

        text = ""
        try:
            with open(last_message_path, encoding="utf-8") as handle:
                text = handle.read().strip()
        except OSError:
            pass
        text = text or agent_text.strip()

        if not text:
            detail = error_detail or " ".join(stderr_lines[-5:]) or f"exit code {process.returncode}"
            if "login" in detail.lower() or "auth" in detail.lower():
                detail += (
                    " - run `codex login` once in a terminal, or untick the "
                    "subscription switch to use OPENAI_API_KEY."
                )
            raise CodexCLIError(f"Codex returned no result: {detail}")

        if error_detail and process.returncode not in (0, None):
            raise CodexCLIError(f"Codex reported an error: {error_detail}")

        return {
            "text": text,
            "session_id": thread_id or "",
            "cost_usd": 0.0,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "num_turns": max(1, num_turns),
            "model": model or "codex-default",
        }

    finally:
        shutil.rmtree(scratch, ignore_errors=True)
