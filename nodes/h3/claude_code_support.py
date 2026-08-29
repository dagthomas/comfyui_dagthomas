# Shared pieces for the H3 nodes that talk to Claude Code directly.
#
# The generic H3 writers reach Claude Code through the model dropdown, which is
# enough to write a prompt but cannot express what the CLI is actually good at:
# researching real references before it writes, and keeping a session open so a
# later node can revise the prompt without re-sending it.
#
# These helpers carry the widgets and the call, so each node file stays focused
# on its own H3 format.

import json
import os
import tempfile
import time
import uuid

from ...utils.claude_code import CLAUDE_CODE_MODELS, RESEARCH_TOOLS, run_claude_code
from ...utils.codex_cli import is_available as codex_available, run_codex
from ...utils.llm_router import (
    CLAUDE_CODE_PROVIDER,
    LOCAL_PROVIDERS,
    call_llm,
    list_local_models,
    split_model,
)

# The H3 skills: short instruction files that are always loaded, each with a
# reference library (grammar, gold examples, style anchors, animation techniques)
# the CLI reads on demand with the Read tool. Same shape as Claude Code SKILL.md
# folders, so they can also be symlinked into ~/.claude/skills. Every node loads
# the core skill; the format skill depends on what the node writes.
SKILLS_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "h3", "skills",
))
CORE_SKILL = "h3-prompt-director"
BASE_SKILLS = (CORE_SKILL, "h3-base-format", "h3-style-craft")
REF_SKILLS = (CORE_SKILL, "h3-ref2va", "h3-style-craft")

# Tools granted when a working directory is attached but research is off: enough
# to read the user's own notes, nothing that reaches the network.
_LOCAL_FILE_TOOLS = ["Read", "Glob", "Grep"]

RESEARCH_DIRECTIVE = (
    "Research first: search the web to ground this in reality before writing - how the "
    "real location actually looks, period-correct wardrobe and props, how light behaves "
    "there, how the physical event in the idea really unfolds. Fold what you learn into "
    "the description as concrete visual detail. Never mention that you researched "
    "anything, never cite a source, and never add commentary: the output is still the "
    "finished prompt and nothing else."
)


def load_skill(name):
    """
    One skill's instructions, ready to append to a system prompt.

    Strips the SKILL.md frontmatter and rewrites its reference table so every
    file name is an absolute path the CLI can open directly. Returns "" if the
    folder is missing, so a node still works without it.
    """
    folder = os.path.join(SKILLS_ROOT, name)
    try:
        with open(os.path.join(folder, "SKILL.md"), encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return ""

    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:]

    references = os.path.join(folder, "references")
    if os.path.isdir(references):
        for file_name in sorted(os.listdir(references)):
            if file_name.endswith(".md"):
                text = text.replace(
                    f"`{file_name}`", f"`{os.path.join(references, file_name)}`"
                )

    return text.strip()


def director_block(skills):
    """The named skills wrapped for a system prompt; "" when none are available."""
    loaded = [(name, load_skill(name)) for name in skills]
    loaded = [(name, text) for name, text in loaded if text]
    if not loaded:
        return ""
    parts = [
        f"=== BEGIN SKILL {name} ===\n{text}\n=== END SKILL {name} ==="
        for name, text in loaded
    ]
    return (
        "\n\n" + "\n\n".join(parts) + "\n\n"
        "The official guide above is authoritative for field names and format; the skills "
        "add the craft and the working rules of this node. Before writing, use the Read tool "
        "on the reference files whose table row matches this task. For the first prompt in "
        "a session that means, at minimum, the prompt-grammar file and the gold-example file "
        "for this task type; open the style files only when a style, medium or reference look "
        "is in play. Read, then write; do not describe what you read."
    )


def _skill_references(name):
    """[(file_name, text)] for every reference .md of one skill, in table order."""
    references = os.path.join(SKILLS_ROOT, name, "references")
    if not os.path.isdir(references):
        return []
    out = []
    for file_name in sorted(os.listdir(references)):
        if not file_name.endswith(".md"):
            continue
        try:
            with open(os.path.join(references, file_name), encoding="utf-8") as handle:
                out.append((file_name, handle.read().strip()))
        except OSError:
            continue
    return out


# What a file-tool-less backend actually needs pasted in, per skill: the prompt
# grammar, the gold examples that teach the format, and the style anchors. The
# rest of each library is catalogue material - style pickers (the node has
# already picked the style), mode/ComfyUI docs, a clean copy of the official
# guide that is already in the system prompt - dead weight at write time that
# costs ~30k extra tokens per run, and on Ollama gets re-prefilled every turn.
_ESSENTIAL_REFERENCES = {
    "h3-prompt-director": ("03_H3_PROMPT_GRAMMAR.md",),
    "h3-base-format": ("11_T2VA_GOLD_EXAMPLES.md",),
    "h3-ref2va": ("13_REF2VA_GOLD_EXAMPLES.md",),
    "h3-style-craft": ("09_H3_STYLE_REFERENCE_ANCHORS.md",),
    "h3-crossover": ("17_CROSSOVER_GOLD_EXAMPLES.md",),
}


def reference_mode(value):
    """
    'off' | 'essential' | 'full' from the H3 LLM Backend widget - or from the
    legacy BOOLEAN it used to be (workflows saved before the combo existed
    restore true/false): False stays off, True now means essentials.
    """
    v = str(value or "").strip().lower()
    if v.startswith("full"):
        return "full"
    if v in ("true", "1", "on") or v.startswith("essent"):
        return "essential"
    return "off"


def director_block_inline(skills, include_references=False):
    """
    The director skills for a backend with no file tools (Ollama, LM Studio, an
    API model): SKILL.md bodies are pasted in, and reference files follow
    `include_references` (any reference_mode() value) - 'off' none,
    'essential' each skill's grammar / gold examples / style anchors
    (~15-20k tokens), 'full' the whole library (~45k tokens).
    """
    mode = reference_mode(include_references)
    loaded = [(name, load_skill(name)) for name in skills]
    loaded = [(name, text) for name, text in loaded if text]
    if not loaded:
        return ""
    parts = []
    included = []
    for name, text in loaded:
        parts.append(f"=== BEGIN SKILL {name} ===\n{text}\n=== END SKILL {name} ===")
        if mode == "off":
            continue
        keep = _ESSENTIAL_REFERENCES.get(name) if mode == "essential" else None
        for file_name, body in _skill_references(name):
            if keep is not None and file_name not in keep:
                continue
            parts.append(
                f"=== BEGIN REFERENCE {name}/{file_name} ===\n{body}\n"
                f"=== END REFERENCE {name}/{file_name} ==="
            )
            included.append(f"{name}/{file_name}")
    if mode == "full":
        note = (
            "The reference files the skills mention are included above in full; use them "
            "as the gold standard for grammar, examples and style."
        )
    elif included:
        note = (
            "The core of each skill's reference library is included above (prompt grammar, "
            "gold examples, style anchors); any other file a skill mentions is NOT available "
            "here - rely on the official guide and the skill rules instead."
        )
    else:
        note = (
            "You have no file tools here: the reference files the skills mention are NOT "
            "available, so rely on the official guide and the skill rules above."
        )
    if mode != "off":
        est = int(sum(len(p) for p in parts) / _CHARS_PER_TOKEN)
        print(
            f"📚 H3 director: {len(loaded)} skill(s) + {len(included)} reference file(s) "
            f"inlined ({mode}) ~{est:,} tokens"
        )
    return (
        "\n\n" + "\n\n".join(parts) + "\n\n"
        "The official guide above is authoritative for field names and format; the skills "
        f"add the craft and the working rules of this node. {note} Write the prompt; do not "
        "describe the rules."
    )


# ---------------------------------------------------------------------------
# Backend selection: Claude Code CLI (default) or anything the LLM router knows
# ---------------------------------------------------------------------------

def list_codex_models():
    """`codex` entries for the dropdowns, only when the Codex CLI is installed."""
    return ["codex"] if codex_available() else []


def is_codex_model(model):
    """
    True when `model` names the OpenAI Codex CLI: the bare alias `codex` (the
    model ~/.codex/config.toml picks) or `codex:<model-id>` for a specific one
    (e.g. `codex:gpt-5.3-codex`).
    """
    model = (model or "").strip().lower()
    return model == "codex" or model.startswith("codex:")


def _model_choices():
    """CLI aliases first, then every local server model answering right now."""
    return list(CLAUDE_CODE_MODELS) + list_codex_models() + list_local_models()


def is_router_model(model):
    """
    True when `model` should go through the LLM router instead of an agent CLI:
    any `provider:model` string (ollama:, lmstudio:, local:, claude:, gpt:,
    gemini:, ...) that is not the claudecode provider or the codex CLI.
    """
    model = (model or "").strip()
    if is_codex_model(model):
        return False
    if ":" not in model:
        return False
    provider, _ = split_model(model)
    return provider != CLAUDE_CODE_PROVIDER


def resolve_backend_model(model, model_override=""):
    """The effective model string: the override box wins over the dropdown."""
    override = (model_override or "").strip()
    if override:
        return override
    model = (model or "").strip()
    if model.startswith(f"{CLAUDE_CODE_PROVIDER}:"):
        return model.split(":", 1)[1]
    return model


LLM_SOCKET_TYPE = "APNEXT_LLM"


def local_llm_options(llm=None):
    """
    The backend settings an H3 node passes through to `run_h3_claude_code`.

    `llm` is the dict an `APNext H3 LLM Backend` node puts on its output (model,
    base_url, temperature, inline_references, max_tokens, and the Ollama-only
    num_ctx / think). None = nothing connected, so the node's own model dropdown
    decides.
    """
    llm = dict(llm or {})
    return {
        "model_override": (llm.get("model") or "").strip(),
        "base_url": (llm.get("base_url") or "").strip(),
        "temperature": float(llm.get("temperature", 1.0) or 0.0),
        "inline_references": reference_mode(llm.get("inline_references", False)),
        "max_tokens": int(llm.get("max_tokens", 8000) or 8000),
        "num_ctx": int(llm.get("num_ctx", 0) or 0),
        "think": llm.get("think"),
        "structured": str(llm.get("structured") or "auto"),
        "unload_after": bool(llm.get("unload_after", True)),
        "api_key": (llm.get("api_key") or "").strip(),
    }


def local_llm_inputs():
    """
    The optional `llm` socket every H3 Claude Code node carries: connect an
    `APNext H3 LLM Backend` node to write with Ollama / LM Studio / any local or
    API model instead of the Claude Code CLI. Leave it unconnected and the
    node's `model` dropdown is used as before.
    """
    return {
        "llm": (LLM_SOCKET_TYPE, {
            "tooltip": (
                "Optional. Connect an APNext H3 LLM Backend node to write with Ollama, LM "
                "Studio, another OpenAI-compatible server or an API model instead of Claude "
                "Code. Overrides the model dropdown while connected."
            ),
        }),
    }


# Text-only conversation memory for backends that have no sessions of their own,
# so resume_session_id / the Refiner / the Continue Writer keep working on Ollama.
# One JSON file per session under the temp folder; survives a ComfyUI restart.

_SESSION_PREFIX = "local-"
_CODEX_SESSION_PREFIX = "codex-"


def _session_dir():
    try:
        import folder_paths  # ComfyUI
        base = folder_paths.get_temp_directory()
    except Exception:
        base = tempfile.gettempdir()
    path = os.path.join(base, "apnext_h3_sessions")
    os.makedirs(path, exist_ok=True)
    return path


def _session_path(session_id):
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in "-_")
    return os.path.join(_session_dir(), f"{safe}.json")


def load_local_session(session_id):
    session_id = (session_id or "").strip()
    if not session_id:
        return None
    try:
        with open(_session_path(session_id), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def save_local_session(session_id, data):
    try:
        with open(_session_path(session_id), "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
    except OSError as exc:
        print(f"⚠️ could not save H3 local session {session_id}: {exc}")


def release_local_llm(local):
    """
    A writer is done with its model: if the backend asked for it and the model
    is an Ollama one, have Ollama unload it now so the VRAM is free for the
    render. Safe to call with None or after a failed run.
    """
    local = local or {}
    model = str(local.get("model_override") or "").strip()
    if not local.get("unload_after") or not model.lower().startswith("ollama:"):
        return False
    from ...utils.llm_router import unload_ollama_model
    ok = unload_ollama_model(model, local.get("base_url") or None)
    if ok:
        print(f"\U0001F9F9 H3: '{model}' unloaded from Ollama - VRAM handed back to the render.")
    return ok


def report_node_error(node_id, message):
    """
    Put a failed run's reason ON the node: web/js/h3_node_error.js listens for
    this event and draws the text as a red banner under the node, so the user
    reads WHY (context overflow, dead backend, timeout) without digging through
    the console. Display only - the caller still raises to stop the graph.
    Safe with node_id None (headless / API runs) and without a PromptServer.
    """
    if not node_id or not str(message or "").strip():
        return
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(
            "apnext.h3.node_error", {"node": str(node_id), "text": str(message)}
        )
    except Exception:
        pass


def _structured_plan(model, local, structured):
    """
    (schema_or_None, ask_for_json) for this call. `structured` names what the
    caller expects back ("scenes"); the backend's `structured_output` mode says
    how far to go: auto = enforce with a schema on Ollama only, on = ask every
    backend (enforced where possible), off = never.
    """
    if not structured:
        return None, False
    mode = str((local or {}).get("structured") or "auto").lower()
    if mode.startswith("off"):
        return None, False
    from .scenes_support import SCENES_JSON_SCHEMA
    on_ollama = str(model or "").lower().startswith("ollama:")
    if on_ollama:
        return SCENES_JSON_SCHEMA, True
    return None, mode.startswith("on")


_CHARS_PER_TOKEN = 3.6          # English prose + tags on Qwen / Llama tokenizers
_IMAGE_TOKENS = 1500            # a downscaled reference picture on a vision model


def _warn_prompt_budget(system_prompt, user_prompt, images, local):
    """
    Say so BEFORE the call when the prompt is about to eat the context. Ollama
    does not refuse an oversized turn - it truncates the input and then cuts the
    reply off wherever the window ends (see llm_router._check_ollama_cutoff).
    """
    num_ctx = int((local or {}).get("num_ctx") or 0)
    if not num_ctx:
        return
    est = int((len(system_prompt or "") + len(user_prompt or "")) / _CHARS_PER_TOKEN) + _IMAGE_TOKENS * len(images or [])
    want = int((local or {}).get("max_tokens") or 8000)
    if est + min(want, 4000) > num_ctx:
        bigger = "65536" if est + 6000 <= 65536 else "131072"
        print(
            f"\u26a0\ufe0f H3: this prompt is ~{est:,} tokens against num_ctx {num_ctx:,} - the reply "
            f"has room for ~{max(0, num_ctx - est):,} tokens and a scene chunk needs 3-4k. Raise num_ctx "
            f"on the H3 LLM Backend node ({bigger}) or shorten the prompt."
        )


def _run_h3_router(system_prompt, user_prompt, images, model, resume_session_id,
                   director, skills, local, structured=None):
    """One H3 turn through the LLM router (Ollama, LM Studio, local, API models)."""
    resume_session_id = (resume_session_id or "").strip()
    session = load_local_session(resume_session_id) if resume_session_id else None
    if resume_session_id and session is None:
        if resume_session_id.startswith(_SESSION_PREFIX):
            raise ValueError(
                f"H3 local session '{resume_session_id}' was not found (temp folder cleared?). "
                "Run the writer again and feed its new session_id."
            )
        raise ValueError(
            f"resume_session_id '{resume_session_id[:12]}…' is a Claude Code session; it cannot "
            f"be continued with '{model}'. Use a Claude Code model, or clear resume_session_id."
        )

    # Web research is a Claude Code tool; strip the directive so a local model
    # does not claim to have searched anything.
    if RESEARCH_DIRECTIVE in (user_prompt or ""):
        user_prompt = user_prompt.replace(RESEARCH_DIRECTIVE, "").strip()

    if session:
        system_prompt = session.get("system") or None
        full_history = list(session.get("messages") or [])
        history = full_history
        # Local sessions replay the whole conversation every turn, so a long
        # serial run grows linearly until num_ctx bursts. A writer that carries
        # its own recap (the Music Video Writer's story-so-far gists) sets
        # local["history_window"] = N: the FIRST exchange (the brief and its
        # synopsis reply) is always replayed, then as many of the most recent
        # exchanges as the context budget affords - never fewer than N - so a
        # short song still sees everything verbatim and a long one degrades to
        # the recap only for its oldest chunks. The full transcript is still
        # saved to disk either way.
        window = int(local.get("history_window") or 0)
        if window > 0 and len(full_history) > 2:
            def est(text):
                return len(text or "") // 3 + 1   # ~3 chars/token, deliberately conservative

            num_ctx = int(local.get("num_ctx") or 0)
            if num_ctx > 0:
                # the 3500 tail: reply headroom slack plus room for the running recap pair
                budget = num_ctx - est(system_prompt) - est(user_prompt) - int(local.get("max_tokens", 8000)) - 3500
            else:
                budget = 32_000                    # no declared window: assume a roomy model
            pairs = [full_history[i:i + 2] for i in range(2, len(full_history), 2)]
            used = est(full_history[0].get("content")) + est(full_history[1].get("content"))
            keep = []
            for pair in reversed(pairs):           # newest first; the tail must stay contiguous
                cost = sum(est(m.get("content")) for m in pair)
                if len(keep) < window or used + cost <= budget:
                    keep.append(pair)
                    used += cost
                else:
                    break
            if len(keep) < len(pairs):
                # Chunks that fall out of the window are not just dropped: they
                # are folded into a running RECAP by one small summarisation
                # call (dense continuity facts - who, where, wardrobe, how each
                # scene ended), stored in the session and updated incrementally.
                tail_start = 2 + 2 * (len(pairs) - len(keep))
                summary = str(session.get("rolled_summary") or "")
                upto = max(2, int(session.get("summary_upto") or 2))
                if upto < tail_start:
                    dropped = full_history[upto:tail_start]
                    body = "\n\n".join(
                        m.get("content") or "" for m in dropped if m.get("role") == "assistant"
                    )
                    try:
                        summary, _model = call_llm(
                            model,
                            (
                                "Update the running recap of a music video's written scenes.\n\n"
                                f"RECAP SO FAR:\n{summary or '(none yet)'}\n\n"
                                f"NEW SCENES TO FOLD IN:\n{body}\n\n"
                                "Rewrite the recap to cover everything so far. One dense line per "
                                "scene: scene number, location, who is on screen with their (Sx) "
                                "ids, wardrobe and prop state, the key action, and the exact state "
                                "the scene ends on. Keep every continuity-relevant fact; drop "
                                "camera moves and style prose. Output only the recap."
                            ),
                            temperature=0.2,
                            max_tokens=1200,
                            base_url=local.get("base_url") or None,
                            num_ctx=local.get("num_ctx", 0),
                            think=local.get("think"),
                            api_key=local.get("api_key") or None,
                        )
                        summary = (summary or "").strip()
                        upto = tail_start
                        print(
                            f"📝 H3 local session: {len(dropped) // 2} older turn(s) folded into "
                            f"the running recap ({len(summary)} chars)."
                        )
                    except Exception as exc:
                        print(
                            f"⚠️ H3 local session: recap update failed ({exc}) - those turns "
                            "reach the writer only as the story-so-far gists."
                        )
                session["rolled_summary"], session["summary_upto"] = summary, upto
                bridge = []
                if summary:
                    bridge = [
                        {"role": "user", "content": "Recap the scenes you have written since the synopsis, before we continue."},
                        {"role": "assistant", "content": summary},
                    ]
                history = full_history[:2] + bridge + [m for pair in reversed(keep) for m in pair]
                print(
                    f"✂️ H3 local session: replaying first + last {len(keep)} of {len(pairs) + 1} "
                    f"turns (~{used:,} est. tokens fit the context budget)"
                    + (" + the running recap of the rest" if summary else "")
                    + "; the full transcript stays in the session file."
                )
    else:
        full_history = history = []
        if director and system_prompt:
            system_prompt = system_prompt + director_block_inline(
                skills, include_references=local.get("inline_references", False)
            )

    schema, ask_json = _structured_plan(model, local, structured)
    if ask_json:
        from .scenes_support import scenes_json_instruction
        user_prompt = (user_prompt or "") + scenes_json_instruction()

    _warn_prompt_budget(system_prompt, user_prompt, images, local)
    started = time.monotonic()
    text, resolved = call_llm(
        model,
        user_prompt,
        system_prompt=system_prompt,
        images=images or None,
        temperature=local.get("temperature", 1.0),
        max_tokens=local.get("max_tokens", 8000),
        base_url=local.get("base_url") or None,
        history=history,
        num_ctx=local.get("num_ctx", 0),
        think=local.get("think"),
        format_schema=schema,
        api_key=local.get("api_key") or None,
    )
    duration = time.monotonic() - started

    session_id = resume_session_id or f"{_SESSION_PREFIX}{uuid.uuid4().hex[:16]}"
    full_history.append({"role": "user", "content": user_prompt})
    full_history.append({"role": "assistant", "content": text})
    save_local_session(session_id, {
        "model": resolved,
        "system": system_prompt,
        "messages": full_history,
        "updated": time.time(),
        # the incrementally-updated recap of turns that no longer fit the window
        **({"rolled_summary": session["rolled_summary"], "summary_upto": session["summary_upto"]}
           if session and session.get("rolled_summary") is not None else {}),
    })

    info = (
        f"model={resolved} | {duration:.1f}s | turns={len(full_history) // 2} | "
        f"session={session_id}"
    )
    print(f"✅ H3 via {resolved} | {info}")
    return text, session_id, info


def _run_h3_codex(system_prompt, user_prompt, images, model, research,
                  use_subscription, timeout_seconds, resume_session_id,
                  working_dir, director, skills):
    """One H3 turn through the OpenAI Codex CLI (`codex exec`)."""
    codex_model = model.split(":", 1)[1].strip() if ":" in (model or "") else ""
    resume_session_id = (resume_session_id or "").strip()
    if resume_session_id.startswith(_SESSION_PREFIX):
        raise ValueError(
            f"resume_session_id '{resume_session_id}' is a local-model session; it cannot "
            f"be continued with Codex. Pick the same ollama:/local: model or clear it."
        )
    if resume_session_id and not resume_session_id.startswith(_CODEX_SESSION_PREFIX):
        raise ValueError(
            f"resume_session_id '{resume_session_id[:12]}…' is a Claude Code session; it "
            f"cannot be continued with Codex. Use a Claude Code model, or clear "
            "resume_session_id."
        )
    thread_id = resume_session_id[len(_CODEX_SESSION_PREFIX):] if resume_session_id else None

    # Codex reads files with its own shell (read-only sandbox), so the skills'
    # reference tables with absolute paths work the same way they do for Claude.
    if director and system_prompt and not thread_id:
        system_prompt = system_prompt + director_block(skills)

    result = run_codex(
        user_prompt,
        system_prompt=None if thread_id else system_prompt,
        images=images or None,
        model=codex_model or None,
        timeout=timeout_seconds,
        research=research,
        working_dir=(working_dir or "").strip() or None,
        resume_session_id=thread_id,
        use_subscription=use_subscription,
        on_progress=lambda note: print(f"   ↳ {note}"),
    )

    session_id = (
        f"{_CODEX_SESSION_PREFIX}{result['session_id']}" if result["session_id"] else ""
    )
    info = (
        f"model=codex:{codex_model or 'default'} | {result['duration_ms'] / 1000:.1f}s | "
        f"turns={result['num_turns']} | session={session_id}"
    )
    print(f"✅ H3 via Codex | {info}")
    return result["text"], session_id, info


def claude_code_inputs():
    """The Claude Code widget block, shared by every H3 node that uses the CLI."""
    return {
        "model": (_model_choices(), {
            "default": "sonnet",
            "tooltip": (
                "Who writes the prompt. sonnet / opus / haiku / fable / default are Claude "
                "Code aliases (`default` = whatever the CLI is configured for). `codex` is "
                "the OpenAI Codex CLI with its configured model (shown when installed; "
                "`codex:<model-id>` in an H3 LLM Backend picks a specific one). ollama: / "
                "lmstudio: / local: entries are whatever your local servers were serving when "
                "the page loaded; pick one to run fully offline. Anything not listed goes in "
                "model_override."
            ),
        }),
        "research": ("BOOLEAN", {
            "default": False,
            "tooltip": (
                "Let the agent CLI (Claude Code or Codex) search the web for real references "
                "before writing - the actual location, wardrobe, lighting and physics. "
                "Slower, and it reaches the internet."
            ),
        }),
        "director": ("BOOLEAN", {
            "default": True,
            "tooltip": (
                "Load the H3 director skills (data/h3/skills): the core writing rules, the "
                "format this node emits, and style/motion craft, each with a reference library "
                "of gold examples and style anchors that Claude Code reads on demand. Costs a "
                "few extra seconds and tokens per run."
            ),
        }),
        "use_subscription": ("BOOLEAN", {
            "default": True,
            "tooltip": (
                "Hide the API key from the CLI so it uses your login and subscription seat "
                "(ANTHROPIC_API_KEY for Claude Code, OPENAI_API_KEY for Codex). Turn off to "
                "bill the API key instead."
            ),
        }),
        "timeout_seconds": ("INT", {
            "default": 1200, "min": 60, "max": 7200, "step": 30,
            "tooltip": (
                "How long to wait PER CALL before the node gives up on the CLI (this is the "
                "node's own watchdog, not a Claude limit). Single H3 prompts take 25-60s; a "
                "multi-scene chunk with director/research on can take 10-20 minutes. The "
                "multi-scene writers retry a timed-out chunk at half size automatically."
            ),
        }),
    }


def claude_code_optional_inputs():
    """Optional Claude Code widgets: session continuity and local context."""
    return {
        "resume_session_id": ("STRING", {
            "default": "",
            "tooltip": (
                "Continue an earlier run by feeding it that node's session_id. The whole "
                "conversation, images included, is still in context. A session sticks to "
                "its backend: Claude Code ids resume with Claude Code, `codex-` ids with "
                "Codex, `local-` ids with the same local model."
            ),
        }),
        "working_dir": ("STRING", {
            "default": "",
            "tooltip": (
                "A folder Claude Code may read while writing - a script, a shot list, lookbook "
                "notes. Empty uses a throwaway scratch folder, which is the safe default."
            ),
        }),
    }


SAME_AS_MODEL = "same as model"


def draft_model_input():
    """
    The `draft_model` widget for writers that split a run into a plan and scene
    chunks: the main `model` plans the video and repairs continuity, while this
    (usually cheaper, faster) model drafts the scene chunks. Output speed is the
    bottleneck of a long run, so drafting with haiku cuts wall-clock time hard.
    """
    return {
        "draft_model": ([SAME_AS_MODEL] + list(CLAUDE_CODE_MODELS), {
            "default": "haiku",
            "tooltip": (
                "Who DRAFTS the scene chunks. The main `model` stays the director: it "
                "plans the video (synopsis, wardrobe/location locks, scene plan) and runs "
                "the continuity repair, while this model writes the scenes from that plan. "
                "haiku drafts several times faster than sonnet. `same as model` turns the "
                "split off. Ignored when the run is backed by Codex, a local server or an "
                "H3 LLM Backend override - those runs use one model throughout."
            ),
        }),
    }


def resolve_draft_model(draft_model, model, local):
    """
    The model that writes the scene chunks. The draft pick applies only when the
    run is a plain Claude Code run: a Codex / router / override backend keeps
    one model throughout (their sessions cannot be shared across backends).
    """
    # Workflows saved before this widget existed restore it as '' - fall back
    # to the widget's default rather than failing validation.
    draft = (draft_model or "").strip() or "haiku"
    if draft == SAME_AS_MODEL:
        return model
    effective = resolve_backend_model(model, (local or {}).get("model_override", ""))
    if ((local or {}).get("model_override") or "").strip() or is_codex_model(effective) \
            or is_router_model(effective):
        print(
            f"ℹ️ H3: draft_model '{draft}' ignored - this run is backed by "
            f"'{effective}', which writes every chunk itself."
        )
        return model
    return draft


# ---------------------------------------------------------------------------
# Project names - a memorable cinematography-flavoured tag for a run, so every
# file the run produces (videos, scene bundles) is visibly from the same
# project. The front-end (web/js/h3_project_name.js) fills the widget with one
# when a node is created and swaps in a new one whenever the seed changes;
# keep the word pools there identical (same words, same order) when editing
# these, since both sides derive the name from the seed the same way.
# ---------------------------------------------------------------------------

_PROJECT_LOOKS = (
    "Golden", "Silver", "Amber", "Noir", "Neon", "Velvet", "Crimson", "Cobalt",
    "Sepia", "Chrome", "Indigo", "Emerald", "Scarlet", "Midnight", "Pastel", "Tungsten",
    "Halide", "Matte", "Anamorphic", "Technicolor", "Ivory", "Onyx", "Coral", "Saffron",
    "Teal", "Mauve", "Ochre", "Umber", "Cyan", "Magenta", "Bronze", "Copper",
    "Ruby", "Sapphire", "Jade", "Opal", "Smoke", "Kodachrome", "Backlit", "Grainy",
)
_PROJECT_MOVES = (
    "Dolly", "Crane", "Zoom", "Orbit", "Boom", "Gimbal", "Rack", "Whip",
    "Glide", "Pan", "Tilt", "Push", "Steadicam", "Tracking", "Vertigo", "Truck",
    "Pedestal", "Arc", "Swish", "Jib", "Handheld", "Drone", "Slider", "Sweep",
    "Pullback", "Drift", "Roll", "Hover", "Snap", "Float",
)
_PROJECT_GEAR = (
    "Slate", "Reel", "Lens", "Shutter", "Bokeh", "Gaffer", "Grip", "Rig",
    "Flare", "Foley", "Scrim", "Frame", "Take", "Clapper", "Montage", "Tripod",
    "Monitor", "Diffuser", "Reflector", "Softbox", "Gel", "Sandbag", "Cstand", "Cable",
    "Marker", "Viewfinder", "Tape", "Cutter", "Barn", "Dailies",
)


_MASK64 = (1 << 64) - 1


def _splitmix64(seed):
    """Deterministic 64-bit stream (splitmix64). Mirrored in
    web/js/h3_project_name.js so a seed names the project identically in the
    UI and in headless/API runs."""
    state = seed & _MASK64
    while True:
        state = (state + 0x9E3779B97F4A7C15) & _MASK64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        yield z ^ (z >> 31)


_TAIL_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
_TAIL_LEN = 4                                   # 36^4 = 1.68 million tails


def _tail(value):
    """4 base-36 characters from a 64-bit value - the part of a project name
    that keeps two seeds apart even when they draw the same three words."""
    n = int(value) % (36 ** _TAIL_LEN)
    out = ""
    for _ in range(_TAIL_LEN):
        out = _TAIL_ALPHABET[n % 36] + out
        n //= 36
    return out


def generate_project_name(seed=None):
    """A name like 'NeonDollyFoley-7k3q': three words from the pools (40 x 30 x
    30) plus a 4-character base-36 tail, ~60 billion combinations. A fixed seed
    (>= 0) always yields the same name - the same one the front-end shows for
    that seed - so seeded re-runs keep their project tag; anything else is random."""
    import random
    if isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0:
        stream = _splitmix64(seed)
        words = "".join(pool[next(stream) % len(pool)]
                        for pool in (_PROJECT_LOOKS, _PROJECT_MOVES, _PROJECT_GEAR))
        return f"{words}-{_tail(next(stream))}"
    rng = random.Random()
    words = rng.choice(_PROJECT_LOOKS) + rng.choice(_PROJECT_MOVES) + rng.choice(_PROJECT_GEAR)
    return f"{words}-{_tail(rng.getrandbits(64))}"


def project_name_input():
    return {
        "project_name": ("STRING", {
            "default": "",
            "tooltip": (
                "A tag for this run - auto-filled with a random name like "
                "'NeonDollyFoley-7k3q' when the node is created; type your own to rename "
                "the project. Wire the node's `project_name` output into Save Video's "
                "`filename_prefix` and every clip of the run lands in its own subfolder "
                "(output/video/<name>/), so the output folder shows at a glance which "
                "videos belong together. Saved "
                "scene bundles carry it too. Empty = a fresh random name each run "
                "(stable when `seed` is fixed)."
            ),
        }),
    }


def resolve_project_name(project_name, seed=-1):
    """The user's name, cleaned for filenames; empty generates one from seed."""
    import re
    name = re.sub(r'[<>:"\\|?*\x00-\x1f]+', "", str(project_name or "")).strip()
    name = name.strip("/").strip()
    return name or generate_project_name(seed)


def project_name_prefix(name):
    """The writers' `project_name` OUTPUT. Wired into Save Video's
    `filename_prefix`, `video/<name>/<name>` puts every run in its own
    subfolder of the output directory; logs and saved bundles keep the plain
    name. A name the user typed with slashes already carries its own folders
    and is passed through untouched."""
    if not name:
        return ""
    return name if "/" in name else f"video/{name}/{name}"


def directions_with_research(extra_instructions, research):
    """Fold the research directive into the user's own extra direction."""
    text = (extra_instructions or "").strip()
    if not research:
        return text
    return f"{RESEARCH_DIRECTIVE}\n{text}" if text else RESEARCH_DIRECTIVE


def run_h3_claude_code(
    system_prompt,
    user_prompt,
    images,
    model,
    research,
    use_subscription,
    timeout_seconds,
    resume_session_id="",
    working_dir="",
    director=False,
    skills=BASE_SKILLS,
    local=None,
    structured=None,
):
    """
    One H3 turn through the CLI - or, when `model` / `local["model_override"]`
    names a router model such as `ollama:qwen3:8b`, through the LLM router with
    a text-only local session so resume still works. Returns (text, session_id,
    info).

    Resuming skips the system prompt: the guide is already in that session's
    context, and re-sending 40 KB of specification every time defeats the point.
    The skills' reference folders stay readable on resume, though, so a revision
    can still consult them. `skills` names which SKILL.md folders under
    data/h3/skills are loaded when `director` is on.
    """
    local = dict(local or {})
    model = resolve_backend_model(model, local.get("model_override", ""))
    if is_codex_model(model):
        return _run_h3_codex(
            system_prompt, user_prompt, images, model, research, use_subscription,
            timeout_seconds, resume_session_id, working_dir, director, skills,
        )
    if is_router_model(model):
        if research:
            print(f"ℹ️  research is a Claude Code feature; '{model}' writes without web research.")
        return _run_h3_router(
            system_prompt, user_prompt, images, model, resume_session_id, director, skills, local,
            structured=structured,
        )

    working_dir = (working_dir or "").strip()
    resume_session_id = (resume_session_id or "").strip()
    if resume_session_id.startswith(_SESSION_PREFIX):
        raise ValueError(
            f"resume_session_id '{resume_session_id}' is a local-model session; it cannot be "
            f"continued with Claude Code. Pick the same ollama:/local: model or clear it."
        )
    if resume_session_id.startswith(_CODEX_SESSION_PREFIX):
        raise ValueError(
            f"resume_session_id '{resume_session_id}' is a Codex session; it cannot be "
            f"continued with Claude Code. Pick a codex model or clear resume_session_id."
        )

    if research:
        tools = list(RESEARCH_TOOLS)
    elif working_dir or director:
        tools = list(_LOCAL_FILE_TOOLS)
    else:
        tools = None

    add_dirs = [SKILLS_ROOT] if director else None
    if director and system_prompt and not resume_session_id:
        system_prompt = system_prompt + director_block(skills)
    _schema, ask_json = _structured_plan(model, local, structured)
    if ask_json:   # 'on': the CLI cannot enforce a schema, but Claude writes JSON reliably
        from .scenes_support import scenes_json_instruction
        user_prompt = (user_prompt or "") + scenes_json_instruction()

    result = run_claude_code(
        user_prompt,
        system_prompt=None if resume_session_id else system_prompt,
        images=images or None,
        model=model,
        timeout=timeout_seconds,
        tools=tools,
        working_dir=working_dir or None,
        resume_session_id=resume_session_id or None,
        use_subscription=use_subscription,
        on_progress=lambda note: print(f"   ↳ {note}"),
        add_dirs=add_dirs,
    )

    info = (
        f"model={result['model']} | {result['duration_ms'] / 1000:.1f}s | "
        f"turns={result['num_turns']} | cost=${result['cost_usd']:.4f} | "
        f"session={result['session_id']}"
    )
    print(f"✅ H3 via Claude Code | {info}")

    return result["text"], result["session_id"], info
