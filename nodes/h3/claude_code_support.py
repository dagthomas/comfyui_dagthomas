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


def director_block_inline(skills, include_references=False):
    """
    The director skills for a backend with no file tools (Ollama, LM Studio, an
    API model): SKILL.md bodies are pasted in, and with `include_references` the
    reference library is pasted in too instead of being read on demand. The
    reference set is large (tens of thousands of tokens), so it is opt-in.
    """
    loaded = [(name, load_skill(name)) for name in skills]
    loaded = [(name, text) for name, text in loaded if text]
    if not loaded:
        return ""
    parts = []
    for name, text in loaded:
        parts.append(f"=== BEGIN SKILL {name} ===\n{text}\n=== END SKILL {name} ===")
        if include_references:
            for file_name, body in _skill_references(name):
                parts.append(
                    f"=== BEGIN REFERENCE {name}/{file_name} ===\n{body}\n"
                    f"=== END REFERENCE {name}/{file_name} ==="
                )
    note = (
        "The reference files the skills mention are included above in full; use them "
        "as the gold standard for grammar, examples and style."
        if include_references
        else "You have no file tools here: the reference files the skills mention are NOT "
        "available, so rely on the official guide and the skill rules above."
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
    base_url, temperature, inline_references, max_tokens). None = nothing
    connected, so the node's own model dropdown decides.
    """
    llm = dict(llm or {})
    return {
        "model_override": (llm.get("model") or "").strip(),
        "base_url": (llm.get("base_url") or "").strip(),
        "temperature": float(llm.get("temperature", 1.0) or 0.0),
        "inline_references": bool(llm.get("inline_references", False)),
        "max_tokens": int(llm.get("max_tokens", 8000) or 8000),
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


def _run_h3_router(system_prompt, user_prompt, images, model, resume_session_id,
                   director, skills, local):
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
        history = list(session.get("messages") or [])
    else:
        history = []
        if director and system_prompt:
            system_prompt = system_prompt + director_block_inline(
                skills, include_references=local.get("inline_references", False)
            )

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
    )
    duration = time.monotonic() - started

    session_id = resume_session_id or f"{_SESSION_PREFIX}{uuid.uuid4().hex[:16]}"
    history.append({"role": "user", "content": user_prompt})
    history.append({"role": "assistant", "content": text})
    save_local_session(session_id, {
        "model": resolved,
        "system": system_prompt,
        "messages": history,
        "updated": time.time(),
    })

    info = (
        f"model={resolved} | {duration:.1f}s | turns={len(history) // 2} | "
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
    draft = (draft_model or "").strip()
    if not draft or draft == SAME_AS_MODEL:
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
