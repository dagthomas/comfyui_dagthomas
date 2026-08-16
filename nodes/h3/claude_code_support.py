# Shared pieces for the H3 nodes that talk to Claude Code directly.
#
# The generic H3 writers reach Claude Code through the model dropdown, which is
# enough to write a prompt but cannot express what the CLI is actually good at:
# researching real references before it writes, and keeping a session open so a
# later node can revise the prompt without re-sending it.
#
# These helpers carry the widgets and the call, so each node file stays focused
# on its own H3 format.

from ...utils.claude_code import CLAUDE_CODE_MODELS, RESEARCH_TOOLS, run_claude_code

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


def claude_code_inputs():
    """The Claude Code widget block, shared by every H3 node that uses the CLI."""
    return {
        "model": (CLAUDE_CODE_MODELS, {
            "default": "sonnet",
            "tooltip": "Claude Code model alias. `default` uses whatever the CLI is configured for.",
        }),
        "research": ("BOOLEAN", {
            "default": False,
            "tooltip": (
                "Let Claude Code search the web for real references before writing - the actual "
                "location, wardrobe, lighting and physics. Slower, and it reaches the internet."
            ),
        }),
        "use_subscription": ("BOOLEAN", {
            "default": True,
            "tooltip": (
                "Hide ANTHROPIC_API_KEY from the CLI so it uses your Claude Code login and "
                "subscription seat. Turn off to bill the API key instead."
            ),
        }),
        "timeout_seconds": ("INT", {
            "default": 900, "min": 60, "max": 3600, "step": 30,
            "tooltip": "How long to wait. H3 prompts take 25-60s; research runs take longer.",
        }),
    }


def claude_code_optional_inputs():
    """Optional Claude Code widgets: session continuity and local context."""
    return {
        "resume_session_id": ("STRING", {
            "default": "",
            "tooltip": (
                "Continue an earlier Claude Code run by feeding it that node's session_id. The "
                "whole conversation, images included, is still in context."
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
):
    """
    One H3 turn through the CLI. Returns (text, session_id, info).

    Resuming skips the system prompt: the guide is already in that session's
    context, and re-sending 40 KB of specification every time defeats the point.
    """
    working_dir = (working_dir or "").strip()
    resume_session_id = (resume_session_id or "").strip()

    if research:
        tools = list(RESEARCH_TOOLS)
    elif working_dir:
        tools = list(_LOCAL_FILE_TOOLS)
    else:
        tools = None

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
    )

    info = (
        f"model={result['model']} | {result['duration_ms'] / 1000:.1f}s | "
        f"turns={result['num_turns']} | cost=${result['cost_usd']:.4f} | "
        f"session={result['session_id']}"
    )
    print(f"✅ H3 via Claude Code | {info}")

    return result["text"], result["session_id"], info
