# APNext Claude Code
#
# Runs a prompt through the locally installed Claude Code CLI, so a workflow can
# use the machine's existing Claude Code login (and its subscription seat)
# instead of an API key. Images are handed over as real files and read by the
# Read tool, which is how Claude Code takes vision input.
#
# Useful in front of the H3 writers: research or draft an idea here, wire the
# text into their `idea` or `extra_instructions`. The H3 writers can also target
# Claude Code directly by picking a `claudecode:` entry in their model dropdown.

from ...utils.apnext_context import (
    build_context,
    context_hidden_inputs,
    context_inputs,
    context_summary,
    with_context,
)
from ...utils.claude_code import (
    CLAUDE_CODE_MODELS,
    RESEARCH_TOOLS,
    ClaudeCodeError,
    find_cli,
    is_interrupt,
    run_claude_code,
)
from ...utils.constants import CUSTOM_CATEGORY
from ...utils.image_utils import tensor2pil

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise creative writing assistant working inside an image and video "
    "generation pipeline. Answer with the finished text only: no preamble, no "
    "commentary, no markdown fences, no offers to help further."
)


class ClaudeCodeNode:
    """
    APNext Claude Code

    Sends a prompt (and optional images) to the Claude Code CLI and returns its
    answer. Authentication comes from the CLI's own login, so no API key is
    needed in ComfyUI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "What to ask. Connect images below to have them described or used as reference.",
                }),
                "model": (CLAUDE_CODE_MODELS, {
                    "default": "sonnet",
                    "tooltip": "Claude Code model alias. `default` uses whatever the CLI is configured for.",
                }),
                "enable_research": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Let Claude Code search the web and read files while it answers "
                        "(WebSearch, WebFetch, Glob, Grep, Read). Slower, and it reaches the "
                        "internet. Off means it answers from the prompt alone."
                    ),
                }),
                "use_subscription": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Hide ANTHROPIC_API_KEY from the CLI so it uses your Claude Code login "
                        "and subscription seat. Turn off to bill the API key instead."
                    ),
                }),
                "timeout_seconds": ("INT", {
                    "default": 600, "min": 30, "max": 3600, "step": 30,
                    "tooltip": "How long to wait before giving up. Research runs take longer.",
                }),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 0xffffffffffffffff,
                    "tooltip": (
                        "Claude Code has no seed of its own - this only controls ComfyUI caching. "
                        "-1 re-runs every queue; any other value reuses the cached answer until "
                        "it changes."
                    ),
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Reference frame(s), sent inline to the CLI as vision input.",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_SYSTEM_PROMPT,
                    "tooltip": "Replaces Claude Code's own system prompt. Long text is moved into the message automatically.",
                }),
                "resume_session_id": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Continue an earlier run by feeding it that node's session_id output. "
                        "Lets a second node refine the first answer - 'make it wilder', "
                        "'shorten shot 2' - with the whole conversation still in context."
                    ),
                }),
                "working_dir": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Run inside this folder so Claude Code can read its files. Empty uses a "
                        "throwaway scratch folder, which is the safe default."
                    ),
                }),
                **context_inputs(),
            },
            "hidden": context_hidden_inputs(),
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "session_id", "info")
    FUNCTION = "run"
    CATEGORY = f"{CUSTOM_CATEGORY}/Claude Code"
    DESCRIPTION = (
        "Runs a prompt through the local Claude Code CLI using its own login. Takes "
        "images, can research the web, and can resume an earlier session to refine an answer."
    )

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        # seed -1 means "always re-run"; ComfyUI compares this against last run.
        return float("nan") if seed == -1 else seed

    def run(
        self,
        prompt,
        model,
        enable_research,
        use_subscription,
        timeout_seconds,
        seed,
        image=None,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        resume_session_id="",
        working_dir="",
        **context_slots,
    ):
        context_text, context_entries = build_context(context_slots, target="the answer")
        try:
            if not prompt.strip() and image is None and not context_entries:
                raise ValueError("Provide a prompt, an image, or both.")

            if not find_cli():
                raise ClaudeCodeError(
                    "The Claude Code CLI was not found. Install it from "
                    "https://claude.com/claude-code and run `claude` once to log in, or set "
                    "CLAUDE_CODE_PATH to the binary."
                )

            images = [tensor2pil(frame) for frame in image] if image is not None else None

            print(
                f"🤖 Claude Code | {model} | {len(images) if images else 0} image(s) | "
                f"research {'on' if enable_research else 'off'} | context: {context_summary(context_entries)}"
            )

            result = run_claude_code(
                with_context(prompt.strip(), context_text),
                system_prompt=system_prompt.strip() or None,
                images=images,
                model=model,
                timeout=timeout_seconds,
                tools=list(RESEARCH_TOOLS) if enable_research else None,
                working_dir=working_dir.strip() or None,
                resume_session_id=resume_session_id.strip() or None,
                use_subscription=use_subscription,
                on_progress=lambda note: print(f"   ↳ {note}"),
            )

            info = (
                f"model={result['model']} | {result['duration_ms'] / 1000:.1f}s | "
                f"turns={result['num_turns']} | cost=${result['cost_usd']:.4f} | "
                f"session={result['session_id']}"
            )
            print(f"✅ Claude Code | {info}")

            return (result["text"], result["session_id"], info)

        except Exception as exc:
            # A cancelled queue must stop the run, not become an error string.
            if is_interrupt(exc):
                raise
            print(f"❌ Claude Code node error: {exc}")
            import traceback

            print(traceback.format_exc())
            return (f"Error occurred while running Claude Code: {exc}", "", "error")
