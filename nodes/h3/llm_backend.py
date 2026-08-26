# APNext H3 LLM Backend
#
# One node that says "write with THIS model" for every H3 Claude Code node:
# Ollama, LM Studio, any OpenAI-compatible server, or a cloud API model. Drag
# its output into the `llm` socket of an H3 writer / refiner / continue /
# crossover / scenes node and that node stops using the Claude Code CLI and
# calls the chosen model through the shared LLM router instead. Session resume
# keeps working (a text-only local session is kept per run), research is off
# (a local model has no web tools), and the director skills are pasted into the
# system prompt since the model cannot Read files.

from ...utils.constants import CUSTOM_CATEGORY
from ...utils.llm_router import (
    AUTO_DETECT,
    CLAUDE_CODE_PROVIDER,
    LOCAL_PROVIDERS,
    claude_models,
    gemini_models,
    gpt_models,
    grok_models,
    groq_models,
    list_local_models,
)
from .claude_code_support import LLM_SOCKET_TYPE, list_codex_models

_CUSTOM = "custom (use model_name below)"

# Hybrid reasoning models (Qwen3, DeepSeek-R1, ...) think before answering.
# The thinking is dead weight here - the H3 format is spelled out in the system
# prompt - and on a local box it can triple the wall-clock time of a run.
_STRUCTURED = [
    "auto (JSON with a schema on Ollama, text envelopes elsewhere)",
    "on (ask every backend for JSON)",
    "off (text envelopes)",
]
_THINKING = ["off (faster - recommended)", "on", "model default"]
_KNOWN_PREFIXES = set(LOCAL_PROVIDERS) | {"claude", "gpt", "gemini", "grok", "groq", "codex", CLAUDE_CODE_PROVIDER}


def _choices():
    local = list_local_models()
    return (
        local
        + list_codex_models()
        + [_CUSTOM]
        + [f"claude:{m}" for m in claude_models]
        + [f"gpt:{m}" for m in gpt_models]
        + [f"gemini:{m}" for m in gemini_models]
        + [f"grok:{m}" for m in grok_models]
        + [f"groq:{m}" for m in groq_models]
        + [AUTO_DETECT]
    )


class H3LLMBackend:
    @classmethod
    def INPUT_TYPES(cls):
        choices = _choices()
        return {
            "required": {
                "model": (choices, {
                    "default": choices[0],
                    "tooltip": (
                        "ollama: / lmstudio: / local: entries are whatever your local servers "
                        "were serving when the page loaded (start the server, reload the page). "
                        "Cloud entries need their API key in the environment. Pick 'custom' to "
                        "type any provider:model string in model_name."
                    ),
                }),
                "model_name": ("STRING", {
                    "default": "ollama:qwen3:8b",
                    "tooltip": (
                        "Used when model = custom. Any router string: 'ollama:qwen3:8b', "
                        "'lmstudio:qwen/qwen3-8b', 'local:my-model', 'claude:claude-sonnet-5', "
                        "'gpt:gpt-5.6', 'gemini:gemini-3.7-flash' - or the Codex CLI: 'codex' "
                        "(its configured model) / 'codex:<model-id>' for a specific one."
                    ),
                }),
                "base_url": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Where to reach a local server, e.g. 'http://192.168.1.10:11434'. Empty "
                        "uses the default for the prefix: ollama 11434, lmstudio 1234, local 8000 "
                        "(or OLLAMA_BASE_URL / LMSTUDIO_BASE_URL / LOCAL_LLM_BASE_URL). Ignored by "
                        "cloud providers."
                    ),
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "max_tokens": ("INT", {
                    "default": 8000, "min": 256, "max": 128000, "step": 256,
                    "tooltip": "Answer length cap. Multi-scene crossover / scenes runs need room.",
                }),
                "inline_skill_references": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "With the H3 node's director on: paste the skills' full reference library "
                        "(gold examples, grammar, style anchors) into the system prompt, not just "
                        "the skill rules. Noticeably better prompts, but tens of thousands of tokens "
                        "- the model needs a big context window. Ollama: raise num_ctx "
                        "(OLLAMA_CONTEXT_LENGTH or a Modelfile), the official guide alone is ~12k tokens."
                    ),
                }),
                # appended LAST so saved workflows keep their widget positions
                "num_ctx": ("INT", {
                    "default": 32768, "min": 0, "max": 1048576, "step": 1024,
                    "tooltip": (
                        "Ollama only: the context window to load the model with. Ollama picks "
                        "its own default from free VRAM - as little as 4k - and an H3 system "
                        "prompt is 9-15k tokens on its own, so on a small default the writing "
                        "rules are silently cut off and the scenes come back unusable. 32k is "
                        "enough for a text-only run, 40k+ with reference images or "
                        "inline_skill_references. 0 = leave the server's default alone. "
                        "Ignored by every other provider (LM Studio, vLLM and the cloud APIs "
                        "set their context elsewhere)."
                    ),
                }),
                "thinking": (_THINKING, {
                    "default": _THINKING[0],
                    "tooltip": (
                        "Ollama only: whether a hybrid reasoning model (Qwen3, DeepSeek-R1, "
                        "gpt-oss, ...) reasons before answering. Off is much faster and the "
                        "H3 rules are already in the system prompt. Any <think> block that "
                        "does arrive is stripped before parsing either way. Models with no "
                        "thinking mode ignore this."
                    ),
                }),
                "unload_after": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Ollama only: once a writer has all its scenes, ask Ollama to drop the model "
                        "from VRAM right away instead of keeping it for its keep_alive window, so the "
                        "memory is back for the video render that follows. Off keeps it loaded (faster "
                        "if another writer runs next)."
                    ),
                }),
                "structured_output": (_STRUCTURED, {
                    "default": _STRUCTURED[0],
                    "tooltip": (
                        "How multi-scene answers travel back. auto: on Ollama the reply is JSON "
                        "constrained by a schema (valid by construction - no envelope to mis-write), "
                        "other backends keep the text envelopes. on: every backend is asked for JSON "
                        "(not enforced off Ollama; the text envelopes remain the fallback). off: text "
                        "envelopes everywhere. The scene text itself is unchanged either way."
                    ),
                }),
            },
        }

    RETURN_TYPES = (LLM_SOCKET_TYPE, "STRING")
    RETURN_NAMES = ("llm", "model_used")
    FUNCTION = "build"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Picks the model that writes for the H3 Claude Code nodes. Connect `llm` to an H3 "
        "writer's `llm` socket to use Ollama, LM Studio, a local OpenAI-compatible server or "
        "an API model instead of the Claude Code CLI."
    )

    def build(self, model, model_name, base_url, temperature, max_tokens,
              inline_skill_references, num_ctx=0, thinking=None, unload_after=True, structured_output=None):
        chosen = (model_name or "").strip() if model == _CUSTOM else (model or "").strip()
        if not chosen:
            raise ValueError("H3 LLM Backend: pick a model or fill in model_name.")
        if model == _CUSTOM:
            prefix = chosen.split(":", 1)[0].lower()
            if prefix not in _KNOWN_PREFIXES:
                # 'qwen3:8b' / 'llama3.1' are Ollama tags, not provider strings
                chosen = f"ollama:{chosen}"
        mode = str(thinking or _THINKING[0])
        llm = {
            "model": chosen,
            "base_url": (base_url or "").strip(),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "inline_references": bool(inline_skill_references),
            "num_ctx": int(num_ctx or 0),
            # None = say nothing and let the model do whatever it does by default
            "think": None if mode.startswith("model default") else mode.startswith("on"),
            "structured": str(structured_output or _STRUCTURED[0]).split(" ", 1)[0],
            "unload_after": True if unload_after is None else bool(unload_after),
        }
        return (llm, chosen)
