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
from .claude_code_support import LLM_SOCKET_TYPE

_CUSTOM = "custom (use model_name below)"
_KNOWN_PREFIXES = set(LOCAL_PROVIDERS) | {"claude", "gpt", "gemini", "grok", "groq", CLAUDE_CODE_PROVIDER}


def _choices():
    local = list_local_models()
    return (
        local
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
                        "'gpt:gpt-5.6', 'gemini:gemini-3.7-flash'."
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

    def build(self, model, model_name, base_url, temperature, max_tokens, inline_skill_references):
        chosen = (model_name or "").strip() if model == _CUSTOM else (model or "").strip()
        if not chosen:
            raise ValueError("H3 LLM Backend: pick a model or fill in model_name.")
        if model == _CUSTOM:
            prefix = chosen.split(":", 1)[0].lower()
            if prefix not in _KNOWN_PREFIXES:
                # 'qwen3:8b' / 'llama3.1' are Ollama tags, not provider strings
                chosen = f"ollama:{chosen}"
        llm = {
            "model": chosen,
            "base_url": (base_url or "").strip(),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "inline_references": bool(inline_skill_references),
        }
        return (llm, chosen)
