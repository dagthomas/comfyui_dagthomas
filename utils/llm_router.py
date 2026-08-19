# Shared multi-provider LLM router
#
# One entry point (`call_llm`) that talks to GPT, Gemini, Claude, Grok and Groq,
# plus any local OpenAI-compatible server (Ollama, LM Studio, vLLM, llama.cpp,
# LocalAI, ...), with an optional system prompt and optional images. Providers
# are selected by a "provider:model" string, the same convention the universal
# nodes already use.

import base64
import io
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

from .constants import (
    gpt_models,
    gemini_models,
    grok_models,
    claude_models,
    groq_models,
)

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

try:
    import httpx
except ImportError:
    httpx = None

from .gemini_client import GEMINI_AVAILABLE, get_gemini_client, gemini_generate
from .claude_code import (
    CLAUDE_CODE_MODELS,
    is_available as claude_code_available,
    is_interrupt,
    run_claude_code,
)

# Claude Code runs through the locally installed CLI and its own login, so it
# needs no API key here.
CLAUDE_CODE_PROVIDER = "claudecode"


# Provider prefix -> (env var names, OpenAI-compatible base url or None)
_OPENAI_COMPATIBLE = {
    "gpt": (("OPENAI_API_KEY",), None),
    "grok": (("XAI_API_KEY", "GROK_API_KEY"), "https://api.x.ai/v1"),
    "groq": (("GROQ_API_KEY",), "https://api.groq.com/openai/v1"),
}

# Local OpenAI-compatible servers: provider prefix -> (env var names, default base url).
# No API key is needed; the URL comes from the node, then the environment, then
# the default port each tool ships with. "local" is the catch-all for anything
# else that speaks the OpenAI API - vLLM, llama.cpp server, LocalAI, TabbyAPI,
# text-generation-webui, or a remote box on the LAN.
_LOCAL_PROVIDERS = {
    "ollama": (("OLLAMA_BASE_URL", "OLLAMA_HOST"), "http://localhost:11434/v1"),
    "lmstudio": (("LMSTUDIO_BASE_URL", "LM_STUDIO_BASE_URL"), "http://localhost:1234/v1"),
    "local": (("LOCAL_LLM_BASE_URL", "LOCAL_BASE_URL"), "http://localhost:8000/v1"),
}

# Preference order used by auto-detect, with the fallback model for each.
# A running local server is the last resort, after every cloud key.
_AUTO_DETECT_ORDER = (
    (("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"), "claude:claude-sonnet-5"),
    (("OPENAI_API_KEY",), "gpt:gpt-5.6"),
    (("GEMINI_API_KEY",), "gemini:gemini-3.7-flash"),
    (("XAI_API_KEY", "GROK_API_KEY"), "grok:grok-4.6"),
    (("GROQ_API_KEY",), "groq:llama-3.3-70b-versatile"),
)

AUTO_DETECT = "auto-detect"
LOCAL_PROVIDERS = tuple(_LOCAL_PROVIDERS)

# Clients are cached per process so repeated node runs reuse connections.
_client_cache = {}


def _first_env(names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


# ----------------------------------------------------------------------
# Local servers
# ----------------------------------------------------------------------

# A dead port does not always refuse the connection quickly - on Windows it is
# routinely swallowed and left to time out - so discovery TCP-probes first with a
# short budget and only speaks HTTP to a port that is actually open.
_LOCAL_PROBE_TIMEOUT = float(os.environ.get("APNEXT_LOCAL_LLM_PROBE_TIMEOUT", "0.5"))
_LOCAL_TIMEOUT = float(os.environ.get("APNEXT_LOCAL_LLM_TIMEOUT", "3.0"))

# Set APNEXT_LOCAL_LLM_DISCOVERY=0 to stop probing entirely. Local models can
# still be used, they just have to be typed in rather than picked from the list.
_LOCAL_DISCOVERY_ENABLED = os.environ.get("APNEXT_LOCAL_LLM_DISCOVERY", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

# ComfyUI rebuilds every dropdown each time the browser asks for /object_info,
# so the discovered list is cached briefly rather than re-probed per call.
_LOCAL_CACHE_TTL = 60.0
_local_models_cache = {"stamp": None, "models": []}


def _normalise_base_url(url):
    """
    Accept the shapes people actually paste: `localhost:11434`, a bare host with
    no path, or a full `.../v1` endpoint. A URL that already has a path is left
    alone, since some servers mount the OpenAI API somewhere other than /v1.
    """
    url = (url or "").strip().rstrip("/")
    if not url:
        return url

    if "://" not in url:
        url = f"http://{url}"
    if urlsplit(url).path in ("", "/"):
        url = f"{url}/v1"

    return url


def resolve_base_url(provider, override=None):
    """Base URL for a local provider: node override, then env var, then default."""
    env_names, default_url = _LOCAL_PROVIDERS[provider]
    return _normalise_base_url(override or _first_env(env_names) or default_url)


def _get_json(url, timeout):
    """GET a small JSON document, or None if anything at all goes wrong."""
    try:
        if httpx is not None:
            response = httpx.get(url, timeout=timeout)
        else:
            import requests

            response = requests.get(url, timeout=timeout)

        if response.status_code != 200:
            return None
        return response.json()
    except Exception:
        return None


def _port_is_open(url, timeout=_LOCAL_PROBE_TIMEOUT):
    """Is anything listening at all? Keeps a dead server to one short timeout."""
    parts = urlsplit(url)
    try:
        # .port raises on a malformed authority, so it stays inside the guard.
        host, port = parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)
        if not host:
            return False
        with socket.create_connection((host, port), timeout):
            return True
    except (OSError, ValueError):
        return False


def discover_local_models(provider, base_url=None, timeout=_LOCAL_TIMEOUT):
    """Model ids one local server is serving right now. Empty when it is not up."""
    url = resolve_base_url(provider, base_url)
    if not _port_is_open(url):
        return []

    payload = _get_json(f"{url}/models", timeout)
    if not isinstance(payload, dict):
        return []

    ids = [
        entry.get("id")
        for entry in payload.get("data", [])
        if isinstance(entry, dict) and entry.get("id")
    ]
    return sorted(ids)


def list_local_models(refresh=False):
    """
    Every `provider:model` string for local servers that answer right now.

    Start a server (or pull a new model), refresh the ComfyUI page, and the
    models show up in the dropdown - no ComfyUI restart needed.
    """
    if not _LOCAL_DISCOVERY_ENABLED:
        return []

    stamp = _local_models_cache["stamp"]
    now = time.monotonic()
    if not refresh and stamp is not None and now - stamp < _LOCAL_CACHE_TTL:
        return list(_local_models_cache["models"])

    # Probed concurrently, so three sleeping servers cost one timeout, not three.
    providers = list(_LOCAL_PROVIDERS)
    with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        per_provider = list(pool.map(discover_local_models, providers))

    found = [
        f"{provider}:{model}"
        for provider, models in zip(providers, per_provider)
        for model in models
    ]

    _local_models_cache["models"] = found
    _local_models_cache["stamp"] = now
    return list(found)


def list_claude_code_models():
    """`claudecode:` entries, but only when the CLI is actually installed."""
    if not claude_code_available():
        return []
    return [f"{CLAUDE_CODE_PROVIDER}:{m}" for m in CLAUDE_CODE_MODELS]


def list_all_models():
    """Every selectable model string, auto-detect first and local servers last."""
    return (
        [AUTO_DETECT]
        + [f"claude:{m}" for m in claude_models]
        + [f"gpt:{m}" for m in gpt_models]
        + [f"gemini:{m}" for m in gemini_models]
        + [f"grok:{m}" for m in grok_models]
        + [f"groq:{m}" for m in groq_models]
        + list_claude_code_models()
        + list_local_models()
    )


def auto_detect_model():
    """Pick the best model whose API key is present, else whatever runs locally."""
    for env_names, model in _AUTO_DETECT_ORDER:
        if _first_env(env_names):
            return model

    # A logged-in Claude Code CLI is a working credential too, and a cheaper one
    # for anyone on a subscription seat.
    if claude_code_available():
        return f"{CLAUDE_CODE_PROVIDER}:sonnet"

    local = list_local_models()
    if local:
        return local[0]

    raise ValueError(
        "No API keys found and no local server responded. Set one of: "
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY, GROQ_API_KEY - "
        "or start a local OpenAI-compatible server (Ollama, LM Studio, vLLM, "
        "llama.cpp) and pick an `ollama:` / `lmstudio:` / `local:` model."
    )


def resolve_model(model_name):
    """Turn 'auto-detect' into a concrete 'provider:model' string."""
    if not model_name or model_name == AUTO_DETECT:
        return auto_detect_model()
    return model_name


def split_model(model_name):
    """'gpt:gpt-4o' -> ('gpt', 'gpt-4o'). Unprefixed names default to gpt."""
    if ":" in model_name:
        provider, model = model_name.split(":", 1)
        return provider.strip().lower(), model.strip()
    return "gpt", model_name.strip()


def _http_client():
    if httpx is None:
        return None
    try:
        return httpx.Client(timeout=180.0)
    except TypeError:
        return httpx.Client()


def _get_openai_compatible_client(provider, base_url_override=None):
    if not OPENAI_AVAILABLE:
        raise ImportError("openai is not installed. Install it with: pip install 'openai<3'")

    if provider in _LOCAL_PROVIDERS:
        base_url = resolve_base_url(provider, base_url_override)
        # Local servers ignore the key, but the OpenAI client refuses to start without one.
        api_key = _first_env(("LOCAL_LLM_API_KEY",)) or "local"
    else:
        env_names, base_url = _OPENAI_COMPATIBLE[provider]
        api_key = _first_env(env_names)
        if not api_key:
            raise ValueError(f"{' or '.join(env_names)} environment variable not set")

    # The URL is part of the key so two nodes pointing at different local servers
    # do not share a client.
    cache_key = f"{provider}|{base_url or ''}"
    cached = _client_cache.get(cache_key)
    if cached is not None:
        return cached

    kwargs = {"api_key": api_key}
    http_client = _http_client()
    if http_client is not None:
        kwargs["http_client"] = http_client
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)
    _client_cache[cache_key] = client
    return client


def _get_claude_client():
    cached = _client_cache.get("claude")
    if cached is not None:
        return cached

    if not ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic is not installed. Install it with: pip install anthropic")

    api_key = _first_env(("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"))
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)
    _client_cache["claude"] = client
    return client


def _get_gemini_client():
    cached = _client_cache.get("gemini")
    if cached is not None:
        return cached

    client = get_gemini_client()
    _client_cache["gemini"] = client
    return client


def _encode_png(image):
    """PIL image -> raw PNG bytes."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _text_history(history):
    """Prior turns as plain text messages (images from earlier turns are not replayed)."""
    out = []
    for turn in history or []:
        role = turn.get("role")
        text = turn.get("content")
        if role in ("user", "assistant") and isinstance(text, str) and text:
            out.append({"role": role, "content": text})
    return out


def _history_as_text(history):
    """Prior turns flattened into a transcript, for providers without a messages array."""
    turns = _text_history(history)
    if not turns:
        return ""
    lines = ["Earlier in this conversation:"]
    for turn in turns:
        label = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"[{label}]\n{turn['content']}")
    lines.append("[User]\n")
    return "\n\n".join(lines)


def _call_openai_compatible(
    provider, model, user_prompt, system_prompt, images, temperature, seed, max_tokens,
    base_url=None, history=None,
):
    client = _get_openai_compatible_client(provider, base_url)

    if images:
        content = [{"type": "text", "text": user_prompt}]
        for image in images:
            encoded = base64.b64encode(_encode_png(image)).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
    else:
        content = user_prompt

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(_text_history(history))
    messages.append({"role": "user", "content": content})

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # Groq rejects the seed parameter on several models; everything else here
    # (GPT, Grok, Ollama, LM Studio, vLLM, llama.cpp) accepts it.
    if seed is not None and seed != -1 and provider != "groq":
        kwargs["seed"] = seed

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        if provider in _LOCAL_PROVIDERS:
            vision_hint = (
                " The model also has to be vision-capable to accept the attached image(s)."
                if images
                else ""
            )
            raise RuntimeError(
                f"Local LLM call to {resolve_base_url(provider, base_url)} failed: {exc}. "
                f"Check the server is running and that '{model}' is available on it."
                f"{vision_hint}"
            ) from exc
        raise

    return (response.choices[0].message.content or "").strip()


def _call_claude(model, user_prompt, system_prompt, images, temperature, max_tokens, history=None):
    client = _get_claude_client()

    if images:
        content = []
        for image in images:
            encoded = base64.b64encode(_encode_png(image)).decode("utf-8")
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": encoded,
                    },
                }
            )
        content.append({"type": "text", "text": user_prompt})
    else:
        content = user_prompt

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": _text_history(history) + [{"role": "user", "content": content}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)

    # Concatenate every text block; thinking-capable models can emit several.
    parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    return "".join(parts).strip()


def _call_gemini(model, user_prompt, system_prompt, images, temperature, max_tokens, history=None):
    client = _get_gemini_client()

    contents = [_history_as_text(history) + user_prompt]
    if images:
        contents.extend(images)

    response = gemini_generate(
        client,
        model,
        contents,
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    return (response.text or "").strip()


def call_llm(
    model_name,
    user_prompt,
    system_prompt=None,
    images=None,
    temperature=1.0,
    seed=-1,
    max_tokens=4000,
    base_url=None,
    history=None,
):
    """
    Send a prompt to whichever provider `model_name` selects.

    `history` is an optional list of earlier `{"role": "user"|"assistant",
    "content": str}` turns, replayed before `user_prompt` so a provider without
    sessions of its own can still continue a conversation.

    `images` is a list of PIL images; providers that support vision receive them
    inline. `base_url` overrides where a local provider (`ollama:`, `lmstudio:`,
    `local:`) is reached and is ignored for the cloud providers. Raises on
    failure so callers can decide how to surface the error.
    """
    resolved = resolve_model(model_name)
    provider, model = split_model(resolved)

    if provider in _OPENAI_COMPATIBLE or provider in _LOCAL_PROVIDERS:
        text = _call_openai_compatible(
            provider,
            model,
            user_prompt,
            system_prompt,
            images,
            temperature,
            seed,
            max_tokens,
            base_url=base_url,
            history=history,
        )
    elif provider == "claude":
        text = _call_claude(
            model, user_prompt, system_prompt, images, temperature, max_tokens, history=history
        )
    elif provider == "gemini":
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai is not installed. Install it with: pip install google-genai")
        text = _call_gemini(
            model, user_prompt, system_prompt, images, temperature, max_tokens, history=history
        )
    elif provider == CLAUDE_CODE_PROVIDER:
        # The CLI owns sampling, so temperature, seed and max_tokens do not apply.
        result = run_claude_code(
            _history_as_text(history) + user_prompt,
            system_prompt=system_prompt,
            images=images,
            model=model,
            timeout=int(os.environ.get("APNEXT_CLAUDE_CODE_TIMEOUT", "600")),
            on_progress=lambda note: print(f"   ↳ {note}"),
        )
        print(
            f"🤖 Claude Code | {model} | {result['duration_ms'] / 1000:.1f}s | "
            f"${result['cost_usd']:.4f} | session {result['session_id'][:8]}"
        )
        text = result["text"]
    else:
        known = ["claude", "gemini", CLAUDE_CODE_PROVIDER] + list(_OPENAI_COMPATIBLE) + list(_LOCAL_PROVIDERS)
        raise ValueError(
            f"Unknown provider '{provider}' in model '{model_name}'. Prefix the model "
            f"with one of: {', '.join(known)} - e.g. 'ollama:qwen3:8b'."
        )

    return text, resolved
