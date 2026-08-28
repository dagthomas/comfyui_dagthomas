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
import re
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
from .codex_cli import is_available as codex_available, run_codex

# Claude Code and Codex run through their locally installed CLIs and their own
# logins, so they need no API key here.
CLAUDE_CODE_PROVIDER = "claudecode"
CODEX_PROVIDER = "codex"


# Provider prefix -> (env var names, OpenAI-compatible base url or None)
_OPENAI_COMPATIBLE = {
    "gpt": (("OPENAI_API_KEY",), None),
    "grok": (("XAI_API_KEY", "GROK_API_KEY"), "https://api.x.ai/v1"),
    "groq": (("GROQ_API_KEY",), "https://api.groq.com/openai/v1"),
    # OpenRouter: one key, every model. Model ids are `vendor/model`, so a
    # router string reads `openrouter:anthropic/claude-fable-5`.
    "openrouter": (("OPENROUTER_API_KEY",), "https://openrouter.ai/api/v1"),
}

# OpenRouter serves 400+ models; these are the ones worth a dropdown entry for
# writing prompts (strong instruction following, long output). Anything else
# goes in as `openrouter:<vendor/model>` through model_name - the full list is
# https://openrouter.ai/models. Checked against the live catalogue on 2026-08-27.
openrouter_models = [
    "anthropic/claude-fable-5",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-opus-4.1",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-4.1",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-3-flash-preview",
    "x-ai/grok-4.6",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-chat-v3.1",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "moonshotai/kimi-k2.5",
    "z-ai/glm-4.7",
    "meta-llama/llama-4-maverick",
    "mistralai/mistral-large-2512",
    "minimax/minimax-m2.7",
]

# The whole catalogue, fetched from OpenRouter's public /models endpoint (no
# key needed) when the node list is built, cached for a while so reloading the
# page does not hit the network every time. Filtered to models that take and
# return TEXT (a writer needs nothing else) and to the interactive endpoints
# (`:batch` variants answer hours later). Curated ids stay at the top of the
# list; everything else follows alphabetically. APNEXT_OPENROUTER_LIST=0 turns
# the fetch off and leaves the curated list.
_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_OPENROUTER_LIST_TTL = 600.0
_OPENROUTER_FETCH_TIMEOUT = 6.0
_openrouter_cache = {"at": 0.0, "ids": None}


def list_openrouter_models(refresh=False):
    """Every text model on OpenRouter as `vendor/model` ids, curated ones first; the curated list offline."""
    import time
    if os.environ.get("APNEXT_OPENROUTER_LIST", "1").strip().lower() in ("0", "false", "off", "no"):
        return list(openrouter_models)
    now = time.monotonic()
    cached = _openrouter_cache["ids"]
    if cached is not None and not refresh and now - _openrouter_cache["at"] < _OPENROUTER_LIST_TTL:
        return list(cached)
    payload = _get_json(_OPENROUTER_MODELS_URL, _OPENROUTER_FETCH_TIMEOUT)
    ids = []
    for entry in (payload or {}).get("data", []) if isinstance(payload, dict) else []:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        mid = str(entry["id"])
        arch = entry.get("architecture") or {}
        outs = arch.get("output_modalities") or ["text"]
        ins = arch.get("input_modalities") or ["text"]
        if "text" not in outs or "text" not in ins or mid.endswith(":batch"):
            continue
        ids.append(mid)
    if not ids:
        return list(cached) if cached else list(openrouter_models)
    rest = sorted(i for i in set(ids) if i not in openrouter_models)
    full = [m for m in openrouter_models if m in set(ids)] + [m for m in openrouter_models if m not in set(ids)] + rest
    _openrouter_cache.update(at=now, ids=full)
    return list(full)


# Extra headers OpenRouter uses for its app rankings; harmless elsewhere.
_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/dagthomas/comfyui_dagthomas",
    "X-Title": "APNext H3 (ComfyUI)",
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
    (("OPENROUTER_API_KEY",), "openrouter:anthropic/claude-fable-5"),
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


# OLLAMA_HOST is Ollama's *bind* address. People set it to `0.0.0.0:11434` (or
# `:11434`) so WSL / the LAN can reach the server - but a wildcard is not a
# destination, and Windows answers a connect() to 0.0.0.0 with WinError 10049.
_WILDCARD_HOSTS = {"0.0.0.0", "::", "0:0:0:0:0:0:0:0"}


def wildcard_host_to_localhost(url):
    """`http://0.0.0.0:11434` / `http://:11434` -> `http://localhost:11434`."""
    parts = urlsplit(url)
    try:
        host = parts.hostname
    except ValueError:
        return url
    if host and host not in _WILDCARD_HOSTS:
        return url
    port = f":{parts.port}" if parts.port else ""
    if not host and not port:
        # `http://:11434` has no hostname; urlsplit keeps the port in netloc.
        port = parts.netloc if parts.netloc.startswith(":") else ""
    netloc = f"localhost{port}"
    return url.replace(parts.netloc, netloc, 1)


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
    url = wildcard_host_to_localhost(url)
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


def list_codex_models():
    """The `codex` entry (the CLI's configured model), only when installed."""
    return [CODEX_PROVIDER] if codex_available() else []


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
        + list_codex_models()
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
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY, GROQ_API_KEY, "
        "OPENROUTER_API_KEY - "
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
    if model_name.strip().lower() == CODEX_PROVIDER:
        # The bare `codex` entry means the Codex CLI with its configured model.
        return CODEX_PROVIDER, CODEX_PROVIDER
    return "gpt", model_name.strip()


# How long one generation may take. A cloud call answers in seconds; a 30B model
# on a home GPU writing four H3 scenes runs for many minutes, so local providers
# get a much longer leash.
_CLOUD_REQUEST_TIMEOUT = float(os.environ.get("APNEXT_LLM_REQUEST_TIMEOUT", "180"))
_LOCAL_REQUEST_TIMEOUT = float(os.environ.get("APNEXT_LOCAL_LLM_REQUEST_TIMEOUT", "900"))


def _request_timeout(provider):
    if provider in _LOCAL_PROVIDERS:
        return _LOCAL_REQUEST_TIMEOUT
    if provider == "openrouter":
        # a routed open model writing six scenes can take minutes; give it the local leash
        return max(_CLOUD_REQUEST_TIMEOUT, _LOCAL_REQUEST_TIMEOUT)
    return _CLOUD_REQUEST_TIMEOUT


def _http_client(timeout=_CLOUD_REQUEST_TIMEOUT):
    if httpx is None:
        return None
    try:
        return httpx.Client(timeout=timeout)
    except TypeError:
        return httpx.Client()


def _get_openai_compatible_client(provider, base_url_override=None, api_key_override=None):
    if not OPENAI_AVAILABLE:
        raise ImportError("openai is not installed. Install it with: pip install 'openai<3'")

    key_override = (api_key_override or "").strip()
    if provider in _LOCAL_PROVIDERS:
        base_url = resolve_base_url(provider, base_url_override)
        # Local servers ignore the key, but the OpenAI client refuses to start without one.
        api_key = key_override or _first_env(("LOCAL_LLM_API_KEY",)) or "local"
    else:
        env_names, base_url = _OPENAI_COMPATIBLE[provider]
        if provider == "openrouter" and (base_url_override or "").strip():
            base_url = _normalise_base_url(base_url_override)      # a gateway in front of OpenRouter
        # the key typed on the LLM Backend node wins over the environment
        api_key = key_override or _first_env(env_names)
        if not api_key:
            raise ValueError(
                f"{' or '.join(env_names)} environment variable not set - or type the key into the "
                f"H3 LLM Backend node's api_key."
            )

    # The URL and the key are part of the cache key so two nodes pointing at
    # different servers, or using different accounts, do not share a client.
    cache_key = f"{provider}|{base_url or ''}|{hash(api_key)}"
    cached = _client_cache.get(cache_key)
    if cached is not None:
        return cached

    kwargs = {"api_key": api_key}
    http_client = _http_client(_request_timeout(provider))
    if http_client is not None:
        kwargs["http_client"] = http_client
    if base_url:
        kwargs["base_url"] = base_url
    if provider == "openrouter":
        kwargs["default_headers"] = dict(_OPENROUTER_HEADERS)

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


# ----------------------------------------------------------------------
# Reasoning models
# ----------------------------------------------------------------------

# Hybrid reasoning models (Qwen3, DeepSeek-R1 distills, ...) put their thinking
# in a <think> block ahead of the answer. Cloud providers return reasoning on a
# separate channel, but an OpenAI-compatible local server usually leaves it
# inline, where it corrupts every downstream parser - an H3 scene envelope
# rehearsed inside a thought would be read as a real scene. Strip it here.
_THINK_BLOCK_RE = re.compile(
    r"\s*<(think|thinking|reasoning)>.*?</\1>\s*", re.DOTALL | re.IGNORECASE
)
_THINK_CLOSE_RE = re.compile(
    r"^.*?</(?:think|thinking|reasoning)>\s*", re.DOTALL | re.IGNORECASE
)
_THINK_OPEN_RE = re.compile(r"<(?:think|thinking|reasoning)>", re.IGNORECASE)


def strip_reasoning(text):
    """The answer with any inline <think>...</think> reasoning removed."""
    if not text or "<" not in text:
        return text

    cleaned = _THINK_BLOCK_RE.sub("\n", text).strip()
    # A server that swallows the opening tag leaves a bare `</think>`: everything
    # ahead of it is thought, not answer.
    if not _THINK_OPEN_RE.search(cleaned) and _THINK_CLOSE_RE.match(cleaned):
        cleaned = _THINK_CLOSE_RE.sub("", cleaned).strip()

    # An unterminated block means the answer was cut off mid-thought; there is
    # nothing better to hand back than what arrived.
    return cleaned or text.strip()


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


def _ollama_api_root(base_url=None):
    """Ollama's own API root; `resolve_base_url` returns the OpenAI-shim `/v1`."""
    url = resolve_base_url("ollama", base_url)
    return url[:-3].rstrip("/") if url.endswith("/v1") else url


def unload_ollama_model(model, base_url=None, timeout=20):
    """
    Ask Ollama to unload `model` (an 'ollama:tag' or bare tag) right now, so its
    VRAM goes back to whatever renders next. Ollama keeps a model loaded for
    `keep_alive` (5 min by default) after the last request; a generate call
    with keep_alive 0 and no prompt is the documented way to release it early.
    Returns True when Ollama acknowledged, False otherwise - never raises.
    """
    tag = (model or "").strip()
    if tag.lower().startswith("ollama:"):
        tag = tag.split(":", 1)[1]
    if not tag:
        return False
    try:
        _post_json(f"{_ollama_api_root(base_url)}/api/generate", {"model": tag, "keep_alive": 0}, timeout)
        return True
    except Exception as exc:
        print(f"\u26a0\ufe0f could not unload '{tag}' from Ollama: {exc}")
        return False


_CONNECT_RETRIES = 2          # a local server can refuse for a moment (restart, reload)
_CONNECT_RETRY_WAIT = 2.0     # seconds between attempts


def _post_json(url, payload, timeout):
    """
    POST JSON and return the decoded reply, raising with the server's text.
    A refused / dropped connection is retried a couple of times before it is
    reported - and the report names the URL, so "10061 refused" at least says
    which host and port nobody was listening on.
    """
    if httpx is not None:
        connect_errors = (httpx.ConnectError, httpx.RemoteProtocolError)
        post = lambda: httpx.post(url, json=payload, timeout=timeout)
    else:
        import requests

        connect_errors = (requests.exceptions.ConnectionError,)
        post = lambda: requests.post(url, json=payload, timeout=timeout)

    last = None
    for attempt in range(_CONNECT_RETRIES + 1):
        try:
            response = post()
            break
        except connect_errors as exc:
            last = exc
            if attempt < _CONNECT_RETRIES:
                print(f"⚠️ {url}: {exc} - retrying in {_CONNECT_RETRY_WAIT:.0f} s "
                      f"({attempt + 1}/{_CONNECT_RETRIES})")
                time.sleep(_CONNECT_RETRY_WAIT)
    else:
        raise RuntimeError(
            f"could not connect to {url} ({last}) - is the server running, and is the "
            "base_url / OLLAMA_HOST pointing at it?"
        ) from last

    if response.status_code != 200:
        raise RuntimeError(f"{url} returned HTTP {response.status_code}: {response.text[:400]}")

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"{url} returned a non-JSON reply: {response.text[:400]}") from exc


def _call_ollama_native(
    model, user_prompt, system_prompt, images, temperature, seed, max_tokens,
    base_url=None, history=None, num_ctx=0, think=None, format_schema=None,
):
    """
    One turn through Ollama's own `/api/chat` instead of its OpenAI shim.

    The shim exposes no way to set `num_ctx`, and Ollama's default context is
    picked from free VRAM - as little as 4k. The H3 system prompt alone is
    9-15k tokens, so on a small default the rules the model must follow are
    silently truncated away and it writes nothing usable. This path sets the
    context explicitly, and can switch a hybrid reasoning model's thinking off
    (`think`), which is both faster and cleaner than stripping it afterwards.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(_text_history(history))

    turn = {"role": "user", "content": user_prompt}
    if images:
        # The native API takes bare base64 strings, not data: URLs.
        turn["images"] = [base64.b64encode(_encode_png(i)).decode("utf-8") for i in images]
    messages.append(turn)

    options = {"temperature": temperature, "num_predict": max_tokens}
    if num_ctx:
        options["num_ctx"] = int(num_ctx)
    if seed is not None and seed != -1:
        options["seed"] = seed

    payload = {"model": model, "messages": messages, "stream": False, "options": options}
    if format_schema:
        # Ollama constrains the sampler to this JSON schema: the reply is valid
        # JSON of that shape by construction, not by the model's good will.
        payload["format"] = format_schema
    if think is not None:
        payload["think"] = bool(think)

    url = f"{_ollama_api_root(base_url)}/api/chat"
    try:
        data = _post_json(url, payload, _LOCAL_REQUEST_TIMEOUT)
    except RuntimeError as exc:
        # Models without a thinking mode reject the flag outright; the request is
        # perfectly good without it.
        if think is not None and "think" in str(exc).lower():
            payload.pop("think", None)
            data = _post_json(url, payload, _LOCAL_REQUEST_TIMEOUT)
        else:
            vision_hint = (
                " The model also has to be vision-capable to accept the attached image(s)."
                if images
                else ""
            )
            raise RuntimeError(
                f"Ollama call to {url} failed: {exc} Check the server is running and that "
                f"'{model}' is pulled (`ollama pull {model}`).{vision_hint}"
            ) from exc

    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"Ollama returned an error: {data['error']}")

    _check_ollama_cutoff(data, model, options)
    message = (data or {}).get("message") or {}
    return strip_reasoning((message.get("content") or "").strip())


def _check_ollama_cutoff(data, model, options):
    """
    Ollama stops quietly when the reply runs out of room - `done_reason` says
    "length" and the text just ends mid-sentence (mid-JSON with a schema). A
    cut-off scene list is worse than no scene list, so this raises with the
    numbers that explain it: how much of the context the prompt took, and
    whether it was num_ctx or num_predict that ran out.
    """
    if not isinstance(data, dict) or data.get("done_reason") != "length":
        return
    prompt_tokens = int(data.get("prompt_eval_count") or 0)
    reply_tokens = int(data.get("eval_count") or 0)
    num_ctx = int(options.get("num_ctx") or 0)
    num_predict = int(options.get("num_predict") or 0)
    if num_predict and reply_tokens >= num_predict - 8:
        why = (f"the reply hit max_tokens ({num_predict}). Raise max_tokens on the H3 LLM Backend "
               f"node, or ask for fewer scenes per call.")
    elif num_ctx:
        room = max(0, num_ctx - prompt_tokens)
        suggest = 65536 if prompt_tokens + 6000 <= 65536 else 131072
        why = (f"the prompt alone took {prompt_tokens:,} of the {num_ctx:,}-token context (num_ctx), "
               f"leaving room for only ~{room:,} reply tokens. Raise num_ctx on the H3 LLM Backend node "
               f"to {suggest:,} (or shorten the prompt: fewer scenes, no inline skill references, "
               f"smaller cast / lyrics).")
    else:
        why = (f"the prompt took {prompt_tokens:,} tokens and the server's default context ran out. "
               f"Set num_ctx on the H3 LLM Backend node (65536 for a long music video).")
    raise RuntimeError(
        f"Ollama cut '{model}' off after {reply_tokens:,} reply tokens (done_reason=length): {why}"
    )


def _call_openai_compatible(
    provider, model, user_prompt, system_prompt, images, temperature, seed, max_tokens,
    base_url=None, history=None, api_key=None, think=None,
):
    client = _get_openai_compatible_client(provider, base_url, api_key)

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
    if provider == "openrouter" and think is not None:
        # OpenRouter's unified reasoning control. Reasoning models (GPT-5,
        # Claude with thinking, DeepSeek-R1, Qwen3 ...) spend `max_tokens` on
        # their hidden thinking FIRST; the H3 format is spelled out in the
        # system prompt, so the backend's default `thinking: off` asks for the
        # least reasoning a model allows, and `on` for the most.
        kwargs["extra_body"] = {"reasoning": {"effort": "high" if think else "minimal"}}

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

    text = (response.choices[0].message.content or "").strip()
    if not text and provider == "openrouter":
        finish = getattr(response.choices[0], "finish_reason", None)
        usage = getattr(response, "usage", None)
        reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
        raise RuntimeError(
            f"OpenRouter returned no text from '{model}' (finish_reason={finish}"
            + (f", reasoning_tokens={reasoning}" if reasoning else "") + "). A reasoning model spends "
            "max_tokens on its hidden thinking first: raise max_tokens on the LLM Backend (8000+), keep "
            "its `thinking` off, or pick a non-reasoning model."
        )
    # open models routed through OpenRouter (Qwen, DeepSeek, GLM, ...) may think out loud
    return strip_reasoning(text) if (provider in _LOCAL_PROVIDERS or provider == "openrouter") else text


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
    num_ctx=0,
    think=None,
    format_schema=None,
    api_key=None,
):
    """
    Send a prompt to whichever provider `model_name` selects.

    `history` is an optional list of earlier `{"role": "user"|"assistant",
    "content": str}` turns, replayed before `user_prompt` so a provider without
    sessions of its own can still continue a conversation.

    `images` is a list of PIL images; providers that support vision receive them
    inline. `base_url` overrides where a local provider (`ollama:`, `lmstudio:`,
    `local:`) is reached and is ignored for the cloud providers.

    `num_ctx` (context window) and `think` (hybrid reasoning on/off) are Ollama
    settings its OpenAI-compatible shim does not expose; passing either sends
    the call through Ollama's own API instead. Both are ignored by every other
    provider. Raises on failure so callers can decide how to surface the error.
    """
    resolved = resolve_model(model_name)
    provider, model = split_model(resolved)

    if provider == "ollama" and (num_ctx or think is not None or format_schema):
        text = _call_ollama_native(
            model,
            user_prompt,
            system_prompt,
            images,
            temperature,
            seed,
            max_tokens,
            base_url=base_url,
            history=history,
            num_ctx=num_ctx,
            think=think,
            format_schema=format_schema,
        )
    elif provider in _OPENAI_COMPATIBLE or provider in _LOCAL_PROVIDERS:
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
            api_key=api_key,
            think=think,
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
    elif provider == CODEX_PROVIDER:
        # The CLI owns sampling, so temperature, seed and max_tokens do not apply.
        # `codex` alone runs the CLI's configured model; `codex:<id>` picks one.
        result = run_codex(
            _history_as_text(history) + user_prompt,
            system_prompt=system_prompt,
            images=images,
            model=model if model != CODEX_PROVIDER else None,
            timeout=int(os.environ.get("APNEXT_CODEX_TIMEOUT", "600")),
            on_progress=lambda note: print(f"   ↳ {note}"),
        )
        print(
            f"🤖 Codex | {result['model']} | {result['duration_ms'] / 1000:.1f}s | "
            f"session {result['session_id'][:8]}"
        )
        text = result["text"]
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
        known = ["claude", "gemini", CLAUDE_CODE_PROVIDER, CODEX_PROVIDER] + list(_OPENAI_COMPATIBLE) + list(_LOCAL_PROVIDERS)
        raise ValueError(
            f"Unknown provider '{provider}' in model '{model_name}'. Prefix the model "
            f"with one of: {', '.join(known)} - e.g. 'ollama:qwen3:8b'."
        )

    return text, resolved
