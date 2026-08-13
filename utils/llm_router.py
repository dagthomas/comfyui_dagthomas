# Shared multi-provider LLM router
#
# One entry point (`call_llm`) that talks to GPT, Gemini, Claude, Grok and Groq
# with an optional system prompt and optional images. Providers are selected by
# a "provider:model" string, the same convention the universal nodes already use.

import base64
import io
import os

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


# Provider prefix -> (env var names, OpenAI-compatible base url or None)
_OPENAI_COMPATIBLE = {
    "gpt": (("OPENAI_API_KEY",), None),
    "grok": (("XAI_API_KEY", "GROK_API_KEY"), "https://api.x.ai/v1"),
    "groq": (("GROQ_API_KEY",), "https://api.groq.com/openai/v1"),
}

# Preference order used by auto-detect, with the fallback model for each.
_AUTO_DETECT_ORDER = (
    (("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"), "claude:claude-sonnet-4.5"),
    (("OPENAI_API_KEY",), "gpt:gpt-4o"),
    (("GEMINI_API_KEY",), "gemini:gemini-2.5-flash"),
    (("XAI_API_KEY", "GROK_API_KEY"), "grok:grok-beta"),
    (("GROQ_API_KEY",), "groq:llama-3.3-70b-versatile"),
)

AUTO_DETECT = "auto-detect"

# Clients are cached per process so repeated node runs reuse connections.
_client_cache = {}


def list_all_models():
    """Every selectable model string, auto-detect first."""
    return (
        [AUTO_DETECT]
        + [f"claude:{m}" for m in claude_models]
        + [f"gpt:{m}" for m in gpt_models]
        + [f"gemini:{m}" for m in gemini_models]
        + [f"grok:{m}" for m in grok_models]
        + [f"groq:{m}" for m in groq_models]
    )


def _first_env(names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def auto_detect_model():
    """Pick the best model whose API key is actually present."""
    for env_names, model in _AUTO_DETECT_ORDER:
        if _first_env(env_names):
            return model

    raise ValueError(
        "No API keys found. Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "GEMINI_API_KEY, XAI_API_KEY, GROQ_API_KEY"
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


def _get_openai_compatible_client(provider):
    cached = _client_cache.get(provider)
    if cached is not None:
        return cached

    if not OPENAI_AVAILABLE:
        raise ImportError("openai is not installed. Install it with: pip install 'openai<3'")

    env_names, base_url = _OPENAI_COMPATIBLE[provider]
    api_key = _first_env(env_names)
    if not api_key:
        raise ValueError(f"{' or '.join(env_names)} environment variable not set")

    kwargs = {"api_key": api_key}
    http_client = _http_client()
    if http_client is not None:
        kwargs["http_client"] = http_client
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)
    _client_cache[provider] = client
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


def _call_openai_compatible(provider, model, user_prompt, system_prompt, images, temperature, seed, max_tokens):
    client = _get_openai_compatible_client(provider)

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
    messages.append({"role": "user", "content": content})

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # Groq rejects the seed parameter on several models, so only GPT and Grok get it.
    if seed is not None and seed != -1 and provider in ("gpt", "grok"):
        kwargs["seed"] = seed

    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def _call_claude(model, user_prompt, system_prompt, images, temperature, max_tokens):
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
        "messages": [{"role": "user", "content": content}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)

    # Concatenate every text block; thinking-capable models can emit several.
    parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    return "".join(parts).strip()


def _call_gemini(model, user_prompt, system_prompt, images, temperature, max_tokens):
    client = _get_gemini_client()

    contents = [user_prompt]
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
):
    """
    Send a prompt to whichever provider `model_name` selects.

    `images` is a list of PIL images; providers that support vision receive them
    inline. Raises on failure so callers can decide how to surface the error.
    """
    resolved = resolve_model(model_name)
    provider, model = split_model(resolved)

    if provider in _OPENAI_COMPATIBLE:
        text = _call_openai_compatible(
            provider, model, user_prompt, system_prompt, images, temperature, seed, max_tokens
        )
    elif provider == "claude":
        text = _call_claude(model, user_prompt, system_prompt, images, temperature, max_tokens)
    elif provider == "gemini":
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai is not installed. Install it with: pip install google-genai")
        text = _call_gemini(model, user_prompt, system_prompt, images, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider '{provider}' in model '{model_name}'")

    return text, resolved
