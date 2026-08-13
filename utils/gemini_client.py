# Shared Gemini helper (google-genai)
#
# These nodes used to talk to the legacy `google-generativeai` SDK, which hard
# pinned google-ai-generativelanguage==0.6.15 and therefore dragged protobuf<6
# plus the whole grpcio / google-api stack into the ComfyUI environment.
# `google-genai` is the current SDK and needs none of that.
#
# Legacy -> current mapping used across the pack:
#   genai.configure(api_key=...)          -> genai.Client(api_key=...)
#   genai.GenerativeModel(m, safety=...)  -> config=types.GenerateContentConfig(...)
#   model.generate_content(contents)      -> client.models.generate_content(
#                                                model=m, contents=..., config=...)

import os

try:
    from google import genai
    from google.genai import types as genai_types

    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    genai_types = None
    GEMINI_AVAILABLE = False


GEMINI_INSTALL_HINT = (
    "google-genai is not installed. Install it with: pip install google-genai"
)

# The permissive safety posture these nodes have always shipped with.
_SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)


def get_gemini_client(api_key=None):
    """
    Build a google-genai client.

    Falls back to the GEMINI_API_KEY environment variable and raises the same
    errors the nodes raised before the migration.
    """
    if not GEMINI_AVAILABLE:
        raise ImportError(GEMINI_INSTALL_HINT)

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    return genai.Client(api_key=key)


def build_gemini_config(**kwargs):
    """
    GenerateContentConfig with every safety filter set to BLOCK_NONE.

    Any extra keyword (temperature, max_output_tokens, system_instruction, ...)
    is forwarded; None values are dropped so callers can pass optionals
    unconditionally.
    """
    if not GEMINI_AVAILABLE:
        raise ImportError(GEMINI_INSTALL_HINT)

    return genai_types.GenerateContentConfig(
        safety_settings=[
            genai_types.SafetySetting(category=category, threshold="BLOCK_NONE")
            for category in _SAFETY_CATEGORIES
        ],
        **{k: v for k, v in kwargs.items() if v is not None},
    )


def gemini_generate(client, model, contents, **config_kwargs):
    """
    client.models.generate_content with the shared safety config applied.

    `contents` may be a string, a PIL image, or a list mixing both - google-genai
    converts PIL images to inline image parts on its own.
    """
    if not isinstance(contents, list):
        contents = [contents]

    return client.models.generate_content(
        model=model,
        contents=contents,
        config=build_gemini_config(**config_kwargs),
    )


def gemini_finished_normally(candidate):
    """
    True when a candidate completed with FinishReason.STOP.

    The legacy SDK exposed finish_reason as the int 1; google-genai returns a
    FinishReason string enum, so compare by name instead.
    """
    reason = getattr(candidate, "finish_reason", None)
    if reason is None:
        return True

    name = getattr(reason, "name", None) or str(reason).rsplit(".", 1)[-1]
    return str(name).upper() == "STOP"
