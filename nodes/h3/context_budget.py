# APNext H3 context budget - server side
#
# Backs the "📏 Show 1024 tokens" button on the H3 LLM Backend node
# (web/js/h3_context_budget.js). Picking `num_ctx` is the difference between an
# Ollama run that writes usable scenes and one that returns noise, and nobody
# has an intuition for what a token is - so the modal shows a real block of
# 1024 tokens, measured with the user's own model, and works out what their
# actual machine can hold and how fast it runs.
#
#   GET  /apnext/h3/context_budget  -> pulled models (layer/KV geometry,
#                                      capabilities) + this box's VRAM and RAM
#   POST /apnext/h3/token_sample    -> a passage cut to EXACTLY N tokens,
#                                      measured with the chosen model
#   POST /apnext/h3/token_benchmark -> real tokens/sec and the GPU/CPU split
#                                      for one model on this machine
#
# Everything degrades: no Ollama, no torch, no psutil - the modal still opens
# and falls back to an estimated sample and the static guidance.

import os

try:
    from server import PromptServer
except Exception:
    PromptServer = None
try:
    from aiohttp import web, ClientSession, ClientTimeout
except Exception:
    web = None
    ClientSession = None
    ClientTimeout = None

from .common import load_guide

# The passage the sample is cut from. The point of the modal is to show what a
# context window holds, so it is cut from the H3 guide the writers actually
# send - prose, field names and example prompts in the same mix as a real run,
# which tokenises like a real run.
_SAMPLE_SOURCE = "guide_base_en.md"

# Rough fallback when no model can measure the sample. English prose runs
# ~4 characters per token on a modern BPE vocabulary.
_CHARS_PER_TOKEN = 4

_DEFAULT_OLLAMA = "http://localhost:11434"

# (target, model) -> {"text", "tokens", "exact"}; a binary search costs several
# model calls, so a sample is measured once per session.
_sample_cache = {}


def _api_root(base_url):
    """Ollama's API root, accepting the shapes people paste into the node."""
    url = (base_url or "").strip().rstrip("/")
    if not url:
        url = (os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST")
               or _DEFAULT_OLLAMA)
        url = url.strip().rstrip("/")
    if "://" not in url:
        url = f"http://{url}"
    if url.endswith("/v1"):
        url = url[:-3].rstrip("/")
    return url


def _model_tag(model):
    """'ollama:qwen3:8b' -> 'qwen3:8b'. Anything else is passed through."""
    tag = (model or "").strip()
    if tag.lower().startswith("ollama:"):
        tag = tag.split(":", 1)[1].strip()
    return tag


async def _get(session, url, timeout=10):
    async with session.get(url, timeout=ClientTimeout(total=timeout)) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        return await response.json()


async def _post(session, url, payload, timeout=600):
    async with session.post(url, json=payload, timeout=ClientTimeout(total=timeout)) as response:
        if response.status != 200:
            body = (await response.text())[:300]
            raise RuntimeError(f"HTTP {response.status}: {body}")
        return await response.json()


# ---------------------------------------------------------------------------
# Model geometry: what one token of context actually costs
# ---------------------------------------------------------------------------

def _geometry(show):
    """
    KV-cache geometry from `/api/show`, as
    {layers, kv_heads, key_length, value_length, context_length, params}.

    The keys are namespaced by architecture ("qwen3.block_count"), so they are
    matched by suffix rather than by a hard-coded model family.
    """
    info = (show or {}).get("model_info") or {}

    def find(suffix):
        for key, value in info.items():
            if key.endswith(suffix) and isinstance(value, (int, float)):
                return int(value)
        return 0

    key_length = find(".attention.key_length")
    value_length = find(".attention.value_length")
    # Models that do not publish head_length use hidden_size / n_heads.
    if not key_length:
        heads = find(".attention.head_count")
        embedding = find(".embedding_length")
        key_length = embedding // heads if heads and embedding else 0
    if not value_length:
        value_length = key_length

    return {
        "layers": find(".block_count"),
        "kv_heads": find(".attention.head_count_kv") or find(".attention.head_count"),
        "key_length": key_length,
        "value_length": value_length,
        "context_length": find(".context_length"),
        "params": find("general.parameter_count"),
    }


def _kv_bytes_per_token(geometry, bytes_per_element=2):
    """
    Bytes of KV cache one token of context costs.

    layers x kv_heads x (key + value dims) x element size - the standard
    grouped-query-attention cache. f16 (2 bytes) is Ollama's default;
    OLLAMA_KV_CACHE_TYPE=q8_0 roughly halves it.

    Models with sliding-window or hybrid attention (Gemma 3, the Mamba
    hybrids) cache far less than this on most layers, so for those the figure
    is an upper bound.
    """
    per_layer = geometry["kv_heads"] * (geometry["key_length"] + geometry["value_length"])
    return geometry["layers"] * per_layer * bytes_per_element


def _hardware():
    """This box's VRAM and RAM, as far as anything installed can report it."""
    out = {}
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            out["gpu"] = {
                "name": torch.cuda.get_device_properties(0).name,
                "total": int(total),
                "free": int(free),
            }
    except Exception:
        pass

    try:
        import psutil

        memory = psutil.virtual_memory()
        out["ram"] = {"total": int(memory.total), "available": int(memory.available)}
    except Exception:
        pass

    out["kv_cache_type"] = os.environ.get("OLLAMA_KV_CACHE_TYPE", "f16")
    out["context_length_env"] = os.environ.get("OLLAMA_CONTEXT_LENGTH", "")
    return out


async def _context_budget(request):
    base_url = _api_root(request.query.get("base_url"))
    payload = {"base_url": base_url, "hardware": _hardware(), "models": [], "loaded": []}

    if ClientSession is None:
        payload["error"] = "aiohttp is unavailable, so Ollama cannot be queried."
        return web.json_response(payload)

    try:
        async with ClientSession() as session:
            tags = await _get(session, f"{base_url}/api/tags", timeout=6)
            payload["running"] = True

            try:
                running = await _get(session, f"{base_url}/api/ps", timeout=6)
                payload["loaded"] = running.get("models") or []
            except Exception:
                pass

            for entry in (tags.get("models") or []):
                name = entry.get("name") or entry.get("model")
                if not name:
                    continue
                model = {
                    "name": name,
                    "size": entry.get("size") or 0,
                    "quantization": (entry.get("details") or {}).get("quantization_level", ""),
                    "parameter_size": (entry.get("details") or {}).get("parameter_size", ""),
                    "family": (entry.get("details") or {}).get("family", ""),
                }
                try:
                    show = await _post(session, f"{base_url}/api/show", {"model": name}, timeout=20)
                    geometry = _geometry(show)
                    model["geometry"] = geometry
                    model["capabilities"] = show.get("capabilities") or []
                    if geometry["layers"] and geometry["kv_heads"]:
                        model["kv_bytes_per_token"] = _kv_bytes_per_token(geometry)
                except Exception:
                    pass
                payload["models"].append(model)
    except Exception as exc:
        payload["running"] = False
        payload["error"] = (
            f"No Ollama at {base_url} ({exc}). Start it with `ollama serve`, or point "
            "the node's base_url at the machine that runs it."
        )

    return web.json_response(payload)


# ---------------------------------------------------------------------------
# The sample: N tokens of real prompt text, measured with the user's model
# ---------------------------------------------------------------------------

async def _prompt_tokens(session, base_url, model, content):
    """`prompt_eval_count` for one user message - the model's own tokenizer."""
    data = await _post(session, f"{base_url}/api/chat", {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "think": False,
        # 1, not 0: some builds treat 0 as "unset" and generate a full answer
        "options": {"num_predict": 1, "temperature": 0},
    }, timeout=600)
    return int(data.get("prompt_eval_count") or 0)


async def _counter(session, base_url, model):
    """
    A `count(text) -> tokens` for this model.

    Ollama reports the count for the whole templated prompt, so the chat
    template's own tokens are measured once up front and subtracted from every
    later probe - the binary search below is one call per step, not two.
    """
    overhead = await _prompt_tokens(session, base_url, model, "")

    async def count(text):
        return max(0, await _prompt_tokens(session, base_url, model, text) - overhead)

    return count


async def _token_sample(request):
    body = await request.json()
    target = max(16, min(8192, int(body.get("target") or 1024)))
    model = _model_tag(body.get("model"))
    base_url = _api_root(body.get("base_url"))

    try:
        source = load_guide(_SAMPLE_SOURCE)
    except Exception as exc:
        return web.json_response({"error": f"could not read the sample text: {exc}"}, status=500)

    cache_key = (target, model)
    if cache_key in _sample_cache:
        return web.json_response(_sample_cache[cache_key])

    estimate = {
        "text": source[: target * _CHARS_PER_TOKEN],
        "tokens": target,
        "exact": False,
        "model": model,
        "note": (
            f"Estimated at ~{_CHARS_PER_TOKEN} characters per token - no model was "
            "available to count it exactly."
        ),
    }
    if not model or ClientSession is None:
        return web.json_response(estimate)

    # Binary-search the cut so the passage is EXACTLY `target` tokens for this
    # model's tokenizer. Token counts are monotonic in the cut but not strict -
    # several cuts share a count - so the search keeps the closest probe seen
    # rather than relying on an exact hit. 14 steps resolve a 12k-character
    # range down to single characters; the first load of a cold model dominates
    # the wall clock, not the probes.
    try:
        async with ClientSession() as session:
            count_tokens = await _counter(session, base_url, model)
            low, high = 0, min(len(source), target * 12)
            best = None
            for _ in range(14):
                if low > high:
                    break
                middle = (low + high) // 2
                count = await count_tokens(source[:middle])
                if best is None or abs(count - target) < abs(best[1] - target):
                    best = (middle, count)
                if count == target:
                    break
                if count < target:
                    low = middle + 1
                else:
                    high = middle - 1

            if best is None:
                raise RuntimeError("the model returned no token counts")

            result = {
                "text": source[: best[0]],
                "tokens": best[1],
                "exact": True,
                "model": model,
                "characters": best[0],
                "note": f"Counted with {model}'s own tokenizer.",
            }
            _sample_cache[cache_key] = result
            return web.json_response(result)
    except Exception as exc:
        estimate["note"] = (
            f"{_CHARS_PER_TOKEN} characters per token (estimated) - {model} could not "
            f"measure it: {exc}"
        )
        return web.json_response(estimate)


# ---------------------------------------------------------------------------
# The benchmark: what this machine actually does with this model
# ---------------------------------------------------------------------------

async def _token_benchmark(request):
    body = await request.json()
    model = _model_tag(body.get("model"))
    base_url = _api_root(body.get("base_url"))
    num_predict = max(8, min(256, int(body.get("num_predict") or 64)))
    num_ctx = int(body.get("num_ctx") or 0)

    if not model:
        return web.json_response({"error": "pick a model first."}, status=400)
    if ClientSession is None:
        return web.json_response({"error": "aiohttp is unavailable."}, status=500)

    options = {"num_predict": num_predict, "temperature": 0.7}
    if num_ctx:
        options["num_ctx"] = num_ctx

    # A real prompt, not a one-liner: an H3 call makes the model read 9-15k
    # tokens before it writes a word, and on a partly-offloaded model that
    # ingestion is most of the wall clock. Measuring it against a two-line
    # question would report fixed overhead as if it were throughput.
    try:
        preamble = load_guide(_SAMPLE_SOURCE)[: 2048 * _CHARS_PER_TOKEN]
    except Exception:
        preamble = ""

    try:
        async with ClientSession() as session:
            data = await _post(session, f"{base_url}/api/chat", {
                "model": model,
                "messages": [{"role": "user", "content":
                              f"{preamble}\n\nIgnore the guide above. Describe a rainy "
                              "neon street in three sentences."}],
                "stream": False,
                "think": False,
                "options": options,
            }, timeout=900)

            eval_count = int(data.get("eval_count") or 0)
            eval_ns = int(data.get("eval_duration") or 0)
            prompt_count = int(data.get("prompt_eval_count") or 0)
            prompt_ns = int(data.get("prompt_eval_duration") or 0)

            read_rate = (prompt_count / (prompt_ns / 1e9)) if prompt_ns else 0
            result = {
                "model": model,
                "num_ctx": num_ctx,
                "eval_count": eval_count,
                "prompt_eval_count": prompt_count,
                "tokens_per_second": (eval_count / (eval_ns / 1e9)) if eval_ns else 0,
                "prompt_tokens_per_second": read_rate,
                # What an H3 call actually costs before the first word: the
                # Ref2VA system prompt is ~15k tokens and is re-read every call.
                "h3_prompt_seconds": (15200 / read_rate) if read_rate else 0,
            }

            # /api/ps says how the loaded model was actually split: size_vram
            # against total size is the GPU/CPU answer for THIS machine.
            try:
                running = await _get(session, f"{base_url}/api/ps", timeout=6)
                for entry in (running.get("models") or []):
                    if (entry.get("name") or entry.get("model")) == model:
                        total = int(entry.get("size") or 0)
                        in_vram = int(entry.get("size_vram") or 0)
                        result["size"] = total
                        result["size_vram"] = in_vram
                        result["on_gpu_fraction"] = (in_vram / total) if total else 0
                        result["context_length"] = entry.get("context_length") or 0
                        break
            except Exception:
                pass

            return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": f"{model} could not run on {base_url}: {exc}"}, status=502)


if PromptServer is not None and web is not None and getattr(PromptServer, "instance", None) is not None:
    PromptServer.instance.routes.get("/apnext/h3/context_budget")(_context_budget)
    PromptServer.instance.routes.post("/apnext/h3/token_sample")(_token_sample)
    PromptServer.instance.routes.post("/apnext/h3/token_benchmark")(_token_benchmark)
