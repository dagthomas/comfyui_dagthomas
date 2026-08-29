# APNext H3 Sound Events - the preview behind the node's 🎚 button
#
# Tuning `min_strength` blind means queueing the whole workflow to find out
# whether a track gives 6 hits a minute or 120. Strength is normalised to each
# track's own loudest hit, so no single value carries over between songs. The
# button on the node posts the upstream Load Audio file here; this runs the
# same detectors once with EVERY kind enabled and hands the full event list
# back, so the modal can filter by kind and strength instantly (no re-run per
# slider step) and show the density the writer would actually see - then
# write the chosen values back into the node's widgets.
#
#   POST /apnext/h3/sound_events_preview
#        {audio, sensitivity, min_gap_seconds, gain_db, dynamics_curve, eq_*_db, on_beat_weight}
#     -> {file, duration, bpm, grid: {bpm, period, phase, on, of}, shaping, on_beat_weight,
#         events: [{t, type, strength, label}], events_no_impacts: [...]}
#
# Two variants come back because the two hit detectors share one label per
# instant: with `impacts` on, a kick that is also bright is called IMPACT;
# with it off, the same kick is a BASS HIT. Filtering one list by kind cannot
# reproduce that, so the modal switches lists when the impacts box changes.
# The other toggles filter cleanly.

import asyncio
import os

try:
    from aiohttp import web
except Exception:  # pragma: no cover - aiohttp always ships with ComfyUI
    web = None

try:
    from server import PromptServer
except Exception:  # pragma: no cover - imported outside ComfyUI
    PromptServer = None

from .sound_events import EVENT_TYPES, SHAPING_KEYS, detect_events
from .stem_split import MODELS as STEM_MODELS, STEM_NAMES, separate_stems

_CACHE = {}
_CACHE_LIMIT = 6
_SHAPING_DEFAULT = {k: (1.0 if k == "dynamics_curve" else 0.0) for k in SHAPING_KEYS}
_SHAPING_RANGE = {k: ((0.25, 4.0) if k == "dynamics_curve" else (-24.0, 24.0)) for k in SHAPING_KEYS}

# Demucs output is expensive (tens of seconds per song), so the separated mono
# stems are cached per (file, mtime, model) - toggling which stems feed the
# detectors then re-runs only detection. Two songs of mono stems ~ 150 MB RAM.
_STEMS_CACHE = {}
_STEMS_CACHE_LIMIT = 2
_STEM_MODEL_NAMES = tuple(m.split(" ")[0] for m in STEM_MODELS)
_ENV_RATE = 25  # envelope samples per second sent to the modal


def _stems_from(body):
    """The validated stems request: {'model': name, 'sources': [...]} or None."""
    raw = body.get("stems")
    if not isinstance(raw, dict):
        return None
    model = str(raw.get("model") or "htdemucs").split(" ")[0]
    if model not in _STEM_MODEL_NAMES:
        model = "htdemucs"
    sources = [s for s in (raw.get("sources") or []) if s in STEM_NAMES]
    if not sources:
        sources = ["drums", "bass"]
    return {"model": model, "sources": sorted(set(sources))}


def _mono_stems(path, name, audio, model):
    """The song's mono stems, separated once per (file, mtime, model)."""
    key = (name, os.path.getmtime(path), model)
    cached = _STEMS_CACHE.get(key)
    if cached is not None:
        return cached
    stems, _model, device, seconds = separate_stems(audio, model)
    mono = {k: v.mean(0).contiguous() for k, v in stems.items()}
    if len(_STEMS_CACHE) >= _STEMS_CACHE_LIMIT:
        _STEMS_CACHE.pop(next(iter(_STEMS_CACHE)))
    _STEMS_CACHE[key] = mono
    return mono


def _envelope(mono, sample_rate):
    """Max-abs per window at _ENV_RATE points/second - the lane the modal draws."""
    import torch
    hop = max(1, int(round(sample_rate / _ENV_RATE)))
    n = mono.shape[-1] // hop * hop
    if n == 0:
        return []
    env = mono[:n].abs().reshape(-1, hop).amax(dim=1)
    peak = float(env.max())
    if peak > 0:
        env = env / peak
    return [round(float(v), 3) for v in env]


def _shaping_from(body):
    """The gain / curve / EQ values from the request, clamped to the node's ranges."""
    out = {}
    for key in SHAPING_KEYS:
        raw = body.get(key)
        value = _SHAPING_DEFAULT[key] if raw is None or raw == "" else float(raw)
        lo, hi = _SHAPING_RANGE[key]
        out[key] = round(min(hi, max(lo, value)), 3)
    return out


def _load_audio(name):
    import folder_paths
    from comfy_extras.nodes_audio import load

    if not folder_paths.exists_annotated_filepath(name):
        raise FileNotFoundError(f"'{name}' is not in the input folder")
    path = folder_paths.get_annotated_filepath(name)
    waveform, sample_rate = load(path)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    return path, {"waveform": waveform, "sample_rate": int(sample_rate)}


def _slim(events):
    return [
        {"t": e["t"], "type": e["type"], "strength": e["strength"], "label": e["label"],
         **({"until": e["until"]} if e.get("until") is not None else {})}
        for e in events
    ]


def _compute(name, sensitivity, min_gap, shaping, on_beat_weight, stems=None):
    path, audio = _load_audio(name)
    sample_rate = int(audio["sample_rate"])
    duration = audio["waveform"].shape[-1] / sample_rate

    # With a Stem Split upstream the node hears the stem mix, not the file -
    # detect on the same mix so the preview matches the run, and hand every
    # stem's envelope back so the modal can draw the lanes.
    stems_payload = None
    if stems:
        mono = _mono_stems(path, name, audio, stems["model"])
        mix = None
        for source in stems["sources"]:
            mix = mono[source].clone() if mix is None else mix + mono[source]
        audio = {"waveform": mix[None, None], "sample_rate": sample_rate}
        stems_payload = {
            "model": stems["model"],
            "sources": stems["sources"],
            "env_rate": _ENV_RATE,
            "env": {k: _envelope(v, sample_rate) for k, v in mono.items()},
            "mix_env": _envelope(mix, sample_rate),
        }

    common = dict(
        sensitivity=sensitivity, min_gap_seconds=min_gap, min_strength=0.0,
        max_events=1_000_000, bass_hits=True, drops_and_stops=True, builds=True,
        sections=True, accents=True, on_beat_weight=on_beat_weight, **shaping,
    )
    # the grid is fitted on the run with impacts - the fuller hit list - and the
    # editor draws that one; the two runs share the tempo, so the phase agrees
    grid = {}
    with_impacts = detect_events(audio, impacts=True, grid_out=grid, **common)
    without_impacts = detect_events(audio, impacts=False, **common)
    return {
        "file": name,
        "bpm": grid.get("bpm", 0.0),
        "grid": grid,
        "on_beat_weight": on_beat_weight,
        "path_mtime": os.path.getmtime(path),
        "duration": round(float(duration), 2),
        "sensitivity": sensitivity,
        "min_gap_seconds": min_gap,
        "shaping": dict(shaping),
        "kinds": list(EVENT_TYPES),
        "stems": stems_payload,
        "events": _slim(with_impacts),
        "events_no_impacts": _slim(without_impacts),
    }


async def _preview(request):
    body = await request.json()
    name = str(body.get("audio") or "").strip()
    if not name:
        return web.json_response(
            {"error": "No audio file: connect a Load Audio node to this node's `audio` input first."},
            status=400,
        )
    try:
        sensitivity = float(body.get("sensitivity") or 1.0)
        min_gap = float(body.get("min_gap_seconds") or 0.18)
        shaping = _shaping_from(body)
        on_beat_weight = round(min(1.0, max(0.0, float(body.get("on_beat_weight") or 0.0))), 3)
    except (TypeError, ValueError):
        return web.json_response({"error": "sensitivity / min_gap_seconds / gain / curve / EQ must be numbers."}, status=400)
    stems = _stems_from(body)

    stems_key = (stems["model"], tuple(stems["sources"])) if stems else None
    key = (name, round(sensitivity, 4), round(min_gap, 4), on_beat_weight, stems_key) + tuple(shaping[k] for k in SHAPING_KEYS)
    cached = _CACHE.get(key)
    if cached is not None:
        try:
            import folder_paths
            still_same = os.path.getmtime(folder_paths.get_annotated_filepath(name)) == cached["path_mtime"]
        except Exception:
            still_same = False
        if still_same:
            return web.json_response(cached)

    loop = asyncio.get_running_loop()
    try:
        payload = await loop.run_in_executor(None, _compute, name, sensitivity, min_gap, shaping, on_beat_weight, stems)
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except Exception as exc:  # decoding / torch errors - show them, do not 500 blindly
        return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = payload
    return web.json_response(payload)


if PromptServer is not None and web is not None and getattr(PromptServer, "instance", None) is not None:
    PromptServer.instance.routes.post("/apnext/h3/sound_events_preview")(_preview)
