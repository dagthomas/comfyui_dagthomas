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
#        {audio, sensitivity, min_gap_seconds}
#     -> {file, duration, events: [{t, type, strength, label}], events_no_impacts: [...]}
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

from .sound_events import EVENT_TYPES, detect_events

_CACHE = {}
_CACHE_LIMIT = 6


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


def _compute(name, sensitivity, min_gap):
    path, audio = _load_audio(name)
    common = dict(
        sensitivity=sensitivity, min_gap_seconds=min_gap, min_strength=0.0,
        max_events=1_000_000, bass_hits=True, drops_and_stops=True, builds=True,
        sections=True, accents=True,
    )
    with_impacts = detect_events(audio, impacts=True, **common)
    without_impacts = detect_events(audio, impacts=False, **common)
    duration = audio["waveform"].shape[-1] / audio["sample_rate"]
    return {
        "file": name,
        "path_mtime": os.path.getmtime(path),
        "duration": round(float(duration), 2),
        "sensitivity": sensitivity,
        "min_gap_seconds": min_gap,
        "kinds": list(EVENT_TYPES),
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
    except (TypeError, ValueError):
        return web.json_response({"error": "sensitivity / min_gap_seconds must be numbers."}, status=400)

    key = (name, round(sensitivity, 4), round(min_gap, 4))
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
        payload = await loop.run_in_executor(None, _compute, name, sensitivity, min_gap)
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
