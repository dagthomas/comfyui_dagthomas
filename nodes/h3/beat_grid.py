# APNext H3 Beat Grid + Beat Emphasis
#
# Two nodes about the PULSE of a song rather than its moments:
#
#   * H3BeatGrid      a straight BPM detector. Tempo from the autocorrelation
#                     of the onset envelope (music_support.estimate_bpm),
#                     phase fitted to the song's own onsets - or to the hits
#                     from Sound Events when they are wired in - and a guess
#                     at the downbeat. Out come the tempo, every beat as a
#                     time, and a JSON grid other nodes can read.
#
#   * H3BeatEmphasis  a CONDITIONING mix for the masked-audio render. H3
#                     generates the picture to match the audio it is given;
#                     sync_check measured ~58% of hits with a picture change
#                     within 80 ms against 49% by chance, i.e. the model nods
#                     along but does not really land on the beat. This node
#                     makes the beats impossible to miss - a short gain spike
#                     on every hit, everything between them ducked, an
#                     optional synthetic click / thump layered on top - for
#                     the audio the model LISTENS to, while the original song
#                     still goes to the output (Save Clip / Scenes Join /
#                     Chain Render's `audio`). Whether the picture follows is
#                     an experiment: render with and without, run Sync Check
#                     on both, keep whichever measures higher.
#
# Pure torch, like the rest of the audio nodes.

import json
import math

import torch

from ...utils.constants import CUSTOM_CATEGORY
from .common import with_advanced_inputs
from .music_support import _mono, analyse, estimate_bpm, fmt_time
from .sound_events import GRID_TOLERANCE, _grid_distance, beat_grid, detect_events, parse_events, shape_signal

# Which grid lines a subdivision puts between two beats.
SUBDIVISIONS = {
    "beats": 1,
    "eighths (2 per beat)": 2,
    "sixteenths (4 per beat)": 4,
    "half notes (every 2nd beat)": 0.5,
}

# The kinds Beat Emphasis hits on. A STOP is energy leaving, a BUILD a ramp,
# a SECTION a turn - none of them is an instant to spike.
EMPHASIS_KINDS = ("BASS HIT", "IMPACT", "ACCENT", "DROP")


# ----------------------------------------------------------------------
# The grid
# ----------------------------------------------------------------------

def _onset_at(onset, hop, times):
    """Onset strength sampled at `times` (s), linear interpolation."""
    n = onset.numel()
    pos = torch.tensor([t / hop for t in times], dtype=torch.float32)
    i0 = torch.clamp(pos.floor().long(), 0, n - 1)
    i1 = torch.clamp(i0 + 1, 0, n - 1)
    frac = pos - i0.float()
    return onset[i0] * (1 - frac) + onset[i1] * frac


def fit_phase_to_onsets(onset, hop, period, duration):
    """
    The phase (0..period) whose grid lines sit on the most onset energy, and
    how clearly it wins: best / mean over the 64 candidates (1.0 = no pulse at
    all, 3+ = a metronome).
    """
    scores = []
    for k in range(64):
        phase = period * k / 64.0
        lines = [phase + m * period for m in range(int((duration - phase) / period) + 1)]
        if not lines:
            scores.append(0.0)
            continue
        scores.append(float(_onset_at(onset, hop, lines).mean()))
    best = max(range(64), key=lambda k: scores[k])
    mean = sum(scores) / len(scores)
    return period * best / 64.0, (scores[best] / mean if mean > 1e-9 else 1.0)


def fit_downbeat(onset, hop, period, phase, beats_per_bar, duration, hits=None):
    """
    Which beat is "one": the offset (0..beats_per_bar-1) whose beats carry the
    most onset energy - or, with hits, the most hit strength. A guess, and
    labelled as one; a 4/4 backbeat can put the snare on 2 and 4 louder than
    the kick on 1.
    """
    if beats_per_bar <= 1:
        return 0
    best, best_score = 0, -1.0
    for off in range(beats_per_bar):
        lines = [phase + (off + m * beats_per_bar) * period for m in range(int((duration - phase) / (period * beats_per_bar)) + 1)]
        lines = [t for t in lines if t <= duration]
        if not lines:
            continue
        if hits:
            score = sum(s for t, s in hits if min(abs(t - l) for l in lines) <= GRID_TOLERANCE)
        else:
            score = float(_onset_at(onset, hop, lines).mean())
        if score > best_score:
            best, best_score = off, score
    return best


def build_grid(audio, events_text="", bpm_override=0.0, beats_per_bar=4, offset_ms=0, subdivision="beats"):
    """
    {"bpm", "period", "phase", "downbeat", "beats_per_bar", "subdivision",
     "confidence", "on", "of", "lines": [{"t", "beat", "bar", "sub"}], "beats": [...], "bars": [...]}
    """
    feats = analyse(audio)
    duration = float(feats["duration"])
    bpm = float(bpm_override or 0.0)
    if bpm <= 0:
        bpm = float(estimate_bpm(feats))
    if bpm <= 0:
        return {"bpm": 0.0, "period": 0.0, "phase": 0.0, "downbeat": 0, "beats_per_bar": beats_per_bar,
                "subdivision": subdivision, "confidence": 0.0, "on": 0, "of": 0, "lines": [], "beats": [], "bars": [],
                "duration": duration}
    period = 60.0 / bpm

    hits = []
    events = parse_events(events_text) if events_text and events_text.strip() else []
    for e in events:
        if e.get("type") in ("BASS HIT", "IMPACT", "ACCENT"):
            hits.append((float(e["t"]), float(e.get("strength", 0.5) or 0.5)))

    if hits:
        fit = beat_grid([(t, "BASS HIT", s) for t, s in hits], bpm)
        phase, on, of = fit["phase"], fit["on"], fit["of"]
        confidence = round(on / of, 3) if of else 0.0
    else:
        phase, ratio = fit_phase_to_onsets(feats["onset"], feats["hop_seconds"], period, duration)
        on, of = 0, 0
        confidence = round(min(1.0, max(0.0, (ratio - 1.0) / 2.0)), 3)   # 1.0x = nothing, 3x = certain

    phase = (phase + float(offset_ms or 0) / 1000.0) % period
    downbeat = fit_downbeat(feats["onset"], feats["hop_seconds"], period, phase, int(beats_per_bar), duration, hits or None)
    # roll the phase so the FIRST grid line is a downbeat where possible
    first_bar_phase = phase + downbeat * period
    while first_bar_phase - period * beats_per_bar >= 0:
        first_bar_phase -= period * beats_per_bar

    per_beat = SUBDIVISIONS.get(subdivision, 1)
    step = period / per_beat if per_beat >= 1 else period * int(round(1 / per_beat))
    lines, beats, bars = [], [], []
    k = 0
    t = phase
    while t <= duration + 1e-6:
        beat_index = int(round((t - phase) / period))
        is_beat = abs(t - (phase + beat_index * period)) < 1e-4
        bar_pos = (beat_index - downbeat) % int(beats_per_bar) if is_beat else -1
        bar_no = (beat_index - downbeat) // int(beats_per_bar) + 1 if is_beat else -1
        line = {"t": round(t, 3), "beat": beat_index + 1 if is_beat else None,
                "bar": bar_no if is_beat and bar_no >= 1 else None,
                "sub": None if is_beat else round(((t - phase) % period) / period, 3)}
        lines.append(line)
        if is_beat:
            beats.append(round(t, 3))
            if bar_pos == 0 and bar_no >= 1:
                bars.append(round(t, 3))
        k += 1
        t = phase + k * step
    return {"bpm": round(bpm, 2), "period": round(period, 5), "phase": round(phase, 4), "downbeat": int(downbeat),
            "beats_per_bar": int(beats_per_bar), "subdivision": subdivision, "confidence": confidence,
            "on": on, "of": of, "lines": lines, "beats": beats, "bars": bars, "duration": round(duration, 3)}


def grid_table(grid):
    """One line per grid line, absolute times - readable and re-parseable."""
    if not grid.get("bpm"):
        return "# no steady pulse found - nothing to grid"
    head = [
        f"# {grid['bpm']:g} BPM, one beat every {grid['period']:.3f} s, first beat at {grid['phase']:.3f} s, "
        f"{grid['beats_per_bar']}/4 bars (downbeat = beat {grid['downbeat'] + 1}, a guess)",
        f"# {len(grid['beats'])} beats, {len(grid['bars'])} bars in {fmt_time(grid['duration'])}"
        + (f"; {grid['on']}/{grid['of']} detected hits sit on the grid" if grid.get("of") else f"; confidence {grid['confidence']:.2f}"),
    ]
    rows = []
    for l in grid["lines"]:
        if l["beat"] is not None:
            bar = f" | bar {l['bar']}" if l["bar"] else ""
            one = " | ONE" if l["bar"] and (l["beat"] - 1 - grid["downbeat"]) % grid["beats_per_bar"] == 0 else ""
            rows.append(f"[{fmt_time(l['t'])}] BEAT {l['beat']}{bar}{one}")
        else:
            rows.append(f"[{fmt_time(l['t'])}]   sub {l['sub']:.2f}")
    return "\n".join(head + rows)


class H3BeatGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The song (Load Audio). Passed through unchanged."}),
            },
            "optional": {
                "events": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Sound Events' `events` (or `events_json`). With it the grid's phase is fitted to the "
                        "detected hits and the table reports how many sit on it; without it the phase is "
                        "fitted to the song's own onset envelope."
                    ),
                }),
                "bpm_override": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1,
                    "tooltip": "Force the tempo (you know the song is 128). 0 = measure it.",
                }),
                "beats_per_bar": ("INT", {
                    "default": 4, "min": 1, "max": 12,
                    "tooltip": "Beats per bar, for the bar numbers and the downbeat guess.",
                }),
                "offset_ms": ("INT", {
                    "default": 0, "min": -500, "max": 500, "step": 5,
                    "tooltip": "Nudge every grid line, positive = later.",
                }),
                "subdivision": (list(SUBDIVISIONS), {
                    "default": "beats",
                    "tooltip": "Which grid lines to list: beats, or eighths / sixteenths between them, or every 2nd beat.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("audio", "bpm", "beat_times", "grid_json", "summary", "count")
    OUTPUT_TOOLTIPS = (
        "The same audio, passed through.",
        "The tempo in BPM (0 = no steady pulse).",
        "One line per grid line: [m:ss.xx] BEAT n | bar b | ONE on the downbeats.",
        "The grid as JSON: bpm, period, phase, downbeat, beats[], bars[], lines[]. Wire into Beat Emphasis.",
        "One line: tempo, first beat, bars, confidence.",
        "How many grid lines.",
    )
    OUTPUT_NODE = True
    FUNCTION = "grid"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "A straight BPM detector: the tempo from the onset envelope's autocorrelation, the beat phase "
        "fitted to the song's onsets (or to Sound Events' hits when wired), a downbeat guess, and every "
        "beat as a time. Feeds Beat Emphasis, and lines up with the grid the Sound Events editor draws."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def grid(self, audio, events="", bpm_override=0.0, beats_per_bar=4, offset_ms=0, subdivision="beats"):
        grid = build_grid(audio, events_text=events or "", bpm_override=bpm_override, beats_per_bar=beats_per_bar,
                          offset_ms=offset_ms, subdivision=subdivision)
        table = grid_table(grid)
        if grid["bpm"]:
            summary = (f"{grid['bpm']:g} BPM, first beat {grid['phase']:.2f}s, {len(grid['beats'])} beats / "
                       f"{len(grid['bars'])} bars in {fmt_time(grid['duration'])}"
                       + (f", {grid['on']}/{grid['of']} hits on the grid" if grid.get("of") else f", confidence {grid['confidence']:.2f}"))
        else:
            summary = f"no steady pulse found in {fmt_time(grid['duration'])}"
        print(f"\U0001F3B5 H3 Beat Grid | {summary}")
        slim = {k: v for k, v in grid.items() if k != "lines"}
        slim["lines"] = grid["lines"]
        return {
            "ui": {"text": [f"{summary}\n{table}"]},
            "result": (audio, float(grid["bpm"]), table, json.dumps(slim), summary, len(grid["lines"])),
        }


with_advanced_inputs(H3BeatGrid, ("audio", "events", "bpm_override"))


# ----------------------------------------------------------------------
# Beat emphasis
# ----------------------------------------------------------------------

LAYERS = ["none", "click", "thump", "click + thump"]


def _synth_click(sr):
    """A 2 ms noise burst into a 3 kHz ping - a rimshot's edge, ~10 ms."""
    n = int(sr * 0.012)
    t = torch.arange(n, dtype=torch.float32) / sr
    ping = torch.sin(2 * math.pi * 3000.0 * t) * torch.exp(-t * 500.0)
    burst = torch.randn(n) * torch.exp(-t * 2500.0) * 0.6
    return (ping + burst) * 0.9


def _synth_thump(sr):
    """A kick: 110 -> 45 Hz sweep, 140 ms, with a little attack transient."""
    n = int(sr * 0.14)
    t = torch.arange(n, dtype=torch.float32) / sr
    freq = 45.0 + 65.0 * torch.exp(-t * 40.0)
    phase = torch.cumsum(2 * math.pi * freq / sr, dim=0)
    body = torch.sin(phase) * torch.exp(-t * 18.0)
    attack = torch.randn(n) * torch.exp(-t * 1500.0) * 0.25
    return body + attack


def emphasis_envelope_db(n, sr, hits, boost_db, duck_db, attack_ms, hold_ms, decay_ms, scale_by_strength):
    """
    Per-sample gain in dB: `duck_db` everywhere, and around each hit a spike
    to `boost_db` (times the hit's strength when scaling) - a linear ramp up
    over `attack_ms`, flat for `hold_ms`, a straight line in dB back down
    over `decay_ms`. Overlaps take the louder one.
    """
    env = torch.full((n,), float(duck_db), dtype=torch.float32)
    a = max(1, int(sr * attack_ms / 1000.0))
    h = max(0, int(sr * hold_ms / 1000.0))
    d = max(1, int(sr * decay_ms / 1000.0))
    ramp_up = torch.linspace(0.0, 1.0, a)
    ramp_down = torch.linspace(1.0, 0.0, d)
    shape = torch.cat([ramp_up, torch.ones(h), ramp_down])          # 0..1
    for t, strength in hits:
        top = float(boost_db) * (float(strength) if scale_by_strength else 1.0)
        start = int(round(t * sr)) - a                                # the peak lands ON the hit
        seg = float(duck_db) + (top - float(duck_db)) * shape
        lo, hi = max(0, start), min(n, start + seg.numel())
        if hi <= lo:
            continue
        env[lo:hi] = torch.maximum(env[lo:hi], seg[lo - start:hi - start])
    return env


def emphasise(audio, hits, boost_db=9.0, duck_db=-6.0, attack_ms=5, hold_ms=40, decay_ms=120,
              layer="thump", layer_level=0.5, scale_by_strength=True, dynamics_curve=1.0, normalize=True):
    wave = audio["waveform"]
    sr = int(audio["sample_rate"])
    squeeze = wave.dim() == 2
    if squeeze:
        wave = wave.unsqueeze(0)
    b, c, n = wave.shape
    out = wave.detach().to("cpu").float().clone()

    env = emphasis_envelope_db(n, sr, hits, boost_db, duck_db, attack_ms, hold_ms, decay_ms, scale_by_strength)
    out = out * (10.0 ** (env / 20.0)).view(1, 1, n)

    if layer != "none" and layer_level > 0 and hits:
        layer_wave = torch.zeros(n, dtype=torch.float32)
        parts = []
        if "click" in layer:
            parts.append(_synth_click(sr))
        if "thump" in layer:
            parts.append(_synth_thump(sr))
        for t, strength in hits:
            amp = float(layer_level) * (float(strength) if scale_by_strength else 1.0)
            start = int(round(t * sr))
            for p in parts:
                lo, hi = max(0, start), min(n, start + p.numel())
                if hi > lo:
                    layer_wave[lo:hi] += p[lo - start:hi - start] * amp
        out = out + layer_wave.view(1, 1, n)

    if abs(float(dynamics_curve) - 1.0) >= 1e-3:
        for bi in range(b):
            for ci in range(c):
                out[bi, ci] = shape_signal(out[bi, ci], sr, {"dynamics_curve": dynamics_curve})

    peak = float(out.abs().max()) if out.numel() else 0.0
    if normalize and peak > 1e-6:
        out = out * (0.98 / peak)
    elif peak > 1.0:
        out = out.clamp(-1.0, 1.0)
    if squeeze:
        out = out[0]
    return {"waveform": out.contiguous(), "sample_rate": sr}, peak


def hits_from(events_text, grid_json, audio, merge_window=0.06):
    """
    [(t, strength)] to emphasise: Sound Events' hits (BASS HIT / IMPACT /
    ACCENT / DROP), the grid's beats (strength 0.6, downbeats 0.8), or both
    (a beat within `merge_window` of a hit is the hit). With neither wired,
    the hit detectors run here with their defaults.
    """
    hits = []
    if events_text and events_text.strip():
        for e in parse_events(events_text):
            if e.get("type") in EMPHASIS_KINDS:
                hits.append((float(e["t"]), max(0.05, min(1.0, float(e.get("strength", 0.6) or 0.6)))))
    grid = None
    if grid_json and grid_json.strip():
        try:
            grid = json.loads(grid_json)
        except Exception:
            grid = None
    if grid and grid.get("beats"):
        bars = set(grid.get("bars") or [])
        for t in grid["beats"]:
            t = float(t)
            if any(abs(t - h) <= merge_window for h, _ in hits):
                continue
            hits.append((t, 0.8 if t in bars else 0.6))
    if not hits and not (events_text and events_text.strip()) and grid is None:
        for e in detect_events(audio, bass_hits=True, impacts=True, drops_and_stops=True, builds=False,
                               sections=False, accents=False):
            if e["type"] in EMPHASIS_KINDS:
                hits.append((float(e["t"]), float(e["strength"])))
    hits.sort()
    return hits


class H3BeatEmphasis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": (
                    "What the render will be conditioned on. For a music video feed the VOCAL STEM "
                    "(AudioSeparation's vocals), not the song: the lips follow the voice, the thumps add the "
                    "beat - the full mix spiked and ducked buries the voice and the lip-sync with it."
                )}),
                "boost_db": ("FLOAT", {
                    "default": 9.0, "min": 0.0, "max": 24.0, "step": 0.5,
                    "tooltip": "Gain spike on every hit, dB above the ducked level. The peak lands exactly on the hit.",
                }),
                "duck_db": ("FLOAT", {
                    "default": -6.0, "min": -36.0, "max": 0.0, "step": 0.5,
                    "tooltip": "Level between hits, dB. Lower = more contrast; -36 leaves almost only the hits.",
                }),
                "layer": (LAYERS, {
                    "default": "thump",
                    "tooltip": "A synthetic transient added on every hit: a click (3 kHz ping), a thump (kick sweep), or both.",
                }),
                "layer_level": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How loud the layer is (0-1, before normalising).",
                }),
            },
            "optional": {
                "events": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Sound Events' `events`: the hits (bass hits, impacts, accents, drops) to emphasise.",
                }),
                "grid_json": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Beat Grid's `grid_json`: every beat gets emphasised (downbeats a little more). With both wired, beats near a hit defer to the hit.",
                }),
                "attack_ms": ("INT", {"default": 5, "min": 1, "max": 100, "tooltip": "Ramp up to the spike, ms (the spike peaks ON the hit)."}),
                "hold_ms": ("INT", {"default": 40, "min": 0, "max": 500, "tooltip": "How long the spike stays at full boost, ms."}),
                "decay_ms": ("INT", {"default": 120, "min": 1, "max": 2000, "tooltip": "How long the spike takes to fall back to the ducked level, ms."}),
                "scale_by_strength": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Scale the boost and the layer by each hit's strength, so a light hit gets a light nudge and a heavy one the full spike.",
                }),
                "dynamics_curve": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                    "tooltip": "|x|^curve after the spikes: above 1 expands (the spikes stand even prouder), below 1 compresses.",
                }),
                "normalize": ("BOOLEAN", {"default": True, "tooltip": "Peak-normalise the result to -0.2 dBFS so the boosts cannot clip."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "INT")
    RETURN_NAMES = ("audio", "summary", "count")
    OUTPUT_TOOLTIPS = (
        "The emphasised mix - wire into the masked-audio node's `master_audio` (or Chain Render's "
        "`conditioning_audio`). NOT into Save Clip: the output video keeps the original song.",
        "What was done.",
        "How many hits were emphasised.",
    )
    OUTPUT_NODE = True
    FUNCTION = "emphasise"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Makes the beats impossible for the video model to miss: a gain spike on every hit, everything "
        "between ducked, an optional click / thump layered on. For the audio H3 is CONDITIONED on (the "
        "masked-audio node / Chain Render's conditioning_audio) - the original song still goes to the "
        "output. Feed it the vocal stem for a music video (voice + thumps: lip-sync AND beat); with a "
        "small boost and no duck the singing stays intact. Measure with Sync Check whether the picture "
        "follows more closely."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def emphasise(self, audio, boost_db, duck_db, layer, layer_level, events="", grid_json="",
                  attack_ms=5, hold_ms=40, decay_ms=120, scale_by_strength=True, dynamics_curve=1.0, normalize=True):
        hits = hits_from(events or "", grid_json or "", audio)
        out, peak = emphasise(
            audio, hits, boost_db=boost_db, duck_db=duck_db, attack_ms=attack_ms, hold_ms=hold_ms,
            decay_ms=decay_ms, layer=layer, layer_level=layer_level, scale_by_strength=scale_by_strength,
            dynamics_curve=dynamics_curve, normalize=normalize,
        )
        source = ("events + grid" if (events and events.strip()) and (grid_json and grid_json.strip())
                  else "events" if events and events.strip() else "grid" if grid_json and grid_json.strip() else "own detection")
        summary = (f"{len(hits)} hit(s) emphasised from {source}: +{boost_db:g} dB spikes over {duck_db:g} dB, "
                   f"{attack_ms}/{hold_ms}/{decay_ms} ms, layer {layer} @ {layer_level:g}"
                   + (f", curve {dynamics_curve:g}" if abs(dynamics_curve - 1) >= 1e-3 else "")
                   + (f", normalised from peak {peak:.2f}" if normalize else ""))
        print(f"\U0001F50A H3 Beat Emphasis | {summary}")
        return {"ui": {"text": [summary]}, "result": (out, summary, len(hits))}


with_advanced_inputs(H3BeatEmphasis, ("audio", "boost_db", "duck_db", "layer", "layer_level", "events", "grid_json"))
