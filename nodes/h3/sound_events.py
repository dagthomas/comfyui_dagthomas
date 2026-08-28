# APNext H3 Sound Events
#
# Finds the moments in a song a video should be cut on - bass hits, impacts,
# drops, builds, stops, section changes - and labels them with the second they
# land at, so the writer can stage the picture ON the music instead of near it.
#
# The detectors are the ones fitted offline against real tracks in graphgen's
# AudioEngine (X:/KODE/graphgen/src/lib/audio/AudioEngine.svelte.ts), ported
# from its real-time Web Audio form to a whole-song torch pass:
#
#   * bass hits    band-limited SPECTRAL FLUX (positive per-bin rises summed
#                  over 20-250 Hz) against an ADAPTIVE MEDIAN floor with a
#                  refractory gap. A kick lifts every low bin at once, while a
#                  bassline note change moves only a few - which is why flux
#                  beats a plain level threshold on a compressed master.
#   * impacts      time-domain RMS rise against the recent minimum. A slam or
#                  a body hit is a broadband transient the FFT smears out, so
#                  it needs the waveform, not the spectrum.
#   * drops/stops  signed loudness novelty: the mean level of the 1.5 s AFTER
#                  a point minus the 1.5 s BEFORE it. Strongly positive is the
#                  beat entering, strongly negative is the music cutting out.
#   * builds       a sustained upward loudness ramp that lands on a drop.
#   * sections     the same novelty over a 4 s window - verse/chorus turns.
#
# What the picture should DO about any of it is not decided here. This node
# reports what the music does and when; the writer stages it.
#
# Offline changes two things for the better: the median window can be CENTRED
# on each frame instead of trailing it, and every threshold is relative to the
# whole track rather than to whatever has played so far, so the first bar is
# scored as accurately as the last.
#
# Pure torch, like music_support - no librosa.

import json
import math
import re

import torch

from ...utils.constants import CUSTOM_CATEGORY
from .common import with_advanced_inputs
from .music_support import _mono, analyse, estimate_bpm, fmt_time, song_profile

# 100 Hz analysis grid: 10 ms is tighter than a listener can place a hit, and
# twice the resolution music_support uses for segmenting (that only has to land
# on a frame boundary; this has to land on a beat).
HOP_SECONDS = 0.01

# Band edges, from the AudioEngine. Sub and bass are split so a sub-drop and a
# kick can be told apart.
BANDS = {
    "sub": (20.0, 60.0),
    "bass": (60.0, 250.0),
    "low_mid": (250.0, 500.0),
    "mid": (500.0, 2000.0),
    "treble": (2000.0, 12000.0),
}

# Event vocabulary. Order is the priority used when the list has to be capped:
# structure first, then the hit stream.
EVENT_TYPES = ("DROP", "STOP", "SECTION", "BUILD", "IMPACT", "BASS HIT", "ACCENT")

_STRUCTURAL = ("DROP", "STOP", "SECTION", "BUILD")

# What each event SOUNDS like, in plain words, at (light, solid, heavy).
#
# The bare token is for the parser and for sorting; this is what the model
# actually reads. "DROP" is a word in a column - "a big DROP in the music" is
# something a director can stage, and the size word is doing real work: the
# same token at 0.2 and at 0.95 are not the same event and must not read alike.
# The token stays in the phrase, in caps, so the prose and the column are
# unmistakably the same moment.
_SOUND = {
    "DROP": (
        "a small DROP - the beat slips back in",
        "a big DROP in the music - full energy returns",
        "a huge DROP - the whole track opens up at once",
    ),
    "STOP": (
        "a brief STOP - the music thins out",
        "a STOP - the music cuts out",
        "a dead STOP - the track falls to nothing",
    ),
    "SECTION": (
        "a light SECTION turn - the arrangement shifts",
        "a SECTION change - a new part of the song",
        "a hard SECTION turn - the song becomes another song",
    ),
    "BUILD": (
        "a short BUILD - a slow lift toward the drop",
        "a BUILD - a riser winding up into the drop",
        "a long BUILD - pressure climbing all the way to the drop",
    ),
    "IMPACT": (
        "a light IMPACT - a sharp crack",
        "an IMPACT - a slam across the whole mix",
        "a huge IMPACT - a blast, the loudest kind of hit",
    ),
    "BASS HIT": (
        "a light BASS HIT - a soft kick",
        "a solid BASS HIT - a kick with weight behind it",
        "a heavy BASS HIT - a low blow you feel in the chest",
    ),
    "ACCENT": (
        "a faint ACCENT - a tick up top",
        "an ACCENT - a bright hat or snare",
        "a hard ACCENT - a cymbal crash",
    ),
}

# How long BEFORE the sound the picture has to START moving, per kind, at
# (light, solid, heavy).
#
# This is the whole reason a "synced" video reads out of sync. What a viewer
# perceives as the moment of an effect is its PEAK, not its first frame - so a
# move told to begin ON the beat peaks a fifth of a second late, on every hit,
# in every clip, and the finished video drifts off the music even though the
# audio is sample-aligned. Every event therefore goes to the writer as a
# WINDOW - start the move here, land it on the beat, settle it by there - and
# these are the lead times. Ordinary animation anticipation, scaled by how big
# the sound is: a cymbal tick needs two frames of warning, a full drop needs a
# held breath.
_LEAD = {
    "DROP": (0.55, 0.85, 1.20),
    "STOP": (0.40, 0.60, 0.85),
    "SECTION": (0.30, 0.40, 0.55),
    "BUILD": (0.90, 1.40, 2.00),
    "IMPACT": (0.32, 0.45, 0.60),
    "BASS HIT": (0.20, 0.28, 0.36),
    "ACCENT": (0.08, 0.12, 0.16),
}

# ...and how long AFTER it the aftermath runs. Dust does not stop in the air on
# the frame after the kick, and a move that ends the instant it lands is the
# other half of looking mechanical.
_TAIL = {
    "DROP": (0.55, 0.80, 1.10),
    "STOP": (0.50, 0.75, 1.10),
    "SECTION": (0.35, 0.45, 0.60),
    "BUILD": (0.0, 0.0, 0.0),   # a build's follow-through IS the drop
    "IMPACT": (0.35, 0.50, 0.70),
    "BASS HIT": (0.22, 0.30, 0.40),
    "ACCENT": (0.10, 0.14, 0.18),
}


def _strength_label(value):
    return "heavy" if value >= 0.66 else "solid" if value >= 0.33 else "light"


def _strength_tier(value):
    return 2 if value >= 0.66 else 1 if value >= 0.33 else 0


def _sound_note(kind, strength):
    """The plain-words description of one event, sized by its strength."""
    tiers = _SOUND.get(kind)
    return tiers[_strength_tier(strength)] if tiers else ""


def _tiered(table, kind, strength):
    """The (light, solid, heavy) value for one event, or 0 for an unknown kind."""
    tiers = table.get(kind)
    return tiers[_strength_tier(strength)] if tiers else 0.0


def event_window(event):
    """
    (start, land, settle) in absolute song seconds for one event.

    `land` is the second the sound is on. `start` is when the picture has to
    begin moving for its peak to arrive there, and `settle` is where the
    aftermath finishes. A BUILD needs no guessed lead: the detector already
    knows the drop it ramps into, so its window is the real ramp.
    """
    kind = event.get("type", "")
    strength = float(event.get("strength", 0.5))
    land = float(event.get("t", 0.0))
    until = event.get("until")
    lead = event.get("lead")
    if kind == "BUILD" and until is not None:
        return land, float(until), float(until)
    # a hand-set window (the editor's drag handles) wins over the tier tables
    start = land - float(lead) if lead is not None else land - _tiered(_LEAD, kind, strength)
    settle = float(until) if until is not None else land + _tiered(_TAIL, kind, strength)
    return max(0.0, start), land, max(land, settle)


# ----------------------------------------------------------------------
# Spectral front end
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Signal shaping: what the detectors get to listen to
# ----------------------------------------------------------------------
# Every detector scores a peak against the audio around it, so on a loud,
# compressed master a kick and a hi-hat can clear the floor by the same
# margin and the list fills with the wrong hits. Shaping the signal FIRST -
# tilting the mix toward the kick, pulling the hats down, bending the
# dynamics so the big hits stand proud of the rest - makes the beats you
# want the loudest thing the detectors see. The chain is gain -> EQ -> curve,
# like pedals in a row, and the editor draws the shaped signal under the
# events (and can monitor it) so what you see and hear is what is scored.

# The five EQ bands: (kwarg, RBJ biquad kind, centre Hz, Q). The editor's
# Web Audio monitor builds the same five BiquadFilterNodes with the same
# numbers; Web Audio and this file share the RBJ cookbook, so the magnitude
# response matches. Here the response is applied in the frequency domain
# (zero phase, no group delay) so hits keep their exact timing.
EQ_BANDS = (
    ("eq_sub_db", "lowshelf", 60.0, 0.707),
    ("eq_bass_db", "peaking", 150.0, 0.9),
    ("eq_low_db", "peaking", 400.0, 0.9),
    ("eq_mid_db", "peaking", 1500.0, 0.9),
    ("eq_high_db", "highshelf", 5000.0, 0.707),
)
SHAPING_KEYS = ("gain_db", "dynamics_curve") + tuple(k for k, *_ in EQ_BANDS)


def _biquad_response(kind, f0, q, gain_db, freqs, sr):
    """|H(f)| of an RBJ cookbook biquad, evaluated on `freqs` (Hz)."""
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    root = 2.0 * math.sqrt(amp) * alpha
    if kind == "peaking":
        b = (1.0 + alpha * amp, -2.0 * cos_w0, 1.0 - alpha * amp)
        a = (1.0 + alpha / amp, -2.0 * cos_w0, 1.0 - alpha / amp)
    elif kind == "lowshelf":
        b = (amp * ((amp + 1) - (amp - 1) * cos_w0 + root),
             2 * amp * ((amp - 1) - (amp + 1) * cos_w0),
             amp * ((amp + 1) - (amp - 1) * cos_w0 - root))
        a = ((amp + 1) + (amp - 1) * cos_w0 + root,
             -2 * ((amp - 1) + (amp + 1) * cos_w0),
             (amp + 1) + (amp - 1) * cos_w0 - root)
    elif kind == "highshelf":
        b = (amp * ((amp + 1) + (amp - 1) * cos_w0 + root),
             -2 * amp * ((amp - 1) + (amp + 1) * cos_w0),
             amp * ((amp + 1) + (amp - 1) * cos_w0 - root))
        a = ((amp + 1) - (amp - 1) * cos_w0 + root,
             2 * ((amp - 1) - (amp + 1) * cos_w0),
             (amp + 1) - (amp - 1) * cos_w0 - root)
    else:
        raise ValueError(f"unknown biquad kind {kind!r}")
    w = 2.0 * math.pi * freqs / sr
    z1 = torch.exp(-1j * w)
    z2 = z1 * z1
    num = b[0] + b[1] * z1 + b[2] * z2
    den = a[0] + a[1] * z1 + a[2] * z2
    return (num / den).abs().float()


def shape_signal(wave, sr, shaping=None):
    """
    gain -> EQ -> dynamics curve on a mono float waveform.

    `shaping` is {gain_db, dynamics_curve, eq_sub_db, eq_bass_db, eq_low_db,
    eq_mid_db, eq_high_db}; anything missing is neutral. The curve raises
    |x| to a power relative to full scale (1.0): above 1 it expands, quiet
    material sinks and the big hits stand proud; below 1 it compresses, the
    soft hits come up. Gain sits in front of it for that reason - a track
    peaking at 0.5 needs lifting before a 2.0 curve can do anything but crush.
    """
    s = shaping or {}
    gain_db = float(s.get("gain_db") or 0.0)
    curve = float(s.get("dynamics_curve") or 1.0)
    eq = [(kind, f0, q, float(s.get(key) or 0.0)) for key, kind, f0, q in EQ_BANDS]
    eq = [b for b in eq if abs(b[3]) >= 0.05]
    if abs(gain_db) < 0.05 and not eq and abs(curve - 1.0) < 1e-3:
        return wave
    out = wave
    if abs(gain_db) >= 0.05:
        out = out * (10.0 ** (gain_db / 20.0))
    if eq:
        n = out.numel()
        spec = torch.fft.rfft(out)
        freqs = torch.fft.rfftfreq(n, 1.0 / sr)
        mag = torch.ones_like(freqs)
        for kind, f0, q, db in eq:
            mag = mag * _biquad_response(kind, f0, q, db, freqs, sr)
        out = torch.fft.irfft(spec * mag, n=n)
    if abs(curve - 1.0) >= 1e-3:
        out = torch.sign(out) * out.abs().clamp(min=0.0).pow(curve)
    return out.float()


def _frames(audio, shaping=None):
    """Magnitude spectrogram, bin frequencies, hop length and the waveform."""
    wave, sr = _mono(audio)
    wave = shape_signal(wave, sr, shaping)
    hop = max(1, int(round(sr * HOP_SECONDS)))
    n_fft = 2048 if sr >= 32000 else 1024
    if wave.numel() < n_fft:
        wave = torch.nn.functional.pad(wave, (0, n_fft - wave.numel()))
    spec = torch.stft(
        wave, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft),
        center=True, return_complex=True,
    ).abs()
    freqs = torch.arange(spec.shape[0], dtype=torch.float32) * sr / n_fft
    return spec, freqs, wave, hop / sr, sr


def _band_mask(freqs, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    if not bool(mask.any()):
        # a very low sample rate can leave a band with no bins at all
        mask[int(torch.argmin((freqs - lo).abs()))] = True
    return mask


def _band_flux(spec, freqs, lo, hi):
    """
    Positive log-magnitude rise summed across one band, per frame.

    Summing the per-bin rises (rather than differencing the band's total
    energy) is what separates a kick - every low bin lifting together - from a
    bassline moving to another note, where a few bins rise and others fall.
    """
    band = torch.log1p(spec[_band_mask(freqs, lo, hi)])
    rise = torch.clamp(band[:, 1:] - band[:, :-1], min=0).mean(dim=0)
    return torch.cat([rise[:1], rise])


def _band_level(spec, freqs, lo, hi):
    return spec[_band_mask(freqs, lo, hi)].mean(dim=0)


def _smooth(x, k):
    if k <= 1:
        return x
    pad = k // 2
    return torch.nn.functional.avg_pool1d(
        torch.nn.functional.pad(x.view(1, 1, -1), (pad, pad), mode="replicate"), k, stride=1
    ).view(-1)


def _rolling_median(x, half):
    """
    Centred rolling median, the adaptive floor every peak is scored against.

    A fixed threshold cannot work across a track: a quiet intro and a loud
    chorus have different noise floors, and a compressed master has almost
    none. The median of the surrounding window IS the local floor.
    """
    n = x.numel()
    if half < 1 or n < 3:
        return x.clone()
    padded = torch.nn.functional.pad(x.view(1, 1, -1), (half, half), mode="replicate").view(-1)
    windows = padded.unfold(0, 2 * half + 1, 1)[:n]
    return windows.median(dim=1).values


# ----------------------------------------------------------------------
# Peak picking
# ----------------------------------------------------------------------

def _pick_peaks(strength, floor, hop, min_gap, ratio, rearm=0.5):
    """
    Local maxima of `strength` that clear `floor * ratio`, at least `min_gap`
    apart, strongest first.

    `rearm` is the AudioEngine's arming rule and it does the work a refractory
    gap cannot: after a hit, the signal has to fall back below `rearm` x the
    threshold before another hit may fire. A kick's decay tail stays over the
    floor for longer than any gap you would dare set - wide enough and you lose
    real sixteenth notes - so without re-arming, one kick reads as two. Pass
    rearm=0 for detectors whose signal is already an isolated impulse.

    Returns [(index, value)]; the value is the raw peak height, which the
    caller normalises into a strength.
    """
    n = strength.numel()
    if n < 3:
        return []
    threshold = floor * ratio + 1e-4
    gap = max(1, int(round(min_gap / hop)))

    candidates = []
    for i in range(1, n - 1):
        v = float(strength[i])
        if v <= float(threshold[i]):
            continue
        if v < float(strength[i - 1]) or v < float(strength[i + 1]):
            continue
        candidates.append((v, i, v))

    # Greedy by height so that when two peaks fall inside one refractory gap
    # the LOUDER one survives - a trailing snare (or the blip before a kick)
    # must never mask the kick. Height, not height-over-threshold: the floor
    # drifts inside a gap and would otherwise let a small early blip outrank
    # the big hit 60 ms behind it.
    candidates.sort(reverse=True)
    kept = []
    for _score, i, v in candidates:
        if all(abs(i - j) > gap for j, _ in kept):
            kept.append((i, v))
    kept.sort()

    if rearm <= 0 or len(kept) < 2:
        return kept

    armed = [kept[0]]
    for i, v in kept[1:]:
        prev = armed[-1][0]
        between = strength[prev + 1:i]
        if between.numel() and not bool((between < threshold[prev + 1:i] * rearm).any()):
            continue  # never disarmed - still the previous hit ringing out
        armed.append((i, v))
    return armed


def _band_tilt(low_level, top_level, hop):
    """
    Per-frame "how much did each band jump", in LINEAR energy, each scaled by
    its own track-wide typical jump.

    Deliberately not the log flux the detectors run on: log1p compresses large
    magnitudes more than small ones, so as a section gets louder the low band's
    measured rise shrinks relative to the top band's and identical kicks start
    reading as cymbals halfway through a track. Linear energy has no such tilt,
    and dividing each band by its own median jump is what makes two bands with
    nothing in common comparable at all.
    """
    back = max(1, int(round(0.03 / hop)))

    def jumps(level):
        prev = torch.nn.functional.pad(
            level.view(1, 1, -1), (back, 0), mode="replicate"
        ).view(-1)[: level.numel()]
        rise = torch.clamp(level - prev, min=0)
        typical = float(rise.median())
        if typical <= 0:
            typical = float(rise.mean()) or 1.0
        return rise / max(typical, 1e-12)

    return jumps(low_level), jumps(top_level)


def _band_excess(signal, index, spread=2):
    """The strongest value of a scaled band-jump signal around one instant."""
    limit = signal.numel()
    lo, hi = max(0, index - spread), min(limit, index + spread + 1)
    return float(signal[lo:hi].max()) if lo < hi else 0.0


def _resolve_hits(bass, impacts, low_tilt, top_tilt, window=0.07, bias=1.5):
    """
    One label per instant for the two hit detectors.

    Where a bass hit and an impact land together they are the same drum, and
    the louder band names it: low-dominant is a BASS HIT, top-dominant is an
    IMPACT. Impacts that stand alone are kept only if they are genuinely
    bright, so a low thump the bass detector missed is not promoted to a slam.

    The top band has to win CLEARLY (`bias`), not just by a hair. IMPACT is the
    rare, notable label - a crash, a slam, a snare that lands like a punch -
    and a marginal call should come back as the ordinary one. Without the bias
    a slightly louder mix of the same kick flips label halfway through a track,
    which reads to the model as a change that is not in the music.
    """
    resolved = list(bass)
    gap = window

    for idx, t, _kind, strength in impacts:
        low = _band_excess(low_tilt, idx)
        top = _band_excess(top_tilt, idx)
        partner = next(
            (k for k, (_, bt, _, _) in enumerate(resolved) if abs(bt - t) <= gap), None
        )
        if partner is None:
            # a lone impact: only a clearly bright transient earns the name
            if top > max(low, 3.0):
                resolved.append((idx, t, "IMPACT", strength))
            continue
        if top > low * bias:
            # the same hit, and the top end clearly wins - relabel, earliest time
            bi, bt, _bk, bs = resolved[partner]
            resolved[partner] = (bi, min(bt, t), "IMPACT", max(bs, strength))
        # else: low dominates, it is the kick already recorded - drop this one

    resolved.sort(key=lambda e: e[1])
    return [(t, kind, s) for _i, t, kind, s in resolved]


# ----------------------------------------------------------------------
# Timing: from "the attack starts" to "the sound is there"
# ----------------------------------------------------------------------

# The detectors time an event at the frame where the spectral flux peaks -
# the frame whose 46 ms window first swallows the attack. That is when the
# sound STARTS, 20-50 ms before it is actually there (a kick's body swells
# for a few tens of ms after its click), so every event read a little early
# against the waveform. The fix is to look at the waveform itself: within a
# short window after the flux peak, the envelope's maximum is the instant
# the sound is on. Hits get a tight window; a DROP - timed from loudness
# over a much longer span - gets a wider one; stops, sections and builds are
# windows in their own right and are left alone.
_SNAP_WINDOW = {
    "BASS HIT": (0.040, 0.080),
    "IMPACT": (0.040, 0.080),
    "ACCENT": (0.005, 0.050),
    "DROP": (0.050, 0.150),
}


def _envelope(wave, sr, width=0.004):
    """Short-window RMS of the waveform - the sound's outline at ~4 ms resolution."""
    k = max(1, int(round(width * sr)))
    x = wave.view(1, 1, -1) ** 2
    x = torch.nn.functional.pad(x, (k // 2, k - 1 - k // 2), mode="replicate")
    return torch.nn.functional.avg_pool1d(x, k, stride=1).view(-1).sqrt()


def _snap_to_transients(found, wave, sr):
    """(t, kind, strength) tuples with each hit moved onto its envelope peak."""
    if not found:
        return found
    env = None
    out = []
    for t, kind, strength in found:
        win = _SNAP_WINDOW.get(kind)
        if win is None:
            out.append((t, kind, strength))
            continue
        if env is None:
            env = _envelope(wave, sr)
        a = max(0, int((t - win[0]) * sr))
        b = min(env.numel(), int((t + win[1]) * sr) + 1)
        if b - a < 2:
            out.append((t, kind, strength))
            continue
        peak = a + int(torch.argmax(env[a:b]))
        out.append((peak / sr, kind, strength))
    return out


def _merge_coincident(events, window=0.07):
    """
    Collapse events that describe the same instant.

    Two detectors legitimately fire on one drum - a snare is both a waveform
    transient and a burst of top end - and a brief that says "BASS HIT at +3.39,
    IMPACT at +3.40" is describing one hit as two. The more specific label wins
    (EVENT_TYPES order), and the earliest time is kept, since that is when the
    hit actually starts.

    `events` is [(t, type, strength)] and comes back the same way, in time order.
    """
    if not events:
        return []
    ordered = sorted(events, key=lambda e: (e[0], EVENT_TYPES.index(e[1])))
    merged = [ordered[0]]
    for t, kind, strength in ordered[1:]:
        prev_t, prev_kind, prev_strength = merged[-1]
        if t - prev_t > window:
            merged.append((t, kind, strength))
            continue
        # same moment: keep the higher-priority label and the louder reading
        if EVENT_TYPES.index(kind) < EVENT_TYPES.index(prev_kind):
            merged[-1] = (prev_t, kind, max(strength, prev_strength))
        else:
            merged[-1] = (prev_t, prev_kind, max(strength, prev_strength))
    return merged


def _normalise(values):
    """Scale a list of raw peak values to 0..1 against the track's own loudest."""
    if not values:
        return []
    top = max(values)
    if top <= 0:
        return [0.0] * len(values)
    return [min(1.0, v / top) for v in values]


# ----------------------------------------------------------------------
# Detectors
# ----------------------------------------------------------------------

_LOUDNESS_SHARE = 0.6     # of a hit's strength: how loud the low band IS at the hit


def _bass_hits(flux, floor, hop, sensitivity, min_gap, level=None):
    """
    Kicks and low hits: 20-250 Hz spectral flux over an adaptive median.

    The flux finds WHERE a kick is; it is a poor measure of how big one is.
    Flux is the low band's RISE, so a soft kick after a rest scores high and a
    heavy kick inside a dense bar - already loud, rising less - scores low.
    Sorted by flux alone, a cap or a strength floor keeps the wrong hits and
    the labels read backwards against the waveform. So the strength is a blend:
    the rise, and the low band's actual level at the hit (`level`, per frame).
    """
    # The AudioEngine's k=3 median factor, divided by sensitivity exactly as its
    # `sens.beat` does: 2 = hair-trigger, 0.5 = strict.
    peaks = _pick_peaks(flux, floor, hop, min_gap, 3.0 / max(0.25, sensitivity))
    rise = _normalise([v for _, v in peaks])
    if level is None or not peaks:
        return [(i, i * hop, "BASS HIT", s) for (i, _), s in zip(peaks, rise)]
    n = level.numel()
    loud = _normalise([float(level[max(0, i - 1):min(n, i + 6)].max()) for i, _ in peaks])
    strengths = [(1.0 - _LOUDNESS_SHARE) * r + _LOUDNESS_SHARE * l for r, l in zip(rise, loud)]
    return [(i, i * hop, "BASS HIT", s) for (i, _), s in zip(peaks, strengths)]


def _impacts(wave, sr, hop, sensitivity, min_gap):
    """
    Slams, crashes and body hits: time-domain RMS rise against the recent
    minimum. Broadband transients smear across the spectrum, so they are found
    on the waveform instead.

    This detector deliberately over-reports - a kick spikes the waveform just
    as a crash does. `_resolve_hits` settles what each one is actually called.
    """
    step = max(1, int(round(sr * hop)))
    win = step * 4
    if wave.numel() < win * 4:
        return []
    frames = wave.unfold(0, win, step)
    rms = torch.sqrt((frames ** 2).mean(dim=1) + 1e-12)
    # rise against the minimum of the last ~80 ms, the AudioEngine's window
    look = max(1, int(round(0.08 / hop)))
    padded = torch.nn.functional.pad(rms.view(1, 1, -1), (look, 0), mode="replicate").view(-1)
    recent_min = -torch.nn.functional.max_pool1d(
        (-padded).view(1, 1, -1), look + 1, stride=1
    ).view(-1)[: rms.numel()]
    rise = torch.clamp(rms - recent_min, min=0)

    floor = _rolling_median(rise, max(1, int(round(1.0 / hop))))
    peaks = _pick_peaks(rise, floor, hop, max(min_gap, 0.15), 4.0 / max(0.25, sensitivity))
    # An impact is meant to be rare and big; keep only rises that are also loud
    # in absolute terms, or every hi-hat qualifies on a quiet track.
    cut = float(torch.quantile(rise, 0.995)) * 0.35
    peaks = [(i, v) for i, v in peaks if v >= cut]
    strengths = _normalise([v for _, v in peaks])
    return [(i, i * hop, "IMPACT", s) for (i, _), s in zip(peaks, strengths)]


def _loudness(spec, hop):
    """Smoothed loudness in dB, the curve drops, stops and builds are read from."""
    rms = torch.sqrt((spec ** 2).mean(dim=0) + 1e-12)
    return 20.0 * torch.log10(_smooth(rms, max(1, int(round(0.05 / hop)))) + 1e-9)


def _novelty(loud_db, hop, window):
    """
    Signed loudness change: mean of the `window` seconds after each frame minus
    the `window` before it, in dB. Positive = energy arriving, negative = energy
    leaving. Unsigned, this is what music_support calls novelty; the SIGN is
    what separates a drop from a stop.
    """
    n = loud_db.numel()
    half = max(1, int(round(window / hop)))
    cum = torch.cat([torch.zeros(1), torch.cumsum(loud_db, 0)])
    idx = torch.arange(n)
    lo = torch.clamp(idx - half, min=0)
    hi = torch.clamp(idx + half, max=n)
    before = (cum[idx] - cum[lo]) / torch.clamp((idx - lo).float(), min=1)
    after = (cum[hi] - cum[idx]) / torch.clamp((hi - idx).float(), min=1)
    # At the very edges one side has (almost) no frames and reads as 0 dB
    # against a real level of -30 dB - a huge fake step. No window, no change.
    valid = ((idx - lo) >= half // 2) & ((hi - idx) >= half // 2)
    return (after - before) * valid.float()


def _slow_median(x, half_seconds, hop):
    """
    Rolling median over a window far longer than the hop (tens of seconds),
    computed on a decimated copy so the unfolded windows stay small, then
    stretched back to every frame. Coarse by design: it is the track's
    "usual" level, not a per-frame measurement.
    """
    n = x.numel()
    half_frames = max(1, int(round(half_seconds / hop)))
    stride = max(1, half_frames // 200)
    coarse = x[::stride]
    med = _rolling_median(coarse, max(1, half_frames // stride))
    return med.repeat_interleave(stride)[:n]


# The intense / calm state machine from scheb/sound-to-light-osc, offline.
# Entering a state takes a bigger margin than staying in it (hysteresis), and
# the condition has to HOLD before the state flips (debounce), so a fill, a
# crash or a two-bar dip cannot toggle it. Margins in dB against the track's
# usual level; scheb's x1.3 / x0.7 enter and x1.1 / x0.9 leave ratios are
# +2.3 / -3.1 and +0.8 / -0.9 dB.
_STATE_ENTER_INTENSE = 2.5
_STATE_LEAVE_INTENSE = 0.8
_STATE_ENTER_CALM = -3.0
_STATE_LEAVE_CALM = -0.9
_STATE_HOLD = 0.5   # s the new condition must hold before the state changes


def _energy_state(loud_db, hop):
    """
    Per frame: +1 intense, 0 neither, -1 calm.

    The level judged is the 1 s median of the loudness (a centred window, so
    the edge sits where the change is) against the 30 s median around it -
    the track's usual level - so a quiet intro in a loud song and a loud
    chorus in a quiet one both read correctly.
    """
    level = _rolling_median(loud_db, max(1, int(round(0.5 / hop))))
    usual = _slow_median(loud_db, 15.0, hop)
    rel = (level - usual).tolist()
    hold = max(1, int(round(_STATE_HOLD / hop)))
    if not rel:
        return []
    # start IN whatever state the opening frames are in - a quiet intro is
    # calm from its first frame, not "neutral until the hold has elapsed"
    # (which would make the opening read as a fall into calm, i.e. a STOP)
    state = 1 if rel[0] >= _STATE_ENTER_INTENSE else (-1 if rel[0] <= _STATE_ENTER_CALM else 0)
    pending = None       # (candidate state, frames it has held)
    out = [0] * len(rel)
    for i, r in enumerate(rel):
        if state == 1:
            want = 1 if r >= _STATE_LEAVE_INTENSE else (-1 if r <= _STATE_ENTER_CALM else 0)
        elif state == -1:
            want = -1 if r <= _STATE_LEAVE_CALM else (1 if r >= _STATE_ENTER_INTENSE else 0)
        else:
            want = 1 if r >= _STATE_ENTER_INTENSE else (-1 if r <= _STATE_ENTER_CALM else 0)
        if want == state:
            pending = None
        elif pending is not None and pending[0] == want:
            pending = (want, pending[1] + 1)
            if pending[1] >= hold:
                state, pending = want, None
        else:
            pending = (want, 1)
        out[i] = state
    return out


def _drops_and_stops(loud_db, hop, sensitivity):
    """
    Energy arriving (DROP) or leaving (STOP), from the signed 1.5 s novelty -
    confirmed against the energy state machine.

    The novelty peak gives the instant (it is sharp, and it gets snapped onto
    the transient later); the state machine says whether anything actually
    CHANGED there. A DROP has to be a state rise - calm to neutral, neutral to
    intense, calm to intense - somewhere between shortly before the peak and
    a couple of seconds after it (the 1 s level median and the hold delay the
    flip); a STOP has to be a state fall. A loud fill that is over in two
    bars, or a dip and return, moves the novelty but never flips the state,
    and used to read as DROP, STOP, DROP - flicker the writer then staged.
    """
    delta = _novelty(loud_db, hop, 1.5)
    # 6 dB is an obvious step to a listener; sensitivity scales it.
    threshold = 6.0 / max(0.25, sensitivity)
    states = _energy_state(loud_db, hop)
    n = len(states)
    before = max(1, int(round(0.75 / hop)))
    after = max(1, int(round(2.5 / hop)))

    def confirmed(i, direction):
        lo = max(0, i - before)
        hi = min(n - 1, i + after)
        s_before = states[lo]
        window = states[lo:hi + 1]
        # the state has to go the right way, and end up there
        return any((s - s_before) * direction > 0 for s in window) and (window[-1] - s_before) * direction > 0

    events = []
    for sign, name in ((1.0, "DROP"), (-1.0, "STOP")):
        signal = delta * sign
        floor = torch.full_like(signal, threshold)
        peaks = _pick_peaks(signal, floor, hop, 2.0, 1.0)
        peaks = [(i, v) for i, v in peaks if confirmed(i, sign)]
        values = [v for _, v in peaks]
        strengths = _normalise(values)
        events += [(i * hop, name, s) for (i, _), s in zip(peaks, strengths)]
    return events


def _sections(loud_db, hop, sensitivity):
    """Verse/chorus turns: the same novelty over a 4 s window."""
    delta = _novelty(loud_db, hop, 4.0).abs()
    threshold = 4.5 / max(0.25, sensitivity)
    peaks = _pick_peaks(delta, torch.full_like(delta, threshold), hop, 6.0, 1.0)
    strengths = _normalise([v for _, v in peaks])
    return [(i * hop, "SECTION", s) for (i, _), s in zip(peaks, strengths)]


def _builds(loud_db, hop, drops):
    """
    Risers: a sustained upward loudness ramp that lands on a drop.

    Anchoring to a drop is what keeps this honest - plenty of passages get
    gradually louder without being a build, but a ramp that ends exactly where
    the beat arrives is one by definition.
    """
    if not drops:
        return []
    curve = _smooth(loud_db, max(1, int(round(0.25 / hop))))
    n = curve.numel()
    look = int(round(6.0 / hop))       # how far back a riser may start
    shortest = int(round(1.0 / hop))
    events = []
    for t, _name, _s in drops:
        end = min(n - 1, int(round(t / hop)))
        start = max(0, end - look)
        if end - start < shortest:
            continue
        # the ramp begins at the quietest point in the run-up
        low = start + int(torch.argmin(curve[start:end + 1]))
        if end - low < shortest:
            continue
        ramp = curve[low:end + 1]
        rise = float(ramp[-1] - ramp[0])
        if rise < 6.0:
            continue
        # ...and it has to RISE, not sit flat and then step. A pad holding
        # steady before the beat drops is not a build, and without this test
        # every drop reports one.
        steps = ramp[1:] - ramp[:-1]
        if float((steps > 0).float().mean()) < 0.6:
            continue
        events.append((low * hop, "BUILD", min(1.0, rise / 18.0)))
    return events


def _accents(flux, floor, hop, sensitivity, min_gap):
    """Hats, snares, cymbals: 2-12 kHz flux. Off by default - it floods."""
    peaks = _pick_peaks(flux, floor, hop, max(min_gap, 0.12), 3.5 / max(0.25, sensitivity))
    strengths = _normalise([v for _, v in peaks])
    return [(i * hop, "ACCENT", s) for (i, _), s in zip(peaks, strengths)]


# ----------------------------------------------------------------------
# The beat grid
# ----------------------------------------------------------------------
# A hit that sits on the pulse is the one a cut can land on; a hit between
# beats is often a ghost note, a fill or a detector misfire. The grid is the
# track's tempo (music_support.estimate_bpm) with its phase fitted to the
# hits themselves - brute force over 64 phases, the way scheb/sound-to-light-osc
# fits its BPM grid by quantisation error. The editor draws the same grid.

GRID_TOLERANCE = 0.045   # s - within this of a grid line counts as ON the beat
_HIT_KINDS = ("BASS HIT", "IMPACT", "ACCENT")


def beat_grid(events, bpm):
    """
    {"bpm", "period", "phase", "on", "of"} for the hits in `events`
    ([(t, kind, s)] or [{"t", "type", ...}]). bpm 0 = no grid.
    """
    hits = [float(e[0] if isinstance(e, (tuple, list)) else e["t"]) for e in events
            if (e[1] if isinstance(e, (tuple, list)) else e["type"]) in _HIT_KINDS]
    if not bpm or bpm <= 0:
        return {"bpm": 0.0, "period": 0.0, "phase": 0.0, "on": 0, "of": len(hits)}
    period = 60.0 / float(bpm)
    best_phase, best_on = 0.0, -1
    for k in range(64):
        phase = period * k / 64.0
        on = sum(1 for t in hits if _grid_distance(t, phase, period) <= GRID_TOLERANCE)
        if on > best_on:
            best_phase, best_on = phase, on
    return {"bpm": round(float(bpm), 2), "period": period, "phase": round(best_phase, 4), "on": best_on, "of": len(hits)}


def _grid_distance(t, phase, period):
    """Seconds from `t` to the nearest grid line."""
    r = (t - phase) / period
    return abs(r - round(r)) * period


def weight_on_beat(events, grid, weight):
    """
    Scale each HIT's strength by how far it sits from the grid: on the line
    (within GRID_TOLERANCE) it keeps its strength; a quarter of a beat away
    or more it is scaled by (1 - weight). weight 1 zeroes the off-beat hits,
    so the strength band or max_events drop them; 0.5 halves them so they
    survive only where nothing on the beat competes. Structural events are
    never touched, and neither is the timing.
    """
    w = max(0.0, min(1.0, float(weight or 0.0)))
    if w <= 0 or not grid or not grid.get("bpm"):
        return events
    period, phase = grid["period"], grid["phase"]
    ramp = max(1e-3, period / 4.0 - GRID_TOLERANCE)
    out = []
    for t, kind, s in events:
        if kind in _HIT_KINDS:
            d = _grid_distance(float(t), phase, period)
            off = min(1.0, max(0.0, d - GRID_TOLERANCE) / ramp)
            s = float(s) * (1.0 - w * off)
        out.append((t, kind, s))
    return out


# ----------------------------------------------------------------------
# The public detector
# ----------------------------------------------------------------------

def detect_events(
    audio,
    sensitivity=1.0,
    min_gap_seconds=0.18,
    min_strength=0.0,
    max_events=600,
    bass_hits=True,
    max_strength=1.0,
    impacts=True,
    drops_and_stops=True,
    builds=True,
    sections=True,
    accents=False,
    time_offset_ms=0,
    gain_db=0.0,
    dynamics_curve=1.0,
    eq_sub_db=0.0,
    eq_bass_db=0.0,
    eq_low_db=0.0,
    eq_mid_db=0.0,
    eq_high_db=0.0,
    on_beat_weight=0.0,
    grid_out=None,
):
    """
    Every labelled moment in the song, as
    [{"t": seconds, "type": str, "strength": 0..1, "label": str, "note": str,
      "sync": str}] - `note` describes what the music does at that instant in
    plain words, `sync` is what the picture should do about it.

    Sorted by time. When `max_events` bites, structural events (drops, stops,
    sections, builds) are kept whole and the hit stream is thinned to the
    strongest - losing a kick costs a cut, losing the drop costs the video.
    """
    shaping = dict(
        gain_db=gain_db, dynamics_curve=dynamics_curve, eq_sub_db=eq_sub_db,
        eq_bass_db=eq_bass_db, eq_low_db=eq_low_db, eq_mid_db=eq_mid_db, eq_high_db=eq_high_db,
    )
    spec, freqs, wave, hop, sr = _frames(audio, shaping)
    duration = wave.numel() / sr
    loud_db = _loudness(spec, hop)

    half = max(1, int(round(0.75 / hop)))  # ~1.5 s median window
    low_flux = _smooth(_band_flux(spec, freqs, 20.0, 250.0), 3)
    low_floor = _rolling_median(low_flux, half)
    top_flux = _smooth(_band_flux(spec, freqs, 2000.0, 12000.0), 3)
    top_floor = _rolling_median(top_flux, half)

    found = []
    low_level = _band_level(spec, freqs, 20.0, 250.0)
    hits = _bass_hits(low_flux, low_floor, hop, sensitivity, min_gap_seconds, level=low_level) if bass_hits else []
    slams = _impacts(wave, sr, hop, sensitivity, min_gap_seconds) if impacts else []
    if hits or slams:
        low_tilt, top_tilt = _band_tilt(
            _band_level(spec, freqs, 20.0, 250.0),
            _band_level(spec, freqs, 2000.0, 12000.0),
            hop,
        )
        found += _resolve_hits(hits, slams, low_tilt, top_tilt)
    structure = _drops_and_stops(loud_db, hop, sensitivity) if drops_and_stops else []
    found += structure

    if builds:
        found += _builds(loud_db, hop, [e for e in structure if e[1] == "DROP"])
    if sections:
        turns = _sections(loud_db, hop, sensitivity)
        # A section change IS usually a drop or a stop. Reporting both puts the
        # same instant in the brief twice under two names, so the more specific
        # label wins.
        marked = [t for t, kind, _ in structure]
        found += [e for e in turns if all(abs(e[0] - t) > 1.0 for t in marked)]
    if accents:
        found += _accents(top_flux, top_floor, hop, sensitivity, min_gap_seconds)

    found = _snap_to_transients(found, wave, sr)
    found = _merge_coincident(found)
    if time_offset_ms:
        shift = float(time_offset_ms) / 1000.0
        found = [(max(0.0, t + shift), kind, s) for t, kind, s in found]

    # The beat grid: fitted to the hits as detected, then (optionally) used to
    # lean the strengths toward the hits that sit on the pulse. `grid_out`
    # lets a caller (the editor's preview) read the fit back.
    if on_beat_weight or grid_out is not None:
        try:
            bpm = float(estimate_bpm(analyse(audio)))
        except Exception:
            bpm = 0.0
        grid = beat_grid(found, bpm)
        if grid_out is not None:
            grid_out.update(grid)
        found = weight_on_beat(found, grid, on_beat_weight)

    # The hit stream is kept inside a strength BAND. A floor alone thins a
    # busy track; a ceiling as well lets a run target one layer of the mix -
    # "the 0.4 kicks, not the 0.9 slams" - or the mid-weight hits between
    # the drops. Structural events (drops, stops, sections, builds) are the
    # song's shape and are never subject to either bound.
    lo = float(min_strength or 0.0)
    hi = 1.0 if max_strength is None else float(max_strength)
    if hi < lo:
        lo, hi = hi, lo
    events = [
        {
            "t": round(float(t), 2),
            "type": kind,
            "strength": round(float(s), 2),
            "label": _strength_label(s),
            "note": _sound_note(kind, s),
        }
        for t, kind, s in found
        if float(t) <= duration and (kind in _STRUCTURAL or lo - 1e-9 <= float(s) <= hi + 1e-9)
    ]

    # A BUILD's own `t` is where the ramp STARTS - the quietest point of the
    # run-up - so on its own it says nothing about where the pressure is meant
    # to be released. Pair it with the drop it climbs into and the writer gets
    # the real window instead of a guessed lead.
    drop_times = sorted(e["t"] for e in events if e["type"] == "DROP")
    for e in events:
        if e["type"] != "BUILD":
            continue
        landing = next((d for d in drop_times if d > e["t"] + 1e-6), None)
        if landing is not None:
            e["until"] = round(float(landing), 2)

    cap = max(1, int(max_events))
    if len(events) > cap:
        # Thin EVENLY: the strongest hits of every few seconds survive, not the
        # strongest of the whole song - a global cut keeps every hit of the
        # loudest chorus and leaves the verses with none to stage.
        keep = [e for e in events if e["type"] in _STRUCTURAL]
        hits_only = [e for e in events if e["type"] not in _STRUCTURAL]
        room = max(0, cap - len(keep))
        window = 4.0
        buckets = {}
        for e in hits_only:
            buckets.setdefault(int(e["t"] // window), []).append(e)
        quota = max(1, room // max(1, len(buckets)))
        chosen, leftovers = [], []
        for _k, bucket in sorted(buckets.items()):
            bucket.sort(key=lambda e: e["strength"], reverse=True)
            chosen += bucket[:quota]
            leftovers += bucket[quota:]
        chosen = chosen[:room]
        leftovers.sort(key=lambda e: e["strength"], reverse=True)
        chosen += leftovers[: max(0, room - len(chosen))]
        events = keep + chosen

    events.sort(key=lambda e: (e["t"], EVENT_TYPES.index(e["type"])))
    return events


# ----------------------------------------------------------------------
# Text form: what the model reads, and what we can read back
# ----------------------------------------------------------------------

_LINE_RE = re.compile(
    r"^\s*\[(\d+):(\d{2}(?:\.\d+)?)\]\s+([A-Z][A-Z ]*[A-Z]|[A-Z])\s*"
    # optional hand-set wind-up start (the editor's left handle)
    r"(?:<-\s*\[(\d+):(\d{2}(?:\.\d+)?)\]\s*)?"
    # optional landing / settle time: a BUILD carries the drop it ramps into,
    # any event may carry a hand-set settle (the editor's right handle)
    r"(?:->\s*\[(\d+):(\d{2}(?:\.\d+)?)\]\s*)?"
    r"(?:\|\s*(\w+))?\s*(?:\|\s*(.*))?$"
)


def events_table(events, duration=0.0, profile=None):
    """
    The readable, re-parseable table. One event per line, absolute times.

    What the picture should DO about each moment is deliberately not here. The
    writer decides that; this only reports what the music does and when.
    """
    lines = []
    if profile:
        from .music_support import profile_line
        lines.append(f"# {profile_line(profile)}")
    if duration:
        lines.append(f"# {fmt_time(duration)} of audio, {len(events)} event(s)")
    for e in events:
        row = f"[{fmt_time(e['t'])}] {e['type']:<9}"
        if e.get("lead") is not None:
            row += f" <- [{fmt_time(max(0.0, float(e['t']) - float(e['lead'])))}]"
        if e.get("until") is not None:
            # where the ramp is released - a build is a window, not an instant
            row += f" -> [{fmt_time(e['until'])}]"
        row += f" | {e['label']}"
        if e.get("note"):
            row += f" | {e['note']}"
        lines.append(row)
    return "\n".join(lines)


def parse_events(text):
    """
    Read events back from either the JSON output or the table.

    Accepting both means the writer's socket takes whichever of this node's two
    string outputs someone happens to wire, and still works if a human edits
    the table by hand or types one from scratch.
    """
    raw = (text or "").strip()
    if not raw:
        return []

    if raw.startswith("[") or raw.startswith("{"):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                data = data.get("events") or []
            out = []
            for e in data:
                if not isinstance(e, dict) or "t" not in e:
                    continue
                strength = float(e.get("strength", 0.5))
                row = {
                    "t": float(e["t"]),
                    "type": str(e.get("type", "EVENT")).upper(),
                    "strength": strength,
                    "label": str(e.get("label") or _strength_label(strength)),
                    "note": str(e.get("note") or _sound_note(
                        str(e.get("type", "")).upper(), strength)),
                }
                if e.get("until") is not None:
                    row["until"] = float(e["until"])
                if e.get("lead") is not None:
                    row["lead"] = float(e["lead"])
                out.append(row)
            out.sort(key=lambda e: e["t"])
            return out
        except (ValueError, TypeError):
            pass  # not JSON after all - fall through to the table reader

    out = []
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        minutes, seconds, kind, lead_min, lead_sec, until_min, until_sec, label, note = m.groups()
        kind = kind.strip()
        strength = {"light": 0.2, "solid": 0.5, "heavy": 0.9}.get((label or "").lower(), 0.5)
        row = {
            "t": int(minutes) * 60 + float(seconds),
            "type": kind,
            "strength": strength,
            "label": (label or "solid").lower(),
            "note": (note or "").strip() or _sound_note(kind, strength),
        }
        if lead_min is not None:
            row["lead"] = max(0.0, row["t"] - (int(lead_min) * 60 + float(lead_sec)))
        if until_min is not None:
            row["until"] = int(until_min) * 60 + float(until_sec)
        out.append(row)
    out.sort(key=lambda e: e["t"])
    return out


def events_for_segment(events, start, end, limit=8):
    """
    The events this clip has to stage, timed FROM THE CLIP'S OWN START.

    That relative time is the whole point: the writer is describing a 9-second
    shot, and "a bass hit at +2.1 s" is something it can stage, while "a bass
    hit at 1:47.3" is not. Capped per clip, strongest first, so one busy bar
    cannot swamp a scene brief - then re-sorted into time order.

    A clip's events are not simply the ones inside it. A drop landing a third
    of a second after a cut needs most of a second of wind-up, and all of that
    wind-up belongs to the OUTGOING clip - which, listing only its own hits,
    would never hear about it and would end at rest. So an event also counts as
    this clip's if its window STARTS here, even though it lands in the next
    one; the outgoing clip climbs into it without resolving, and the incoming
    clip (which lists the same event, opening mid-move) releases it. That
    hand-off across the cut is the difference between a video that hits the
    drop and one that arrives just after it.
    """
    inside = [
        e for e in events
        if start <= e["t"] < end
        or (e["t"] >= end and event_window(e)[0] < end - 1e-6)
    ]
    if len(inside) > limit:
        ranked = sorted(
            inside,
            key=lambda e: (e["type"] not in _STRUCTURAL, -e["strength"]),
        )
        inside = ranked[:limit]
    inside.sort(key=lambda e: e["t"])

    span = max(0.0, end - start)
    out = []
    for e in inside:
        cue, land, settle = event_window(e)
        out.append(dict(
            e,
            offset=round(e["t"] - start, 2),
            # the staging window, clamped into the clip: nothing outside these
            # bounds exists as far as this render is concerned
            cue_offset=round(min(span, max(0.0, cue - start)), 2),
            land_offset=round(min(span, max(0.0, land - start)), 2),
            settle_offset=round(min(span, max(0.0, settle - start)), 2),
            # ...but the fact that it was clamped is itself a staging note. A
            # wind-up that began in the previous clip means this one opens
            # mid-move, and a peak past the end means this one must not resolve.
            opens_wound_up=cue < start - 1e-6,
            lands_after=land > end + 1e-6,
        ))
    # Ordered by where the MOVE starts, not where the sound is. The writer
    # turns this list into one chronological run of clauses, and the first
    # clause of every event is its wind-up - so a drop whose pressure starts
    # building before an earlier kick has to be read, and written, first.
    out.sort(key=lambda e: (e["cue_offset"], e["land_offset"]))
    return out


# The hit budget of one brief follows the clip's length. Calibrated with the
# Sound Events preview panel: 6 slots per ~9 s piece (one every 1.5 s) fills a
# piece with its 2-4 real hits and still leaves room; drops / stops / builds
# take their slots first, so the cap only ever thins the plain hit stream.
EVENT_SLOTS_PER_SECOND = 6 / 9


def event_slots_for(seconds):
    """How many hits a brief for a clip of `seconds` may list - never below 1."""
    return max(1, math.ceil(float(seconds) * EVENT_SLOTS_PER_SECOND - 1e-9))


def segment_event_lines(events, start, end, limit=8):
    """
    `events_for_segment` as the indented window lines the writer's brief uses.

    THREE numbers, not one - start the move / land it / settle it - all timed
    from the clip's own start. Handing over the bare instant is what makes a
    "synced" video read late: a move told to begin on the beat peaks after it.

    Type and size only. WHAT lands on the moment is the writer's decision, not
    a lookup - the plain-words note the table carries would just be a stronger
    hint toward the same handful of images in every scene.
    """
    lines = []
    for e in events_for_segment(events, start, end, limit):
        row = (
            f"    [+{e['cue_offset']:5.2f} ->+{e['land_offset']:5.2f}"
            f" ->+{e['settle_offset']:5.2f}s] {e['type']:<9} | {e['label']}"
        )
        if e.get("opens_wound_up"):
            row += "  <opens already moving>"
        if e.get("lands_after"):
            row += "  <lands in the next clip>"
        lines.append(row)
    return lines


def parse_edits(text):
    """The editor's `edits` JSON -> dict; anything unreadable is an empty edit set."""
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        print("⚠️ H3 Sound Events: `edits` is not valid JSON - ignored")
        return {}
    return data if isinstance(data, dict) else {}


def apply_edits(events, edits, tolerance=0.08):
    """
    Apply the editor's hand edits to a detected list: moved / resized /
    retyped / re-weighted events (matched by the time they were DETECTED at,
    so the edit survives a re-detection with the same settings) and events
    the editor added. Returns (events, changed_count, added_count).
    """
    if not edits:
        return events, 0, 0
    out = [dict(e) for e in events]
    changed = 0
    for op in edits.get("events", []) or []:
        try:
            at = float(op.get("at"))
        except (TypeError, ValueError):
            continue
        best = min(out, key=lambda e: abs(float(e["t"]) - at), default=None)
        if best is None or abs(float(best["t"]) - at) > tolerance:
            continue
        touched = False
        if op.get("type") in EVENT_TYPES:
            best["type"] = op["type"]; touched = True
        for key in ("t", "strength", "lead", "until"):
            if key in op and op[key] is not None:
                try:
                    best[key] = round(float(op[key]), 3)
                    touched = True
                except (TypeError, ValueError):
                    pass
        if touched:
            best["strength"] = max(0.0, min(1.0, float(best.get("strength", 0.5))))
            best["label"] = _strength_label(best["strength"])
            best["note"] = _sound_note(best["type"], best["strength"])
            best["edited"] = True
            changed += 1
    added = 0
    for op in edits.get("added", []) or []:
        try:
            t = float(op.get("t"))
        except (TypeError, ValueError):
            continue
        kind = op.get("type") if op.get("type") in EVENT_TYPES else "BASS HIT"
        strength = max(0.0, min(1.0, float(op.get("strength", 0.6) or 0.6)))
        row = {"t": round(t, 3), "type": kind, "strength": round(strength, 2),
               "label": _strength_label(strength), "note": _sound_note(kind, strength), "edited": True}
        for key in ("lead", "until"):
            if op.get(key) is not None:
                try:
                    row[key] = round(float(op[key]), 3)
                except (TypeError, ValueError):
                    pass
        out.append(row)
        added += 1
    out.sort(key=lambda e: float(e["t"]))
    return out, changed, added


def parse_rejected(text):
    """'12.4 33.08 1:05.2' -> [12.4, 33.08, 65.2]; anything unreadable is skipped."""
    out = []
    for tok in (text or "").replace(",", " ").split():
        try:
            if ":" in tok:
                m, s = tok.split(":", 1)
                out.append(int(m) * 60 + float(s))
            else:
                out.append(float(tok))
        except ValueError:
            continue
    return out


def events_summary(events):
    """One line: how many of each kind."""
    if not events:
        return "no events detected"
    counts = {}
    for e in events:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    parts = [f"{counts[k]} {k.lower()}{'s' if counts[k] != 1 else ''}"
             for k in EVENT_TYPES if k in counts]
    return ", ".join(parts)


# ----------------------------------------------------------------------
# The node
# ----------------------------------------------------------------------

class H3SoundEvents:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "The song (Load Audio). Passed through unchanged on the audio output.",
                }),
                "sensitivity": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 2.0, "step": 0.05,
                    "tooltip": (
                        "How easily an event triggers. Every detector scores a peak against "
                        "the median of the surrounding audio, and this divides that factor: "
                        "2.0 is a hair-trigger (every sixteenth), 0.5 is strict (only the "
                        "big ones). Raise it for a sparse, quiet track; lower it for a wall "
                        "of sound where everything clears the floor."
                    ),
                }),
                "min_gap_seconds": ("FLOAT", {
                    "default": 0.18, "min": 0.05, "max": 2.0, "step": 0.01,
                    "tooltip": (
                        "Refractory gap: the shortest time between two hits of the same "
                        "kind. 0.18 s keeps kicks and their trailing snare apart at most "
                        "tempos. Raise it to thin a busy track down to the downbeats."
                    ),
                }),
                "max_events": ("INT", {
                    "default": 600, "min": 4, "max": 2000, "step": 4,
                    "tooltip": (
                        "Cap on the whole list. A song has a kick on every beat - 300-400 in "
                        "three minutes - and the writer already limits how many hits each "
                        "scene's brief lists, so the cap only needs to catch runaway tracks; "
                        "120 threw two thirds of a normal song's kicks away. Drops, stops, "
                        "sections and builds are always kept; only the hit stream is thinned, "
                        "strongest (loudest) first."
                    ),
                }),
            },
            "optional": {
                "min_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Drop hits weaker than this (0-1, 1 = the loudest hit in this track). "
                        "With max_strength it is a band: 0.35-0.45 keeps only the hits around "
                        "0.4. Structural events (drops, stops, sections, builds) are never dropped."
                    ),
                }),
                "bass_hits": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Kicks and low hits: 20-250 Hz spectral flux over an adaptive floor.",
                }),
                "impacts": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Slams, crashes, body hits: broadband RMS rise on the waveform.",
                }),
                "drops_and_stops": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "The beat arriving (DROP) or cutting out (STOP), from signed loudness novelty.",
                }),
                "builds": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Risers: a sustained loudness ramp that lands on a drop.",
                }),
                "sections": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Verse/chorus turns, from the same novelty over a 4 s window.",
                }),
                "accents": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Hats, snares and cymbals (2-12 kHz flux). Off by default: on most "
                        "tracks this alone is hundreds of events and it crowds out the rest."
                    ),
                }),
                "rejected": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Hits you have struck out on the preview timeline (click a tick in "
                        "\U0001F39A Preview events), as seconds separated by spaces. Anything within "
                        "50 ms of a listed time is dropped from the output. Clear to keep everything."
                    ),
                }),
                "max_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Drop hits STRONGER than this (0-1). Together with min_strength it is a "
                        "band: 0.35-0.45 targets the hits around 0.4 and leaves both the soft "
                        "kicks and the big slams out. 1.0 = no ceiling. Structural events are "
                        "never dropped."
                    ),
                }),
                # appended last so saved workflows keep their widget positions
                "edits": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": (
                        "Hand edits from the 🎚 editor, as JSON: events moved, stretched "
                        "(wind-up / settle handles), retyped, re-weighted, and events you added. "
                        "Each edit names the time the event was DETECTED at, so it survives a "
                        "re-detection with the same settings. Clear to go back to the detector's list."
                    ),
                }),
                "time_offset_ms": ("INT", {
                    "default": 0, "min": -250, "max": 250, "step": 5,
                    "tooltip": (
                        "Global nudge for every event, in milliseconds - positive = later. Events are "
                        "already snapped onto the waveform's own peaks, so 0 is where the sound is; use "
                        "this if the whole list still reads early or late against the editor's waveform "
                        "(check with the playhead), or to pre-empt a render that lands its moves late."
                    ),
                }),
                # Signal shaping, gain -> EQ -> curve, in front of every detector.
                # Appended after everything else so saved workflows keep their widget positions.
                "gain_db": ("FLOAT", {
                    "default": 0.0, "min": -24.0, "max": 24.0, "step": 0.5,
                    "tooltip": (
                        "Input gain in dB, first in the chain. Mostly matters with a dynamics_curve "
                        "above 1: the curve bends relative to full scale, so a quiet master needs "
                        "lifting first or the curve crushes everything. Also decides how much of the "
                        "quiet spectrum the hit detectors still notice."
                    ),
                }),
                "dynamics_curve": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                    "tooltip": (
                        "Bends the waveform's dynamics: |x| raised to this power (relative to full "
                        "scale). 1.0 = untouched. Above 1 EXPANDS - the soft material sinks and the big "
                        "hits stand proud, so a dense track gives fewer, cleaner beats. Below 1 "
                        "COMPRESSES - the soft hits come up, for a sparse or quiet track."
                    ),
                }),
                "eq_sub_db": ("FLOAT", {
                    "default": 0.0, "min": -24.0, "max": 24.0, "step": 0.5,
                    "tooltip": "EQ: low shelf at 60 Hz, dB. Sub-bass and 808 tails. Cut it if the sub drone keeps the bass floor high.",
                }),
                "eq_bass_db": ("FLOAT", {
                    "default": 0.0, "min": -24.0, "max": 24.0, "step": 0.5,
                    "tooltip": "EQ: bell at 150 Hz, dB. The kick's body. Boost to make the kick the loudest thing the detectors see.",
                }),
                "eq_low_db": ("FLOAT", {
                    "default": 0.0, "min": -24.0, "max": 24.0, "step": 0.5,
                    "tooltip": "EQ: bell at 400 Hz, dB. Low mids - toms, bass guitar, the snare's body.",
                }),
                "eq_mid_db": ("FLOAT", {
                    "default": 0.0, "min": -24.0, "max": 24.0, "step": 0.5,
                    "tooltip": "EQ: bell at 1.5 kHz, dB. Vocals, guitars, the snare's crack. Cut to stop a busy midrange triggering impacts.",
                }),
                "eq_high_db": ("FLOAT", {
                    "default": 0.0, "min": -24.0, "max": 24.0, "step": 0.5,
                    "tooltip": "EQ: high shelf at 5 kHz, dB. Hats, cymbals, air. Cut to keep hats out of the impact and accent detectors.",
                }),
                "on_beat_weight": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Prefer hits that sit on the beat. The track's tempo is measured and a "
                        "beat grid is phase-fitted to the hits; a hit within 45 ms of a grid line "
                        "keeps its strength, one a quarter-beat or more away is scaled by (1 - this). "
                        "0 = off. 0.5 halves the off-beat hits so they only survive where nothing on "
                        "the beat competes; 1.0 removes them (given any min_strength). Structural "
                        "events and timing are never touched. The editor draws the same grid."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("audio", "events", "events_json", "summary", "count")
    OUTPUT_TOOLTIPS = (
        "The same audio, passed through - wire the writer from here.",
        "The readable table, one event per line. Wire into the Music Video Writer's "
        "`sound_events` socket; it slices the events per clip automatically.",
        "The same events as JSON, for other nodes or your own scripting.",
        "One line: how many of each kind, plus the measured BPM and character.",
        "How many events were found.",
    )
    OUTPUT_NODE = True
    FUNCTION = "detect"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Finds and labels the moments worth cutting on - bass hits, impacts, drops, "
        "builds, stops, section changes - with the second each one lands at. Wire "
        "`events` into the Music Video Writer's `sound_events` socket and every scene's "
        "brief carries the hits inside its own clip, timed from the clip's start, so the "
        "picture is staged ON the music."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def detect(self, audio, sensitivity, min_gap_seconds, max_events,
               min_strength=0.0, bass_hits=True, impacts=True, drops_and_stops=True,
               builds=True, sections=True, accents=False, rejected="", max_strength=1.0, edits="",
               time_offset_ms=0, gain_db=0.0, dynamics_curve=1.0, eq_sub_db=0.0, eq_bass_db=0.0,
               eq_low_db=0.0, eq_mid_db=0.0, eq_high_db=0.0, on_beat_weight=0.0):
        grid = {}
        events = detect_events(
            audio,
            sensitivity=sensitivity,
            min_gap_seconds=min_gap_seconds,
            min_strength=min_strength,
            max_strength=max_strength,
            max_events=max_events,
            bass_hits=bass_hits,
            impacts=impacts,
            drops_and_stops=drops_and_stops,
            builds=builds,
            sections=sections,
            accents=accents,
            time_offset_ms=int(time_offset_ms or 0),
            gain_db=gain_db, dynamics_curve=dynamics_curve, eq_sub_db=eq_sub_db,
            eq_bass_db=eq_bass_db, eq_low_db=eq_low_db, eq_mid_db=eq_mid_db, eq_high_db=eq_high_db,
            on_beat_weight=on_beat_weight, grid_out=grid if on_beat_weight else None,
        )
        if on_beat_weight and grid.get("bpm"):
            print(f"\U0001F3B5 H3 Sound Events | beat grid {grid['bpm']:.1f} BPM, {grid['on']}/{grid['of']} hits on it; "
                  f"off-beat hits weighted x{1 - float(on_beat_weight):.2f}")
        elif on_beat_weight:
            print("\U0001F3B5 H3 Sound Events | on_beat_weight set but no steady pulse found - hits left unweighted")
        shaped = [f"{k}={float(v):g}" for k, v in (("gain", gain_db), ("curve", dynamics_curve), ("sub", eq_sub_db),
                  ("bass", eq_bass_db), ("low", eq_low_db), ("mid", eq_mid_db), ("high", eq_high_db))
                  if abs(float(v) - (1.0 if k == "curve" else 0.0)) >= 0.05]
        if shaped:
            print(f"\U0001F39B H3 Sound Events | signal shaped before detection: {', '.join(shaped)}")
        struck = parse_rejected(rejected)
        if struck:
            before = len(events)
            events = [e for e in events if not any(abs(float(e["t"]) - t) <= 0.08 for t in struck)]
            print(f"\U0001F941 H3 Sound Events | {before - len(events)} hit(s) struck out on the timeline")
        events, changed, added = apply_edits(events, parse_edits(edits))
        if changed or added:
            print(f"\u270F\uFE0F H3 Sound Events | hand edits: {changed} event(s) changed, {added} added")
        feats = analyse(audio)
        profile = song_profile(feats)
        duration = float(feats["duration"])

        table = events_table(events, duration=duration, profile=profile)
        summary = f"{events_summary(events)} in {fmt_time(duration)}"
        print(f"🥁 H3 Sound Events | {summary}")

        return {
            "ui": {"text": [f"{summary}\n{table}"]},
            "result": (
                audio,
                table,
                json.dumps(events, ensure_ascii=False),
                summary,
                len(events),
            ),
        }


with_advanced_inputs(
    H3SoundEvents,
    ("audio", "sensitivity", "min_gap_seconds", "max_events"),
)
