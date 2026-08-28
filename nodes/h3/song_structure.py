# APNext H3 - song structure: beats, bars, sections, and which sections repeat
#
# music_support.analyse() hears energy and hits. It does not hear *form*: it
# cannot tell the writer that 0:29-0:52 is the chorus and that it comes back
# at 1:11, so the writer's strongest musical rule - every chorus returns to
# ONE signature look that escalates - only fired when the lyrics carried a
# [Chorus] tag, which transcribed lyrics never do. This module adds form:
#
#   tempo      onset-envelope tempo, checked against half / double / triplet
#              alternatives with a comb score so a harmonic cannot win
#   beats      beat positions tracked on the onset envelope
#   meter      2/3/4/6-beat bar candidates: for each, the phase that carries
#              the most accent (onset + bass onset + chord change), scored as
#              a z-statistic so a bar with fewer samples cannot win by noise
#   sections   a novelty curve over the beat grid - Gaussian checkerboards on
#              the chroma and on the MFCC self-similarity plus a short energy
#              step - peaks snapped to downbeats, slivers merged, very long
#              stretches split at their strongest interior change
#   repeats    two sections are the same material when the self-similarity
#              matrix (chroma + timbre) shows a diagonal stripe between them;
#              linked sections form groups; the recurring group with the
#              most energy is the chorus, the earliest other recurring group
#              the verse, one-offs are intro / bridge / outro by position
#
# Every claim carries a confidence and the callers only print what clears
# the bar: when the repeats are not clear enough to name a chorus the
# sections come back unlabelled, when no phase carries the accent there are
# no downbeats. A wrong "chorus" would steer a whole video; "unknown" costs
# nothing. Calibrated on the tracks in the input folder against the chorus
# times in their Whisper transcripts (mean boundary error 1.3-3.0 s, chorus
# vs verse repetition margin >= 0.39).
#
# librosa is optional at import time (it ships with the portable build):
# without it song_structure() returns None and everything degrades to the
# energy-only behaviour. The first call in a process pays librosa's numba
# warm-up (~30 s); later tracks take about a second.

import hashlib
import math

import numpy as np

SR = 22050
HOP = 512

MIN_LABEL_CONFIDENCE = 0.55   # below this, sections are returned without names
MIN_METER_CONFIDENCE = 0.45   # below this, no downbeats are reported
LINK_THRESHOLD = 0.42         # diagonal-stripe similarity that makes two sections "the same"
MAX_SECTION_SECONDS = 45.0

_CACHE = {}
_CACHE_LIMIT = 8


# ---------------------------------------------------------------------------
# audio in
# ---------------------------------------------------------------------------

def _fingerprint(audio):
    import torch
    wav = audio["waveform"]
    h = hashlib.sha1()
    h.update(str((tuple(wav.shape), int(audio.get("sample_rate", 0)))).encode())
    flat = wav.reshape(-1)
    step = max(1, flat.numel() // 4096)
    h.update(flat[::step].to(torch.float32).cpu().numpy().tobytes())
    return h.hexdigest()


def _mono_22k(audio):
    import torch
    wav = audio["waveform"]
    if wav.dim() == 3:
        wav = wav[0]
    mono = wav.to(torch.float32).mean(dim=0).cpu()
    sr = int(audio["sample_rate"])
    if sr != SR:
        # torchaudio's resampler: always present here, no resampy / soxr needed
        import torchaudio
        mono = torchaudio.functional.resample(mono, sr, SR)
    return np.ascontiguousarray(mono.numpy(), dtype=np.float32)


# ---------------------------------------------------------------------------
# tempo
# ---------------------------------------------------------------------------

def _refine_tempo(onset_env, tempo):
    """
    Score `tempo` against its half / double / triplet relatives on the onset
    autocorrelation with a comb (lag, 2 lag, 4 lag) and a gentle prior around
    dance tempos. Returns (bpm, confidence 0..1).
    """
    import librosa
    if not tempo or tempo <= 0:
        return 0.0, 0.0
    ac = librosa.autocorrelate(onset_env - onset_env.mean())
    ac = ac / (ac[0] + 1e-9)
    fps = SR / HOP

    def comb(bpm):
        lag = 60.0 * fps / bpm
        total, weight = 0.0, 0.0
        for k, w in ((1, 1.0), (2, 0.7), (4, 0.4)):
            pos = lag * k
            i = int(pos)
            if i + 1 >= len(ac):
                break
            frac = pos - i
            total += w * ((1 - frac) * ac[i] + frac * ac[i + 1])
            weight += w
        return total / weight if weight else 0.0

    def prior(bpm):
        return math.exp(-0.5 * ((math.log2(bpm / 115.0)) / 0.9) ** 2)

    candidates = {}
    for factor in (0.5, 2.0 / 3.0, 0.75, 1.0, 4.0 / 3.0, 1.5, 2.0):
        bpm = tempo * factor
        if 40 <= bpm <= 240:
            candidates[bpm] = comb(bpm) * prior(bpm)
    if not candidates:
        return float(tempo), 0.3
    ranked = sorted(candidates.items(), key=lambda kv: -kv[1])
    best_bpm, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = max(0.0, min(1.0, (best - second) / (abs(best) + 1e-9))) if best > 0 else 0.0
    return float(best_bpm), float(confidence)


# ---------------------------------------------------------------------------
# meter / downbeats
# ---------------------------------------------------------------------------

def _meter(accent):
    """
    accent: one standardised value per beat. For every bar length, the phase
    whose beats carry the most accent, scored as a z-statistic (mean gap to
    the other phases times sqrt(samples)) so a longer bar cannot win just
    because each phase has fewer, noisier samples. Returns
    (beats_per_bar, downbeat_phase, confidence 0..1).
    """
    n = len(accent)
    if n < 16:
        return 4, 0, 0.0
    a = (accent - accent.mean()) / (accent.std() + 1e-9)
    stats = {}
    for bar in (2, 3, 4, 6):
        means = [float(a[ph::bar].mean()) for ph in range(bar)]
        count = n // bar
        best_phase, best_z = 0, -1e9
        for ph, m in enumerate(means):
            others = [x for j, x in enumerate(means) if j != ph]
            z = (m - float(np.mean(others))) * math.sqrt(max(1, count))
            if z > best_z:
                best_phase, best_z = ph, z
        stats[bar] = (best_phase, best_z)
    # 4/4 is the default reading of a duple accent, 6 of a triple one, unless
    # the shorter bar is clearly the better fit
    duple = 4 if stats[4][1] >= stats[2][1] * 0.9 else 2
    triple = 6 if stats[6][1] >= stats[3][1] * 0.9 else 3
    bar = duple if stats[duple][1] >= stats[triple][1] else triple
    phase, z = stats[bar]
    # z ~ 3 is a real "one"; 1.5 is noise
    confidence = max(0.0, min(1.0, (z - 1.5) / 3.0))
    return bar, phase, float(confidence)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def _checkerboard_novelty(R, half):
    """Gaussian checkerboard kernel slid along the diagonal of a self-similarity matrix."""
    n = R.shape[0]
    L = 2 * half
    ax = np.arange(L) - half + 0.5
    g = np.exp(-0.5 * (ax / (half / 2.0)) ** 2)
    K = np.outer(g * np.sign(ax), g * np.sign(ax))
    padded = np.pad(R, half, mode="constant")
    nov = np.zeros(n)
    for t in range(n):
        nov[t] = float((padded[t:t + L, t:t + L] * K).sum())
    nov = np.clip(nov, 0, None)
    return nov / (nov.max() + 1e-9)


def _energy_step(energy, w=4):
    n = len(energy)
    out = np.zeros(n)
    for i in range(2, n):
        out[i] = abs(float(energy[i:i + w].mean()) - float(energy[max(0, i - w):i].mean()))
    return out / (out.max() + 1e-9)


def _stripe(R, a1, b1, a2, b2, min_cover=0.6):
    """Best mean along any diagonal of R[a1:b1, a2:b2] covering >= min_cover of the shorter side."""
    B = R[a1:b1, a2:b2]
    if B.size == 0:
        return 0.0
    h, w = B.shape
    need = max(3, int(min(h, w) * min_cover))
    best = 0.0
    for d in range(-(w - 1), h):
        diag = np.diagonal(B, offset=-d)
        if len(diag) >= need:
            best = max(best, float(diag.mean()))
    return best


def _link_groups(R, spans):
    """Union-find over sections whose stripe similarity clears LINK_THRESHOLD."""
    n = len(spans)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    scores = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = _stripe(R, spans[i][0], spans[i][1], spans[j][0], spans[j][1])
            scores[i, j] = scores[j, i] = s
            if s >= LINK_THRESHOLD:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    # a section that only made it into a group through one strong neighbour
    # (transitive linking) is detached again when it is, on average, not
    # like the rest of the group - the verse that follows a chorus over the
    # same loop is the usual case
    members = {}
    for i, r in enumerate(roots):
        members.setdefault(r, []).append(i)
    for r, m in members.items():
        if len(m) < 3:
            continue
        for i in m:
            mean_to_rest = float(np.mean([scores[i, j] for j in m if j != i]))
            if mean_to_rest < LINK_THRESHOLD * 0.8:
                roots[i] = -1 - i          # its own singleton group
    ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return [ids[r] for r in roots], scores


def _label_sections(n, groups, energies, scores):
    """
    Read form out of the groups. Returns (labels, confidence). Labels are
    None when there is no recurring group at all.
    """
    members = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    recurring = {g: m for g, m in members.items() if len(m) >= 2}
    if not recurring:
        return [None] * n, 0.0

    def chorus_key(g):
        # the chorus is the part that comes back most - and hits hardest when
        # it does. Count first: a loud two-time outro must not outrank the
        # hook that returns six times.
        m = recurring[g]
        return len(m) + 2.0 * float(np.mean([energies[i] for i in m]))
    chorus = max(recurring, key=chorus_key)
    others = sorted((g for g in recurring if g != chorus), key=lambda g: recurring[g][0])
    verse = others[0] if others else None

    labels = [None] * n
    for i in recurring[chorus]:
        labels[i] = "chorus"
    if verse is not None:
        for i in recurring[verse]:
            labels[i] = "verse"
        for g in others[1:]:
            before = all((i + 1 < n and labels[i + 1] == "chorus") for i in recurring[g])
            for i in recurring[g]:
                labels[i] = "pre-chorus" if before else "verse"
    first_form = min(i for i in range(n) if labels[i] is not None)
    last_form = max(i for i in range(n) if labels[i] is not None)
    for i in range(n):
        if labels[i] is not None:
            continue
        if i < first_form:
            labels[i] = "intro" if i == 0 else "verse"
        elif i > last_form:
            labels[i] = "outro" if i == n - 1 else "verse"
        else:
            labels[i] = "verse"
    # one bridge at most: the last one-off section that sits right before a
    # chorus with at least two choruses already behind it
    for i in range(n - 1, -1, -1):
        if (labels[i] == "verse" and len(members.get(groups[i], [])) == 1
                and i + 1 < n and labels[i + 1] == "chorus"
                and sum(1 for j in range(i) if labels[j] == "chorus") >= 2):
            labels[i] = "bridge"
            break

    # confidence: the chorus repeats have to be clearly more alike than any
    # chorus is to a non-chorus section
    ch = recurring[chorus]
    within = [scores[i, j] for i in ch for j in ch if i < j]
    across = [scores[i, j] for i in ch for j in range(n) if labels[j] != "chorus"]
    within_typ = float(np.median(within)) if within else 0.0
    across_typ = float(np.percentile(across, 90)) if across else 0.0
    margin = within_typ - across_typ
    conf = 0.4 + margin * 1.5
    return labels, float(max(0.0, min(1.0, conf)))


def _number(labels):
    counts = {}
    out = []
    for lab in labels:
        if lab in ("chorus", "verse", "pre-chorus"):
            counts[lab] = counts.get(lab, 0) + 1
            out.append(f"{lab} {counts[lab]}")
        else:
            out.append(lab)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def song_structure(audio):
    """
    {
      "duration", "bpm", "bpm_confidence",
      "meter", "meter_confidence", "beats": [s], "downbeats": [s], "bar_seconds",
      "sections": [{"start", "end", "label" | None, "numbered" | None, "group", "energy"}],
      "label_confidence", "labelled": bool,
    }
    or None when librosa is unavailable.
    """
    try:
        import librosa
    except Exception:
        return None
    key = _fingerprint(audio)
    if key in _CACHE:
        return _CACHE[key]

    y = _mono_22k(audio)
    duration = len(y) / SR
    out = {"duration": round(float(duration), 2), "bpm": 0.0, "bpm_confidence": 0.0, "meter": 4,
           "meter_confidence": 0.0, "beats": [], "downbeats": [], "bar_seconds": 0.0,
           "sections": [], "label_confidence": 0.0, "labelled": False}
    if duration < 12:
        return _remember(key, out)

    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, aggregate=np.median)
    tempo0, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=SR, hop_length=HOP, trim=False)
    tempo0 = float(np.atleast_1d(tempo0)[0])
    bpm, bpm_conf = _refine_tempo(onset_env, tempo0)
    if bpm and abs(bpm - tempo0) > 2.0:
        _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=SR, hop_length=HOP,
                                                 start_bpm=bpm, tightness=200, trim=False)
    beat_frames = np.asarray(beat_frames, dtype=int)
    if len(beat_frames) < 8:
        return _remember(key, out)
    beat_times = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP)
    out.update(bpm=round(bpm, 1), bpm_confidence=round(bpm_conf, 2),
               beats=[round(float(t), 3) for t in beat_times])

    # ---- beat-synchronous features ---------------------------------------
    def sync(F, agg):
        S = librosa.util.sync(F, beat_frames, aggregate=agg)
        return S[:, 1:len(beat_frames) + 1]

    chroma = sync(librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP), np.median)
    mfcc = sync(librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=20)[1:], np.mean)
    energy = sync(librosa.feature.rms(y=y, hop_length=HOP), np.mean)[0]
    onset_b = sync(onset_env[None, :], np.max)[0]
    spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    low = spec[(freqs >= 40) & (freqs <= 160)].sum(axis=0)
    bass_b = sync(np.maximum(np.diff(low, prepend=low[0]), 0)[None, :], np.max)[0]
    nb = min(chroma.shape[1], mfcc.shape[1], len(energy), len(onset_b), len(bass_b), len(beat_times))
    chroma, mfcc, energy, onset_b, bass_b = chroma[:, :nb], mfcc[:, :nb], energy[:nb], onset_b[:nb], bass_b[:nb]
    beat_times = beat_times[:nb]
    mfcc_z = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-9)

    def z(v):
        return (v - v.mean()) / (v.std() + 1e-9)

    chord_change = np.zeros(nb)
    for i in range(1, nb):
        a, b = chroma[:, i], chroma[:, i - 1]
        chord_change[i] = 1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    # ---- meter ------------------------------------------------------------
    accent = z(onset_b) + z(bass_b) + 0.8 * z(chord_change)
    meter, phase, meter_conf = _meter(accent)
    downbeat_idx = list(range(phase, nb, meter))
    trusted_meter = meter_conf >= MIN_METER_CONFIDENCE
    out.update(meter=meter, meter_confidence=round(meter_conf, 2),
               downbeats=[round(float(beat_times[i]), 3) for i in downbeat_idx] if trusted_meter else [],
               bar_seconds=round(60.0 / bpm * meter, 3) if bpm else 0.0)

    # ---- boundaries ---------------------------------------------------------
    R_chroma = np.asarray(librosa.segment.recurrence_matrix(chroma, width=3, mode="affinity", sym=True, metric="cosine"), float)
    R_mfcc = np.asarray(librosa.segment.recurrence_matrix(mfcc_z, width=3, mode="affinity", sym=True, metric="cosine"), float)
    half = 8
    novelty = (0.5 * _checkerboard_novelty(R_chroma, half) + 1.0 * _checkerboard_novelty(R_mfcc, half)
               + 0.6 * _energy_step(energy, 4)) / 2.1
    peaks = librosa.util.peak_pick(novelty, pre_max=6, post_max=6, pre_avg=12, post_avg=12, delta=0.04, wait=8)
    bounds = sorted({0, nb} | {int(p) for p in peaks if 0 < p < nb})
    if trusted_meter and downbeat_idx:
        dbs = np.asarray(downbeat_idx)
        snapped = {0, nb}
        for b in bounds:
            if b in (0, nb):
                continue
            j = int(dbs[np.argmin(np.abs(dbs - b))])
            snapped.add(j if abs(j - b) <= meter else b)
        bounds = sorted(snapped)
    # merge slivers shorter than two bars
    min_len = max(4, 2 * meter)
    changed = True
    while changed and len(bounds) > 2:
        changed = False
        for i in range(1, len(bounds) - 1):
            if bounds[i] - bounds[i - 1] < min_len or bounds[i + 1] - bounds[i] < min_len:
                del bounds[i]
                changed = True
                break
    # split stretches longer than MAX_SECTION_SECONDS at their strongest interior change
    spb = 60.0 / bpm if bpm else 0.5
    max_beats = int(MAX_SECTION_SECONDS / spb)
    changed = True
    while changed:
        changed = False
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            if b - a > max_beats and b - a >= 2 * min_len + 2:
                inner = novelty[a + min_len:b - min_len]
                if inner.size:
                    cut = a + min_len + int(np.argmax(inner))
                    bounds.insert(i + 1, cut)
                    changed = True
                    break

    def beat_time(i):
        return float(beat_times[i]) if i < nb else float(duration)

    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    energies = [float(energy[a:b].mean()) if b > a else 0.0 for a, b in spans]
    e_max = max(energies) or 1.0
    energies = [e / e_max for e in energies]

    # ---- repeats -> form ------------------------------------------------------
    R_both = np.asarray(librosa.segment.recurrence_matrix(
        np.vstack([chroma * 2.0, mfcc_z * 0.5]), width=3, mode="affinity", sym=True, metric="cosine"), float)
    groups, scores = _link_groups(R_both, spans) if len(spans) > 1 else ([0], np.zeros((1, 1)))
    labels, label_conf = _label_sections(len(spans), groups, energies, scores)
    labelled = label_conf >= MIN_LABEL_CONFIDENCE and sum(1 for l in labels if l == "chorus") >= 2
    numbered = _number(labels) if labelled else [None] * len(spans)
    out["sections"] = [
        {"start": round(beat_time(a), 2), "end": round(beat_time(b), 2),
         "label": labels[i] if labelled else None, "numbered": numbered[i],
         "group": int(groups[i]), "energy": round(energies[i], 2)}
        for i, (a, b) in enumerate(spans)
    ]
    out["label_confidence"] = round(label_conf, 2)
    out["labelled"] = labelled
    return _remember(key, out)


def _remember(key, value):
    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = value
    return value


# ---------------------------------------------------------------------------
# readouts the writer and the Song Analysis node print
# ---------------------------------------------------------------------------

def _fmt(t):
    m = int(t // 60)
    return f"{m}:{t - m * 60:05.2f}"


def structure_lines(structure):
    """Summary lines for a prompt or a readout, honouring the confidence gates."""
    if not structure or not structure.get("sections"):
        return []
    lines = []
    if structure["labelled"]:
        parts = [f"{s['numbered']} {_fmt(s['start'])}-{_fmt(s['end'])}" for s in structure["sections"]]
        lines.append(
            "SONG STRUCTURE (measured from the audio - the choruses are where the signature look "
            "returns and escalates; verses travel): " + " · ".join(parts)
        )
    else:
        parts = [f"{_fmt(s['start'])}-{_fmt(s['end'])}" for s in structure["sections"]]
        lines.append("SONG SECTIONS (measured; the repeats are not clear enough to name a chorus): " + " · ".join(parts))
    if structure.get("downbeats"):
        lines.append(
            f"METER: {structure['meter']}/4 at ~{structure['bpm']:g} BPM, a bar is {structure['bar_seconds']:.2f}s; "
            "each piece lists its downbeats - land the big gestures on them."
        )
    elif structure.get("bpm"):
        lines.append(f"TEMPO: ~{structure['bpm']:g} BPM (no clear downbeat measured - do not count bars).")
    return lines


def section_for_span(structure, start, end):
    """The numbered label of the section covering most of [start, end), or None."""
    if not structure or not structure.get("labelled"):
        return None
    best, best_overlap = None, 0.0
    for s in structure["sections"]:
        overlap = min(end, s["end"]) - max(start, s["start"])
        if overlap > best_overlap:
            best, best_overlap = s, overlap
    return best["numbered"] if best and best_overlap > 0 else None


def beats_in_span(structure, start, end):
    """Every beat inside [start, end), absolute seconds."""
    if not structure:
        return []
    return [t for t in structure.get("beats", []) if start - 1e-3 <= t < end - 0.05]


def downbeats_in_span(structure, start, end):
    if not structure:
        return []
    return [t for t in structure.get("downbeats", []) if start - 1e-3 <= t < end - 0.05]


def boundary_times(structure):
    """Section starts (excluding 0) - the cuts a video wants most."""
    if not structure:
        return []
    return [s["start"] for s in structure.get("sections", [])[1:]]


def summary_line(structure):
    """One line for the Song Analysis readout."""
    if not structure or not structure.get("sections"):
        return "structure: not measured"
    if structure["labelled"]:
        form = " · ".join(f"{s['numbered']} {_fmt(s['start'])}" for s in structure["sections"])
        head = f"form ({structure['label_confidence']:.0%} sure): {form}"
    else:
        head = f"{len(structure['sections'])} sections, no clear chorus repeat"
    if structure.get("downbeats"):
        head += f" | {structure['meter']}/4, bar {structure['bar_seconds']:.2f}s ({structure['meter_confidence']:.0%})"
    else:
        head += " | no clear downbeat"
    return head
