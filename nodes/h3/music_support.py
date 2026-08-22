# Audio-side helpers for the H3 Music Video Writer.
#
# A song comes in as a ComfyUI AUDIO dict ({"waveform": [B, C, T], "sample_rate"})
# and leaves as a list of segments, each at most `max_seconds` long, cut where
# the music actually changes (onsets / energy steps / section boundaries,
# lyric-line starts when timed lyrics are given). Segment lengths are snapped
# to MiniMax-H3's frame grid (5 + 17k frames at 24 fps, 124..362 frames trained)
# so every rendered clip is exactly as long as its audio slice and the stitched
# video never drifts against the song. Pure torch/numpy - no librosa needed.

import math
import re

import torch

FPS = 24
FRAME_STEP = 17            # H3 latent stride: valid frame counts are 5 + 17k
MIN_FRAMES = 124           # ~5.17 s - the shortest trained clip
MAX_FRAMES = 362           # ~15.08 s - the longest trained clip
HOP_SECONDS = 0.02         # analysis hop (50 Hz)

SEGMENT_MODES = [
    "Auto (cut on the music)",
    "Fixed (as long as allowed, then the rest)",
    "Lyric lines (cut before lyric lines when possible)",
]


# ----------------------------------------------------------------------
# Frame grid

def frames_for_seconds(seconds, *, round_up=False):
    """Nearest valid H3 frame count for a duration (snapped to 5 + 17k)."""
    raw = seconds * FPS
    k = (raw - 5) / FRAME_STEP
    k = math.ceil(k) if round_up else round(k)
    frames = 5 + FRAME_STEP * max(0, int(k))
    return max(MIN_FRAMES, min(MAX_FRAMES, frames))


def seconds_for_frames(frames):
    return frames / FPS


def grid_durations(max_seconds, min_seconds):
    """Every allowed segment duration (seconds) between min and max, ascending."""
    lo = max(MIN_FRAMES, frames_for_seconds(min_seconds, round_up=True))
    hi = min(MAX_FRAMES, frames_for_seconds(max_seconds))
    if hi < lo:
        hi = lo
    return [seconds_for_frames(f) for f in range(lo, hi + 1, FRAME_STEP)]


# ----------------------------------------------------------------------
# Analysis

def _mono(audio):
    wave = audio["waveform"]
    if wave.dim() == 3:
        wave = wave[0]
    if wave.dim() == 2:
        wave = wave.mean(dim=0)
    return wave.float().cpu(), int(audio["sample_rate"])


def analyse(audio):
    """
    Per-hop features for the song:
      rms      [N]   loudness envelope (linear)
      onset    [N]   spectral-flux onset strength, normalised 0..1
      novelty  [N]   long-window energy change (section boundaries), 0..1
      times    [N]   hop centre times in seconds
    """
    wave, sr = _mono(audio)
    hop = max(1, int(sr * HOP_SECONDS))
    n_fft = 2048 if sr >= 32000 else 1024
    if wave.numel() < n_fft:
        wave = torch.nn.functional.pad(wave, (0, n_fft - wave.numel()))
    window = torch.hann_window(n_fft)
    spec = torch.stft(wave, n_fft=n_fft, hop_length=hop, window=window, center=True, return_complex=True).abs()
    # [F, N] -> log-magnitude, spectral flux (positive differences only)
    logmag = torch.log1p(spec)
    flux = torch.clamp(logmag[:, 1:] - logmag[:, :-1], min=0).sum(dim=0)
    flux = torch.cat([flux[:1], flux])
    rms = torch.sqrt((spec ** 2).mean(dim=0) + 1e-9)
    n = rms.numel()
    times = torch.arange(n) * hop / sr

    # smooth a touch, then normalise
    def _smooth(x, k):
        if k <= 1:
            return x
        pad = k // 2
        return torch.nn.functional.avg_pool1d(
            torch.nn.functional.pad(x.view(1, 1, -1), (pad, pad), mode="replicate"), k, stride=1
        ).view(-1)

    onset = _smooth(flux, 3)
    onset = onset - _smooth(onset, 51)  # remove the slow trend
    onset = torch.clamp(onset, min=0)
    onset = onset / (onset.max() + 1e-9)

    loud = torch.log(rms + 1e-6)
    win = max(2, int(2.0 / HOP_SECONDS))  # 2 s before vs 2 s after
    cum = torch.cat([torch.zeros(1), torch.cumsum(loud, 0)])
    idx = torch.arange(n)
    a0 = torch.clamp(idx - win, min=0)
    b1 = torch.clamp(idx + win, max=n)
    before = (cum[idx] - cum[a0]) / torch.clamp((idx - a0).float(), min=1)
    after = (cum[b1] - cum[idx]) / torch.clamp((b1 - idx).float(), min=1)
    novelty = (after - before).abs()
    novelty = novelty / (novelty.max() + 1e-9)

    return {"rms": rms, "onset": onset, "novelty": novelty, "times": times, "hop_seconds": hop / sr,
            "duration": wave.numel() / sr, "sample_rate": sr}


# ----------------------------------------------------------------------
# Song character: tempo and aggression, from the envelopes analyse() already
# computes. Used to tell the writer whether the staging should be tender or
# punchy, and what tempo the cuts and choreography should sit on.

def estimate_bpm(feats):
    """
    Tempo from the autocorrelation of the onset envelope, folded into the
    70-180 BPM range. Returns 0.0 when the song has no usable pulse (ambient
    pads, spoken word), so callers can simply skip the tempo line.
    """
    onset, hop = feats["onset"], feats["hop_seconds"]
    x = onset - onset.mean()
    n = x.numel()
    lo = max(2, int(round(60.0 / 200.0 / hop)))   # 200 BPM
    hi = min(n // 2, int(round(60.0 / 60.0 / hop)))  # 60 BPM
    if hi <= lo + 2:
        return 0.0
    denom = float((x * x).sum()) + 1e-9
    ac = torch.tensor([float((x[:n - lag] * x[lag:]).sum()) / denom for lag in range(lo, hi + 1)])
    # a real beat repeats: reward lags whose double also correlates
    score = ac.clone()
    for i in range(score.numel()):
        lag2 = 2 * (lo + i) - lo
        if 0 <= lag2 < score.numel():
            score[i] += 0.5 * ac[lag2]
    j = int(torch.argmax(score))
    if float(ac[j]) < 0.05:  # no periodicity worth reporting
        return 0.0
    # parabolic interpolation for sub-hop precision
    lag = float(lo + j)
    if 0 < j < score.numel() - 1:
        y0, y1, y2 = float(score[j - 1]), float(score[j]), float(score[j + 1])
        d = y0 - 2 * y1 + y2
        if abs(d) > 1e-9:
            lag += 0.5 * (y0 - y2) / d
    bpm = 60.0 / (lag * hop)
    while bpm < 70.0:
        bpm *= 2.0
    while bpm > 180.0:
        bpm /= 2.0
    return round(bpm, 1)


def song_profile(feats):
    """
    The song's character as numbers and words:
      bpm            0.0 = no clear pulse
      spikes_per_sec onset peaks per second (transient density)
      dynamics_db    loudness spread p95-p15 in dB (small = compressed wall)
      intensity      0-100 aggression score
      label          gentle / laid-back / mid-energy / driving / aggressive
      dynamics_label compressed / moderate / wide
    """
    onset, rms, hop = feats["onset"], feats["rms"], feats["hop_seconds"]
    n = onset.numel()
    duration = max(1e-6, float(feats["duration"]))

    # spikes: local maxima of the onset envelope above a floor, >=100 ms apart
    spikes = 0
    gap = max(1, int(round(0.1 / hop)))
    last = -gap
    for i in range(1, n - 1):
        v = float(onset[i])
        if v > 0.25 and v >= float(onset[i - 1]) and v >= float(onset[i + 1]) and i - last >= gap:
            spikes += 1
            last = i
    spikes_per_sec = spikes / duration

    loud_db = 20.0 * torch.log10(rms + 1e-9)
    active = loud_db[loud_db > float(loud_db.max()) - 60.0]  # ignore true silence
    if active.numel() < 4:
        active = loud_db
    q = torch.quantile(active, torch.tensor([0.15, 0.5, 0.95]))
    dynamics_db = float(q[2] - q[0])
    # sustained loudness: how close the typical level sits to the loud peaks
    sustain = max(0.0, min(1.0, 1.0 - (float(q[2] - q[1]) / 12.0)))
    punch = float(torch.quantile(onset, 0.9))

    intensity = int(round(100.0 * min(1.0, (
        0.45 * min(1.0, spikes_per_sec / 3.0)  # transient density
        + 0.35 * sustain                        # wall-of-sound loudness
        + 0.20 * punch                          # how hard the hits hit
    ))))
    label = (
        "gentle" if intensity < 20 else
        "laid-back" if intensity < 40 else
        "mid-energy" if intensity < 60 else
        "driving" if intensity < 80 else
        "aggressive"
    )
    dynamics_label = (
        "compressed" if dynamics_db < 6.0 else
        "moderate" if dynamics_db < 15.0 else
        "wide"
    )
    return {
        "bpm": estimate_bpm(feats),
        "spikes_per_sec": round(spikes_per_sec, 2),
        "dynamics_db": round(dynamics_db, 1),
        "intensity": intensity,
        "label": label,
        "dynamics_label": dynamics_label,
    }


def profile_line(profile):
    """One human-readable line, for the console, the segment table and the model."""
    bpm = f"~{profile['bpm']:g} BPM" if profile["bpm"] else "no steady pulse"
    return (
        f"{bpm} | {profile['label']} (intensity {profile['intensity']}/100) | "
        f"{profile['dynamics_label']} dynamics ({profile['dynamics_db']:g} dB) | "
        f"{profile['spikes_per_sec']:g} onset spikes/s"
    )


def _peak_near(values, times, t, radius):
    """Max of `values` within +-radius seconds of t, and the time where it sits."""
    n = values.numel()
    if n == 0:
        return 0.0, t
    hop = float(times[1] - times[0]) if n > 1 else HOP_SECONDS
    c = int(round(t / hop))
    r = max(1, int(round(radius / hop)))
    a, b = max(0, c - r), min(n, c + r + 1)
    if a >= b:
        return 0.0, t
    win = values[a:b]
    j = int(torch.argmax(win))
    return float(win[j]), float(times[a + j])


# ----------------------------------------------------------------------
# Lyrics

_TS_RE = re.compile(r"^\s*[\[(]?(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?[\])]?\s*[-–>:]?\s*(.*)$")


def parse_lyrics(text):
    """
    [(start_seconds or None, line), ...]. Accepts LRC `[mm:ss.xx] line`,
    `m:ss line`, `m:ss - line`, or plain lines (start = None). Section tags
    like [Chorus] are kept as lines with no time but flagged with a leading '#'.
    """
    out = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _TS_RE.match(line)
        if m and m.group(4) is not None:
            start = int(m.group(1)) * 60 + int(m.group(2))
            frac = m.group(3)
            if frac:
                start += int(frac) / (10 ** len(frac))
            body = m.group(4).strip()
            if body:
                out.append((float(start), body))
            else:
                out.append((float(start), ""))  # timed silence marker
            continue
        if re.match(r"^\[[A-Za-z][^\]]*\]$", line) or re.match(r"^\([A-Za-z][^)]*\)$", line):
            out.append((None, "#" + line.strip("[]()")))
            continue
        out.append((None, line))
    return out


def place_untimed_lyrics(lyrics, duration, lead_in=0.0):
    """
    Spread untimed lines evenly over the song (after an optional lead-in) so
    each segment still gets the lines that roughly belong to it. Already-timed
    lines keep their time; untimed lines between two timed ones are spread
    between them.
    """
    if not lyrics:
        return []
    placed = []
    # anchors: (index, time) for timed lines, plus virtual start/end
    anchors = [(-1, lead_in)] + [(i, t) for i, (t, _) in enumerate(lyrics) if t is not None] + [(len(lyrics), duration)]
    for (i0, t0), (i1, t1) in zip(anchors, anchors[1:]):
        gap = [k for k in range(i0 + 1, i1) if lyrics[k][0] is None and not lyrics[k][1].startswith("#")]
        if i0 >= 0:
            placed.append((t0, lyrics[i0][1], True))
        for j, k in enumerate(gap):
            t = t0 + (t1 - t0) * (j + 0.5) / (len(gap) + 1) if gap else t0
            placed.append((t, lyrics[k][1], False))
    placed.sort(key=lambda p: p[0])
    return placed  # [(time, line, exact)]


# ----------------------------------------------------------------------
# Segmentation

def segment_song(audio, max_seconds=15.0, min_seconds=5.2, mode=SEGMENT_MODES[0], lyric_times=()):
    """
    Cut the song into [(start, end), ...] in seconds. Every segment length is
    on the H3 frame grid except the last one, which takes whatever is left
    (rounded up to the grid for rendering; the audio slice is zero-padded).
    """
    feats = analyse(audio)
    total = feats["duration"]
    choices = grid_durations(max_seconds, min_seconds)
    longest = choices[-1]
    shortest = choices[0]
    onset, novelty, times = feats["onset"], feats["novelty"], feats["times"]
    lyric_times = sorted(t for t in lyric_times if t is not None)

    def lyric_bonus(t):
        # cutting just before a lyric line is good, cutting right after one starts is bad
        bonus = 0.0
        for lt in lyric_times:
            if -0.35 <= lt - t <= 0.6:
                bonus = max(bonus, 1.0)
            elif 0 < t - lt < 1.2:
                bonus = min(bonus, -0.8)
        return bonus

    segments = []
    start = 0.0
    while total - start > 1e-3:
        remaining = total - start
        if remaining <= longest + 1e-6:
            segments.append((start, total))
            break
        if mode == SEGMENT_MODES[1]:
            end = start + longest
            # do not leave a stub shorter than the shortest allowed clip
            if total - end < shortest:
                end = start + choices[max(0, len(choices) // 2)]
            segments.append((start, end))
            start = end
            continue
        best, best_end = -1e9, start + longest
        for dur in choices:
            end = start + dur
            if total - end < shortest and total - end > 1e-3:
                # would leave a stub: allowed only if we can still fit one more clip
                continue
            o, ot = _peak_near(onset, times, end, 0.18)
            nv, _ = _peak_near(novelty, times, end, 0.35)
            score = 1.0 * o + 1.6 * nv + 0.015 * (dur / longest) * len(choices)
            if lyric_times:
                weight = 1.4 if mode == SEGMENT_MODES[2] else 0.8
                score += weight * lyric_bonus(end)
            if score > best:
                best, best_end = score, end
        segments.append((start, best_end))
        start = best_end

    return _merge_short_tail(segments, total, longest, shortest), feats


def _merge_short_tail(segments, total, longest, shortest):
    """A tail shorter than the shortest clip is folded into its predecessor."""
    if len(segments) >= 2 and (segments[-1][1] - segments[-1][0]) < shortest:
        s0, _ = segments[-2]
        if total - s0 <= longest + 1e-6:
            segments[-2:] = [(s0, total)]
        else:
            mid = s0 + seconds_for_frames(frames_for_seconds((total - s0) / 2))
            segments[-2:] = [(s0, mid), (mid, total)]
    return segments


def segment_by_lyrics(audio, max_seconds=15.0, min_seconds=5.2, lyric_times=()):
    """
    Lyrics-driven cut: [(start, end), ...] where (almost) every segment starts
    where a lyric phrase starts. The song is walked front to back; each cut
    snaps to the H3 grid duration that lands just before the first lyric line
    at least min_seconds away, so a new scene begins right as its line begins.
    Stretches without lyric lines cut on the music (onsets / energy changes)
    like Auto. Needs timed lyric starts to mean anything - the caller decides
    the fallback when none exist.
    """
    feats = analyse(audio)
    total = feats["duration"]
    choices = grid_durations(max_seconds, min_seconds)
    longest, shortest = choices[-1], choices[0]
    onset, novelty, times = feats["onset"], feats["novelty"], feats["times"]
    lyric_times = sorted(t for t in lyric_times if t is not None)

    def musical_cut(start):
        best, best_end = -1e9, start + longest
        for dur in choices:
            end = start + dur
            if 1e-3 < total - end < shortest:
                continue  # would leave a stub shorter than the shortest clip
            o, _ = _peak_near(onset, times, end, 0.18)
            nv, _ = _peak_near(novelty, times, end, 0.35)
            score = 1.0 * o + 1.6 * nv + 0.015 * (dur / longest) * len(choices)
            if score > best:
                best, best_end = score, end
        return best_end

    segments = []
    start = 0.0
    while total - start > 1e-3:
        if total - start <= longest + 1e-6:
            segments.append((start, total))
            break
        # the first lyric start reachable with a valid clip length
        target = next(
            (lt for lt in lyric_times if start + shortest - 0.35 <= lt <= start + longest + 0.6),
            None,
        )
        if target is None:
            end = musical_cut(start)
        else:
            # grid duration landing as close as possible, ideally just BEFORE
            # the line starts (cutting into a started line is heavily penalised)
            best, end = -1e9, None
            for dur in choices:
                cand = start + dur
                if 1e-3 < total - cand < shortest:
                    continue
                d = target - cand  # >0: cut lands before the line
                score = -d if d >= 0 else -0.6 + 2.0 * d
                if score > best:
                    best, end = score, cand
            if end is None:
                end = musical_cut(start)
        segments.append((start, end))
        start = end

    return _merge_short_tail(segments, total, longest, shortest), feats


def energy_labels(feats, segments):
    """'quiet' / 'medium' / 'loud' / 'peak' per segment, relative to the song."""
    rms, times = feats["rms"], feats["times"]
    means = []
    for s, e in segments:
        mask = (times >= s) & (times < e)
        means.append(float(rms[mask].mean()) if mask.any() else 0.0)
    if not means:
        return []
    ref = sorted(means)
    q = lambda p: ref[min(len(ref) - 1, int(p * len(ref)))]
    labels = []
    for m in means:
        if m >= q(0.85):
            labels.append("peak")
        elif m >= q(0.55):
            labels.append("loud")
        elif m >= q(0.25):
            labels.append("medium")
        else:
            labels.append("quiet")
    return labels


def slice_audio(audio, start, end, frames=None):
    """
    AUDIO dict for [start, end) seconds. With `frames`, the slice is padded or
    trimmed to exactly frames/24 seconds so it matches the rendered clip.
    """
    wave = audio["waveform"]
    sr = int(audio["sample_rate"])
    a = int(round(start * sr))
    b = int(round(end * sr))
    piece = wave[..., a:b]
    if frames is not None:
        want = int(round(frames / FPS * sr))
        if piece.shape[-1] < want:
            piece = torch.nn.functional.pad(piece, (0, want - piece.shape[-1]))
        else:
            piece = piece[..., :want]
    return {"waveform": piece.contiguous(), "sample_rate": sr}


def lyrics_for_segment(placed, start, end):
    """Lines whose (placed) time falls inside [start, end). Returns [(time, line, exact)]."""
    return [(t, line, exact) for t, line, exact in placed if start - 1e-6 <= t < end - 1e-6]


def fmt_time(t):
    m = int(t // 60)
    s = t - m * 60
    return f"{m}:{s:05.2f}"


def segment_table(segments, labels, frames, placed_lyrics):
    """Human-readable list of the cuts (also what the model gets)."""
    lines = []
    for i, ((s, e), label, fr) in enumerate(zip(segments, labels, frames), 1):
        lyr = [line for _, line, _ in lyrics_for_segment(placed_lyrics, s, e) if line and not line.startswith("#")]
        lyric_text = " / ".join(lyr) if lyr else "[instrumental]"
        lines.append(
            f"{i:02d}  {fmt_time(s)} – {fmt_time(e)}  ({e - s:5.2f}s, {fr} frames)  energy: {label:6s}  lyrics: {lyric_text}"
        )
    return "\n".join(lines)
