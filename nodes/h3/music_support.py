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

# Which side of a beat the cut sits on. The frame grid moves a cut in 0.71 s
# steps, so a cut is never ON a hit - it is just before it (the hit is the
# first thing in the new scene: the classic music-video cut) or just after
# it (the hit is the last thing in the outgoing scene, and the new one opens
# on the release). Auto is what the cutter always did: onsets and downbeats
# from either side, drops opening the new scene.
CUT_PLACEMENTS = [
    "Auto (nearest onset; drops open the new scene)",
    "Before the beat (the hit opens the new scene)",
    "After the beat (the hit closes the outgoing scene)",
]
BEAT_REACH = 0.72   # s - one grid step (17 frames): a one-sided window this wide
                    # always holds exactly one candidate cut for any beat


def placement_side(placement):
    """'before' / 'after' / None for a CUT_PLACEMENTS entry (or a bare word)."""
    p = str(placement or "").strip().lower()
    if p.startswith("before"):
        return "before"
    if p.startswith("after"):
        return "after"
    return None


def beat_fits(beat, cut, side, reach=BEAT_REACH, back=None):
    """
    Does `beat` sit on the wanted side of `cut`? before: the beat lands within
    `reach` s AFTER the cut; after: within `reach` s BEFORE it; None: within
    `reach` on either side (or `back` behind / `reach` ahead when `back` is given).
    """
    d = beat - cut          # >0: the beat comes after the cut
    if side == "before":
        return 0.0 <= d <= reach
    if side == "after":
        return -reach <= d <= 0.0
    lo = -(reach if back is None else back)
    return lo <= d <= reach


# ----------------------------------------------------------------------
# Frame grid

def frames_for_seconds(seconds, *, round_up=False, round_down=False):
    """Nearest valid H3 frame count for a duration (snapped to 5 + 17k)."""
    raw = seconds * FPS
    k = (raw - 5) / FRAME_STEP
    if round_up:
        k = math.ceil(k)
    elif round_down:
        k = math.floor(k)
    else:
        k = round(k)
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


def _peak_near(values, times, t, radius, side=None):
    """
    Max of `values` within +-radius seconds of t, and the time where it sits.
    `side` narrows the window to one side of t: 'before' looks only AFTER t
    (a beat the cut sits before), 'after' only BEFORE t.
    """
    n = values.numel()
    if n == 0:
        return 0.0, t
    hop = float(times[1] - times[0]) if n > 1 else HOP_SECONDS
    c = int(round(t / hop))
    r = max(1, int(round(radius / hop)))
    back, fwd = (0, r) if side == "before" else (r, 0) if side == "after" else (r, r)
    a, b = max(0, c - back), min(n, c + fwd + 1)
    if a >= b:
        return 0.0, t
    win = values[a:b]
    j = int(torch.argmax(win))
    return float(win[j]), float(times[a + j])


# ----------------------------------------------------------------------
# Lyrics

_TS_RE = re.compile(r"^\s*[\[(]?(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?[\])]?\s*[-–>:]?\s*(.*)$")
# a range `0:02 - 0:04 line` / `[0:02-0:04] line` / `0:02 --> 0:04 line`: the second
# time is the line's END - keep the start, drop the end so it never lands in the text
_TS_END_RE = re.compile(r"^(?:[-–—>→]*\s*)?\d{1,2}:\d{2}(?:[.:]\d{1,3})?[\])]?\s*[-–>:]?\s+(.*)$")


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
            m_end = _TS_END_RE.match(body)
            if m_end:
                body = m_end.group(1).strip()
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

def _structure_bonus_fn(structure, events=None, placement=None):
    """
    Cuts want to land where the song changes. A measured section boundary
    (chorus in, verse in) is worth more than any onset; a DROP wants the cut
    just BEFORE it so it lands at the top of the new scene; a STOP closes a
    scene; a downbeat beats the beat before it. Structure comes from
    song_structure() and events from the Sound Events node; either may be
    absent, and then that part is a no-op. `placement` (CUT_PLACEMENTS) puts
    drops and downbeats on one side of the cut; sections and stops are
    boundaries, not beats, and stay symmetric.
    """
    side = placement_side(placement)
    boundaries = [s["start"] for s in (structure or {}).get("sections", [])[1:]]
    downbeats = (structure or {}).get("downbeats") or []
    by_kind = {}
    for e in events or []:
        by_kind.setdefault(e.get("type"), []).append(float(e.get("t", 0.0)))
    drops, stops, section_events = by_kind.get("DROP", []), by_kind.get("STOP", []), by_kind.get("SECTION", [])

    def bonus(t):
        best = 0.0
        if any(beat_fits(d, t, side, reach=BEAT_REACH if side else 0.6, back=0.15) for d in drops):
            best = max(best, 2.5)
        if any(abs(b - t) <= 0.4 for b in boundaries) or any(abs(x - t) <= 0.4 for x in section_events):
            best = max(best, 2.0)
        if any(abs(x - t) <= 0.3 for x in stops):
            best = max(best, 1.5)
        if best == 0.0 and any(beat_fits(d, t, side, reach=BEAT_REACH if side else 0.2) for d in downbeats):
            best = 0.5
        return best
    return bonus


def _lookahead_fn(events, choices, placement=None, forced=()):
    """
    The cutter is greedy - one cut at a time - so a cut that is fine on its
    own can leave the NEXT drop (or the next tapped cut) unreachable: too
    close for the shortest clip, too far for the longest, or between two
    grid lengths. Penalise a candidate end when the nearest target ahead is
    within two more clips and no one- or two-clip run from there lands on
    it. A tap is the editor's explicit will and weighs twice a drop.
    """
    side = placement_side(placement)
    drops = [(float(e.get("t", 0.0)), 1.5) for e in events or [] if e.get("type") == "DROP"]
    taps = [(float(f), 3.0) for f in forced or ()]
    targets = sorted(drops + taps)
    longest = choices[-1] if choices else 0.0
    offsets = sorted(set(choices) | {a + b for a in choices for b in choices})
    reach, back = (BEAT_REACH, 0.15) if side else (0.6, 0.15)

    def penalty(end):
        for t, pen in targets:
            if t <= end - back:
                continue                    # already behind us
            if t - end > 2 * longest + reach:
                return 0.0                  # too far ahead to be decided here
            if beat_fits(t, end, side, reach=reach, back=back):
                return 0.0                  # this cut lands on it
            if any(beat_fits(t, end + o, side, reach=reach, back=back) for o in offsets):
                return 0.0                  # a later cut still can
            return pen
        return 0.0
    return penalty


def cut_reason(t, structure=None, events=None, lyric_times=(), forced=(), placement=None):
    """Why a cut sits where it sits - for the Cut Plan readout."""
    side = placement_side(placement)
    by_kind = {}
    for e in events or []:
        by_kind.setdefault(e.get("type"), []).append(float(e.get("t", 0.0)))
    if any(abs(f - t) <= 0.5 for f in (forced or ())):
        return "your tap"
    if any(beat_fits(d, t, side, reach=BEAT_REACH if side else 0.6, back=0.15) for d in by_kind.get("DROP", [])):
        return "drop closes the scene" if side == "after" else "drop lands here"
    if any(abs(s["start"] - t) <= 0.4 for s in (structure or {}).get("sections", [])[1:]):
        return "section start"
    if any(abs(x - t) <= 0.4 for x in by_kind.get("SECTION", [])):
        return "section change"
    if any(abs(x - t) <= 0.3 for x in by_kind.get("STOP", [])):
        return "stop"
    if any(-0.35 <= lt - t <= 0.6 for lt in lyric_times):
        return "lyric line"
    if any(beat_fits(d, t, side, reach=BEAT_REACH if side else 0.2) for d in (structure or {}).get("downbeats") or []):
        return "downbeat"
    return "onset"


def _forced_end(start, forced, shortest, longest, placement=None):
    """
    A hand-placed cut inside this clip's allowed span, on the frame grid - or
    None. The grid rounds the tap to the nearest valid length; with a
    placement it rounds DOWN (before: the cut lands ahead of the tapped hit)
    or UP (after: the hit stays in the outgoing scene).
    """
    side = placement_side(placement)
    for f in forced:
        if start + shortest - 0.3 <= f <= start + longest + 0.3:
            frames = frames_for_seconds(f - start, round_down=(side == "before"), round_up=(side == "after"))
            frames = max(MIN_FRAMES, min(MAX_FRAMES, frames))
            return start + seconds_for_frames(frames)
    return None


def segment_song(audio, max_seconds=15.0, min_seconds=5.2, mode=SEGMENT_MODES[0], lyric_times=(), structure=None, events=None, forced=(), placement=None):
    """
    Cut the song into [(start, end), ...] in seconds. Every segment length is
    on the H3 frame grid except the last one, which takes whatever is left
    (rounded up to the grid for rendering; the audio slice is zero-padded).
    `placement` (CUT_PLACEMENTS) decides which side of a beat the cut sits on.
    """
    side = placement_side(placement)
    onset_reach = BEAT_REACH if side else 0.18
    feats = analyse(audio)
    total = feats["duration"]
    choices = grid_durations(max_seconds, min_seconds)
    longest = choices[-1]
    shortest = choices[0]
    onset, novelty, times = feats["onset"], feats["novelty"], feats["times"]
    lyric_times = sorted(t for t in lyric_times if t is not None)
    structure_bonus = _structure_bonus_fn(structure, events, placement)

    def lyric_bonus(t):
        # cutting just before a lyric line is good, cutting right after one starts is bad
        bonus = 0.0
        for lt in lyric_times:
            if -0.35 <= lt - t <= 0.6:
                bonus = max(bonus, 1.0)
            elif 0 < t - lt < 1.2:
                bonus = min(bonus, -0.8)
        return bonus

    forced = sorted(f for f in (forced or ()) if 0.5 < f < total - 0.5)
    lookahead = _lookahead_fn(events, choices, placement, forced)
    segments = []
    start = 0.0
    while total - start > 1e-3:
        remaining = total - start
        if remaining <= longest + 1e-6 and not any(start + shortest <= f <= total - shortest for f in forced):
            segments.append((start, total))
            break
        hand = _forced_end(start, forced, shortest, longest, placement)
        if hand is not None and total - hand > 1e-3:
            segments.append((start, hand))
            start = hand
            continue
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
            if any(1e-3 < f - end < shortest - 0.3 for f in forced):
                continue  # would strand a tapped cut too close ahead to reach
            o, ot = _peak_near(onset, times, end, onset_reach, side)
            nv, _ = _peak_near(novelty, times, end, 0.35)
            score = 1.0 * o + 1.6 * nv + 0.015 * (dur / longest) * len(choices) + structure_bonus(end) - lookahead(end)
            if lyric_times:
                weight = 1.4 if mode == SEGMENT_MODES[2] else 0.8
                score += weight * lyric_bonus(end)
            if score > best:
                best, best_end = score, end
        segments.append((start, best_end))
        start = best_end

    return _merge_short_tail(segments, total, longest, shortest), feats


CUT_PLAN_HEADER = "CUT PLAN"
_CUT_LINE = re.compile(r"^\s*(\d+)\s+(\d+:\d{1,2}(?:\.\d+)?)\s*-\s*(\d+:\d{1,2}(?:\.\d+)?)")


def _clock(text):
    m, s = text.split(":")
    return int(m) * 60 + float(s)


def format_cut_plan(segments, total, min_seconds, max_seconds, *, structure=None, events=None,
                    lyric_times=(), labels=None, forced=(), placement=None, beat_offsets=None, beat_unit="beat"):
    """
    The scene list as text - one line per scene, `NN  m:ss.ss - m:ss.ss  dur  energy  section  cut: why`.
    Human-readable, hand-editable, and parse_cut_plan() reads it back. Only the
    number and the two clock times matter to the parser; everything after is
    commentary. `beat_offsets` (from snap_segments_to_beats) marks the plan
    beat-exact in the header and says per scene how far its end moved.
    """
    try:
        from .song_structure import section_for_span
    except Exception:  # pragma: no cover
        section_for_span = lambda *_: None  # noqa: E731
    side = placement_side(placement)
    head = f"{CUT_PLAN_HEADER}: {len(segments)} scenes | {min_seconds:g}-{max_seconds:g} s per scene | {fmt_time(total)} total"
    if side:
        head += " | cuts " + ("before" if side == "before" else "after") + " the beat"
    if beat_offsets is not None:
        head += f" | {BEAT_EXACT_MARK}: every scene opens on the {beat_unit}, frame-exact lengths (render with Chain Render - it trims the pad)"
    lines = [head]
    for i, (s, e) in enumerate(segments, 1):
        section = section_for_span(structure, s, e) if structure else None
        energy = labels[i - 1] if labels and i - 1 < len(labels) else ""
        why = "end of song" if i == len(segments) else cut_reason(e, structure, events, lyric_times, forced, placement)
        if beat_offsets is not None and i < len(segments):
            moved = beat_offsets[i - 1] if i - 1 < len(beat_offsets) else None
            why += f" -> on the {beat_unit} ({moved * 1000:+.0f} ms)" if moved is not None else f" (no {beat_unit} in reach - left on the grid)"
        lines.append(
            f"{i:02d}  {fmt_time(s)} - {fmt_time(e)}  {e - s:5.2f}s  {energy:<6} {(section or ''):<12} cut: {why}"
        )
    return "\n".join(lines)


def parse_cut_plan(text):
    """[(start, end), ...] from a cut plan (or any lines shaped like one); [] when nothing parses."""
    out = []
    for line in (text or "").splitlines():
        m = _CUT_LINE.match(line)
        if not m:
            continue
        try:
            s, e = _clock(m.group(2)), _clock(m.group(3))
        except ValueError:
            continue
        if e > s:
            out.append((s, e))
    out.sort()
    return out


def normalise_cut_plan(segments, total):
    """
    Make a (possibly hand-edited) plan contiguous and inside the song: starts
    at 0, no gaps or overlaps (each scene starts where the previous ends),
    nothing past the end, no empty scenes. The scene lengths are what the
    editor asked for; only the seams are repaired.
    """
    cleaned = []
    cursor = 0.0
    for _s, e in sorted(segments):
        e = min(float(e), float(total))
        if e - cursor >= 0.5:
            cleaned.append((cursor, e))
            cursor = e
    if cleaned and total - cursor > 1e-3:
        if total - cursor < 1.0:
            s0, _ = cleaned[-1]
            cleaned[-1] = (s0, float(total))
        else:
            cleaned.append((cursor, float(total)))
    return cleaned


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


# ----------------------------------------------------------------------
# Beat-exact cuts
# ----------------------------------------------------------------------
# The frame grid moves a cut in 0.71 s steps, so a scene boundary is never
# exactly on a beat - the cutter can only get within a third of a second.
# But the grid is a RENDER constraint, not a delivery one: Chain Render
# renders the next grid length up and trims the pad, so a scene may be any
# whole number of frames. This pass takes the cutter's grid-placed ends and
# moves each one onto the nearest beat (or downbeat), rounded to the frame,
# so every scene opens on the pulse to within half a frame (21 ms). The plan
# is marked `beat-exact` in its header and the writer then keeps the frame
# counts exact instead of snapping them back to the grid.

BEAT_SNAP_MODES = [
    "off (frame grid)",
    "nearest beat (frame-exact - Chain Render)",
    "nearest downbeat (frame-exact - Chain Render)",
]
BEAT_EXACT_MARK = "beat-exact"


def snap_segments_to_beats(segments, beats, total, min_seconds, max_seconds, reach=None):
    """
    [(start, end)] with every end (but the last) moved onto the nearest of
    `beats` (s) within `reach`, rounded to the frame; each scene starts where
    the previous one ends. A snap that would push a scene outside
    [min_seconds, max_seconds] is skipped. Returns (segments, offsets) where
    offsets[i] is how far scene i's end moved in seconds (None = not snapped).
    """
    beats = sorted(float(b) for b in beats or [])
    if not beats or len(segments) < 2:
        return list(segments), [None] * len(segments)
    if reach is None:
        gaps = [b - a for a, b in zip(beats, beats[1:]) if b - a > 0.05]
        reach = (min(gaps) / 2.0) if gaps else 0.4
    out, offsets = [], []
    start = float(segments[0][0])
    for i, (_s, e) in enumerate(segments):
        if i == len(segments) - 1:
            out.append((start, float(total)))
            offsets.append(None)
            break
        e = float(e)
        # the cutter may itself have left this scene a little outside the range
        # (a merged tail, a forced cut); a snap must not make that worse, but it
        # is allowed to stay as far out as the scene already was
        orig_len = e - start
        lo = min(min_seconds, orig_len) - 0.05
        hi = max(max_seconds, orig_len) + 0.05
        moved = None
        for nearest in sorted(beats, key=lambda b: abs(b - e))[:2]:     # nearest, then the other side
            if abs(nearest - e) > reach + 1e-6:
                break
            snapped = round(nearest * FPS) / FPS
            length = snapped - start
            if lo <= length <= hi and total - snapped >= 1.0:
                moved = snapped - e
                e = snapped
                break
        out.append((start, e))
        offsets.append(moved)
        start = e
    return out, offsets


def cut_plan_is_beat_exact(text):
    """Does this cut plan carry frame-exact (beat-snapped) scene lengths?"""
    head = (text or "").strip().splitlines()[:1]
    return bool(head) and head[0].startswith(CUT_PLAN_HEADER) and BEAT_EXACT_MARK in head[0]


def exact_frames(seconds):
    """A scene length as a whole number of frames, no grid - for beat-exact plans."""
    return max(1, int(round(float(seconds) * FPS)))


def segment_by_lyrics(audio, max_seconds=15.0, min_seconds=5.2, lyric_times=(), structure=None, events=None, forced=(), placement=None):
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
    structure_bonus = _structure_bonus_fn(structure, events, placement)
    side = placement_side(placement)
    onset_reach = BEAT_REACH if side else 0.18

    def musical_cut(start):
        best, best_end = -1e9, start + longest
        for dur in choices:
            end = start + dur
            if 1e-3 < total - end < shortest:
                continue  # would leave a stub shorter than the shortest clip
            if any(1e-3 < f - end < shortest - 0.3 for f in forced):
                continue  # would strand a tapped cut too close ahead to reach
            o, _ = _peak_near(onset, times, end, onset_reach, side)
            nv, _ = _peak_near(novelty, times, end, 0.35)
            score = 1.0 * o + 1.6 * nv + 0.015 * (dur / longest) * len(choices) + structure_bonus(end) - lookahead(end)
            if score > best:
                best, best_end = score, end
        return best_end

    forced = sorted(f for f in (forced or ()) if 0.5 < f < total - 0.5)
    lookahead = _lookahead_fn(events, choices, placement, forced)
    segments = []
    start = 0.0
    while total - start > 1e-3:
        hand = _forced_end(start, forced, shortest, longest, placement)
        if hand is not None and total - hand > 1e-3:
            segments.append((start, hand))
            start = hand
            continue
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
