# APNext H3 Sync Check - does the picture actually move on the hits?
#
# Timestamps in a prompt are a request, not a guarantee, and a video model
# can nod along in the text while the frames do their own thing. The only
# honest answer is to measure the render: take the clip's frames and its
# audio, find the hits in the audio, find the moments the picture changes,
# and count how often a hit has a picture change within a couple of frames -
# against the same count for the same hits shifted to random times. A lift
# over chance is real sync; no lift means whatever sync you feel is coming
# from the cuts between clips, not from inside them.
#
# Measured on 18 H3 clips of one video (26 Aug): 58% of hits had a picture
# change within ±80 ms, chance 49%. Onset-envelope / picture-change
# correlation +0.05. That is why the Cut Plan puts the hits on the cuts.
#
# The score is per clip. Feed the decoded frames and the clip's own audio
# piece (what the writer / Save Clip carried), or a whole joined video and
# the whole song - the maths is the same.

import numpy as np

from ...utils.constants import CUSTOM_CATEGORY
from .sound_events import detect_events, parse_events

HIT_KINDS = ("BASS HIT", "IMPACT", "DROP", "STOP")


def visual_change_curve(images, size=(54, 96)):
    """Per-frame mean absolute difference of a small greyscale copy, 0..1."""
    import torch
    x = images
    if x.dim() == 4 and x.shape[-1] in (1, 3, 4):
        x = x[..., :3].mean(dim=-1)                      # [N,H,W] grey
    x = x.to(torch.float32)
    x = torch.nn.functional.interpolate(x[:, None], size=size, mode="area")[:, 0]
    d = (x[1:] - x[:-1]).abs().mean(dim=(1, 2))
    d = torch.cat([torch.zeros(1), d.cpu()])
    d = d.numpy()
    return d / (d.max() + 1e-9)


def change_peaks(curve, fps, floor=0.15, min_gap=0.12):
    gap = max(1, int(min_gap * fps))
    out = []
    for i in range(1, len(curve) - 1):
        if curve[i] >= curve[i - 1] and curve[i] > curve[i + 1] and curve[i] > floor:
            if not out or i - out[-1] >= gap:
                out.append(i)
    return np.array(out, dtype=float) / fps


def hit_rate(hits, peaks, tol):
    if len(hits) == 0 or len(peaks) == 0:
        return 0.0, np.array([])
    offs = np.array([peaks[np.argmin(np.abs(peaks - t))] - t for t in hits])
    return float(np.mean(np.abs(offs) <= tol)), offs


def chance_rate(hits, peaks, duration, tol, n=300, seed=7):
    if len(hits) == 0 or len(peaks) == 0 or duration <= 1.0:
        return 0.0
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        shift = rng.uniform(0.3, duration - 0.3)
        vals.append(hit_rate((hits + shift) % duration, peaks, tol)[0])
    return float(np.mean(vals))


def onset_correlation(audio, curve, fps):
    """Correlation of the audio onset envelope with the picture-change curve, and the best lag."""
    try:
        import librosa
    except Exception:
        return None
    wav = audio["waveform"]
    y = (wav[0] if wav.dim() == 3 else wav).mean(dim=0).cpu().numpy().astype(np.float32)
    sr = int(audio["sample_rate"])
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=max(1, int(sr / fps)))
    n = min(len(env), len(curve))
    if n < 12:
        return None
    a, b = env[:n] - env[:n].mean(), curve[:n] - curve[:n].mean()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    zero = float(np.corrcoef(a, b)[0, 1])
    best, best_lag = zero, 0
    for lag in range(-int(0.5 * fps), int(0.5 * fps) + 1):
        aa = a[max(0, -lag):n - max(0, lag)]
        bb = b[max(0, lag):n - max(0, -lag)]
        if len(aa) > 8:
            c = float(np.corrcoef(aa, bb)[0, 1])
            if c > best:
                best, best_lag = c, lag
    return {"r0": zero, "best_r": best, "best_lag_ms": best_lag / fps * 1000.0}


def measure_sync(images, audio, fps, events=None, tol_frames=2.0, min_strength=0.5):
    curve = visual_change_curve(images)
    duration = len(curve) / fps
    peaks = change_peaks(curve, fps)
    if events is None:
        events = detect_events(audio, bass_hits=True, impacts=True, drops_and_stops=True,
                               builds=False, sections=False, accents=False, min_strength=min_strength)
    hits = np.array([float(e["t"]) for e in events if e.get("type") in HIT_KINDS and 0 <= float(e["t"]) < duration])
    tol = tol_frames / fps
    rate, offs = hit_rate(hits, peaks, tol)
    base = chance_rate(hits, peaks, duration, tol)
    corr = onset_correlation(audio, curve, fps)
    per_hit = [(float(t), float(o)) for t, o in zip(hits, offs)]
    return {
        "duration": duration, "frames": int(len(curve)), "fps": float(fps),
        "hits": int(len(hits)), "visual_peaks": int(len(peaks)),
        "hit_rate": rate, "chance": base, "lift": rate - base,
        "median_offset_ms": float(np.median(np.abs(offs)) * 1000.0) if len(offs) else float("nan"),
        "tolerance_ms": tol * 1000.0, "correlation": corr, "per_hit": per_hit,
    }


def verdict(m):
    if m["hits"] == 0:
        return "no hits in this audio at this strength - nothing to check"
    lift = m["lift"]
    if lift >= 0.25:
        return "REAL SYNC: the picture changes on the hits far more than chance"
    if lift >= 0.10:
        return "some sync: a modest lift over chance"
    return "NO IN-CLIP SYNC: hits land on picture changes no more than chance - the sync you feel is the cuts"


def report(m):
    lines = [
        f"SYNC CHECK | {m['frames']} frames @ {m['fps']:g} fps = {m['duration']:.2f}s | {m['hits']} hits | {m['visual_peaks']} picture changes",
        f"hits with a picture change within ±{m['tolerance_ms']:.0f} ms: {m['hit_rate']:.0%}  vs chance {m['chance']:.0%}  -> lift {m['lift']:+.0%}",
    ]
    if m["hits"]:
        lines.append(f"median |offset| to the nearest picture change: {m['median_offset_ms']:.0f} ms")
    if m["correlation"]:
        c = m["correlation"]
        lines.append(f"onset envelope vs picture change: r = {c['r0']:+.2f} at 0 lag, best r = {c['best_r']:+.2f} at {c['best_lag_ms']:+.0f} ms picture lag")
    lines.append(verdict(m))
    if m["per_hit"]:
        lines.append("per hit (audio time -> nearest picture change offset):")
        lines.append("  " + "  ".join(f"{t:5.2f}s {o*1000:+4.0f}ms" for t, o in m["per_hit"][:24]) + ("  …" if len(m["per_hit"]) > 24 else ""))
    return "\n".join(lines)


class H3SyncCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The rendered frames (decoded clip, or a joined video)."}),
                "audio": ("AUDIO", {"tooltip": "The audio those frames were rendered to - the clip's own piece, or the whole song for a joined video."}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0, "step": 0.01, "tooltip": "Frame rate of `images`."}),
                "tolerance_frames": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 8.0, "step": 0.5,
                                               "tooltip": "How close (in frames) a picture change must be to a hit to count."}),
                "min_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                           "tooltip": "When no events are connected: only hits at least this strong are checked."}),
            },
            "optional": {
                "sound_events": ("STRING", {"forceInput": True,
                                            "tooltip": "Optional: the Sound Events list for THIS audio (timed from its start). Otherwise hits are detected here."}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("report", "hit_rate", "chance", "lift")
    OUTPUT_NODE = True
    FUNCTION = "check"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Measure whether a rendered clip's picture actually changes on the audio hits, against "
        "chance. A lift over chance is real in-clip sync; none means the sync comes from the cuts."
    )

    def check(self, images, audio, fps, tolerance_frames, min_strength, sound_events=""):
        events = parse_events(sound_events) if (sound_events or "").strip() else None
        m = measure_sync(images, audio, fps, events=events, tol_frames=tolerance_frames, min_strength=min_strength)
        text = report(m)
        print(f"🎯 H3 Sync Check | {text.splitlines()[1]} | {verdict(m)}")
        return {"ui": {"text": [text]}, "result": (text, float(m["hit_rate"]), float(m["chance"]), float(m["lift"]))}
