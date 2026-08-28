"""Measure the MiniMax H3 audio VAE's timing behaviour, so H3 Masked Song Latent's
defaults are grounded rather than guessed.

Three questions, answered on a synthetic click-and-tone track (or on a real
song given as the first argument):

  1. ROUND-TRIP DELAY - encode -> decode and cross-correlate against the input.
     A non-zero lag means every masked clip's audio sits early/late by a
     constant, which the node can compensate in its slice start.

  2. PRE-ROLL - the encoder is a DAC conv stack (same-padded, non-causal) with
     a CAUSAL attention block on top.  A slice cut hard at `clip_start` gives
     the first tokens zero-padded past and nothing to attend to.  Encoding
     from `clip_start - N ticks` and dropping N tokens should bring the head
     tokens back to what a continuous encode of the song would have produced.
     The reference is the full-track encode; the A/B is the per-token error of
     the head tokens for N in PREROLLS.

  3. LOOKAHEAD - the mirror image at the tail (the convs see the future, the
     attention does not), so the last tokens of a hard-cut slice may differ
     from the continuous encode too.

Usage (from the ComfyUI root, with ComfyUI's python):

    python custom_nodes/comfyui_dagthomas/scripts/h3_audio_vae_probe.py [song.wav|mp3] [--vae PATH]

Prints a report; nothing is written.
"""

import argparse
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
COMFY_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)

SR = 32000
HOP = 800                      # samples per latent tick (40 Hz)
PREROLLS = (0, 1, 2, 4, 8, 16, 40, 80)     # ticks = 0 / 25 ms / 50 ms / ... / 2 s
LOOKAHEADS = (0, 1, 2, 4, 8, 16)
HEAD_TOKENS = 24               # tokens inspected at the head of a slice (0.6 s)
TAIL_TOKENS = 12


def synthetic_track(seconds=40.0, sr=SR, seed=7):
    """Clicks every 0.5 s on a moving tone bed with a little noise - enough
    transients to locate, enough sustain for the encoder to have a past."""
    g = torch.Generator().manual_seed(seed)
    n = int(seconds * sr)
    t = torch.arange(n, dtype=torch.float64) / sr
    bed = 0.15 * torch.sin(2 * math.pi * (220 + 40 * torch.sin(2 * math.pi * 0.11 * t)) * t)
    bed = bed + 0.08 * torch.sin(2 * math.pi * 3.0 * 220 * t) * (0.5 + 0.5 * torch.sin(2 * math.pi * 0.23 * t))
    noise = 0.02 * torch.randn(n, generator=g, dtype=torch.float64)
    clicks = torch.zeros(n, dtype=torch.float64)
    for k in range(int(seconds * 2)):
        i = int(k * 0.5 * sr) + 37          # deliberately off the tick grid
        if i + 64 < n:
            env = torch.exp(-torch.arange(64, dtype=torch.float64) / 6.0)
            clicks[i:i + 64] += 0.8 * env * torch.cos(2 * math.pi * 3000 * torch.arange(64, dtype=torch.float64) / sr)
    x = (bed + noise + clicks).clamp(-1, 1).float()
    return x.unsqueeze(0).repeat(2, 1).unsqueeze(0)   # [1, 2, L]


def load_track(path):
    import torchaudio
    wav, sr = torchaudio.load(path)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    wav = wav[:2]
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.unsqueeze(0).float()


def load_vae(path):
    import comfy.sd
    import comfy.utils
    sd = comfy.utils.load_torch_file(path)
    return comfy.sd.VAE(sd=sd)


def encode(vae, wav):
    """wav [1, 2, L] -> [1, 32, 2, T] on CPU float32."""
    return vae.encode(wav.movedim(1, -1)).float().cpu()


def decode(vae, z):
    """[1, 32, 2, T] -> [1, 2, L] on CPU float32."""
    out = vae.decode(z)
    if out.ndim == 3 and out.shape[1] != 2 and out.shape[-1] == 2:
        out = out.movedim(-1, 1)
    return out.float().cpu()


def xcorr_lag(a, b, max_lag):
    """Lag (in samples) at which b best matches a, searched in [-max_lag, max_lag].
    Positive = b is late relative to a."""
    a = a - a.mean()
    b = b - b.mean()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    best, best_lag = -1e30, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = torch.dot(a[: n - lag], b[lag:])
        else:
            c = torch.dot(a[-lag:], b[: n + lag])
        if c > best:
            best, best_lag = c, lag
    return best_lag


def click_onsets(x, sr=SR, period=0.5, window=0.02):
    """Onset of each click: the first sample in a +-window around the nominal
    time whose envelope crosses half the local peak."""
    env = x.abs()
    onsets = []
    n = x.numel()
    for k in range(int(n / sr / period)):
        c = int((k * period) * sr) + 37
        lo, hi = max(0, c - int(window * sr)), min(n, c + int(window * sr))
        seg = env[lo:hi]
        if seg.numel() == 0:
            continue
        peak = seg.max()
        idx = int((seg >= 0.5 * peak).nonzero()[0])
        onsets.append(lo + idx)
    return onsets


def rel_err(a, b):
    """Per-token relative error between two [1, 32, 2, T] latents: ||a-b|| / ||b||."""
    num = (a - b).flatten(0, 2).norm(dim=0)
    den = b.flatten(0, 2).norm(dim=0).clamp(min=1e-6)
    return (num / den)


def fmt_row(label, values):
    return f"  {label:<14}" + " ".join(f"{v:6.3f}" for v in values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song", nargs="?", help="a song file; default = synthetic clicks + tone")
    ap.add_argument("--vae", default=os.path.join(COMFY_ROOT, "models", "vae", "minimax_h3_audio_vae_fp32.safetensors"))
    ap.add_argument("--start", type=float, default=10.0, help="slice start in seconds (tick-aligned) for the pre-roll A/B")
    ap.add_argument("--length", type=float, default=5.0, help="slice length in seconds")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    print(f"audio VAE: {args.vae}")
    vae = load_vae(args.vae)
    print(f"  audio_sample_rate={getattr(vae, 'audio_sample_rate', '?')}  device={vae.device}")

    x = load_track(args.song) if args.song else synthetic_track()
    print(f"track: {'synthetic' if not args.song else args.song}  {x.shape[-1] / SR:.1f}s")

    # ---- 1. round-trip delay --------------------------------------------------
    z_full = encode(vae, x)
    y = decode(vae, z_full)
    n = min(x.shape[-1], y.shape[-1])
    print(f"\n1. ROUND-TRIP: encode {x.shape[-1]} samples -> {z_full.shape[-1]} ticks "
          f"(expected {math.ceil(x.shape[-1] / HOP)}) -> decode {y.shape[-1]} samples")
    lag = xcorr_lag(x[0, 0, :n].double(), y[0, 0, :n].double(), max_lag=HOP * 2)
    print(f"   cross-correlation lag: {lag:+d} samples = {lag / SR * 1000:+.2f} ms  (positive = decoded audio is LATE)")
    if not args.song:
        ox, oy = click_onsets(x[0, 0]), click_onsets(y[0, 0, :x.shape[-1]])
        d = [b - a for a, b in zip(ox, oy)]
        if d:
            d_t = torch.tensor(d, dtype=torch.float64)
            print(f"   click onsets: {len(d)} clicks, median shift {d_t.median():+.0f} samples "
                  f"({d_t.median() / SR * 1000:+.2f} ms), min {min(d):+d} max {max(d):+d}")

    # ---- 2. pre-roll ----------------------------------------------------------
    s_tick = int(round(args.start * 40))
    n_ticks = int(round(args.length * 40))
    s0, e0 = s_tick * HOP, (s_tick + n_ticks) * HOP
    ref = z_full[..., s_tick:s_tick + n_ticks]
    print(f"\n2. PRE-ROLL A/B: slice {args.start:.2f}s-{args.start + args.length:.2f}s "
          f"(ticks {s_tick}-{s_tick + n_ticks}); reference = the same ticks of the full-track encode")
    print(f"   per-token relative error of the first {HEAD_TOKENS} tokens (columns = tokens 0..{HEAD_TOKENS - 1}):")
    steady = None
    summary = []
    for pre in PREROLLS:
        a = max(0, s0 - pre * HOP)
        got = pre if a == s0 - pre * HOP else (s0 - a) // HOP
        z = encode(vae, x[..., a:e0])[..., got:got + n_ticks]
        err = rel_err(z, ref)
        head = err[:HEAD_TOKENS]
        mid = err[HEAD_TOKENS: max(HEAD_TOKENS + 1, n_ticks - TAIL_TOKENS)]
        steady = float(mid.mean()) if mid.numel() else float("nan")
        summary.append((pre, [float(v) for v in head], steady))
        print(fmt_row(f"pre {pre:>3} ticks", head[:12].tolist()) + f"   | mid-slice {steady:.3f}")
    # ticks whose error stays within 1.5x of the mid-slice floor, per pre-roll
    print("   head tokens worse than 1.5x the mid-slice floor:")
    for pre, head, st in summary:
        bad = sum(1 for v in head if v > 1.5 * st)
        print(f"     pre {pre:>3} ticks ({pre * 25:>5} ms): {bad:>2} of {HEAD_TOKENS} head tokens off"
              + ("   <- clean" if bad == 0 else ""))

    # ---- 3. lookahead ---------------------------------------------------------
    print(f"\n3. LOOKAHEAD A/B: last {TAIL_TOKENS} tokens of the same slice, encoded with M extra ticks after the cut:")
    for look in LOOKAHEADS:
        b = min(x.shape[-1], e0 + look * HOP)
        z = encode(vae, x[..., s0:b])[..., :n_ticks]
        err = rel_err(z, ref)[-TAIL_TOKENS:]
        print(fmt_row(f"look {look:>3} ticks", err.tolist()))

    # ---- 4. does a partial-tick start matter? ---------------------------------
    print("\n4. OFF-GRID START: the slice started 400 samples (half a tick) early, first 8 tokens vs reference")
    z = encode(vae, x[..., s0 - 400:e0])[..., :n_ticks]
    print(fmt_row("half tick", rel_err(z, ref)[:8].tolist()))
    print("   (the slice's tokens sit on their own grid, so a big number here is a re-tokenisation, not damage;\n"
          "    it does mean pre-roll and lookahead must be whole ticks relative to the clip start)")

    # ---- 5. is the hard-start head AUDIBLY wrong? -----------------------------
    print("\n5. DECODED HEAD: decode each slice encode and compare its first 100 / 500 ms to the input audio (SNR, dB)")
    target = x[0, 0, s0:e0].double()

    def snr(y, ms):
        k = int(SR * ms / 1000)
        a, b = target[:k], y[:k]
        return float(10 * torch.log10(a.pow(2).sum() / (a - b).pow(2).sum().clamp(min=1e-12)))

    for pre in (0, 8, 40, 80):
        a = max(0, s0 - pre * HOP)
        got = (s0 - a) // HOP
        z = encode(vae, x[..., a:e0 + 8 * HOP])[..., got:got + n_ticks]
        y = decode(vae, z)[0, 0, :n_ticks * HOP].double()
        print(f"   pre {pre:>3} ticks: SNR first 100 ms {snr(y, 100):6.1f} dB | first 500 ms {snr(y, 500):6.1f} dB | whole slice {snr(y, args.length * 1000):6.1f} dB")
    y_ref = decode(vae, ref)[0, 0, :n_ticks * HOP].double()
    print(f"   full-track ref: SNR first 100 ms {snr(y_ref, 100):6.1f} dB | first 500 ms {snr(y_ref, 500):6.1f} dB | whole slice {snr(y_ref, args.length * 1000):6.1f} dB")


if __name__ == "__main__":
    main()
