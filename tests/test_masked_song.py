"""CPU-only checks for H3 Masked Song Latent (nodes/h3/masked_song.py).

A fake audio VAE stands in for MiniMax's: it turns every 800-sample tick into
one token whose value is the tick's index in the WHOLE song (plus a marker for
how much past it was encoded with), so the test can verify which song ticks
landed in which latent slots, that pre-roll tokens were dropped, and that the
lookahead window was cut back to the target length.

Run:  python -m pytest tests/test_masked_song.py -q      (from the pack root)
      or plain  python tests/test_masked_song.py
"""

import os
import sys
import types

import torch

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.dirname(os.path.dirname(PACK))
for p in (COMFY, os.path.dirname(PACK)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import comfy.nested_tensor  # noqa: F401  - the real one when ComfyUI is importable
except Exception:  # pragma: no cover - minimal stand-in
    comfy = types.ModuleType("comfy")
    nt = types.ModuleType("comfy.nested_tensor")

    class NestedTensor:
        def __init__(self, xs):
            self.tensors = list(xs)
            self.is_nested = True

        def unbind(self):
            return tuple(self.tensors)

    nt.NestedTensor = NestedTensor
    comfy.nested_tensor = nt
    sys.modules["comfy"] = comfy
    sys.modules["comfy.nested_tensor"] = nt

# the pack's relative imports need it loaded as a package; utils.constants is tiny
sys.modules.setdefault("comfyui_dagthomas", types.ModuleType("comfyui_dagthomas"))
sys.modules["comfyui_dagthomas"].__path__ = [PACK]
from comfyui_dagthomas.nodes.h3 import masked_song as ms  # noqa: E402

SR, HOP = 32000, 800


class FakeAudioVAE:
    """[B, L, 2] -> [B, 32, 2, ceil(L/800)]; channel 0 = absolute tick index of the
    window's first sample / 800 + token position, channel 1 = number of ticks
    of past in the window (so a hard-cut head is distinguishable)."""

    audio_sample_rate = SR

    def __init__(self):
        self.calls = []

    def encode(self, x):
        b, length, ch = x.shape
        assert ch == 2
        t = -(-length // HOP)
        self.calls.append(length)
        z = torch.zeros(b, 32, 2, t)
        # the window's absolute start is smuggled in sample 0 of channel 1 by the test
        start_tick = float(x[0, 0, 1].item())
        z[:, 0] = start_tick + torch.arange(t).float()
        z[:, 1] = float(t)
        return z


def song(seconds=30.0):
    """A song whose channel-1 samples carry their own absolute tick index (for the fake VAE)."""
    n = int(seconds * SR)
    wave = torch.zeros(1, 2, n)
    wave[0, 0] = torch.sin(torch.arange(n).float() * 0.01)
    wave[0, 1] = (torch.arange(n) // HOP).float()
    return {"waveform": wave, "sample_rate": SR}


def av_latent(frames=124, h=48, w=84):
    steps = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
    ticks_ = round(frames / 24 * 40)
    import comfy.nested_tensor
    video = torch.randn(1, 24, steps, h, w)
    audio = torch.randn(1, 32, 2, ticks_)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, steps, ticks_


def unbind(x):
    return tuple(x.unbind()) if hasattr(x, "unbind") else tuple(x.tensors)


def test_exact_slice_with_preroll_and_lookahead():
    node = ms.H3MaskedSongLatent()
    vae = FakeAudioVAE()
    latent, steps, ticks_ = av_latent(124)
    out, trim, clip = node.prepare(latent, vae, song(), clip_start_seconds=10.0, context_length=0,
                                   preroll_seconds=1.0, lookahead_seconds=0.2)
    video, audio = unbind(out["samples"])
    assert audio.shape == (1, 32, 2, ticks_) and video.shape[2] == steps
    # slot k holds song tick 400 + k: pre-roll tokens were dropped, lookahead cut back
    expect = 400 + torch.arange(ticks_).float()
    assert torch.equal(audio[0, 0, 0], expect), audio[0, 0, 0][:5]
    # the encode window was 40 ticks of past + the grid + 8 ticks ahead
    assert vae.calls == [(40 + ticks_ + 8) * HOP]
    assert audio[0, 1, 0, 0].item() == 40 + ticks_ + 8
    # masks: video all generated, audio fully frozen
    vmask, amask = unbind(out["noise_mask"])
    assert vmask.shape == (1, 1, steps, 48, 84) and vmask.min() == 1.0
    assert amask.shape == (1, 1, 2, ticks_) and amask.max() == 0.0
    assert trim == 0
    # clip_audio is the exact picture-length slice at the VAE rate
    assert clip["sample_rate"] == SR
    assert clip["waveform"].shape == (1, 2, round((10.0 + 124 / 24) * SR) - 10 * SR)


def test_zero_preroll_matches_old_hard_cut_and_audio_denoise_mask():
    node = ms.H3MaskedSongLatent()
    vae = FakeAudioVAE()
    latent, _, ticks_ = av_latent(124)
    out, _, _ = node.prepare(latent, vae, song(), clip_start_seconds=2.5, context_length=0,
                             preroll_seconds=0.0, lookahead_seconds=0.0, audio_denoise=0.1)
    _, audio = unbind(out["samples"])
    assert torch.equal(audio[0, 0, 0], 100 + torch.arange(ticks_).float())
    assert vae.calls == [ticks_ * HOP]
    _, amask = unbind(out["noise_mask"])
    assert torch.allclose(amask, torch.full_like(amask, 0.1))


def test_preroll_is_capped_by_the_song_start():
    node = ms.H3MaskedSongLatent()
    vae = FakeAudioVAE()
    latent, _, ticks_ = av_latent(124)
    out, _, _ = node.prepare(latent, vae, song(), clip_start_seconds=0.5, context_length=0,
                             preroll_seconds=2.0, lookahead_seconds=0.0)
    _, audio = unbind(out["samples"])
    assert torch.equal(audio[0, 0, 0], 20 + torch.arange(ticks_).float())
    assert vae.calls == [(20 + ticks_) * HOP]          # only the 0.5 s that exists


def test_latent_prefix_copies_source_tail_and_masks_it():
    node = ms.H3MaskedSongLatent()
    vae = FakeAudioVAE()
    latent, steps, _ = av_latent(124)
    src, src_steps, _ = av_latent(124)
    src_video, _ = unbind(src["samples"])
    out, trim, _ = node.prepare(latent, vae, song(), clip_start_seconds=4.0, context_length=39,
                                source_latent=src)
    video, _ = unbind(out["samples"])
    assert trim == 39
    pre_steps = ms.video_steps_for_frames(39)          # 12
    assert torch.equal(video[:, :, :pre_steps], src_video[:, :, src_steps - pre_steps:])
    vmask, _ = unbind(out["noise_mask"])
    assert vmask[0, 0, :pre_steps].max() == 0.0 and vmask[0, 0, pre_steps:].min() == 1.0


def test_voice_gate_freezes_sung_ticks_and_loosens_gaps():
    node = ms.H3MaskedSongLatent()
    vae = FakeAudioVAE()
    latent, _, ticks_ = av_latent(124)
    # a vocal stem: silence, then a 1 s phrase at 11.0-12.0 s, then silence
    n = int(30 * SR)
    v = torch.zeros(1, 2, n)
    v[0, :, 11 * SR:12 * SR] = 0.5 * torch.sin(torch.arange(SR).float() * 0.05)
    voice = {"waveform": v, "sample_rate": SR}
    out, _, _ = node.prepare(latent, vae, song(), clip_start_seconds=10.0, context_length=0,
                             voice=voice, audio_denoise=0.0, gap_denoise=0.15, gate_hold_seconds=0.1)
    _, amask = unbind(out["noise_mask"])
    row = amask[0, 0, 0]
    # ticks 40..79 are the phrase (11.0-12.0 s from a 10.0 s start); hold = 4 ticks either side,
    # plus the detector's 150 ms release tail after the phrase
    assert row[44:76].max() == 0.0, "sung ticks must be frozen"
    assert row[:35].min() == 0.15 and row[95:].min() == 0.15, "gaps must sit at gap_denoise"
    assert row[36:40].max() == 0.0 and row[80:84].max() == 0.0, "hold extends the frozen region"
    assert 44 <= int((row == 0.0).sum()) <= 60, "frozen span = phrase + hold + release"
    assert torch.equal(amask[0, 0, 0], amask[0, 0, 1]), "both stereo rows share the gate"


def test_prefix_snaps_to_h3_runs():
    assert ms.snap_context_length(40, 200, 124) == 39
    assert ms.snap_context_length(22, 200, 124) == 22
    assert ms.pixel_frames(ms.video_steps_for_frames(56)) == 56


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
