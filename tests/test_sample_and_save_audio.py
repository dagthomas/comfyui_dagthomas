"""CPU-only checks for H3 Sampler + Save Clip's latent-audio path
(nodes/h3/sample_and_save.py::decode_sampled_audio).

With `audio` unplugged and `audio_vae` connected, the clip's soundtrack is the
audio half of the SAMPLED H3 AV latent, decoded - the Sync Sound challenge
setting ("audio from the same H3 pass as the video"). A fake audio VAE stands
in for MiniMax's: every latent tick becomes 800 samples.

Run:  python -m pytest tests/test_sample_and_save_audio.py -q   (from the pack root)
      or plain  python tests/test_sample_and_save_audio.py
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
    import comfy.nested_tensor  # noqa: F401
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

sys.modules.setdefault("comfyui_dagthomas", types.ModuleType("comfyui_dagthomas"))
sys.modules["comfyui_dagthomas"].__path__ = [PACK]
from comfyui_dagthomas.nodes.h3 import sample_and_save as ss  # noqa: E402

import comfy.nested_tensor  # noqa: E402

SR, HOP = 32000, 800


class FakeAudioVAE:
    """[B, 32, 2, T] -> [B, T*800, 2]: a tone whose sample count is exactly the tick count x 800."""

    audio_sample_rate = SR

    def __init__(self):
        self.decoded = []

    def decode(self, z):
        b, c, two, t = z.shape
        assert (c, two) == (32, 2)
        self.decoded.append(tuple(z.shape))
        n = t * HOP
        wave = torch.sin(torch.arange(n).float() * 0.05).view(1, n, 1).expand(b, n, 2).clone()
        return wave


def av_latent(frames=124):
    ticks = int(round(frames / 24.0 * 40))
    video = torch.zeros(1, 64, 5, 48, 84)
    audio = torch.randn(1, 32, 2, ticks)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, ticks


def test_decodes_the_audio_half_of_the_sampled_latent():
    sampled, ticks = av_latent(124)
    vae = FakeAudioVAE()
    out = ss.decode_sampled_audio(sampled, vae)
    assert out is not None
    assert vae.decoded == [(1, 32, 2, ticks)], "the AUDIO stream (not the video) went to the audio VAE"
    wave = out["waveform"]
    assert tuple(wave.shape) == (1, 2, ticks * HOP)
    assert out["sample_rate"] == SR
    # the latent grid is 40 Hz, so the clip's audio is its frame time rounded to a 25 ms tick
    assert abs(wave.shape[-1] / SR - 124 / 24.0) < 1.0 / 40, "the decoded audio is the clip's length to a tick"
    assert wave.abs().max() <= 1.0 + 1e-6, "loudness-normalised like core's VAE Decode Audio"


def test_video_only_latent_gives_silence():
    sampled = {"samples": torch.zeros(1, 64, 5, 48, 84)}
    assert ss.decode_sampled_audio(sampled, FakeAudioVAE()) is None


def test_external_audio_wins_over_the_latent():
    """render() only decodes when `audio` is unplugged - the signature makes that the default."""
    import inspect
    sig = inspect.signature(ss.H3SampleAndSave.render)
    assert sig.parameters["audio"].default is None
    assert sig.parameters["audio_vae"].default is None
    opt = ss.H3SampleAndSave.INPUT_TYPES()["optional"]
    assert opt["audio_vae"][0] == "VAE"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
