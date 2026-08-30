"""CPU-only checks for H3 Voice Over Music (nodes/h3/voice_mix.py).

The bed is two tones - 60 Hz (a kick / bass fundamental) and 1 kHz (inside the
vocal band) - under a voice that sounds the whole time. Full-band ducking must
pull both down by `duck_db`; vocal-band ducking must pull only the 1 kHz tone
down and leave the 60 Hz tone alone. Per-stem trims scale their stem before
the sum.

Run:  python -m pytest tests/test_voice_mix.py -q      (from the pack root)
      or plain  python tests/test_voice_mix.py
"""

import math
import os
import sys
import types

import torch

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.dirname(os.path.dirname(PACK))
for p in (COMFY, os.path.dirname(PACK)):
    if p not in sys.path:
        sys.path.insert(0, p)

sys.modules.setdefault("comfyui_dagthomas", types.ModuleType("comfyui_dagthomas"))
sys.modules["comfyui_dagthomas"].__path__ = [PACK]
from comfyui_dagthomas.nodes.h3 import voice_mix as vm  # noqa: E402

SR = 32000
SECONDS = 2.0
N = int(SR * SECONDS)
T = torch.arange(N).float() / SR


def tone(hz, amp=0.4):
    return {"waveform": (amp * torch.sin(2 * math.pi * hz * T)).view(1, 1, N).repeat(1, 2, 1), "sample_rate": SR}


def magnitude(wave, hz):
    """|X(hz)| of the first channel, normalised so a full-scale sine reads 1.0."""
    spec = torch.fft.rfft(wave[0, 0])
    k = int(round(hz * N / SR))
    return float(spec[k].abs()) * 2.0 / N


def db(x):
    return 10.0 ** (x / 20.0)


VOICE = tone(3000.0, 0.5)          # sounds the whole time -> the duck is fully engaged
LOW, MID = tone(60.0), tone(1000.0)


def test_full_band_duck_pulls_the_whole_bed():
    out, stats = vm.mix_voice_over_music(VOICE, [LOW, MID], music_db=0.0, duck_db=-12.0, normalize=False,
                                         duck_band=vm.DUCK_BANDS[0])
    w = out["waveform"]
    assert not stats["banded"]
    assert abs(magnitude(w, 60.0) / 0.4 - db(-12.0)) < 0.03
    assert abs(magnitude(w, 1000.0) / 0.4 - db(-12.0)) < 0.03


def test_vocal_band_duck_leaves_the_kick_alone():
    out, stats = vm.mix_voice_over_music(VOICE, [LOW, MID], music_db=0.0, duck_db=-12.0, normalize=False,
                                         duck_band=vm.DUCK_BANDS[1])
    w = out["waveform"]
    assert stats["banded"]
    assert abs(magnitude(w, 60.0) / 0.4 - 1.0) < 0.03, "60 Hz is outside the vocal band: untouched"
    assert abs(magnitude(w, 1000.0) / 0.4 - db(-12.0)) < 0.03, "1 kHz is inside the vocal band: ducked"
    assert abs(magnitude(w, 3000.0) / 0.5 - 1.0) < 0.03, "the voice itself is never ducked"


def test_split_band_is_exact_and_linear_phase():
    bed = LOW["waveform"] + MID["waveform"]
    inside, outside = vm.split_band(bed, SR)
    assert torch.allclose(inside + outside, bed, atol=1e-5)
    assert magnitude(inside, 1000.0) > 0.39 and magnitude(inside, 60.0) < 0.01
    assert magnitude(outside, 60.0) > 0.39 and magnitude(outside, 1000.0) < 0.01


def test_per_stem_trims_scale_before_the_sum():
    out, _ = vm.mix_voice_over_music(VOICE, [LOW, MID], music_db=0.0, duck_db=0.0, normalize=False,
                                     music_dbs=[-6.0, 0.0])
    w = out["waveform"]
    assert abs(magnitude(w, 60.0) / 0.4 - db(-6.0)) < 0.03
    assert abs(magnitude(w, 1000.0) / 0.4 - 1.0) < 0.03


def test_node_widgets_are_appended_after_the_old_ones():
    """Saved workflows carry [music_db, duck_db, voice_db, normalize]; the new widgets must come after."""
    it = vm.H3VoiceOverMusic.INPUT_TYPES()
    assert list(it["required"].keys()) == ["voice", "music_db", "duck_db", "voice_db", "normalize"]
    assert list(it["optional"].keys()) == ["music_1", "music_2", "music_3", "duck_band", "music_1_db", "music_2_db", "music_3_db"]
    out, info = vm.H3VoiceOverMusic().mix(VOICE, -3.0, -4.0, 0.0, True, music_1=LOW, music_2=MID,
                                          duck_band=vm.DUCK_BANDS[1], music_3_db=-3.0)
    assert "vocal band" in info and out["sample_rate"] == SR


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
