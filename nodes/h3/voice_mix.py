# H3 Voice Over Music - the conditioning mix for a music video render.
#
# H3 lip-syncs to whatever audio sits in its latent and moves the picture to
# whatever rhythm it hears there. The two extremes both lose something: the
# full song buries the consonants (mouths drift), a bare vocal stem has no beat
# in it (nothing to cut on), and a synthetic thump track is not the song. This
# node does what a mix engineer would: the voice on top at full level, the real
# music underneath at a set level, and - optionally - the music ducked a few dB
# more only while the voice is actually sounding (a sidechain), so every word
# stays readable and the drums come back up between the phrases.
#
# Wire the result into H3 Chain Render's `conditioning_audio` (or the masked
# audio node's `master_audio`). The original song still goes to the output -
# this mix is only what the model listens to.

import torch

from ...utils.constants import CUSTOM_CATEGORY


def _as_bcn(audio):
    wave = audio["waveform"]
    if wave.dim() == 2:
        wave = wave.unsqueeze(0)
    return wave.detach().to("cpu").float(), int(audio["sample_rate"])


def _match(wave, sr, target_sr, channels, length):
    """Resample / re-channel / pad-trim `wave` [B, C, N] onto the voice's grid."""
    if sr != target_sr:
        import torchaudio
        wave = torchaudio.functional.resample(wave, sr, target_sr)
    b, c, n = wave.shape
    if c != channels:
        wave = wave.mean(dim=1, keepdim=True).expand(b, channels, n) if c != 1 else wave.expand(b, channels, n)
    if n < length:
        wave = torch.nn.functional.pad(wave, (0, length - n))
    elif n > length:
        wave = wave[:, :, :length]
    return wave.contiguous()


def _db(x):
    return 10.0 ** (float(x) / 20.0)


def voice_envelope(voice, sr, attack_ms=5.0, release_ms=150.0, window_ms=20.0):
    """0..1 'how much voice is sounding right now', per sample, smoothed like a compressor detector."""
    mono = voice.mean(dim=(0, 1))                       # [N]
    hop = max(1, int(sr * window_ms / 1000.0))
    n = mono.numel()
    frames = (n + hop - 1) // hop
    padded = torch.nn.functional.pad(mono, (0, frames * hop - n))
    rms = padded.view(frames, hop).pow(2).mean(dim=1).sqrt()          # [frames]
    peak = float(rms.max()) if rms.numel() else 0.0
    if peak <= 1e-6:
        return torch.zeros(n)
    level = (rms / peak).clamp(0.0, 1.0)
    # -40 dB below the loudest frame counts as silence; full duck from -20 dB up
    lo, hi = _db(-40.0), _db(-20.0)
    level = ((level - lo) / (hi - lo)).clamp(0.0, 1.0)
    # attack / release smoothing, frame by frame
    a = torch.exp(torch.tensor(-hop / (sr * attack_ms / 1000.0))).item()
    r = torch.exp(torch.tensor(-hop / (sr * release_ms / 1000.0))).item()
    out = torch.zeros_like(level)
    cur = 0.0
    for i in range(frames):
        target = float(level[i])
        coeff = a if target > cur else r
        cur = coeff * cur + (1.0 - coeff) * target
        out[i] = cur
    return out.repeat_interleave(hop)[:n]


def mix_voice_over_music(voice, musics, voice_db=0.0, music_db=-9.0, duck_db=-6.0, normalize=True):
    """
    voice, musics: AUDIO dicts. Returns (AUDIO, info dict). The music tracks are
    summed, resampled onto the voice's grid, put `music_db` under the voice and
    ducked a further `duck_db` while the voice sounds.
    """
    v, sr = _as_bcn(voice)
    b, c, n = v.shape
    bed = torch.zeros_like(v)
    used = 0
    for m in musics:
        if m is None:
            continue
        w, msr = _as_bcn(m)
        bed = bed + _match(w, msr, sr, c, n)[:b] if w.shape[0] >= b else bed + _match(w, msr, sr, c, n).expand(b, c, n)
        used += 1
    out = v * _db(voice_db)
    ducked_pct = 0.0
    if used:
        gain = torch.full((n,), _db(music_db))
        if duck_db < 0:
            env = voice_envelope(v, sr)
            gain = gain * (1.0 + env * (_db(duck_db) - 1.0))       # 1.0 -> duck while the voice sounds
            ducked_pct = float((env > 0.5).float().mean()) * 100.0
        out = out + bed * gain.view(1, 1, n)
    peak = float(out.abs().max()) if out.numel() else 0.0
    if normalize and peak > 1e-6:
        out = out * (0.98 / peak)
    elif peak > 1.0:
        out = out.clamp(-1.0, 1.0)
    if voice["waveform"].dim() == 2:
        out = out[0]
    return {"waveform": out.contiguous(), "sample_rate": sr}, {"music_tracks": used, "peak": peak, "ducked_pct": ducked_pct}


class H3VoiceOverMusic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice": ("AUDIO", {"tooltip": (
                    "The vocal stem (AudioSeparation's Vocals). Stays on top at full level - this is what "
                    "the mouth follows."
                )}),
                "music_db": ("FLOAT", {
                    "default": -9.0, "min": -40.0, "max": 6.0, "step": 0.5,
                    "tooltip": (
                        "How far under the voice the music sits, in dB. -9 keeps every drum hit audible "
                        "while the words stay on top; -3 is nearly the full mix, -18 is a whisper of beat."
                    ),
                }),
                "duck_db": ("FLOAT", {
                    "default": -6.0, "min": -24.0, "max": 0.0, "step": 0.5,
                    "tooltip": (
                        "Sidechain: while the voice is actually sounding, the music dips this much further "
                        "(5 ms in, 150 ms out) so consonants stay readable, and comes back up between "
                        "phrases where the beat is what matters. 0 = no ducking."
                    ),
                }),
                "voice_db": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Trim on the voice itself. Leave 0 unless the stem is unusually quiet or hot.",
                }),
                "normalize": ("BOOLEAN", {"default": True, "tooltip": "Peak-normalise the mix to -0.2 dBFS so nothing clips."}),
            },
            "optional": {
                "music_1": ("AUDIO", {"tooltip": (
                    "The music under the voice: the whole song (Load Audio), or a stem. Several inputs are "
                    "summed - e.g. AudioSeparation's Drums + Bass + Other for the instrumental without a "
                    "second copy of the voice."
                )}),
                "music_2": ("AUDIO", {"tooltip": "Another music stem to sum in (Bass)."}),
                "music_3": ("AUDIO", {"tooltip": "Another music stem to sum in (Other)."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    OUTPUT_TOOLTIPS = (
        "The conditioning mix - wire into H3 Chain Render's `conditioning_audio` (or the masked-audio node's "
        "`master_audio`). The original song still goes to the output; this is only what H3 listens to.",
        "What was mixed.",
    )
    FUNCTION = "mix"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "The voice on top, the real music under it, ducked a little more while the voice sounds - so H3 "
        "reads the mouth from the vocal stem AND moves on the actual beat. Feed Vocals into `voice` and the "
        "song or its other stems into `music_1..3`; send the result to Chain Render's `conditioning_audio`."
    )

    def mix(self, voice, music_db, duck_db, voice_db, normalize, music_1=None, music_2=None, music_3=None):
        musics = [m for m in (music_1, music_2, music_3) if m is not None]
        out, stats = mix_voice_over_music(voice, musics, voice_db=voice_db, music_db=music_db,
                                          duck_db=duck_db, normalize=normalize)
        if not musics:
            info = "voice only - connect the song or its stems to music_1..3 for the beat"
        else:
            info = (f"voice {voice_db:+.1f} dB over {stats['music_tracks']} music track(s) at {music_db:+.1f} dB"
                    + (f", ducked {duck_db:+.1f} dB while the voice sounds ({stats['ducked_pct']:.0f}% of the song)"
                       if duck_db < 0 else "")
                    + (f", normalised from peak {stats['peak']:.2f}" if normalize else ""))
        print(f"\U0001f399️ H3 Voice Over Music | {info}")
        return (out, info)
