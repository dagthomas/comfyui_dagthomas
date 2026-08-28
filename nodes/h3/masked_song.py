# APNext H3 Masked Song Latent - the exact song slice, frozen in the H3 audio latent.
#
# H3 lip-syncs to whatever sits in its audio latent. This node writes the
# master song's slice for this clip into the target AV latent and protects it
# from denoising, so the picture is generated against the real, frozen vocal -
# the lip-sync is enforced by the model, not asked for in the prompt. A
# drop-in for MiniMaxH3SongMaskedAVContext (ComfyUI-H3-Motion-Context-MultiRef):
# same inputs, same three outputs, same chain-continuation paths - without the
# third-party pack, and with the audio VAE driven the way its timing needs
# (measured by scripts/h3_audio_vae_probe.py on the real H3 audio VAE):
#
#   * the VAE's round-trip delay is 0 ms - nothing to compensate;
#   * its encoder is a same-padded conv stack under a CAUSAL attention block,
#     so a slice cut hard at clip_start gives the first tokens a zero past and
#     nothing to attend to: token 0 is ~57 % off what a continuous encode of
#     the song gives, and the error takes ~0.5 s to decay. Encoding from
#     `preroll_seconds` earlier and dropping those tokens brings the head to
#     the floor (1 s -> ~1 %, 2 s -> at the floor). The old node cut hard.
#   * the convs also look ahead, so the last ~200 ms of a hard-cut slice are
#     off too; `lookahead_seconds` of real song after the cut (silence once
#     the song has ended) fixes the tail. The old node did this only as an
#     error-recovery retry.
#   * both must be whole 25 ms ticks relative to the clip start - the latent
#     grid is 40 Hz and an off-grid window is a different tokenisation.
#
# `audio_denoise` is the nested-mask value for the audio rows: 0 keeps the
# old fully-frozen behaviour; a small value (0.05-0.15) lets the model touch
# the audio rows at that noise level while the vocal stays. Core handles a
# fractional mask natively (per-row timesteps), so this is an A/B knob.
#
# Mechanism (all in ComfyUI core, nothing patched): the latent's `noise_mask`
# is a NestedTensor (video mask, audio mask); comfy.model_base.MiniMaxH3 turns
# it into per-row timesteps and re-injects the preserved rows from the input
# latent every step (scale_latent_inpaint).

import logging
import math

import torch

from ...utils.constants import CUSTOM_CATEGORY

_LOG = logging.getLogger("apnext.h3.masked_song")

FPS = 24.0
AUDIO_HZ = 40                                  # H3 audio latent ticks per second
TICK = 1.0 / AUDIO_HZ
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)              # H3 video VAE temporal groups (cyclic)

try:  # the authoritative constant when core is importable
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN as _CORE_FPT
    FRAME_PER_TOKEN = tuple(int(x) for x in _CORE_FPT)
except Exception:  # pragma: no cover - tests / standalone
    pass


# ----------------------------------------------------------------------------- grid helpers

def pixel_frames(latent_t):
    """Pixel frames covered by `latent_t` H3 video latent steps."""
    return sum(FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)] for k in range(int(latent_t)))


def video_steps_for_frames(frame_count):
    """Exact H3 video-latent step count for a valid pixel-frame run (5, 22, 39, ...)."""
    frame_count, total = int(frame_count), 0
    for steps in range(1, 100000):
        total += FRAME_PER_TOKEN[(steps - 1) % len(FRAME_PER_TOKEN)]
        if total == frame_count:
            return steps
        if total > frame_count:
            break
    raise ValueError(f"H3 Masked Song Latent: {frame_count} frames is not an exact H3 temporal run (5, 22, 39, ...)")


def largest_h3_run(n):
    """Largest exact H3 run (5 + 17k) that fits in `n` frames, 0 if none."""
    n = int(n)
    return 0 if n < 5 else 5 + ((n - 5) // 17) * 17


def snap_context_length(requested, available, target_frames):
    cap = min(int(requested), int(available), int(target_frames) - 1)
    run = largest_h3_run(cap)
    if run < 5:
        raise ValueError(
            "H3 Masked Song Latent: a visual prefix needs at least 5 source frames and a target longer than the prefix"
        )
    if run != int(requested):
        _LOG.warning(
            "H3 Masked Song Latent: context_length %d -> exact H3 prefix %d (valid runs are 5, 22, 39, 56, ...)",
            int(requested), run,
        )
    return run


def unbind_av(latent):
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(f"H3 Masked Song Latent: expected a MiniMax H3 AV latent, got {type(samples)!r}")
    if len(parts) < 2:
        raise ValueError("H3 Masked Song Latent: expected joint H3 video+audio latent streams")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(f"H3 Masked Song Latent: video latent must be [B,C,T,H,W], got {tuple(video.shape)}")
    if audio.ndim != 4:
        raise ValueError(f"H3 Masked Song Latent: audio latent must be [B,C,2,T], got {tuple(audio.shape)}")
    return video, audio


# ----------------------------------------------------------------------------- audio helpers

def stereo_first(audio, label="master_audio"):
    """AUDIO dict -> ([1, 2, L] float32 CPU, sample_rate)."""
    wave = audio["waveform"]
    if wave.ndim == 2:
        wave = wave.unsqueeze(0)
    if wave.ndim != 3:
        raise ValueError(f"H3 Masked Song Latent: {label} waveform must be [B,C,L], got {tuple(wave.shape)}")
    wave = wave[:1].detach().to("cpu").float()
    channels = int(wave.shape[1])
    if channels == 1:
        wave = wave.repeat(1, 2, 1)
    elif channels != 2:
        raise ValueError(
            f"H3 Masked Song Latent: {label} has {channels} channels - downmix to stereo before this node."
        )
    return wave, int(audio["sample_rate"])


def resample(wave, sr, target_sr):
    if int(sr) == int(target_sr):
        return wave
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError(
            f"H3 Masked Song Latent: master audio is {sr} Hz but the audio VAE wants {target_sr} Hz and torchaudio is unavailable"
        ) from exc
    return torchaudio.functional.resample(wave, int(sr), int(target_sr))


def window(wave, start, end):
    """wave[..., start:end] with silence wherever the song does not reach (before 0 or past its end)."""
    length = int(wave.shape[-1])
    lo, hi = max(0, start), min(length, end)
    piece = wave[..., lo:hi] if hi > lo else wave[..., :0]
    pad_l, pad_r = lo - start, end - hi
    if pad_l or pad_r:
        piece = torch.nn.functional.pad(piece, (max(0, pad_l), max(0, pad_r)))
    return piece


def ticks(seconds):
    return int(round(float(seconds) * AUDIO_HZ))


def voice_gate(voice, clip_start_seconds, n_ticks, hold_ticks, threshold=0.25):
    """
    Per-tick bool [n_ticks]: True where the vocal stem is sounding inside this clip,
    grown by `hold_ticks` on both sides. The envelope is Voice Over Music's
    sidechain detector (relative to the stem's loudest moment, -40 dB = silence),
    averaged onto the 40 Hz latent grid.
    """
    from .voice_mix import voice_envelope
    wave, sr = stereo_first(voice, "voice")
    env = voice_envelope(wave, sr)                                   # [N] 0..1, per sample
    start = int(round(float(clip_start_seconds) * sr))
    hop = sr / AUDIO_HZ
    level = torch.zeros(int(n_ticks))
    for k in range(int(n_ticks)):
        a, b = start + int(round(k * hop)), start + int(round((k + 1) * hop))
        a, b = max(0, a), min(env.numel(), b)
        if b > a:
            level[k] = env[a:b].mean()
    active = level > float(threshold)
    if hold_ticks > 0 and active.any():
        kernel = 2 * int(hold_ticks) + 1
        grown = torch.nn.functional.max_pool1d(active.float().view(1, 1, -1), kernel, stride=1, padding=int(hold_ticks))
        active = grown.view(-1)[: int(n_ticks)] > 0.5
    return active


# ----------------------------------------------------------------------------- the node

class H3MaskedSongLatent:
    """Write the exact master-song slice into the H3 audio latent and protect it from denoising."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Target AV latent from MiniMax H3 Reference to Video / Image to Video."}),
                "audio_vae": ("VAE", {"tooltip": "The MiniMax H3 audio VAE - encodes the song slice into the audio latent."}),
                "master_audio": ("AUDIO", {
                    "tooltip": "The whole song (or the conditioning mix from H3 Voice Over Music). The slice this clip "
                               "covers is written into the latent and protected from denoising; the rest is ignored."
                }),
                "clip_start_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 99999.0, "step": 0.001,
                    "tooltip": "Where this clip starts in the song (the writer's `clip_starts`). For a continuing clip "
                               "this already includes the pinned prefix (Chain Render does that).",
                }),
                "context_length": ("INT", {
                    "default": 39, "min": 0, "max": 9999,
                    "tooltip": "Visual prefix copied from the previous clip (source_latent / source_frames): exact H3 runs "
                               "5 / 22 / 39 / 56 ...; 0 = no prefix (independent clip).",
                }),
                "source_fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "FPS of source_frames (the decoded-frames continuation path).",
                }),
                "crop": (["disabled", "center"], {"default": "disabled"}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "The H3 video VAE - only for the source_frames path."}),
                "source_frames": ("IMAGE", {"tooltip": "Decoded frames of the previous clip; its tail is encoded as the prefix."}),
                "source_latent": ("LATENT", {
                    "tooltip": "The previous clip's sampled AV latent; its tail is copied straight into the new latent "
                               "(no decode / re-encode). Preferred for chaining.",
                }),
                "preroll_seconds": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.025,
                    "tooltip": "Song audio encoded BEFORE the clip start and then dropped, so the first tokens have a "
                               "past (the encoder's attention is causal). Measured: 0 s leaves token 0 ~57 % off and "
                               "the first 0.5 s wrong; 1 s brings the head to ~1 %; 2 s is the floor. 0 = the old "
                               "hard-cut behaviour, for A/B.",
                }),
                "lookahead_seconds": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.025,
                    "tooltip": "Song audio encoded AFTER the clip end and dropped, so the last tokens see what follows "
                               "(the encoder's convs look ahead ~200 ms). Silence once the song has ended.",
                }),
                "audio_denoise": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Noise-mask value of the audio rows. 0 = the song slice is fully frozen (default). A small "
                               "value (0.05-0.15) lets the model re-touch the audio at that noise level while the "
                               "vocal stays - an A/B knob; the output still gets the real song. With `voice` connected "
                               "this is the value while the voice is SINGING; the gaps get `gap_denoise`.",
                }),
                "voice": ("AUDIO", {
                    "tooltip": "The vocal stem (AudioSeparation's vocals) - the VOICE GATE. Wherever the voice is sounding "
                               "the audio rows are held at `audio_denoise` (0 = frozen, so the lips have the exact vocal); "
                               "between phrases they are held at `gap_denoise` instead, so the model has a little freedom "
                               "over the music bed where no lip-sync is at stake. Leave unconnected for one value everywhere.",
                }),
                "gap_denoise": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Noise-mask value of the audio rows BETWEEN sung phrases (voice connected). 0 = frozen there "
                               "too; 0.1-0.25 lets the model re-touch the music bed in the gaps.",
                }),
                "gate_hold_seconds": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.025,
                    "tooltip": "How far the frozen region extends before and after every sung stretch (whole 25 ms ticks) "
                               "- room for the mouth opening ahead of the word and closing after it.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "AUDIO")
    RETURN_NAMES = ("latent", "trim_frames", "clip_audio")
    FUNCTION = "prepare"
    CATEGORY = CUSTOM_CATEGORY
    DESCRIPTION = (
        "Write the exact master-song slice for this clip into the H3 audio latent and protect it from "
        "denoising, so the picture is generated against the real, frozen vocal. Pre-roll and lookahead "
        "encode the slice in context (the audio VAE's head/tail tokens are wrong without them). Optional "
        "visual prefix from the previous clip's latent or frames for chaining. Drop-in for "
        "MiniMaxH3SongMaskedAVContext."
    )

    def prepare(self, latent, audio_vae, master_audio, clip_start_seconds=0.0, context_length=39,
                source_fps=24.0, crop="disabled", vae=None, source_frames=None, source_latent=None,
                preroll_seconds=1.0, lookahead_seconds=0.2, audio_denoise=0.0,
                voice=None, gap_denoise=0.15, gate_hold_seconds=0.2):
        import comfy.nested_tensor

        target_video, target_audio = unbind_av(latent)
        if int(target_video.shape[0]) != 1 or int(target_audio.shape[0]) != 1:
            raise ValueError("H3 Masked Song Latent: batch size 1 only")
        target_frames = pixel_frames(int(target_video.shape[2]))
        audio_ticks = int(target_audio.shape[-1])
        nominal = int(round(target_frames / FPS * AUDIO_HZ))
        if audio_ticks != nominal:
            _LOG.info("H3 Masked Song Latent: target latent has %d audio ticks for %d frames (nominal %d) - using the latent's",
                      audio_ticks, target_frames, nominal)

        # ---- the song slice, on the audio VAE's clock ----------------------------
        vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
        hop = vae_sr // AUDIO_HZ
        if hop * AUDIO_HZ != vae_sr:
            raise ValueError(f"H3 Masked Song Latent: audio VAE rate {vae_sr} Hz is not a multiple of the {AUDIO_HZ} Hz latent grid")
        wave, sr = stereo_first(master_audio)
        wave = resample(wave, sr, vae_sr)
        song_samples = int(wave.shape[-1])

        clip_start_seconds = float(clip_start_seconds)
        if clip_start_seconds < 0:
            raise ValueError("H3 Masked Song Latent: clip_start_seconds must be >= 0")
        start = int(round(clip_start_seconds * vae_sr))
        picture_end = int(round((clip_start_seconds + target_frames / FPS) * vae_sr))
        if picture_end <= start:
            raise RuntimeError("H3 Masked Song Latent: the clip covers no song samples")
        if start >= song_samples:
            _LOG.warning("H3 Masked Song Latent: clip starts at %.2fs, after the song ends (%.2fs) - the slice is silence",
                         clip_start_seconds, song_samples / vae_sr)
        elif picture_end > song_samples:
            _LOG.warning("H3 Masked Song Latent: the clip runs %.2fs past the end of the song - padding silence",
                         (picture_end - song_samples) / vae_sr)
        clip_wave = window(wave, start, picture_end)

        # ---- encode in context: pre-roll + the latent grid + lookahead -----------
        pre = min(ticks(preroll_seconds), start // hop)          # only as much past as the song has
        look = ticks(lookahead_seconds)
        enc_start = start - pre * hop
        enc_end = start + (audio_ticks + look) * hop
        z = audio_vae.encode(window(wave, enc_start, enc_end).movedim(1, -1))
        if getattr(z, "ndim", 0) != 4:
            raise ValueError(f"H3 Masked Song Latent: audio VAE returned {tuple(getattr(z, 'shape', ()))}, expected [B,C,2,T]")
        got = int(z.shape[-1])
        if got < pre + audio_ticks:
            raise RuntimeError(
                f"H3 Masked Song Latent: the audio VAE produced {got} ticks for a {pre + audio_ticks + look}-tick window "
                f"(need {pre + audio_ticks}); is this the MiniMax H3 audio VAE?"
            )
        song_latent = z[:1, :, :, pre:pre + audio_ticks]

        out_video = target_video.clone()
        out_audio = target_audio.clone()
        out_audio.copy_(song_latent.to(device=out_audio.device, dtype=out_audio.dtype))

        # ---- the visual prefix (chain continuation) ------------------------------
        if source_latent is not None and source_frames is not None:
            raise ValueError("H3 Masked Song Latent: connect either source_latent or source_frames, not both")
        prefix_frames, prefix_steps = 0, 0
        if source_latent is not None:
            prefix_frames, prefix_steps = self._prefix_from_latent(out_video, source_latent, context_length, target_frames)
        elif source_frames is not None:
            prefix_frames, prefix_steps = self._prefix_from_frames(out_video, source_frames, vae, context_length,
                                                                   target_frames, source_fps, crop)

        # ---- the nested noise mask -----------------------------------------------
        video_mask = torch.ones((1, 1) + tuple(out_video.shape[2:]), dtype=torch.float32)
        if prefix_steps > 0:
            video_mask[:, :, :prefix_steps] = 0.0
        sung = float(max(0.0, min(1.0, audio_denoise)))
        audio_mask = torch.full((1, 1, int(out_audio.shape[2]), int(out_audio.shape[3])), sung, dtype=torch.float32)
        gate_note = f"audio {'frozen' if sung <= 0 else f'mask {sung:.2f}'}"
        if voice is not None:
            gap = float(max(0.0, min(1.0, gap_denoise)))
            active = voice_gate(voice, clip_start_seconds, audio_ticks, ticks(gate_hold_seconds))
            row = torch.where(active, torch.full_like(audio_mask[0, 0, 0], sung), torch.full_like(audio_mask[0, 0, 0], gap))
            audio_mask[0, 0] = row.unsqueeze(0).expand(int(out_audio.shape[2]), -1)
            pct = float(active.float().mean()) * 100.0
            gate_note = (f"voice gate: {pct:.0f}% of ticks sung ({'frozen' if sung <= 0 else f'mask {sung:.2f}'}), "
                         f"gaps at mask {gap:.2f}, hold {ticks(gate_hold_seconds) * TICK:.2f}s")
            if pct == 0.0:
                _LOG.info("H3 Masked Song Latent: no voice detected in this clip - every audio row sits at gap_denoise %.2f", gap)

        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
        clip_audio = {"waveform": clip_wave.contiguous(), "sample_rate": vae_sr}

        print(
            f"🎤 H3 Masked Song Latent | song {clip_start_seconds:.3f}s → {clip_start_seconds + target_frames / FPS:.3f}s "
            f"({target_frames} frames, {audio_ticks} ticks) encoded with {pre * TICK:.2f}s pre-roll + {look * TICK:.2f}s lookahead"
            f" | {gate_note}"
            + (f" | visual prefix {prefix_frames} frames ({prefix_steps} steps)" if prefix_steps else " | no visual prefix")
        )
        return (out, prefix_frames, clip_audio)

    # ------------------------------------------------------------------ prefix paths

    @staticmethod
    def _prefix_from_latent(out_video, source_latent, context_length, target_frames):
        source_video, _ = unbind_av(source_latent)
        if int(source_video.shape[0]) != 1:
            raise ValueError("H3 Masked Song Latent: source_latent batch size 1 only")
        if int(context_length) <= 0:
            raise ValueError("H3 Masked Song Latent: context_length must be > 0 when source_latent is connected")
        if tuple(source_video.shape[1:2]) + tuple(source_video.shape[3:]) != tuple(out_video.shape[1:2]) + tuple(out_video.shape[3:]):
            raise ValueError("H3 Masked Song Latent: source_latent and target latent must share channels and resolution")
        available = pixel_frames(int(source_video.shape[2]))
        n = snap_context_length(context_length, available, target_frames)
        steps = video_steps_for_frames(n)
        if steps >= int(out_video.shape[2]):
            raise ValueError("H3 Masked Song Latent: the visual prefix would fill the whole target latent")
        if steps > int(source_video.shape[2]):
            raise RuntimeError("H3 Masked Song Latent: source_latent is shorter than the requested prefix")
        tail_start = int(source_video.shape[2]) - steps
        if tail_start % len(FRAME_PER_TOKEN) != 0:
            raise ValueError(
                f"H3 Masked Song Latent: the source tail starts at H3 temporal phase {tail_start % len(FRAME_PER_TOKEN)}; "
                "use an H3-grid source length or the source_frames path"
            )
        out_video[:, :, :steps] = source_video[:, :, tail_start:].to(device=out_video.device, dtype=out_video.dtype)
        return n, steps

    @staticmethod
    def _prefix_from_frames(out_video, source_frames, vae, context_length, target_frames, source_fps, crop):
        import comfy.utils
        if vae is None:
            raise ValueError("H3 Masked Song Latent: vae is required when source_frames is connected")
        if getattr(source_frames, "ndim", 0) != 4 or int(source_frames.shape[0]) < 1:
            raise ValueError("H3 Masked Song Latent: source_frames must be IMAGE [N,H,W,C]")
        if int(context_length) <= 0:
            raise ValueError("H3 Masked Song Latent: context_length must be > 0 when source_frames is connected")
        # constant-frame-rate index map onto 24 fps
        count, fps = int(source_frames.shape[0]), float(source_fps)
        out_n = max(1, int(round(count * FPS / fps)))
        if out_n == count and abs(fps - FPS) < 1e-6:
            idx = torch.arange(count, dtype=torch.long)
        else:
            t = (torch.arange(out_n, dtype=torch.float64) + 0.5) / FPS
            idx = torch.round(t * fps - 0.5).to(torch.long).clamp_(0, count - 1)
        n = snap_context_length(context_length, int(idx.numel()), target_frames)
        tail = source_frames.index_select(0, idx[-n:].to(source_frames.device))
        width, height = int(out_video.shape[4]) * 16, int(out_video.shape[3]) * 16
        tail = comfy.utils.common_upscale(tail[..., :3].movedim(-1, 1), width, height, "lanczos", crop).movedim(1, -1)
        prefix = vae.encode(tail)
        if getattr(prefix, "ndim", 0) != 5:
            raise ValueError(f"H3 Masked Song Latent: video VAE returned {tuple(getattr(prefix, 'shape', ()))}, expected [B,C,T,H,W]")
        steps = int(prefix.shape[2])
        if pixel_frames(steps) != n:
            raise RuntimeError(f"H3 Masked Song Latent: {n} prefix frames encoded to {steps} steps covering "
                               f"{pixel_frames(steps)} frames - refusing a phase-shifted seam")
        if steps >= int(out_video.shape[2]):
            raise ValueError("H3 Masked Song Latent: the visual prefix would fill the whole target latent")
        out_video[:, :, :steps] = prefix[:1].to(device=out_video.device, dtype=out_video.dtype)
        return n, steps
