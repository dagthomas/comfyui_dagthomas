# APNext H3 Refine Encode
#
# Builds the CLEAN full-length AV latent for a face-refine v2v pass: the
# upscaled pass-1 frames are encoded with the H3 video VAE and paired with the
# pass-1 audio (copied verbatim from the pass-1 latent, or encoded from an
# AUDIO track). Feed the result to APNext H3 Mouth Guard, then sample with
# RandomNoise + BasicScheduler (denoise ~0.35-0.5) - the sampler adds the
# partial noise itself. Never pre-noise this latent (no DisableNoise flows):
# protected regions are restored from this latent as-is.

import logging

import torch

try:
    import comfy.model_management
    import comfy.nested_tensor
except ImportError:  # imported outside ComfyUI (widget checks) - node can't run there anyway
    pass

from ...utils.constants import CUSTOM_CATEGORY
from .mouth_guard import unbind_av

FPS = 24
AUDIO_LATENT_FPS = 40


def _snap_17k5(n):
    """Largest frame count on the 17k+5 grid that fits in n."""
    if n < 5:
        raise ValueError(f"H3 Refine Encode: needs at least 5 frames, got {n}")
    while n % 17 != 5:
        n -= 1
    return n


class H3RefineEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": (
                        "Pass-1 frames, upscaled to the refine resolution. Width and height "
                        "must be multiples of 32; the frame count is trimmed down to the "
                        "17k+5 grid if it is off it."
                    ),
                }),
                "vae": ("VAE", {"tooltip": "The MiniMax H3 video VAE."}),
            },
            "optional": {
                "source_latent": ("LATENT", {
                    "tooltip": (
                        "The pass-1 AV latent: its audio stream is copied verbatim (the exact "
                        "lip-sync source). Preferred over the audio input."
                    ),
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Fallback soundtrack, encoded with audio_vae when source_latent is not connected.",
                }),
                "audio_vae": ("VAE", {"tooltip": "The MiniMax H3 audio VAE (required with audio)."}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("latent", "frame_count", "width", "height")
    FUNCTION = "encode"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Encodes upscaled pass-1 frames (+ the pass-1 audio) into a clean MiniMax H3 AV "
        "latent for a mouth-guarded refine pass. frame_count/width/height mirror the "
        "encoded geometry - wire them into MiniMax H3 Reference to Video."
    )

    def _audio_latent(self, n, source_latent, audio, audio_vae):
        want_t = round(n / FPS * AUDIO_LATENT_FPS)
        if source_latent is not None:
            _, src_audio = unbind_av(source_latent["samples"])
            have_t = int(src_audio.shape[-1])
            if have_t < want_t:
                raise ValueError(
                    f"H3 Refine Encode: source_latent audio covers {have_t} latent ticks but "
                    f"{n} frames need {want_t} - the pass-1 latent is shorter than the frames."
                )
            if have_t > want_t:
                logging.warning(f"H3 Refine Encode: source_latent audio trimmed {have_t} -> {want_t} ticks.")
            return src_audio[:1, :, :, :want_t].float().cpu()
        if audio is not None:
            if audio_vae is None:
                raise ValueError("H3 Refine Encode: audio_vae is required when audio is connected")
            import torchaudio
            waveform = audio["waveform"]
            sr = int(audio["sample_rate"])
            vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
            if sr != vae_sr:
                waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
            # a hair beyond the clip length so the encode never comes up short
            need = int(round((n / FPS + 0.2) * vae_sr))
            z = audio_vae.encode(waveform[:1, :, :need].movedim(1, -1))  # [1,32,2,T]
            have_t = int(z.shape[-1])
            if have_t < want_t:
                logging.warning(
                    f"H3 Refine Encode: audio covers {have_t}/{want_t} latent ticks - padding the tail with silence."
                )
                z = torch.nn.functional.pad(z, (0, want_t - have_t))
            return z[:, :, :, :want_t].float().cpu()
        raise ValueError(
            "H3 Refine Encode: connect source_latent (preferred) or audio + audio_vae - "
            "the refine pass needs the pass-1 soundtrack to preserve lip-sync."
        )

    def encode(self, frames, vae, source_latent=None, audio=None, audio_vae=None):
        n_in = int(frames.shape[0])
        height, width = int(frames.shape[1]), int(frames.shape[2])
        if width % 32 or height % 32:
            raise ValueError(
                f"H3 Refine Encode: frame size {width}x{height} is not a multiple of 32 - "
                "scale with a /32-snapped target (H3 canvases are; a clean 2x upscale stays on the grid)."
            )
        n = _snap_17k5(n_in)
        if n != n_in:
            logging.warning(f"H3 Refine Encode: {n_in} frames trimmed to {n} (H3 17k+5 grid).")

        video = vae.encode(frames[:n, :, :, :3])
        if getattr(video, "ndim", 0) != 5 or int(video.shape[1]) != 24:
            raise ValueError(
                f"H3 Refine Encode: the video VAE returned {tuple(getattr(video, 'shape', ()))} - "
                "connect the MiniMax H3 video VAE."
            )
        expected_t = 2 if n <= 5 else ((n - 5) // 17) * 5 + 2
        if int(video.shape[2]) != expected_t:
            raise ValueError(
                f"H3 Refine Encode: {n} frames encoded to {int(video.shape[2])} latent frames, "
                f"expected {expected_t} - refusing a phase-shifted latent."
            )

        audio_lat = self._audio_latent(n, source_latent, audio, audio_vae)
        device = comfy.model_management.intermediate_device()
        latent = {"samples": comfy.nested_tensor.NestedTensor(
            (video[:1].float().to(device), audio_lat.to(device)))}
        logging.info(
            f"H3 Refine Encode: {n} frames @ {width}x{height} -> video latent "
            f"{tuple(video.shape[2:])}, audio latent {tuple(audio_lat.shape[2:])}"
        )
        return (latent, n, width, height)
