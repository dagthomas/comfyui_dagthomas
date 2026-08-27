# APNext H3 Scenes Join
#
# The batch workflows render every scene from the H3 Scenes / Crossover Writer
# as its own clip (ComfyUI runs the sampler once per list element). This node
# sits right before Create Video and gathers all those per-scene IMAGE batches
# (and their AUDIO) back into ONE frame batch + ONE audio track, so a single
# Create Video → Save Video renders the whole story as one continuous file.

import torch

from ...utils.constants import CUSTOM_CATEGORY


def _first(v, default=None):
    if isinstance(v, (list, tuple)):
        return v[0] if v else default
    return v if v is not None else default


class H3ScenesJoin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Per-scene frame batches (connect the VAE Decode output that runs once per scene).",
                }),
                "crossfade_frames": ("INT", {
                    "default": 0, "min": 0, "max": 120,
                    "tooltip": "0 = hard cuts. N = overlap the last N frames of a scene with the first N of the next and blend them (shortens the total by N frames per cut).",
                }),
                "size_mismatch": (["resize to first scene", "error"], {
                    "default": "resize to first scene",
                    "tooltip": "What to do if scenes come out at different resolutions.",
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Per-scene audio (the VAE Decode Audio output). Joined in the same order; sample rates are unified to the first scene's.",
                }),
                "replace_audio": ("AUDIO", {
                    "tooltip": (
                        "Use this track as the whole video's soundtrack instead of the joined "
                        "per-scene audio - e.g. the original song from the H3 Music Video Writer. "
                        "Connect ONE audio here (not a per-scene list)."
                    ),
                }),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "INT")
    RETURN_NAMES = ("images", "audio", "frame_count", "scene_count")
    FUNCTION = "join"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Concatenates every rendered scene (frames + audio) from a batch run into one "
        "IMAGE batch and one AUDIO track, so one Create Video / Save Video outputs the "
        "whole video instead of a file per scene."
    )

    # ---- frames -------------------------------------------------------------

    @staticmethod
    def _match_size(img, h, w, mode):
        if img.shape[1] == h and img.shape[2] == w:
            return img
        if mode == "error":
            raise ValueError(
                f"H3 Scenes Join: scene resolution {img.shape[2]}x{img.shape[1]} does not match "
                f"first scene {w}x{h}. Set size_mismatch to 'resize to first scene' or fix the workflow."
            )
        x = img.permute(0, 3, 1, 2)
        x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).clamp(0, 1)

    @staticmethod
    def _crossfade(a, b, n):
        n = min(n, a.shape[0], b.shape[0])
        if n <= 0:
            return torch.cat([a, b], dim=0)
        t = torch.linspace(0, 1, n + 2, device=a.device, dtype=a.dtype)[1:-1].view(n, 1, 1, 1)
        blend = a[-n:] * (1 - t) + b[:n] * t
        return torch.cat([a[:-n], blend, b[n:]], dim=0)

    def _join_images(self, images, crossfade, mode):
        first = images[0]
        h, w = first.shape[1], first.shape[2]
        out = self._match_size(first, h, w, mode)
        for img in images[1:]:
            img = self._match_size(img.to(out.device, out.dtype), h, w, mode)
            out = self._crossfade(out, img, crossfade)
        return out

    # ---- audio --------------------------------------------------------------

    @staticmethod
    def _resample(wave, sr_from, sr_to):
        if sr_from == sr_to:
            return wave
        try:
            import torchaudio
            return torchaudio.functional.resample(wave, sr_from, sr_to)
        except Exception:
            # fallback: linear interpolation along time
            n = int(round(wave.shape[-1] * sr_to / sr_from))
            return torch.nn.functional.interpolate(wave, size=n, mode="linear", align_corners=False)

    def _join_audio(self, audios, frames_total, fps_hint=None):
        audios = [a for a in audios if a is not None and a.get("waveform") is not None]
        if not audios:
            return None
        sr = int(audios[0]["sample_rate"])
        first = audios[0]["waveform"]
        channels = first.shape[1]
        parts = []
        for a in audios:
            wav = a["waveform"].to(first.device, first.dtype)
            wav = self._resample(wav, int(a["sample_rate"]), sr)
            if wav.shape[1] != channels:
                wav = wav.mean(dim=1, keepdim=True).expand(-1, channels, -1)
            parts.append(wav.clone())
        # Declick every seam: a hard splice between two clips' audio lands on
        # arbitrary sample values and clicks. A 5 ms raised-cosine out / in at
        # each boundary is inaudible on its own and removes the click; lengths
        # are unchanged. (A chain render's `audio` output, wired into
        # `replace_audio`, is the better seam - crossfaded over real overlap.)
        n = max(2, int(round(0.005 * sr)))
        for k in range(len(parts)):
            m = min(n, parts[k].shape[-1])
            if m < 2:
                continue
            t = torch.linspace(0.0, 1.0, m, device=parts[k].device, dtype=parts[k].dtype)
            ramp = 0.5 - 0.5 * torch.cos(t * 3.141592653589793)
            if k > 0:
                parts[k][..., :m] = parts[k][..., :m] * ramp
            if k < len(parts) - 1:
                parts[k][..., -m:] = parts[k][..., -m:] * (1.0 - ramp)
        return {"waveform": torch.cat(parts, dim=2), "sample_rate": sr}

    # ---- node ---------------------------------------------------------------

    def join(self, images, crossfade_frames, size_mismatch, audio=None, replace_audio=None):
        images = [i for i in (images or []) if i is not None]
        if not images:
            raise ValueError("H3 Scenes Join: no images received.")
        crossfade = int(_first(crossfade_frames, 0))
        mode = _first(size_mismatch, "resize to first scene")

        frames = self._join_images(images, crossfade, mode)

        replacement = _first(replace_audio, None)
        if replacement is not None and replacement.get("waveform") is not None:
            wav = replacement["waveform"]
            return (frames, {"waveform": wav, "sample_rate": int(replacement["sample_rate"])}, int(frames.shape[0]), len(images))

        joined_audio = self._join_audio(list(audio or []), frames.shape[0])
        if joined_audio is None:
            # Create Video needs something; emit a silent stereo track.
            joined_audio = {
                "waveform": torch.zeros(1, 2, 1),
                "sample_rate": 44100,
            }
        elif crossfade > 0 and len(images) > 1:
            # Frames got shorter by crossfade*(n-1); trim audio proportionally so A/V stay aligned.
            total_raw = sum(i.shape[0] for i in images)
            keep = frames.shape[0] / max(total_raw, 1)
            wav = joined_audio["waveform"]
            joined_audio["waveform"] = wav[..., : int(wav.shape[-1] * keep)]

        return (frames, joined_audio, int(frames.shape[0]), len(images))
