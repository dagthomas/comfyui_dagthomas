# APNext H3 Mouth Guard
#
# The face-refine second pass (upscale -> ref-conditioned H3 v2v re-render)
# restores likeness but re-denoises the mouth, stripping the lip-sync the
# first pass generated. This node protects the mouth: it takes a per-frame
# lips mask (e.g. Draw Face Mask (MediaPipe) with regions=custom, lips only),
# reduces it onto the H3 video latent grid, and installs it as the latent's
# noise mask (0 = preserve, 1 = generate) together with an audio mask that
# locks the pass-1 soundtrack. The sampler then keeps the mouth (and audio)
# from the input latent at every step while the rest of the frame is refined.
#
# IMPORTANT: the input latent must be CLEAN (the encoded pass-1 content, via
# APNext H3 Refine Encode) and the refine pass must add its own noise
# (RandomNoise + BasicScheduler with denoise < 1). Do NOT use a pre-noised /
# DisableNoise flow: the sampler's inpaint blend restores the *input* latent
# in protected regions, so a pre-noised input would preserve noise, not lips.
#
# Quality note: ComfyUI's sampler-level inpaint blending works with any model.
# When ComfyUI-H3-Motion-Context-MultiRef is installed, this node additionally
# enables its H3 mask engine, which pins protected tokens at the model's
# clean-conditioning timestep (the same mechanism keyframes use) for a much
# cleaner boundary. Without it the mouth is still preserved, just seamier.

import logging
import sys

import torch
import torch.nn.functional as F

try:
    import comfy.model_management
    import comfy.nested_tensor
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN
except ImportError:  # imported outside ComfyUI (widget checks) - node can't run there anyway
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)

from ...utils.constants import CUSTOM_CATEGORY


def pixel_frames_for(latent_t):
    """Pixel frames covered by latent_t H3 video latent frames (17k+5 grid)."""
    return sum(FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)] for k in range(int(latent_t)))


def unbind_av(samples):
    """(video, audio) streams of an H3 AV latent, validated."""
    if not getattr(samples, "is_nested", False):
        raise ValueError("H3 Mouth Guard expects a MiniMax H3 AV latent (video+audio NestedTensor)")
    tensors = samples.unbind()
    if len(tensors) != 2 or tensors[0].ndim != 5 or tensors[0].shape[1] != 24:
        raise ValueError("H3 Mouth Guard expects a MiniMax H3 AV latent (video [B,24,T,H/16,W/16])")
    return tensors[0], tensors[1]


def _grow_pixel_mask(masks, grow):
    """Dilate a [N,H,W] mask batch by `grow` pixels (chunked, GPU when possible)."""
    if grow <= 0:
        return masks
    kernel = 2 * int(grow) + 1
    device = comfy.model_management.get_torch_device()
    out = []
    for start in range(0, masks.shape[0], 16):
        chunk = masks[start:start + 16].unsqueeze(1)
        try:
            grown = F.max_pool2d(chunk.to(device), kernel, stride=1, padding=int(grow)).cpu()
        except Exception:
            grown = F.max_pool2d(chunk, kernel, stride=1, padding=int(grow))
        out.append(grown.squeeze(1))
    return torch.cat(out, dim=0)


def _gaussian_blur_2d(mask, sigma):
    """Separable gaussian on [T,1,H,W]; sigma in cells."""
    radius = max(1, int(3.0 * sigma + 0.5))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kx = kernel.view(1, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1)
    mask = F.conv2d(mask, kx, padding=(0, radius))
    mask = F.conv2d(mask, ky, padding=(radius, 0))
    return mask


def _install_mask_engine():
    """Best-effort enable of the MultiRef H3 mask engine (per-token cond
    timesteps for protected regions). Returns True when active."""
    for name, module in list(sys.modules.items()):
        if name.rsplit(".", 1)[-1] == "h3_compat" and hasattr(module, "ensure_existing_video_compat"):
            try:
                module.ensure_existing_video_compat()
                return True
            except Exception as exc:
                logging.warning(f"H3 Mouth Guard: H3 mask engine present but failed to enable: {exc}")
                return False
    logging.warning(
        "H3 Mouth Guard: ComfyUI-H3-Motion-Context-MultiRef not found - falling back to "
        "sampler-level inpaint blending only. The mouth and audio are still preserved in "
        "the output, but the model won't see them as clean context (seamier boundary)."
    )
    return False


class H3MouthGuard:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": (
                        "H3 AV latent whose samples already contain the CLEAN encoded pass-1 "
                        "content (from APNext H3 Refine Encode). Sample it with RandomNoise + "
                        "BasicScheduler denoise < 1 - never a pre-noised / DisableNoise flow."
                    ),
                }),
                "masks": ("MASK", {
                    "tooltip": (
                        "Per-frame lips mask batch (white = lips), one mask per pixel frame - "
                        "e.g. Detect Face Landmarks + Draw Face Mask (MediaPipe) with "
                        "regions=custom, lips only. Any resolution; it is reduced onto the "
                        "latent grid."
                    ),
                }),
                "grow_pixels": ("INT", {
                    "default": 8, "min": 0, "max": 256,
                    "tooltip": "Dilate the lips mask this many source pixels first (safety margin around the mouth).",
                }),
                "protect_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Lock the input latent's audio stream so the pass-1 soundtrack (and "
                        "its lip-sync timing) survives the refine. Off = the audio is fully "
                        "regenerated."
                    ),
                }),
            },
            "optional": {
                "grow_cells": ("INT", {
                    "default": 1, "min": 0, "max": 8,
                    "tooltip": "Extra dilation in latent cells (one cell covers 16 source pixels; the model reads 2x2 cells per token).",
                }),
                "feather_cells": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 4.0, "step": 0.1,
                    "tooltip": (
                        "Gaussian feather (in latent cells) on the protection edge. Fractional "
                        "mask values run at intermediate strength when the H3 mask engine "
                        "(ComfyUI-H3-Motion-Context-MultiRef) is installed."
                    ),
                }),
                "existing_mask": (["merge", "replace"], {
                    "default": "merge",
                    "tooltip": "merge = combine with a noise mask already on the latent (protected stays protected).",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "MASK", "STRING")
    RETURN_NAMES = ("latent", "mask_preview", "report")
    FUNCTION = "guard"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Protects the mouth (and the soundtrack) during an H3 face-refine v2v pass: reduces "
        "a per-frame lips mask onto the H3 latent grid and installs it as the latent's noise "
        "mask, so the sampler keeps the pass-1 lip-sync while the rest of the frame is "
        "re-rendered toward the reference likeness."
    )

    def guard(self, latent, masks, grow_pixels, protect_audio,
              grow_cells=1, feather_cells=0.5, existing_mask="merge"):
        video, audio = unbind_av(latent["samples"])
        t_lat, lat_h, lat_w = int(video.shape[2]), int(video.shape[3]), int(video.shape[4])
        expected_frames = pixel_frames_for(t_lat)

        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        masks = masks.float().cpu()
        n_masks = int(masks.shape[0])
        if n_masks != expected_frames:
            logging.warning(
                f"H3 Mouth Guard: {n_masks} mask frame(s) for a latent covering "
                f"{expected_frames} pixel frames - indices are clamped."
            )

        # pixel-space safety margin, then reduce onto the latent grid (max
        # semantics: any lip pixel in a cell/group protects it)
        grown = _grow_pixel_mask(masks, int(grow_pixels))
        spatial = F.adaptive_max_pool2d(grown.unsqueeze(1), (lat_h, lat_w)).squeeze(1)  # [N,h,w]

        # temporal reduce over the model's cyclic 1,4,4,4,4 frame groups
        protection = torch.zeros((t_lat, lat_h, lat_w), dtype=torch.float32)
        start = 0
        for k in range(t_lat):
            group = FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)]
            lo = min(start, n_masks - 1)
            hi = max(lo + 1, min(start + group, n_masks))
            protection[k] = spatial[lo:hi].amax(dim=0)
            start += group

        for _ in range(int(grow_cells)):
            protection = F.max_pool2d(protection.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
        if feather_cells > 0:
            # grow already ran, so the mouth core stays fully protected; the
            # blur only softens the boundary. The model amaxes 2x2 cells per
            # token itself, so no token snapping is needed here.
            protection = _gaussian_blur_2d(protection.unsqueeze(1), float(feather_cells)).squeeze(1)
        protection = protection.clamp(0.0, 1.0)

        video_mask = (1.0 - protection).view(1, 1, t_lat, lat_h, lat_w)
        audio_shape = (1, 1, int(audio.shape[2]), int(audio.shape[3]))
        audio_mask = torch.zeros(audio_shape) if protect_audio else torch.ones(audio_shape)

        prior = latent.get("noise_mask")
        if prior is not None and existing_mask == "merge":
            try:
                if getattr(prior, "is_nested", False):
                    prior_video, prior_audio = (t.float().cpu() for t in prior.unbind()[:2])
                    audio_mask = torch.minimum(audio_mask, prior_audio)
                else:
                    prior_video = prior.float().cpu()
                video_mask = torch.minimum(video_mask, prior_video)
            except Exception as exc:
                logging.warning(f"H3 Mouth Guard: could not merge the latent's existing noise mask ({exc}) - replacing it.")

        out = latent.copy()
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

        engine = _install_mask_engine()

        # preview: the protected region painted back onto pixel-resolution frames
        per_cell = 1.0 - video_mask[0, 0]  # [T,h,w]
        preview_lat = per_cell.repeat_interleave(
            torch.tensor([FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)] for k in range(t_lat)]), dim=0)
        preview = F.interpolate(preview_lat.unsqueeze(1),
                                size=(int(masks.shape[1]), int(masks.shape[2])),
                                mode="nearest-exact").squeeze(1)

        protected = 1.0 - video_mask[0, 0]
        per_frame = protected.flatten(1).sum(dim=1)
        report = (
            f"masks in: {n_masks} (expected {expected_frames}) | latent grid {t_lat}x{lat_h}x{lat_w} | "
            f"protected cells/frame min {int(per_frame.min())} max {int(per_frame.max())} "
            f"({100.0 * protected.mean():.2f}% of video latent) | "
            f"audio: {'locked' if protect_audio else 'regenerated'} | "
            f"H3 mask engine: {'active' if engine else 'FALLBACK (sampler blend only)'}"
        )
        logging.info(f"H3 Mouth Guard: {report}")
        return (out, preview, report)
