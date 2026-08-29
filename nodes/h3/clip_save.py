# APNext H3 Save Clip
#
# Decode-and-save in ONE node, one clip at a time. The stock chain
# (VAE Decode -> Create Video -> Save Video) maps each node over the WHOLE
# scene list before the next node runs, so every decoded clip of an 18-scene
# music video sits in RAM at once (tens of GB at 1344x768) and nothing reaches
# disk until the very end - one OOM and the entire run's renders are lost.
# This node decodes one latent, muxes it with its audio slice, writes the mp4,
# and frees the frames before the next list item, so peak memory is a single
# clip and every finished clip is already on disk if a later one crashes.

import logging
import os
from fractions import Fraction

try:
    import folder_paths
    from comfy_api.latest import InputImpl, Types
except ImportError:  # imported outside ComfyUI (widget checks) - node can't run there anyway
    pass

from ...utils.constants import CUSTOM_CATEGORY

_FORMATS = ["mp4", "mkv", "webm"]
_CODEC_FOR = {"mp4": "h264", "mkv": "h264", "webm": "av1"}


def write_wav(path, wave_ct, sample_rate):
    """
    Write a [C, T] float tensor as 16-bit PCM WAV with the stdlib `wave`
    module. torchaudio.save is NOT used on purpose: on current torch it
    delegates to torchcodec, whose ffmpeg DLLs are not present in every
    ComfyUI install - the stdlib writer has no dependency to break.
    """
    import wave as wave_module

    import torch
    pcm = (wave_ct.detach().to("cpu").float().clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16)
    with wave_module.open(path, "wb") as fh:
        fh.setnchannels(int(pcm.shape[0]))
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm.transpose(0, 1).contiguous().numpy().tobytes())
    return path


def save_clip_wav(audio, video_path):
    """
    The clip's exact audio as a WAV next to its video file; returns the path
    or None. The mp4's AAC stream carries ~23 ms of encoder priming and tail
    padding that only players trim, so butt-joining the FILES in an editor
    leaves a 2-14 ms hole in the sound at every seam. The WAVs are the
    sample-exact slices: lay them under the mp4s and manual assembly is
    gapless - contiguous slices concatenate back into the original song.
    """
    if audio is None or audio.get("waveform") is None:
        return None
    try:
        path = os.path.splitext(video_path)[0] + ".wav"
        return write_wav(path, audio["waveform"][0], int(audio["sample_rate"]))
    except Exception as exc:
        print(f"⚠️ H3: could not write the clip's wav sidecar: {exc}")
        return None


def save_frames(images, audio, filename_prefix, fps, format):
    """Mux one clip's frames (+ optional AUDIO) and write it under the output
    directory; returns the file path. Shared by H3SaveClip and H3DeRopeSave."""
    width, height = int(images.shape[2]), int(images.shape[1])
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix, folder_paths.get_output_directory(), width, height
    )
    file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(format)}"
    path = os.path.join(full_output_folder, file)

    frame_count = int(images.shape[0])
    video = InputImpl.VideoFromComponents(
        Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps))
    )
    video.save_to(
        path,
        format=Types.VideoContainer(format),
        codec=Types.VideoCodec(_CODEC_FOR[format]),
    )
    del video
    logging.info(f"💾 H3 clip saved: {path} ({frame_count} frames @ {width}x{height})")
    return path


class H3SaveClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {
                    "default": "video/H3",
                    "tooltip": (
                        "Where the clips go under the output directory - wire a writer's "
                        "`project_name` output here so every run lands in its own folder."
                    ),
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0,
                    "tooltip": "H3 renders at 24 fps.",
                }),
                "format": (_FORMATS, {"tooltip": "mp4/mkv use H.264, webm uses AV1."}),
            },
            "optional": {
                "samples": ("LATENT", {
                    "tooltip": (
                        "The sampler's output latent (the H3 AV latent). With a list of "
                        "scenes wired in, the node runs once per scene and saves each "
                        "clip before decoding the next. Needs `vae`. Ignored when "
                        "`images` is wired."
                    ),
                }),
                "vae": ("VAE", {"tooltip": "The MiniMax H3 video VAE (required with samples)."}),
                "images": ("IMAGE", {
                    "tooltip": (
                        "Already-decoded frames (e.g. from H3 Exact Recover after a "
                        "de-rope pass). Wired, they are saved directly and "
                        "samples/vae are ignored."
                    ),
                }),
                "audio": ("AUDIO", {
                    "tooltip": (
                        "This clip's soundtrack - wire the writer's `audio_segments` list "
                        "so each scene is muxed with its own slice of the song. Empty = "
                        "silent clips."
                    ),
                }),
                # appended last so saved workflows keep their widget positions
                "wav_sidecar": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Also write the clip's audio as a WAV next to the video file (same "
                        "name). For MANUAL assembly in an editor: AAC in mp4 leaves a 2-14 ms "
                        "hole at every butt-join (encoder priming/padding), while the WAVs are "
                        "sample-exact - lay them under the clips and the joins are gapless. "
                        "H3 Stitch Clips does not need them."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Decodes one H3 latent and writes the clip straight to disk (replaces VAE "
        "Decode + Create Video + Save Video). Because each list item is decoded, "
        "saved and freed before the next, an 18-scene run needs the RAM of ONE clip "
        "instead of all of them - and clips already saved survive a mid-run OOM."
    )

    def save(self, filename_prefix, fps, format, samples=None, vae=None, audio=None, images=None,
             wav_sidecar=True):
        if images is None:
            if samples is None or vae is None:
                raise ValueError(
                    "H3 Save Clip: wire either samples + vae (decode here) or "
                    "images (already decoded)."
                )
            latent = samples["samples"]
            if latent.is_nested:  # H3 AV latent: video stream first, same as core VAE Decode
                latent = latent.unbind()[0]
            images = vae.decode(latent)
            if len(images.shape) == 5:  # combine video batches, same as core VAE Decode
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        path = save_frames(images, audio, filename_prefix, fps, format)
        if wav_sidecar and audio is not None:
            save_clip_wav(audio, path)
        del images  # free this clip's frames before the next list item decodes
        return (path,)
