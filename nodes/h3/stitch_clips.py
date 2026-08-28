# APNext H3 Stitch Clips - one gapless file from the clips already on disk.
#
# H3 Sample + Save (and Save Clip) write every scene as its own mp4 as soon as
# it is rendered, which is what keeps a 20-scene run at the memory of one clip.
# Putting those files back together is where the seams come from: every AAC
# stream carries ~23 ms of encoder priming and tail padding that a player only
# trims at the START and END of a file, so a stream-copy concatenation (ffmpeg
# `-c copy`, most editors' "append") leaves a 2-14 ms hole at every join. The
# clips are sample-exact; the container is not.
#
# This node stitches the saved files after the last scene: the H.264 packets are
# copied as they are (no re-encode, no quality loss, no frames in RAM) with
# their timestamps shifted onto one timeline, and the audio is DECODED, cut to
# exactly each clip's frame count, concatenated and encoded ONCE - or replaced
# by the original song, muxed once over the whole video. Either way there is
# nothing at the joins for a player to trim.
#
# Wire H3 Sample + Save's `file_path` (a list - one per scene) into `file_paths`
# and, for the cleanest sound, the Load Audio into `song`.

import os
import re
from fractions import Fraction

import torch

from ...utils.constants import CUSTOM_CATEGORY

_CLIP_NAME_RE = re.compile(r"^(?P<stem>.*?)_(?P<n>\d{5})_\.(?P<ext>\w+)$")


def _as_list(v):
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _first(v, default=None):
    if isinstance(v, (list, tuple)):
        return v[0] if v else default
    return v if v is not None else default


def stitched_name(paths, suffix="_full"):
    """`<folder>/<stem><suffix>.mp4` next to the clips: the stem is the clips' common
    name without the _00001_ counter (`Project-abcd_00001_.mp4` -> `Project-abcd_full.mp4`)."""
    first = paths[0]
    folder, name = os.path.split(first)
    m = _CLIP_NAME_RE.match(name)
    stem = m.group("stem") if m else os.path.splitext(name)[0]
    return os.path.join(folder, f"{stem}{suffix}.mp4")


def _song_stereo(song):
    wave = song["waveform"]
    if wave.dim() == 2:
        wave = wave.unsqueeze(0)
    wave = wave[0].detach().to("cpu").float()          # [C, N]
    if wave.shape[0] == 1:
        wave = wave.repeat(2, 1)
    elif wave.shape[0] > 2:
        wave = wave[:2]
    return wave, int(song["sample_rate"])


def stitch_files(paths, out_path, song=None, audio_bitrate=192_000, sync_offset_ms=0.0, log=print):
    """
    Concatenate H3 clips into one mp4: video packets copied with shifted
    timestamps, audio decoded + trimmed to each clip's exact length (or the
    `song` AUDIO dict) and encoded once. `sync_offset_ms` slides the whole
    soundtrack against the picture: negative = sound earlier (use it when the
    lips move before the word arrives), positive = sound later. Returns
    (out_path, frames, seconds).
    """
    import av
    import numpy as np

    paths = [p for p in paths if p and os.path.isfile(p)]
    if not paths:
        raise ValueError("H3 Stitch Clips: no clip files to stitch - wire H3 Sample + Save's `file_path` list.")

    probe = av.open(paths[0])
    in_v = probe.streams.video[0]
    first_codec, first_size = in_v.codec_context.name, (in_v.width, in_v.height)
    fps = Fraction(in_v.average_rate or in_v.guessed_rate or Fraction(24, 1))
    src_audio = probe.streams.audio[0] if probe.streams.audio else None
    if song is not None:
        song_wave, sr = _song_stereo(song)
        layout = "stereo"
    elif src_audio is not None:
        song_wave, sr = None, int(src_audio.rate)
        layout = "stereo" if src_audio.layout.nb_channels >= 2 else "mono"
    else:
        song_wave, sr, layout = None, None, None

    out = av.open(out_path, "w")
    out_v = out.add_stream_from_template(in_v)
    out_v.time_base = in_v.time_base
    out_a = None
    if sr:
        out_a = out.add_stream("aac", rate=sr, layout=layout)
        out_a.bit_rate = int(audio_bitrate)
        out_a.time_base = Fraction(1, sr)
    probe.close()

    channels = 2 if layout == "stereo" else 1
    offset_frames = 0                 # video frames already written
    audio_written = 0                 # audio samples already encoded
    total_frames = 0
    for path in paths:                # the whole timeline, for the last clip's trim
        c = av.open(path)
        total_frames += sum(1 for pk in c.demux(c.streams.video[0]) if pk.dts is not None or pk.pts is not None)
        c.close()
    total_samples = int(round(Fraction(total_frames) / fps * sr)) if sr else 0
    shift = int(round(float(sync_offset_ms) / 1000.0 * sr)) if sr else 0   # + = sound later

    def encode_audio(block):          # block: float32 [channels, n]
        nonlocal audio_written
        if block.shape[1] == 0:
            return
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(block, dtype=np.float32), format="fltp", layout=layout)
        frame.sample_rate = sr
        frame.time_base = Fraction(1, sr)
        frame.pts = audio_written
        audio_written += block.shape[1]
        for pkt in out_a.encode(frame):
            out.mux(pkt)

    for path in paths:
        c = av.open(path)
        v = c.streams.video[0]
        if (v.width, v.height) != first_size or v.codec_context.name != first_codec:
            c.close(); out.close()
            raise ValueError(f"H3 Stitch Clips: {os.path.basename(path)} is {v.width}x{v.height} {v.codec_context.name}; "
                             f"the first clip is {first_size[0]}x{first_size[1]} {first_codec} - clips must match to be copied.")
        a = c.streams.audio[0] if (song_wave is None and c.streams.audio) else None
        offset = Fraction(offset_frames) / fps          # seconds
        n_frames = 0
        decoded = []
        streams = [v] + ([a] if a is not None else [])
        for pkt in c.demux(*streams):
            if pkt.dts is None and pkt.pts is None:
                continue
            if pkt.stream.type == "video":
                tb = pkt.time_base or v.time_base
                ts_shift = int(round(offset / tb))
                if pkt.pts is not None:
                    pkt.pts += ts_shift
                if pkt.dts is not None:
                    pkt.dts += ts_shift
                pkt.stream = out_v
                out.mux(pkt)
                n_frames += 1
            else:
                for fr in pkt.decode():
                    arr = fr.to_ndarray()
                    if fr.format.is_planar:
                        decoded.append(arr.astype(np.float32))
                    else:                                  # packed [1, n*ch]
                        decoded.append(arr.reshape(-1, fr.layout.nb_channels).T.astype(np.float32))
        if a is not None:
            for fr in a.decode():                          # flush the decoder
                arr = fr.to_ndarray()
                decoded.append(arr.astype(np.float32) if fr.format.is_planar else arr.reshape(-1, fr.layout.nb_channels).T.astype(np.float32))
        c.close()

        want = int(round(Fraction(n_frames) / fps * sr)) if sr else 0
        if out_a is not None:
            if song_wave is not None:
                start = int(round(Fraction(offset_frames) / fps * sr))
                if shift < 0 and offset_frames == 0:
                    # sound earlier: the first clip's window starts |shift| samples into the song;
                    # the head-drop below then removes the silence put in front of it
                    block = np.concatenate([np.zeros((channels, -shift), dtype=np.float32),
                                            song_wave[:, -shift:-shift + want].numpy()], axis=1)
                else:
                    block = song_wave[:, start:start + want].numpy()
            else:
                block = np.concatenate(decoded, axis=1) if decoded else np.zeros((channels, 0), dtype=np.float32)
                if block.shape[0] == 1 and channels == 2:
                    block = np.repeat(block, 2, axis=0)
                block = block[:channels, :want]
            if block.shape[1] < want:
                block = np.concatenate([block, np.zeros((channels, want - block.shape[1]), dtype=np.float32)], axis=1)
            if offset_frames == 0 and shift:
                # slide the whole soundtrack: later = silence first, earlier = drop the head
                block = (np.concatenate([np.zeros((channels, shift), dtype=np.float32), block], axis=1) if shift > 0
                         else block[:, -shift:])
            if offset_frames + n_frames >= total_frames:
                # the last clip: land the soundtrack exactly on the video's length
                need = max(0, total_samples - audio_written)
                block = (block[:, :need] if block.shape[1] >= need
                         else np.concatenate([block, np.zeros((channels, need - block.shape[1]), dtype=np.float32)], axis=1))
            encode_audio(block)
        log(f"🧵 H3 Stitch Clips | {os.path.basename(path)} | {n_frames} frames @ {float(offset):.3f}s"
            + (f" | audio {want} samples" if sr else ""))
        offset_frames += n_frames

    if out_a is not None:
        for pkt in out_a.encode(None):
            out.mux(pkt)
    out.close()
    seconds = float(Fraction(offset_frames) / fps)
    return out_path, offset_frames, seconds


class H3StitchClips:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_paths": ("STRING", {
                    "forceInput": True,
                    "tooltip": "The saved clips in order - H3 Sample + Save's (or Save Clip's) `file_path` output, one per scene.",
                }),
                "suffix": ("STRING", {
                    "default": "_full",
                    "tooltip": "Appended to the clips' common name: `Project-abcd_00001_.mp4` ... -> `Project-abcd_full.mp4`, in the same folder.",
                }),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Off = pass the clips through untouched, write nothing."}),
                "sync_offset_ms": ("FLOAT", {
                    "default": 0.0, "min": -500.0, "max": 500.0, "step": 1.0,
                    "tooltip": "Slides the whole soundtrack against the picture. NEGATIVE = the sound comes earlier - use it "
                               "when the lips move before you hear the word (try -40 to -80). Positive = later. One frame is "
                               "41.7 ms; audio that leads the picture is noticed from ~45 ms, audio that lags from ~100 ms.",
                }),
            },
            "optional": {
                "song": ("AUDIO", {
                    "tooltip": "The original song (Load Audio). Muxed once over the whole video from 0:00, replacing the clips' "
                               "own audio - the cleanest sound. Unconnected: the clips' audio is decoded, cut to each clip's exact "
                               "frame count and encoded once, which also has no gaps.",
                }),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "report")
    OUTPUT_NODE = True
    FUNCTION = "stitch"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "After the last scene, joins the clips already on disk into one mp4 with nothing at the seams: video packets "
        "copied (no re-encode, no frames in RAM), audio decoded and encoded once - or the original song muxed once. "
        "A plain concatenation of the per-clip files leaves a 2-14 ms AAC priming gap at every join; this does not."
    )

    def stitch(self, file_paths, suffix, enabled, sync_offset_ms=0.0, song=None):
        paths = [str(p) for p in _as_list(file_paths) if p]
        suffix = str(_first(suffix, "_full") or "_full")
        song = _first(song, None)
        sync_offset_ms = float(_first(sync_offset_ms, 0.0) or 0.0)
        if not bool(_first(enabled, True)):
            return (paths[-1] if paths else "", "H3 Stitch Clips: disabled")
        if not paths:
            raise ValueError("H3 Stitch Clips: no clip files - wire H3 Sample + Save's `file_path` into `file_paths`.")
        out_path = stitched_name(paths, suffix)
        lines = []
        stitch_files(paths, out_path, song=song, sync_offset_ms=sync_offset_ms, log=lambda s: (print(s), lines.append(s)))
        report = (f"STITCH | {len(paths)} clip(s) → {os.path.basename(out_path)} | "
                  f"{'original song muxed once' if song is not None else 'clip audio decoded + encoded once'}"
                  + (f" | sound {'earlier' if sync_offset_ms < 0 else 'later'} by {abs(sync_offset_ms):.0f} ms" if sync_offset_ms else "")
                  + "\n" + "\n".join(lines))
        print(f"✅ H3 Stitch Clips | {os.path.basename(out_path)} written from {len(paths)} clip(s)")
        return (out_path, report)
