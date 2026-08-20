# APNext H3 scenes on disk
#
# save_scene_bundle(): the writers call this (toggle `save_scenes`) after a
# successful generation - everything needed to re-render lands as one JSON in
# `output/apnext_scenes/` (scenes, synopsis, segments, durations, frame
# lengths, clip starts, cast, tables). LLM output is expensive; the bundle is
# the insurance.
#
# H3ScenesLoad: the file picker - re-render a saved run without any LLM call.
# Its outputs mirror the Music Video Writer's core outputs (same names, same
# order, lists where the writer has lists), so it drops into the same graph:
# scenes → review/render, lengths → length, clip_starts → masked-audio
# context. Connect the same song to `audio` and the per-clip `audio_segments`
# (ref_audio_1) are re-sliced from the saved segment times.

import json
import os
import re
import time

from ...utils.constants import CUSTOM_CATEGORY
from .music_support import slice_audio

_KIND = "apnext_h3_scenes"
_NO_FILES = "(no saved scenes found)"


def scenes_dir():
    import folder_paths
    d = os.path.join(folder_paths.get_output_directory(), "apnext_scenes")
    os.makedirs(d, exist_ok=True)
    return d


def _slug(text, fallback):
    m = re.search(r"^\s*Title:\s*(.+)$", text or "", re.MULTILINE)
    s = re.sub(r"[^a-z0-9]+", "-", (m.group(1) if m else fallback).lower()).strip("-")
    return (s or fallback)[:48]


def save_scene_bundle(source, synopsis, scenes, segments, durations, lengths,
                      clip_starts, cast, song_seconds, scenes_text, segment_table, info):
    """Write one self-contained JSON bundle; returns the path (or None on failure)."""
    try:
        bundle = {
            "kind": _KIND,
            "version": 1,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "synopsis": synopsis or "",
            "scenes": [str(s) for s in scenes],
            "segments": [[float(a), float(b)] for a, b in (segments or [])],
            "durations": [float(d) for d in (durations or [])],
            "lengths": [int(f) for f in (lengths or [])],
            "clip_starts": [float(c) for c in (clip_starts or [])],
            "cast": cast or "",
            "scene_count": len(scenes),
            "song_seconds": float(song_seconds or 0.0),
            "scenes_text": scenes_text or "",
            "segment_table": segment_table or "",
            "info": info or "",
        }
        name = f"{time.strftime('%Y%m%d_%H%M%S')}_{_slug(synopsis, source.lower())}.json"
        path = os.path.join(scenes_dir(), name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=1, ensure_ascii=False)
        print(f"💾 H3 scenes saved: {path}")
        return path
    except Exception as exc:  # never fail the run over a backup
        print(f"⚠️ H3 scenes save failed: {exc}")
        return None


def _list_saved():
    try:
        d = scenes_dir()
        files = [f for f in os.listdir(d) if f.lower().endswith(".json")]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
        return files
    except Exception:
        return []


class H3ScenesLoad:
    @classmethod
    def INPUT_TYPES(cls):
        files = _list_saved()
        return {
            "required": {
                "file": (files or [_NO_FILES], {
                    "tooltip": (
                        "A scenes bundle saved by a writer's `save_scenes` toggle "
                        "(output/apnext_scenes/), newest first. Refresh the browser to "
                        "re-scan the folder."
                    ),
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": (
                        "The same song the scenes were written for. Connect it to rebuild "
                        "the per-clip `audio_segments` (ref_audio_1) from the saved segment "
                        "times. Not needed for the masked-audio workflow (clip_starts + the "
                        "master song cover it)."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "AUDIO", "STRING", "STRING", "STRING",
                    "STRING", "INT", "FLOAT", "STRING", "FLOAT")
    RETURN_NAMES = ("scenes", "durations", "lengths", "audio_segments", "segment_table",
                    "scenes_text", "synopsis", "cast", "scene_count", "song_seconds",
                    "info", "clip_starts")
    OUTPUT_IS_LIST = (True, True, True, True, False, False, False, False, False, False, False, True)
    FUNCTION = "load"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Load a saved scenes bundle from output/apnext_scenes/ and re-render it without "
        "any LLM call. Outputs mirror the Music Video Writer's core outputs, so it drops "
        "into the same graph; connect the song to `audio` to rebuild the per-clip audio "
        "segments for ref_audio workflows."
    )

    @classmethod
    def IS_CHANGED(cls, file="", **kwargs):
        try:
            return os.path.getmtime(os.path.join(scenes_dir(), file))
        except Exception:
            return float("nan")

    def load(self, file, audio=None):
        if not file or file == _NO_FILES:
            raise ValueError(
                "No saved scenes yet. Run a writer with `save_scenes` on first - bundles "
                "land in output/apnext_scenes/."
            )
        path = os.path.join(scenes_dir(), os.path.basename(file))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("kind") != _KIND:
            raise ValueError(f"{file} is not an APNext H3 scenes bundle.")

        scenes = [str(s) for s in data.get("scenes", [])]
        if not scenes:
            raise ValueError(f"{file} contains no scenes.")
        durations = [float(d) for d in data.get("durations", [])] or [10.0] * len(scenes)
        lengths = [int(x) for x in data.get("lengths", [])] or [round(d * 24) for d in durations]
        clip_starts = [float(c) for c in data.get("clip_starts", [])]
        segments = [tuple(s) for s in data.get("segments", [])]
        if not segments and clip_starts:
            segments = [(c, c + d) for c, d in zip(clip_starts, durations)]
        if not clip_starts and segments:
            clip_starts = [s for s, _ in segments]

        audio_segments = []
        info = f"loaded {file} | {len(scenes)} scene(s) | saved {data.get('created', '?')}"
        if audio is not None and segments:
            try:
                wave = audio.get("waveform")
                have = wave.shape[-1] / int(audio.get("sample_rate", 1)) if wave is not None else 0.0
                want = float(data.get("song_seconds") or 0.0)
                if want and abs(have - want) > 0.5:
                    print(
                        f"⚠️ H3 Scenes Load: the connected audio is {have:.1f}s but the scenes "
                        f"were written for a {want:.1f}s song - the slices may not line up."
                    )
                audio_segments = [
                    slice_audio(audio, s, e, fr) for (s, e), fr in zip(segments, lengths)
                ]
            except Exception as exc:
                print(f"⚠️ H3 Scenes Load: could not slice audio: {exc}")

        print(f"📂 H3 Scenes Load | {info}")
        return (
            scenes, durations, lengths, audio_segments,
            data.get("segment_table", ""), data.get("scenes_text", ""),
            data.get("synopsis", ""), data.get("cast", ""), len(scenes),
            float(data.get("song_seconds") or 0.0), info, clip_starts,
        )
