# APNext H3 Song Analysis
#
# The Music Video Writer's sound measurement as a standalone, visible node:
# drop it next to Load Audio and the song's tempo and character appear right
# on the node - BPM (autocorrelated onset envelope), aggression intensity
# 0-100 (transient spikes + sustained loudness + punch), the gentle..aggressive
# label and the dynamics spread in dB.
#
# The writers measure this themselves internally; this node is the readout -
# use it to sanity-check a song before a long run, to decide wildness /
# performance settings, or wire `profile` into any STRING input (a context
# socket, extra_instructions) and `audio` straight through to the writer.

from ...utils.constants import CUSTOM_CATEGORY
from .music_support import analyse, profile_line, song_profile


class H3SongAnalysis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "The song (Load Audio). Passed through unchanged on the audio output.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("audio", "profile", "bpm", "intensity", "label")
    OUTPUT_TOOLTIPS = (
        "The same audio, passed through - wire the writer from here.",
        "One line: BPM | label (intensity /100) | dynamics | onset spikes/s.",
        "Estimated tempo; 0.0 = no steady pulse detected.",
        "Aggression 0-100 from transient density, sustained loudness and punch.",
        "gentle / laid-back / mid-energy / driving / aggressive.",
    )
    OUTPUT_NODE = True
    FUNCTION = "measure"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Measures a song's BPM and hard/softness (onset spikes, loudness, dynamics) and "
        "shows the result on the node. The Music Video Writer measures this itself to "
        "steer its staging; this node is the visible readout, with the audio passed "
        "through so it drops between Load Audio and a writer."
    )

    def measure(self, audio):
        profile = song_profile(analyse(audio))
        line = profile_line(profile)
        print(f"🎚️ H3 Song Analysis | {line}")
        return {
            "ui": {"text": [line]},
            "result": (audio, line, float(profile["bpm"]), int(profile["intensity"]),
                       str(profile["label"])),
        }
