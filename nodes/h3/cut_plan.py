# APNext H3 Cut Plan - the scene list, decided by the music before any writing
#
# Give it the song and how long a scene may be (5-15 s by default) and it
# returns every scene the video needs: how many, and each one's start and
# end on the H3 frame grid, placed where the music wants a cut - a section
# start (chorus in, verse in), a drop landing at the top of the new scene, a
# stop, a downbeat, a lyric phrase start, or failing all that the strongest
# onset. It is the same cutter the Music Video Writer runs internally, with
# the song's measured form and the Sound Events list folded in, exposed as a
# node so the plan can be SEEN and hand-edited before a single scene is
# written. Wire `cut_plan` into a writer and the writer uses these scenes
# verbatim: its own segment_mode / max / min widgets are ignored (and greyed
# out on the canvas).
#
# The plan is plain text, one scene per line, so a hand edit is a normal
# edit: change an end time, delete a line, and the writer follows.

from ...utils.constants import CUSTOM_CATEGORY
import json

from .music_support import (
    BEAT_SNAP_MODES,
    CUT_PLACEMENTS,
    SEGMENT_MODES,
    analyse,
    energy_labels,
    fmt_time,
    format_cut_plan,
    parse_cut_plan,
    parse_lyrics,
    segment_by_lyrics,
    segment_song,
    snap_segments_to_beats,
)
from .song_structure import song_structure, summary_line
from .sound_events import parse_events, parse_rejected


def snap_to_onsets(times, audio, window=0.15):
    """A tapped time lands a little after the sound; pull each one onto the nearest onset."""
    if not times:
        return []
    feats = analyse(audio)
    onset, t_axis = feats["onset"], feats["times"]
    out = []
    for t in times:
        lo = int(max(0, (t - window) / feats["hop_seconds"]))
        hi = int(min(len(onset) - 1, (t + window) / feats["hop_seconds"]))
        if hi > lo:
            k = lo + int(onset[lo:hi + 1].argmax())
            out.append(float(t_axis[k]) if float(onset[k]) > 0.2 else float(t))
        else:
            out.append(float(t))
    return sorted(set(round(x, 3) for x in out))


class H3CutPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The song. Passed through so the node drops in between Load Audio and a writer."}),
                "segment_mode": (SEGMENT_MODES, {
                    "default": SEGMENT_MODES[0],
                    "tooltip": (
                        "Auto cuts on the music (section starts, drops, stops, downbeats, onsets). "
                        "Fixed makes every scene as long as allowed. Lyric lines cuts just before a "
                        "lyric phrase whenever one is in reach (needs timed lyrics below)."
                    ),
                }),
                "max_seconds": ("FLOAT", {
                    "default": 15.0, "min": 3.0, "max": 15.0, "step": 0.1,
                    "tooltip": "Longest scene allowed. H3 renders up to 15 s per clip.",
                }),
                "min_seconds": ("FLOAT", {
                    "default": 5.2, "min": 3.0, "max": 15.0, "step": 0.1,
                    "tooltip": "Shortest scene allowed. Below ~5 s a scene has no room for a move and its landing.",
                }),
            },
            "optional": {
                "manual_cuts": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Cuts you placed yourself - seconds or m:ss separated by spaces - e.g. from "
                        "\U0001F3AE Tap the cuts. Each one is snapped to the nearest onset within 150 ms and "
                        "becomes a scene boundary; stretches between your cuts longer than "
                        "max_seconds are cut on the music as usual."
                    ),
                }),
                "lyrics": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": (
                        "Optional timed lyrics (`[0:15] line`). Cuts then avoid landing just after a "
                        "line has started, and the Lyric lines mode cuts right before lines. Empty = "
                        "take the `lyrics_in` socket (e.g. H3 Lyrics Transcribe)."
                    ),
                }),
                "lyrics_in": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Timed lyrics from another node - the `lyrics` output of an APNext H3 Lyrics "
                        "Transcribe node. Used only while the `lyrics` box is empty; typed text wins."
                    ),
                }),
                "sound_events": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Optional: the `events` output of an APNext H3 Sound Events node. A drop then "
                        "lands at the top of a new scene and a stop closes one."
                    ),
                }),
                # appended last so saved workflows keep their widget positions
                "cut_placement": (CUT_PLACEMENTS, {
                    "default": CUT_PLACEMENTS[0],
                    "tooltip": (
                        "Which side of a beat the cut sits on. The frame grid moves a cut in 0.71 s "
                        "steps, so a cut is never exactly on a hit. Before: the cut lands just ahead "
                        "of the hit and the hit is the first thing in the new scene (the classic "
                        "music-video cut; drops, downbeats and onsets all open the new scene, tapped "
                        "cuts round down onto the grid). After: the hit is the last thing in the "
                        "outgoing scene and the new one opens on the release (tapped cuts round up). "
                        "Auto: the cutter's usual mix - onsets and downbeats from either side, drops "
                        "opening the new scene."
                    ),
                }),
                "beat_snap": (BEAT_SNAP_MODES, {
                    "default": BEAT_SNAP_MODES[0],
                    "tooltip": (
                        "After the cutter has placed every cut on the frame grid, move each one onto the "
                        "nearest beat (or downbeat), rounded to the frame - so every scene opens ON the "
                        "pulse to within half a frame (21 ms) instead of up to a third of a second off. "
                        "Scene lengths are then frame-exact, not on the 17-frame grid: render with H3 "
                        "Chain Render, which renders the next grid length up and trims the pad. The "
                        "single-clip path (Save Clip) would leave the pad in. The beats come from "
                        "`beat_grid` when wired, else from the song's measured structure."
                    ),
                }),
                "beat_grid": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Optional: Beat Grid's `grid_json` - its beats and downbeats (with your BPM override "
                        "and offset) are what beat_snap snaps to, instead of the measured structure's."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "INT", "STRING")
    RETURN_NAMES = ("audio", "cut_plan", "count", "summary")
    OUTPUT_NODE = True
    FUNCTION = "plan"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Decide the scenes before writing them: how many, and each one's start and end, "
        "placed where the music wants a cut within your min/max length. Wire `cut_plan` into "
        "a writer and it uses exactly these scenes - its own segment settings are ignored."
    )

    def plan(self, audio, segment_mode, max_seconds, min_seconds, lyrics="", sound_events="", manual_cuts="",
             cut_placement=None, beat_snap=None, beat_grid="", lyrics_in=""):
        placement = cut_placement or CUT_PLACEMENTS[0]
        if not (lyrics or "").strip() and (lyrics_in or "").strip():
            lyrics = lyrics_in          # the socket feeds the box while it is empty; typed text wins
        snap_mode = str(beat_snap or BEAT_SNAP_MODES[0])
        if min_seconds > max_seconds:
            min_seconds, max_seconds = max_seconds, min_seconds
        try:
            structure = song_structure(audio)
        except Exception as exc:
            print(f"⚠️ H3 Cut Plan: song structure not measured ({type(exc).__name__}: {exc})")
            structure = None
        events = parse_events(sound_events) if (sound_events or "").strip() else []
        parsed_lyrics = parse_lyrics(lyrics or "")
        timed = [t for t, _ in parsed_lyrics if t is not None]
        forced = snap_to_onsets(parse_rejected(manual_cuts), audio) if (manual_cuts or "").strip() else []

        if segment_mode == SEGMENT_MODES[2] and timed:
            segments, feats = segment_by_lyrics(audio, max_seconds, min_seconds, lyric_times=timed,
                                                structure=structure, events=events, forced=forced,
                                                placement=placement)
        else:
            segments, feats = segment_song(audio, max_seconds, min_seconds, segment_mode,
                                           lyric_times=timed, structure=structure, events=events, forced=forced,
                                           placement=placement)
        # beat-exact: move every grid-placed cut onto the nearest beat / downbeat
        beat_offsets, beat_unit = None, "beat"
        if not snap_mode.startswith("off"):
            beat_unit = "downbeat" if "downbeat" in snap_mode else "beat"
            grid = None
            if (beat_grid or "").strip():
                try:
                    grid = json.loads(beat_grid)
                except Exception as exc:
                    print(f"⚠️ H3 Cut Plan: beat_grid is not JSON ({exc}) - using the measured structure")
            if grid and grid.get("beats"):
                lines = grid.get("bars") if beat_unit == "downbeat" else grid.get("beats")
                origin = f"Beat Grid ({grid.get('bpm', 0):g} BPM)"
            elif structure and structure.get("beats"):
                lines = structure.get("downbeats") if beat_unit == "downbeat" else structure.get("beats")
                origin = f"measured structure ({structure.get('bpm', 0):g} BPM)"
            else:
                lines, origin = [], ""
            if lines:
                segments, beat_offsets = snap_segments_to_beats(segments, lines, feats["duration"], min_seconds, max_seconds)
                moved = [o for o in beat_offsets if o is not None]
                print(f"🎵 H3 Cut Plan | {len(moved)}/{max(0, len(segments) - 1)} cuts snapped onto the {beat_unit} "
                      f"from the {origin}; largest move {max((abs(o) for o in moved), default=0.0) * 1000:.0f} ms")
            else:
                print(f"⚠️ H3 Cut Plan: beat_snap is on but no {beat_unit}s were found (no steady pulse) - cuts stay on the grid")
        labels = energy_labels(feats, segments)
        text = format_cut_plan(segments, feats["duration"], min_seconds, max_seconds,
                               structure=structure, events=events, lyric_times=timed, labels=labels, forced=forced,
                               placement=placement, beat_offsets=beat_offsets, beat_unit=beat_unit)
        count = len(segments)
        form = summary_line(structure) if structure else "structure: not measured"
        summary = f"{count} scenes, {min_seconds:g}-{max_seconds:g} s, {fmt_time(feats['duration'])} | {form}"
        print(f"✂️ H3 Cut Plan | {summary}")
        # a round trip guards the format: what we print must parse back to the same cuts
        assert len(parse_cut_plan(text)) == count, "cut plan text did not round-trip"
        return {
            "ui": {"text": [text]},
            "result": (audio, text, count, summary),
        }
