# APNext H3 Music Video (Minimal)
#
# The whole music-video pipeline behind four knobs: drop in the song, paste the
# lyrics, pick a cinematic look, set three sliders - go. Everything else
# (concept, cast, segmentation, wardrobe and location locks, chunking, saving)
# is the full H3 Music Video Writer running underneath with opinionated
# defaults: the model invents the concept and the performer, the song is cut on
# the music, the imagery is staged from the lyric lines, and reference images
# (the performer) pass straight through to the video node. Writes with Claude
# Code, the Codex CLI or anything on the `llm` socket, exactly like the full
# writer.

from ...utils.constants import CUSTOM_CATEGORY
from .claude_code_support import claude_code_inputs, local_llm_inputs, project_name_input
from .claude_code_music_video_writer import (
    AUDIO_MODES,
    H3ClaudeCodeMusicVideoWriter,
    PERFORMANCE_MODES,
    PROMPT_MODES,
)
from .music_support import SEGMENT_MODES
from .common import (
    AUTO,
    DIALOGUE_LANGUAGES,
    VISUAL_STYLES,
    reference_image_inputs,
    reference_image_outputs,
)

# Slider -> performance_mode: story visuals at the bottom, the singer lip-syncing
# on camera at the top, a blend in the middle band.
_PERFORMANCE_SLIDER_TOOLTIP = (
    "How much the singer is on camera. 0-33 = Narrative (story visuals, nobody "
    "sings on camera), 34-66 = Mixed (performance and story alternate), 67-100 = "
    "Performance (the singer lip-syncs the lyrics on camera)."
)

_PACE_TOOLTIP = (
    "How fast the video cuts. 0 = long, slow pieces (up to ~15 s per clip), 100 = "
    "quick cuts (pieces down to ~6 s). The song is still cut ON the music inside "
    "that range."
)

_TIMEOUT_SECONDS = 1800


def _performance_mode(slider):
    slider = max(0, min(100, int(slider)))
    if slider <= 33:
        return PERFORMANCE_MODES[1]  # Narrative
    if slider <= 66:
        return PERFORMANCE_MODES[2]  # Mixed
    return PERFORMANCE_MODES[0]      # Performance


def _max_segment_seconds(pace):
    pace = max(0, min(100, int(pace)))
    return round(15.0 - 9.0 * pace / 100.0, 1)  # 15.0 .. 6.0


class H3MusicVideoMinimal:
    @classmethod
    def INPUT_TYPES(cls):
        cc = claude_code_inputs()
        required = {
            "audio": ("AUDIO", {
                "tooltip": "The song. It is cut into pieces on the music and every piece becomes one clip.",
            }),
            "lyrics": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": (
                    "Lyrics, one line per line. Timestamps make the sync exact: `[0:15] "
                    "line` (or LRC `[00:15.20] line`); section tags like [Chorus] are "
                    "kept; untimed lines are spread evenly. Empty = instrumental video. "
                    "The imagery of every scene is staged from its lyric lines."
                ),
            }),
            "visual_style": (VISUAL_STYLES, {
                "default": "Live-action, 35mm cinematic film aesthetic",
                "tooltip": (
                    "The look of the whole video - the curated cinematic looks (35mm, Wes "
                    "Anderson, neon noir, ...) each fix style, camera, lenses and colour. "
                    "Auto lets the model pick one to fit the song."
                ),
            }),
            "performance": ("INT", {
                "default": 80, "min": 0, "max": 100, "step": 1,
                "tooltip": _PERFORMANCE_SLIDER_TOOLTIP,
            }),
            "pace": ("INT", {
                "default": 30, "min": 0, "max": 100, "step": 1,
                "tooltip": _PACE_TOOLTIP,
            }),
            "wildness": ("INT", {
                "default": 45, "min": 0, "max": 100, "step": 1,
                "tooltip": "0 = grounded performance video, 100 = fully surreal. Above 40 seeds surreal events.",
            }),
            "model": cc["model"],
            "seed": ("INT", {
                "default": -1, "min": -1, "max": 0xffffffffffffffff,
                "tooltip": "Seeds the surreal picks and controls caching. -1 re-runs every queue.",
            }),
            "prompt_mode": (PROMPT_MODES, {
                "default": PROMPT_MODES[1],  # Ref2VA - the pre-switch behaviour
                "tooltip": (
                    "REF writes against the reference-image guide (<Picture N> binds the "
                    "performer photo); FL writes against the base guide, everything from "
                    "scratch in words. Auto: REF when a picture is connected, FL otherwise."
                ),
            }),
        }
        optional = {}
        optional.update(local_llm_inputs())
        optional.update(reference_image_inputs())
        # appended LAST so saved workflows keep their widget positions
        optional.update(project_name_input())
        return {"required": required, "optional": optional}

    _IMAGE_OUTPUT_TYPES, _IMAGE_OUTPUT_NAMES = reference_image_outputs()
    RETURN_TYPES = (
        ("STRING", "FLOAT", "INT", "AUDIO", "STRING", "STRING", "STRING")
        + _IMAGE_OUTPUT_TYPES
        + ("FLOAT", "STRING")
    )
    RETURN_NAMES = (
        "scenes",
        "durations",
        "lengths",
        "audio_segments",
        "scenes_text",
        "session_id",
        "info",
    ) + _IMAGE_OUTPUT_NAMES + ("clip_starts", "project_name")
    OUTPUT_IS_LIST = (True, True, True, True, False, False, False) + (False,) * len(_IMAGE_OUTPUT_NAMES) + (True, False)
    FUNCTION = "write_video"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "The one-box music video: song + lyrics + a cinematic look + three sliders "
        "(performance, pace, wildness) - the model invents the concept and the performer "
        "and writes the whole video. Attach reference images to fix the performer's face. "
        "The full H3 Music Video Writer runs underneath with sensible defaults; use that "
        "node when you need cast, locks, briefs or the masked-audio path."
    )

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        return float("nan") if seed == -1 else seed

    @classmethod
    def VALIDATE_INPUTS(cls, prompt_mode=None):
        # Workflows saved before this widget existed restore it as '' - accept
        # anything; the full writer coerces empty values to Auto.
        return True

    def write_video(self, audio, lyrics, visual_style, performance, pace, wildness,
                    model, seed, prompt_mode=PROMPT_MODES[1], llm=None, project_name="",
                    **image_slots):
        writer = H3ClaudeCodeMusicVideoWriter()
        result = writer.write_video(
            audio=audio,
            direction="",  # the model invents the concept from the song and lyrics
            lyrics=lyrics,
            performance_mode=_performance_mode(performance),
            segment_mode=SEGMENT_MODES[0],  # Auto (cut on the music)
            max_segment_seconds=_max_segment_seconds(pace),
            min_segment_seconds=5.2,
            shots_per_scene=AUTO,
            visual_style=visual_style,
            dialogue_language=DIALOGUE_LANGUAGES[0],  # Auto (match the lyrics)
            wildness=wildness,
            model=model,
            research=False,
            director=True,
            use_subscription=True,
            timeout_seconds=_TIMEOUT_SECONDS,
            seed=seed,
            llm=llm,
            scenes_from_lyrics=bool((lyrics or "").strip()),
            audio_mode=AUDIO_MODES[0],
            prompt_mode=prompt_mode,
            project_name=project_name,
            **{name: image_slots.get(name) for name in self._IMAGE_OUTPUT_NAMES},
        )
        # Full-writer tuple: scenes, durations, lengths, audio_segments, table,
        # scenes_text, synopsis, cast, n, song_seconds, session_id, info,
        # image_1..9, clip_starts, project_name. Keep the rendering essentials.
        images = result[12:12 + len(self._IMAGE_OUTPUT_NAMES)]
        return (
            result[0], result[1], result[2], result[3],  # scenes, durations, lengths, audio
            result[5],                                    # scenes_text
            result[10], result[11],                       # session_id, info
        ) + images + (result[-2], result[-1])             # clip_starts, project_name
