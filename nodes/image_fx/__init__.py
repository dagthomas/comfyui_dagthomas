# APNext FX Image Effects Nodes

from .bloom import APNextBloom
from .sharpen import APNextSharpen  
from .noise import APNextNoise
from .rough import APNextRough
from .color_grading import APNextColorGrading
from .cross_processing import APNextCrossProcessing
from .split_toning import APNextSplitToning
from .hdr_tone_mapping import APNextHDRToneMapping
from .glitch_art import APNextGlitchArt
from .film_halation import APNextFilmHalation

NODE_CLASS_MAPPINGS = {
    "APNextBloom": APNextBloom,
    "APNextSharpen": APNextSharpen,
    "APNextNoise": APNextNoise,
    "APNextRough": APNextRough,
    "APNextColorGrading": APNextColorGrading,
    "APNextCrossProcessing": APNextCrossProcessing,
    "APNextSplitToning": APNextSplitToning,
    "APNextHDRToneMapping": APNextHDRToneMapping,
    "APNextGlitchArt": APNextGlitchArt,
    "APNextFilmHalation": APNextFilmHalation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APNextBloom": "APNext Bloom FX",
    "APNextSharpen": "APNext Sharpen FX", 
    "APNextNoise": "APNext Noise FX",
    "APNextRough": "APNext Rough FX",
    "APNextColorGrading": "APNext Color Grading FX",
    "APNextCrossProcessing": "APNext Cross Processing FX",
    "APNextSplitToning": "APNext Split Toning FX",
    "APNextHDRToneMapping": "APNext HDR Tone Mapping FX",
    "APNextGlitchArt": "APNext Glitch Art FX",
    "APNextFilmHalation": "APNext Film Halation FX",
}
