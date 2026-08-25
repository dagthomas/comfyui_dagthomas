# APNext MiniMax-H3 Nodes

from . import dashboard  # noqa: F401  (registers the /apnext/h3/api_workflows route)
from . import context_budget  # noqa: F401  (registers the /apnext/h3/context_budget routes)
from .base_prompt_writer import H3BasePromptWriter
from .claude_code_base_writer import H3ClaudeCodeBaseWriter
from .claude_code_continue_writer import H3ClaudeCodeContinueWriter
from .claude_code_ref_writer import H3ClaudeCodeRefWriter
from .claude_code_refiner import H3ClaudeCodeRefiner
from .ref_prompt_writer import H3RefPromptWriter
from .prompt_preview import H3PromptPreview
from .characters import H3Characters
from .claude_code_crossover_writer import H3ClaudeCodeCrossoverWriter
from .claude_code_scenes_writer import H3ClaudeCodeScenesWriter
from .scene_pick import H3ScenePick
from .scene_counter import H3SceneCounter
from .scenes_to_chain_plan import H3ScenesToChainPlan
from .scenes_join import H3ScenesJoin
from .llm_backend import H3LLMBackend
from .claude_code_music_video_writer import H3ClaudeCodeMusicVideoWriter
from .claude_code_presentation_writer import H3ClaudeCodePresentationWriter
from .claude_code_short_film_writer import H3ClaudeCodeShortFilmWriter
from .music_video_minimal import H3MusicVideoMinimal
from .scenes_review import H3ScenesReview
from .scenes_review_gate import H3ScenesReviewGate
from .song_analysis import H3SongAnalysis
from .sound_events import H3SoundEvents
from .scene_brief import H3SceneBrief
from .scenes_store import H3ScenesLoad
from .manual_scenes import H3ManualScenes
from .mouth_guard import H3MouthGuard
from .refine_encode import H3RefineEncode
from .clip_save import H3SaveClip
from .derope import H3DeRopeSave

# Vendored V3-API node by Fred Bliss (fbjr) - needs comfy_api.latest (any
# current ComfyUI); guarded so the rest of the pack survives without it.
try:
    from .awq_encoder_loader import MiniMaxH3AWQEncoderLoader
except ImportError:
    MiniMaxH3AWQEncoderLoader = None

NODE_CLASS_MAPPINGS = {
    "H3BasePromptWriter": H3BasePromptWriter,
    "H3RefPromptWriter": H3RefPromptWriter,
    "H3ClaudeCodeBaseWriter": H3ClaudeCodeBaseWriter,
    "H3ClaudeCodeRefWriter": H3ClaudeCodeRefWriter,
    "H3ClaudeCodeRefiner": H3ClaudeCodeRefiner,
    "H3ClaudeCodeContinueWriter": H3ClaudeCodeContinueWriter,
    "H3PromptPreview": H3PromptPreview,
    "H3Characters": H3Characters,
    "H3ClaudeCodeCrossoverWriter": H3ClaudeCodeCrossoverWriter,
    "H3ClaudeCodeScenesWriter": H3ClaudeCodeScenesWriter,
    "H3ScenePick": H3ScenePick,
    "H3SceneCounter": H3SceneCounter,
    "H3ScenesToChainPlan": H3ScenesToChainPlan,
    "H3ScenesJoin": H3ScenesJoin,
    "H3LLMBackend": H3LLMBackend,
    "H3ClaudeCodeMusicVideoWriter": H3ClaudeCodeMusicVideoWriter,
    "H3ClaudeCodePresentationWriter": H3ClaudeCodePresentationWriter,
    "H3ClaudeCodeShortFilmWriter": H3ClaudeCodeShortFilmWriter,
    "H3MusicVideoMinimal": H3MusicVideoMinimal,
    "H3ScenesReview": H3ScenesReview,
    "H3ScenesReviewGate": H3ScenesReviewGate,
    "H3SongAnalysis": H3SongAnalysis,
    "H3SoundEvents": H3SoundEvents,
    "H3SceneBrief": H3SceneBrief,
    "H3ScenesLoad": H3ScenesLoad,
    "H3ManualScenes": H3ManualScenes,
    "H3MouthGuard": H3MouthGuard,
    "H3RefineEncode": H3RefineEncode,
    "H3SaveClip": H3SaveClip,
    "H3DeRopeSave": H3DeRopeSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3BasePromptWriter": "APNext H3 Prompt Writer",
    "H3RefPromptWriter": "APNext H3 Reference Prompt Writer",
    "H3ClaudeCodeBaseWriter": "APNext H3 Claude Code Writer",
    "H3ClaudeCodeRefWriter": "APNext H3 Claude Code Reference Writer",
    "H3ClaudeCodeRefiner": "APNext H3 Claude Code Refiner",
    "H3ClaudeCodeContinueWriter": "APNext H3 Claude Code Continue Writer",
    "H3PromptPreview": "APNext H3 Prompt Preview",
    "H3Characters": "APNext H3 Characters",
    "H3ClaudeCodeCrossoverWriter": "APNext H3 Crossover Writer",
    "H3ClaudeCodeScenesWriter": "APNext H3 Claude Code Scenes Writer",
    "H3ScenePick": "APNext H3 Scene Pick",
    "H3SceneCounter": "APNext H3 Scene Counter",
    "H3ScenesToChainPlan": "APNext H3 Scenes → Contex Loop Plan",
    "H3ScenesJoin": "APNext H3 Scenes Join",
    "H3LLMBackend": "APNext H3 LLM Backend (Ollama / local / API)",
    "H3ClaudeCodeMusicVideoWriter": "APNext H3 Music Video Writer",
    "H3ClaudeCodePresentationWriter": "APNext H3 Presentation Writer",
    "H3ClaudeCodeShortFilmWriter": "APNext H3 Short Film Writer",
    "H3MusicVideoMinimal": "APNext H3 Music Video (Minimal)",
    "H3ScenesReview": "APNext H3 Scenes Review (edit before render)",
    "H3ScenesReviewGate": "APNext H3 Dailies Gate (print / punch up / cut)",
    "H3SongAnalysis": "APNext H3 Song Analysis (BPM / intensity)",
    "H3SoundEvents": "APNext H3 Sound Events (bass hits / drops / stops)",
    "H3SceneBrief": "APNext H3 Scene Brief (manual scene)",
    "H3ScenesLoad": "APNext H3 Scenes Load (from disk)",
    "H3ManualScenes": "APNext H3 Manual Scenes (script → lists)",
    "H3MouthGuard": "APNext H3 Mouth Guard (protect lips in refine)",
    "H3RefineEncode": "APNext H3 Refine Encode (v2v AV latent)",
    "H3SaveClip": "APNext H3 Save Clip (decode → disk, low memory)",
    "H3DeRopeSave": "APNext H3 De-Rope + Save Clip (Motion Lab by matlowai)",
}

if MiniMaxH3AWQEncoderLoader is not None:
    NODE_CLASS_MAPPINGS["MiniMaxH3AWQEncoderLoader"] = MiniMaxH3AWQEncoderLoader
    NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3AWQEncoderLoader"] = (
        "Load MiniMax H3 Compressed-Tensors AWQ Encoder (by fbjr)"
    )

# ---------------------------------------------------------------------------
# Simple vs advanced form. Everything not listed here is hidden behind the
# node's "Show advanced inputs" toggle (ComfyUI frontend). Autogrow sockets
# (context_N, image_N, cast_N) and the llm socket always stay visible.
# ---------------------------------------------------------------------------
from .common import with_advanced_inputs  # noqa: E402

_TYPED_REFS = tuple(f"{k}_{i}" for k in ("subject", "scenery", "object") for i in (1, 2, 3))
_CREATIVE = ("visual_style", "include_dialogue", "dialogue_language")
_CC = ("model", "seed")

with_advanced_inputs(H3BasePromptWriter,
    ("idea", "task_type", "duration_seconds", *_CREATIVE, "model", "seed", "image", "extra_instructions"))
with_advanced_inputs(H3RefPromptWriter,
    ("idea", "task_type", "reference_role", "duration_seconds", *_CREATIVE, "model", "seed", "reference_notes", "extra_instructions"))
with_advanced_inputs(H3ClaudeCodeBaseWriter,
    ("idea", "task_type", "duration_seconds", *_CREATIVE, *_CC, "image", "extra_instructions"),
    also_advanced=_TYPED_REFS)
with_advanced_inputs(H3ClaudeCodeRefWriter,
    ("idea", "task_type", "reference_role", "duration_seconds", *_CREATIVE, *_CC, "reference_notes", "extra_instructions"))
with_advanced_inputs(H3ClaudeCodeContinueWriter,
    ("frames", "idea", "continuation_mode", "duration_seconds", *_CREATIVE, *_CC, "previous_prompt", "extra_instructions"))
with_advanced_inputs(H3ClaudeCodeRefiner,
    ("h3_prompt", "instruction", *_CC, "session_id", "image"))
with_advanced_inputs(H3ClaudeCodeScenesWriter,
    ("idea", "scene_count", "duration_mode", "continuity_mode", "scene_duration", *_CREATIVE, *_CC,
     "image", "wardrobe", "locations", "extra_instructions", "project_name"),
    also_advanced=_TYPED_REFS)
with_advanced_inputs(H3ClaudeCodeCrossoverWriter,
    ("direction", "extra_cast", "scene_count", "duration_mode", "continuity_mode", "scene_duration",
     "visual_style", "dialogue_language", *_CC, "wardrobe", "locations", "extra_instructions", "image_notes",
     "project_name"))
with_advanced_inputs(H3ClaudeCodeMusicVideoWriter,
    ("audio", "sound_events", "direction", "lyrics", "performance_mode", "segment_mode", "max_segment_seconds",
     "visual_style", "dialogue_language", *_CC, "extra_cast", "wardrobe", "locations", "extra_instructions", "image_notes",
     "project_name"))
with_advanced_inputs(H3ClaudeCodeShortFilmWriter,
    ("manuscript", "length_mode", "scene_count", "target_minutes", "continuity_mode",
     "visual_style", "dialogue_language", "prompt_mode", *_CC,
     "extra_cast", "wardrobe", "locations", "extra_instructions", "image_notes", "project_name"))
with_advanced_inputs(H3ClaudeCodePresentationWriter,
    ("source_material", "direction", "presentation_format", "scene_count", "duration_mode",
     "scene_duration", "visual_aids", "visual_style", "dialogue_language", *_CC,
     "extra_cast", "wardrobe", "locations", "extra_instructions", "image_notes", "project_name"))
with_advanced_inputs(H3LLMBackend, ("model", "model_name", "num_ctx", "thinking"))
with_advanced_inputs(H3MouthGuard, ("latent", "masks", "grow_pixels", "protect_audio"))
with_advanced_inputs(H3ScenesJoin, ("images", "crossfade_frames", "audio", "replace_audio"))
with_advanced_inputs(H3ScenesToChainPlan, ("scenes", "durations", "prompt_prefix"))
