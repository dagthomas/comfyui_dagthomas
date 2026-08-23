# Verifies widgets_values in the generated workflows against the actual node
# definitions: count, order, and combo membership.
import json
import os
import sys
import types

for name in ("folder_paths",):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.get_temp_directory = lambda: r"X:\Temp"
        mod.get_output_directory = lambda: r"X:\Temp"
        sys.modules[name] = mod

sys.path.insert(0, r"X:\comfyui\comfyui\ComfyUI_windows_portable\ComfyUI\custom_nodes")

from comfyui_dagthomas import NODE_CLASS_MAPPINGS

EX = r"X:\comfyui\comfyui\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui_dagthomas\examples\h3"

PRIMITIVES = ("STRING", "INT", "FLOAT", "BOOLEAN")


def widget_list(cls):
    """[(name, type_or_options)] for every widget, in declaration order."""
    it = cls.INPUT_TYPES()
    out = []
    for group in ("required", "optional"):
        for name, spec in (it.get(group) or {}).items():
            typ = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if opts.get("forceInput"):
                continue
            if isinstance(typ, list):
                out.append((name, typ))
            elif typ in PRIMITIVES:
                out.append((name, typ))
                if name in ("seed", "noise_seed"):
                    out.append((name + ".control", "STRING"))
    return out


def check(fname, node_types):
    with open(os.path.join(EX, fname), encoding="utf-8") as f:
        wf = json.load(f)
    for n in wf["nodes"]:
        if n["type"] not in node_types:
            continue
        cls = NODE_CLASS_MAPPINGS[n["type"]]
        expected = widget_list(cls)
        values = n.get("widgets_values") or []
        assert len(values) == len(expected), (
            f"{fname} node {n['id']} {n['type']}: {len(values)} widget values, "
            f"expected {len(expected)}: {[e[0] for e in expected]}"
        )
        for (wname, wtype), val in zip(expected, values):
            if isinstance(wtype, list):
                assert val in wtype, (
                    f"{fname} node {n['id']} {n['type']} widget {wname}: "
                    f"{val!r} not a valid combo option"
                )
        print(f"  ok: {fname} node {n['id']} {n['type']} ({len(values)} widgets)")


check("h3_music_video.json", {"H3SongAnalysis"})
check("h3_music_video_masked_audio.json", {"H3SongAnalysis"})
check("h3_presentation.json", {"H3ClaudeCodePresentationWriter", "H3Characters", "H3ScenesReview"})
check("h3_music_video_minimal.json", {"H3MusicVideoMinimal", "H3ScenesReview", "H3SongAnalysis"})
check("h3_music_video_masked_audio_briefs.json",
      {"H3ClaudeCodeMusicVideoWriter", "H3SceneBrief", "H3Characters", "H3ScenesReview"})
check("h3_music_video_dailies_gate.json",
      {"H3ClaudeCodeMusicVideoWriter", "H3Characters", "H3ScenesReviewGate"})
check("h3_short_film.json", {"H3ClaudeCodeShortFilmWriter", "H3Characters", "H3ScenesReview"})
check("h3_music_video_minimal_dailies_gate.json", {"H3MusicVideoMinimal", "H3ScenesReviewGate"})
check("h3_music_video_masked_audio_briefs_dailies_gate.json",
      {"H3ClaudeCodeMusicVideoWriter", "H3SceneBrief", "H3ScenesReviewGate"})
check("h3_presentation_dailies_gate.json",
      {"H3ClaudeCodePresentationWriter", "H3Characters", "H3ScenesReviewGate"})
check("h3_short_film_manual.json", {"H3ManualScenes", "H3ScenesReview"})
check("h3_face_refine_mouthguard.json", {"H3MouthGuard", "H3RefineEncode"})
print("WIDGETS-OK")
