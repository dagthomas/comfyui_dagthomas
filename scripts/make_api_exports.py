# Converts the canvas example workflows into ComfyUI API-format prompts
# (what POST /prompt expects), so a custom front-end can queue them directly.
#
# Widget values are mapped to named inputs three ways, in order of trust:
#   1. our own nodes: widget order derived from the live INPUT_TYPES
#   2. widgets_values_named stored in the canvas JSON (core nodes)
#   3. small hand maps for the few core nodes without either
# Connections become ["<src id>", <slot>]. UI-only nodes (notes, the preview)
# are dropped, control_after_generate is dropped, and in non-gate exports the
# Scenes Review node is forced to Bypass so a headless run never interrupts.
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
OUT = os.path.join(EX, "api")
os.makedirs(OUT, exist_ok=True)

SKIP_TYPES = {"MarkdownNote", "Note", "H3PromptPreview"}
PRIMITIVES = ("STRING", "INT", "FLOAT", "BOOLEAN")
HAND_MAPS = {
    "LoadAudio": ["audio"],
    "LoadImage": ["image"],
}
BYPASS_REVIEW = "Bypass (pass scenes through)"


def widget_names(cls):
    """Widget input names in declaration order (None = UI-only slot to skip)."""
    it = cls.INPUT_TYPES()
    out = []
    for group in ("required", "optional"):
        for name, spec in (it.get(group) or {}).items():
            typ = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if opts.get("forceInput"):
                continue
            if isinstance(typ, list) or typ in PRIMITIVES:
                out.append(name)
                if name in ("seed", "noise_seed"):
                    out.append(None)  # control_after_generate: UI only
    return out


def export(src, dst, bypass_review):
    with open(os.path.join(EX, src), encoding="utf-8") as f:
        wf = json.load(f)
    nodes = [n for n in wf["nodes"] if n["type"] not in SKIP_TYPES and n.get("mode", 0) == 0]
    keep = {n["id"] for n in nodes}
    links = {l[0]: l for l in wf["links"]}
    prompt = {}
    for n in nodes:
        inputs = {}
        values = n.get("widgets_values") or []
        cls = NODE_CLASS_MAPPINGS.get(n["type"])
        if cls is not None:
            names = widget_names(cls)
            assert len(names) == len(values), (
                f"{src} node {n['id']} {n['type']}: {len(values)} widget values vs "
                f"{len(names)} widgets"
            )
            for name, value in zip(names, values):
                if name is not None:
                    inputs[name] = value
        elif n.get("widgets_values_named"):
            for name, value in n["widgets_values_named"].items():
                if name != "control_after_generate" and value is not None:
                    inputs[name] = value
        elif n["type"] in HAND_MAPS:
            for name, value in zip(HAND_MAPS[n["type"]], values):
                if value is not None:
                    inputs[name] = value
        elif values:
            raise AssertionError(f"{src}: no widget map for {n['type']} (node {n['id']})")

        for inp in n.get("inputs", []):
            lid = inp.get("link")
            if lid is None:
                continue
            _l, s, sslot, _d, _dslot, _t = links[lid]
            if s in keep:
                inputs[inp["name"]] = [str(s), sslot]

        if bypass_review and n["type"] == "H3ScenesReview":
            inputs["mode"] = BYPASS_REVIEW

        prompt[str(n["id"])] = {"class_type": n["type"], "inputs": inputs}

    # sanity: every connection points at an exported node
    for nid, entry in prompt.items():
        for name, value in entry["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in prompt, f"{src}: {nid}.{name} -> missing node {value[0]}"

    with open(os.path.join(OUT, dst), "w", encoding="utf-8") as f:
        json.dump(prompt, f, indent=2, ensure_ascii=False)
    print(f"exported {dst}: {len(prompt)} nodes")


export("h3_music_video.json", "h3_music_video.api.json", bypass_review=True)
export("h3_music_video_minimal.json", "h3_music_video_minimal.api.json", bypass_review=True)
export("h3_presentation.json", "h3_presentation.api.json", bypass_review=True)
export("h3_music_video_dailies_gate.json", "h3_music_video_dailies_gate.api.json", bypass_review=False)
export("h3_music_video_minimal_dailies_gate.json", "h3_music_video_minimal_dailies_gate.api.json", bypass_review=False)
export("h3_presentation_dailies_gate.json", "h3_presentation_dailies_gate.api.json", bypass_review=False)
export("h3_short_film.json", "h3_short_film.api.json", bypass_review=True)
print("API-EXPORTS-OK")
