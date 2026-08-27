# Brings every example workflow up to date with the CURRENT node definitions.
#
# LiteGraph restores `widgets_values` by position, so a node that gained a
# widget since a workflow was saved loads with the new widget at its default -
# fine on the canvas, but the saved file then fails check_widgets.py, the API
# exports miss the input, and a node that gained a widget in the MIDDLE would
# shift every later value. This script, for every node of ours in every
# example:
#   * appends the default of each widget the file does not have yet (the
#     usual case: widgets are appended last exactly so this is safe)
#   * strips trailing nulls left by JS button widgets of older versions
#   * replaces a combo value the node no longer offers with the default
#   * reports (and leaves alone) anything it cannot reconcile
# then re-runs the link-table validation. Run it after adding widgets, before
# make_api_exports.py.
#
#   python scripts/update_workflow_widgets.py            # update in place
#   python scripts/update_workflow_widgets.py --check    # report only

import glob
import json
import os
import sys
import types

for name in ("folder_paths",):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.get_temp_directory = lambda: r"X:\Temp"
        mod.get_output_directory = lambda: r"X:\Temp"
        mod.get_filename_list = lambda *_a, **_k: []
        sys.modules[name] = mod

sys.path.insert(0, r"X:\comfyui\comfyui\ComfyUI_windows_portable\ComfyUI\custom_nodes")
from comfyui_dagthomas import NODE_CLASS_MAPPINGS  # noqa: E402

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "h3")
PRIMITIVES = ("STRING", "INT", "FLOAT", "BOOLEAN")
# nodes whose JS extension adds a SERIALIZED widget of its own (a DOM panel): so many extra values are fine
JS_WIDGETS = {"H3PromptPreview": 1}
# v3-schema nodes whose widgets come from ComfyUI's model folders - not readable outside ComfyUI
SKIP_TYPES = {"MiniMaxH3AWQEncoderLoader"}


def widget_specs(cls):
    """[(name, type_or_options, default)] for every widget, in declaration order."""
    it = cls.INPUT_TYPES()
    out = []
    for group in ("required", "optional"):
        for name, spec in (it.get(group) or {}).items():
            typ = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if opts.get("forceInput"):
                continue
            if isinstance(typ, list):
                out.append((name, typ, opts.get("default", typ[0] if typ else "")))
            elif typ in PRIMITIVES:
                default = opts.get("default", {"STRING": "", "INT": 0, "FLOAT": 0.0, "BOOLEAN": False}[typ])
                out.append((name, typ, default))
                if name in ("seed", "noise_seed"):
                    out.append((name + ".control", "STRING", "fixed"))
    return out


def reconcile(node, specs, fname):
    values = list(node.get("widgets_values") or [])
    notes = []
    while values and values[-1] is None and len(values) > len(specs):
        values.pop()
        notes.append("stripped a trailing null")
    if len(values) < len(specs):
        added = [s[0] for s in specs[len(values):]]
        values.extend(s[2] for s in specs[len(values):])
        notes.append(f"appended defaults for {', '.join(added)}")
    elif len(values) > len(specs):
        if len(values) - len(specs) <= JS_WIDGETS.get(node.get("type"), 0):
            return values, notes, True          # the JS panel's own value rides at the end
        notes.append(f"UNRESOLVED: {len(values)} values for {len(specs)} widgets - left as is")
        return values, notes, False
    for k, (wname, wtype, default) in enumerate(specs):
        if isinstance(wtype, list) and values[k] not in wtype:
            notes.append(f"{wname}: {values[k]!r} is no longer offered -> {default!r}")
            values[k] = default
    return values, notes, True


def validate(wf, name):
    nodes = {n["id"]: n for n in wf["nodes"]}
    seen = set()
    for lid, src, sslot, dst, dslot, _t in wf["links"]:
        assert lid not in seen, f"{name}: duplicate link {lid}"
        seen.add(lid)
        s, d = nodes[src], nodes[dst]
        assert lid in (s["outputs"][sslot].get("links") or []), f"{name}: link {lid} missing on {src}.outputs[{sslot}]"
        assert d["inputs"][dslot].get("link") == lid, f"{name}: link {lid} mismatch on {dst}.inputs[{dslot}]"
    for n in wf["nodes"]:
        for i, inp in enumerate(n.get("inputs", [])):
            if inp.get("link") is not None:
                assert inp["link"] in seen, f"{name}: node {n['id']} input {i} dangling link {inp['link']}"


def main(check_only=False):
    files = sorted(glob.glob(os.path.join(EX, "*.json")))
    changed_files, problems = 0, 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            wf = json.load(f)
        if "nodes" not in wf:
            continue
        fname = os.path.basename(path)
        changed = False
        for n in wf["nodes"]:
            cls = NODE_CLASS_MAPPINGS.get(n["type"])
            if cls is None or n["type"] in SKIP_TYPES:
                continue
            try:
                specs = widget_specs(cls)
            except Exception as exc:   # a v3 node needing the real ComfyUI (model folders) - skip it
                print(f"  {fname} #{n['id']} {n['type']}: cannot read widgets outside ComfyUI ({type(exc).__name__}) - skipped")
                continue
            values, notes, ok = reconcile(n, specs, fname)
            if not ok:
                problems += 1
            if notes:
                print(f"  {fname} #{n['id']} {n['type']}: " + "; ".join(notes))
            if ok and values != list(n.get("widgets_values") or []):
                n["widgets_values"] = values
                changed = True
        validate(wf, fname)
        if changed and not check_only:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(wf, f, indent=2, ensure_ascii=False)
            changed_files += 1
    print(f"{'would update' if check_only else 'updated'} {changed_files} workflow(s); {problems} unresolved")
    return problems


if __name__ == "__main__":
    sys.exit(1 if main(check_only="--check" in sys.argv) else 0)
