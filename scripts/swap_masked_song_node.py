"""Swap the third-party masked-audio node for this pack's drop-in in every example workflow.

    MiniMaxH3SongMaskedAVContext  (ComfyUI-H3-Motion-Context-MultiRef)
        -> H3MaskedSongLatent      (nodes/h3/masked_song.py)

Both nodes share the input names, widget order (clip_start_seconds,
context_length, source_fps, crop) and the three outputs (latent, trim_frames,
clip_audio), so only the node's type changes; the new widgets (preroll_seconds,
lookahead_seconds, audio_denoise) are appended with their defaults so the saved
widget list matches what the front-end expects. Links, positions and groups
are left untouched. The `Needs ComfyUI-H3-Motion-Context-MultiRef` line in the
workflow notes is updated too.

Usage:  python scripts/swap_masked_song_node.py              # all examples/h3/*.json + api/*.json
        python scripts/swap_masked_song_node.py NAME.json    # just one
        python scripts/swap_masked_song_node.py --dry-run    # report only
"""
import glob
import json
import os
import sys

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "h3")
OLD, NEW = "MiniMaxH3SongMaskedAVContext", "H3MaskedSongLatent"
NEW_WIDGETS = [1.0, 0.2, 0.0, 0.15, 0.2]   # preroll_seconds, lookahead_seconds, audio_denoise, gap_denoise, gate_hold_seconds
NEW_INPUTS = {"preroll_seconds": 1.0, "lookahead_seconds": 0.2, "audio_denoise": 0.0,
              "gap_denoise": 0.15, "gate_hold_seconds": 0.2}
NOTE_OLD = "Needs ComfyUI-H3-Motion-Context-MultiRef and audio-separation-nodes-comfyui."
NOTE_NEW = "Needs audio-separation-nodes-comfyui (the masked-audio node is this pack's own H3 Masked Song Latent)."


def swap_ui(wf):
    n = 0
    for node in wf.get("nodes", []):
        if node.get("type") == OLD:
            node["type"] = NEW
            wv = node.get("widgets_values")
            if isinstance(wv, list) and len(wv) == 4:
                wv.extend(NEW_WIDGETS)
            props = node.get("properties") or {}
            if props.get("Node name for S&R") == OLD:
                props["Node name for S&R"] = NEW
            n += 1
        if node.get("type") == "MarkdownNote":
            wv = node.get("widgets_values") or []
            if wv and isinstance(wv[0], str) and NOTE_OLD in wv[0]:
                wv[0] = wv[0].replace(NOTE_OLD, NOTE_NEW)
    return n


def swap_api(wf):
    n = 0
    for node in wf.values():
        if isinstance(node, dict) and node.get("class_type") == OLD:
            node["class_type"] = NEW
            inputs = node.setdefault("inputs", {})
            for k, v in NEW_INPUTS.items():
                inputs.setdefault(k, v)
            n += 1
    return n


def main(argv):
    dry = "--dry-run" in argv
    names = [a for a in argv if not a.startswith("--")]
    paths = ([os.path.join(EX, n) if not os.path.isabs(n) else n for n in names]
             or sorted(glob.glob(os.path.join(EX, "*.json")) + glob.glob(os.path.join(EX, "api", "*.json"))))
    total = 0
    for path in paths:
        raw = open(path, encoding="utf-8").read()
        wf = json.loads(raw)
        n = swap_ui(wf) if "nodes" in wf else swap_api(wf)
        if not n:
            continue
        total += n
        print(f"{'would swap' if dry else 'swapped'} {n} node(s) in {os.path.relpath(path, EX)}")
        if not dry:
            indent = 2 if raw.lstrip().startswith('{\n  "') else 4
            open(path, "w", encoding="utf-8", newline="\n").write(json.dumps(wf, indent=indent, ensure_ascii=False) + "\n")
    print(f"{total} node(s) in total")


if __name__ == "__main__":
    main(sys.argv[1:])
