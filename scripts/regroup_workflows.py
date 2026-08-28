"""Re-layout every example H3 workflow into function-specific, colour-coded groups.

Every node is classified by what it does (load models, attention patches,
conditioning, sampling, decode, save, user inputs, audio analysis, prompt
writer, prompt preview, notes ...).  Each function becomes one LiteGraph group
with its own colour; nodes inside a group are re-packed into tidy columns and
the groups are laid out in two rows with clear gaps between them:

    row 1  (render path):  Load Models -> Sol Attention / Speed -> Conditioning
                           -> Sampling -> Chain Render -> Decode & Video -> Save
    row 2  (prompt path):  Notes -> User Inputs -> Audio Analysis
                           -> Prompt Writer (LLM) -> Prompt Preview

Node sizes, links, widgets and everything else are left untouched - only
``pos``, ``groups`` and the canvas viewport (``extra.ds``) are rewritten.

Usage:  python scripts/regroup_workflows.py            # all examples/h3/*.json
        python scripts/regroup_workflows.py NAME.json  # just one
"""
import glob
import json
import os
import sys

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "h3")

# Hand-built loop layout with its own bespoke groups - only fill in the
# missing group there instead of re-laying it out.
HAND_LAID = {"h3_crossover_contex_chain.json": ("USER INPUTS + WRITER", "#4f8a3a")}

GRID = 10
NODE_TITLE = 30          # LiteGraph draws the node title bar above pos[1]
PAD_L = PAD_R = PAD_B = 40
PAD_T = 80               # room for the group title bar above the first node title
VGAP = 40                # gap between a node's bottom and the next node's title
COL_GAP = 60             # gap between columns inside a group
GROUP_GAP = 150          # gap between groups in a row
ROW_GAP = 180            # gap between the two rows
MAX_COL_H = 1300         # a column taller than this overflows into a new one
ORIGIN = (-1900, 4400)   # keep the canvas roughly where the old layouts lived
COLLAPSED_W, COLLAPSED_H = 180, 0
TITLE_CHAR_W = 15        # ~px per character of a 24 px group title (min group width)
TITLE_TEXT_PX = 9        # ~px per character of a node's bold 14 px title (measured max ~10 for short caps)
TITLE_TEXT_PAD = 60      # collapse-dot offset before the title text + safety margin

# category -> (title, colour, row, columns of node types in stacking order)
CATEGORIES = {
    "models": ("LOAD MODELS", "#3a5fa8", 0, [
        ["UNETLoader", "CLIPLoader", "MiniMaxH3AWQEncoderLoader", "VAELoader",
         "LoraLoaderModelOnly", "LoadMediaPipeFaceLandmarker"],
    ]),
    "attention": ("SOL ATTENTION / SPEED PATCHES", "#c26a1f", 0, [
        ["PathchSageAttentionKJ", "MiniMaxH3MemoryEfficientSageAttentionPatch",
         "ModelAttentionBackend", "MiniMaxLowVRAMAttention", "MiniMaxChunkFeedForward",
         "EasyCache", "MiniMaxH3SigmaShift"],
        ["MiniMaxH3MemoryEfficientSolAttentionPatch", "SolAttnPatch"],
        ["SpectrumApplyMiniMaxH3"],
    ]),
    "conditioning": ("CONDITIONING", "#b59a1f", 0, [
        ["ImageScaleBy", "MediaPipeFaceLandmarker", "MediaPipeFaceMask", "H3RefineEncode"],
        ["H3MaskedSongLatent", "MiniMaxH3SongMaskedAVContext", "MiniMaxH3ReferenceToVideo", "MiniMaxH3ImageToVideo"],
        ["H3MouthGuard"],
    ]),
    "sampling": ("SAMPLING", "#b03a3a", 0, [
        ["RandomNoise", "BasicGuider", "KSamplerSelect", "BasicScheduler"],
        ["SamplerCustomAdvanced"],
    ]),
    "render": ("CHAIN RENDER", "#c2337a", 0, [
        ["H3MusicVideoChainRender", "H3ShortFilmChainRender", "H3SceneRetake"],
    ]),
    "decode": ("DECODE & VIDEO", "#2aa3b8", 0, [
        ["VAEDecode", "VAEDecodeAudio"],
        ["ImageScale", "RTXVideoSuperResolution", "MaskToImage", "PreviewImage"],
        ["ImageCompositeMasked", "CreateVideo"],
    ]),
    "output": ("SAVE / OUTPUT", "#2a6e4a", 0, [
        ["H3SaveClip", "H3DeRopeSave", "SaveVideo", "H3StitchClips"],
        ["H3SyncCheck"],
    ]),
    "notes": ("NOTES", "#4a4a4a", 1, [
        ["MarkdownNote", "Note"],
    ]),
    "inputs": ("USER INPUTS", "#4f8a3a", 1, [
        ["LoadVideo", "GetVideoComponents", "LoadAudio", "LoadImage",
         "ResolutionSelector", "H3ResolutionSelector", "PrimitiveFloat", "ComfyMathExpression"],
        ["H3Characters", "H3SceneBrief", "H3CutPlan"],
        ["TimePromptNode", "ScenePromptNode", "FeelingsPromptNode", "CinematicPromptNode"],
    ]),
    "audio": ("AUDIO ANALYSIS", "#1f8f6a", 1, [
        ["AudioSeparation", "H3LyricsTranscribe", "H3SongAnalysis", "H3SoundEvents", "H3VoiceOverMusic"],
        ["H3BeatGrid", "H3BeatEmphasis"],
    ]),
    "llm": ("PROMPT WRITER (LLM)", "#7a3fa0", 1, [
        ["H3LLMBackend", "PrimitiveInt", "H3ScenePick", "H3SceneCounter", "H3ScenesToChainPlan", "H3ScenesLoad"],
        ["H3ClaudeCodeBaseWriter", "H3BasePromptWriter", "H3ClaudeCodeCrossoverWriter",
         "H3ClaudeCodeMusicVideoWriter", "H3ClaudeCodePresentationWriter",
         "H3ClaudeCodeScenesWriter", "H3ClaudeCodeShortFilmWriter",
         "H3MusicVideoMinimal", "H3ManualScenes"],
        ["H3ClaudeCodeRefiner", "H3ScenesReviewGate"],
    ]),
    "preview": ("PROMPT PREVIEW", "#5b6b8f", 1, [
        ["H3PromptPreview"],
    ]),
    "other": ("OTHER", "#666666", 1, [[]]),
}

# Notes whose title mentions one of these stay next to the nodes they explain
# (placed in that group's last column); every other note goes to NOTES.
NOTE_ROUTING = [
    ("masked-audio", "conditioning"),
    ("chain render", "render"),
    ("cut plan", "inputs"),
    ("sync check", "output"),
    ("de-rop", "output"),
    ("temporal upsampling", "output"),
]

TYPE_INDEX = {}
for _cat, (_t, _c, _r, _cols) in CATEGORIES.items():
    for _ci, _col in enumerate(_cols):
        for _i, _typ in enumerate(_col):
            TYPE_INDEX[_typ] = (_cat, _ci, _i)


def snap_up(v):
    return -(-int(round(v)) // GRID) * GRID


def as_list(v):
    return [v["0"], v["1"]] if isinstance(v, dict) else list(v)


def node_size(n):
    """Footprint the node really occupies on the canvas.  LiteGraph draws the
    title text unclipped, so a long title sticks out past ``size[0]`` - the
    width returned here is the larger of the body and the title text."""
    title_w = len(n.get("title") or n["type"]) * TITLE_TEXT_PX + TITLE_TEXT_PAD
    if n.get("flags", {}).get("collapsed"):
        return max(COLLAPSED_W, title_w), COLLAPSED_H
    w, h = as_list(n.get("size") or [200, 100])
    return max(int(round(w)), title_w), int(round(h))


def classify(n):
    """Return (category, column, order-key) for a node."""
    typ, title = n["type"], (n.get("title") or "").lower()
    if typ in ("MarkdownNote", "Note"):
        for key, cat in NOTE_ROUTING:
            if key in title:
                return cat, len(CATEGORIES[cat][3]) - 1, 999
        return "notes", 0, 0
    if "sync check" in title:            # the muted VAEDecode feeding H3SyncCheck
        return "output", 1, -1
    if typ in TYPE_INDEX:
        return TYPE_INDEX[typ]
    if "Loader" in typ:
        return "models", 0, 500
    if "Save" in typ:
        return "output", 0, 500
    if "Decode" in typ:
        return "decode", 0, 500
    print(f"    ! unknown node type {typ!r} -> OTHER")
    return "other", 0, 500


def pack_columns(members):
    """members: list of (col, order, node). Returns list of columns, each a list
    of (node, w, h) already overflowed to MAX_COL_H."""
    by_col = {}
    for col, order, n in members:
        by_col.setdefault(col, []).append((order, n["id"], n))
    columns = []
    for col in sorted(by_col):
        cur, cur_h = [], 0
        for _o, _i, n in sorted(by_col[col], key=lambda t: (t[0], t[1])):
            w, h = node_size(n)
            need = NODE_TITLE + h + (VGAP if cur else 0)
            if cur and cur_h + need > MAX_COL_H:
                columns.append(cur)
                cur, cur_h = [], 0
                need = NODE_TITLE + h
            cur.append((n, w, h))
            cur_h += need
        columns.append(cur)
    return columns


def place_group(columns, gx, gy, title=""):
    """Position nodes for one group whose top-left is (gx, gy). Returns (w, h)."""
    x = gx + PAD_L
    max_h = 0
    for col in columns:
        col_w = max(w for _n, w, _h in col)
        y = gy + PAD_T
        for n, _w, h in col:
            n["pos"] = [x, y + NODE_TITLE]
            y = snap_up(y + NODE_TITLE + h + VGAP)
        max_h = max(max_h, y - VGAP - (gy + PAD_T))
        x = snap_up(x + col_w + COL_GAP)
    w = snap_up(x - COL_GAP - gx + PAD_R)
    w = max(w, snap_up(len(title) * TITLE_CHAR_W + PAD_L + PAD_R))  # title must fit
    h = snap_up(PAD_T + max_h + PAD_B)
    return w, h


def relayout(wf):
    buckets = {}
    for n in wf["nodes"]:
        cat, col, order = classify(n)
        buckets.setdefault(cat, []).append((col, order, n))

    rows = {0: [], 1: []}
    for cat in CATEGORIES:                       # keeps the pipeline order
        if cat in buckets:
            rows[CATEGORIES[cat][2]].append(cat)

    groups, gid = [], 1
    x0, y0 = ORIGIN
    row_y = y0
    for r in (0, 1):
        x, row_h = x0, 0
        for cat in rows[r]:
            title, color, _row, _cols = CATEGORIES[cat]
            w, h = place_group(pack_columns(buckets[cat]), x, row_y, title)
            groups.append({"id": gid, "title": title, "bounding": [x, row_y, w, h],
                           "color": color, "font_size": 24, "flags": {}})
            gid += 1
            x += w + GROUP_GAP
            row_h = max(row_h, h)
        row_y += row_h + ROW_GAP
    wf["groups"] = groups

    xs = [g["bounding"][0] for g in groups] + [g["bounding"][0] + g["bounding"][2] for g in groups]
    ys = [g["bounding"][1] for g in groups] + [g["bounding"][1] + g["bounding"][3] for g in groups]
    total_w = max(xs) - min(xs)
    ds = wf.setdefault("extra", {}).setdefault("ds", {})
    ds["scale"] = round(min(0.6, 1800 / max(total_w, 1)), 4)
    ds["offset"] = [-min(xs) + 60, -min(ys) + 60]


def node_rect(n):
    x, y = as_list(n["pos"])
    w, h = node_size(n)
    return x, y - NODE_TITLE, w, h + NODE_TITLE


def add_group_for_ungrouped(wf, title, color):
    """For hand-laid workflows: wrap every node outside all groups in one group."""
    loose = []
    for n in wf["nodes"]:
        rx, ry, rw, rh = node_rect(n)
        cx, cy = rx + rw / 2, ry + rh / 2
        inside = any(bx <= cx <= bx + bw and by <= cy <= by + bh
                     for bx, by, bw, bh in (g["bounding"] for g in wf.get("groups", [])))
        if not inside:
            loose.append((rx, ry, rw, rh))
    if not loose:
        return False
    x0 = (min(r[0] for r in loose) - PAD_L) // GRID * GRID
    y0 = (min(r[1] for r in loose) - PAD_T) // GRID * GRID
    x1 = snap_up(max(r[0] + r[2] for r in loose) + PAD_R)
    y1 = snap_up(max(r[1] + r[3] for r in loose) + PAD_B)
    gid = max([g.get("id", 0) for g in wf.get("groups", [])] + [0]) + 1
    wf.setdefault("groups", []).append(
        {"id": gid, "title": title, "bounding": [x0, y0, x1 - x0, y1 - y0],
         "color": color, "font_size": 24, "flags": {}})
    return True


def process(path):
    with open(path, "rb") as f:
        raw = f.read()
    crlf = b"\r\n" in raw
    wf = json.loads(raw.decode("utf-8"))
    if "nodes" not in wf:
        return "skipped (not a UI workflow)"

    name = os.path.basename(path)
    if name in HAND_LAID:
        title, color = HAND_LAID[name]
        if not add_group_for_ungrouped(wf, title, color):
            return "unchanged (hand-laid, nothing loose)"
        note = "hand-laid: added missing group"
    else:
        relayout(wf)
        note = f"{len(wf['groups'])} groups"

    text = json.dumps(wf, indent=2, ensure_ascii=False)
    if crlf:
        text = text.replace("\n", "\r\n")
    with open(path, "wb") as f:
        f.write(text.encode("utf-8"))
    return note


def main(argv):
    names = argv[1:]
    paths = ([os.path.join(EX, n) for n in names] if names
             else sorted(glob.glob(os.path.join(EX, "*.json"))))
    for path in paths:
        print(f"{os.path.basename(path)}: {process(path)}")


if __name__ == "__main__":
    main(sys.argv)
