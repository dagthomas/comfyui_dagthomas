# Normalises the canvas geometry of every example workflow:
#   - node positions snapped to LiteGraph's 10 px grid (sizes to whole pixels)
#   - every group's bounding recomputed around its member nodes with uniform
#     padding (60 px above the topmost node title for the group's own title
#     bar, 30 px on the sides and bottom), snapped to the same grid
# Membership is decided BEFORE resizing, from the group each node's centre
# currently sits in, so the visual intent of the layout is preserved.
import glob
import json
import os

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "h3")

GRID = 10
SIDE = 30      # left/right/bottom padding between nodes and the group border
TOP = 60       # room above the topmost node title for the group title bar
NODE_TITLE = 30  # LiteGraph draws a node's title above pos[1]
COLLAPSED_W, COLLAPSED_H = 180, 0  # collapsed nodes render as a title bar only


def snap(v, up=False, down=False):
    if up:
        return -(-int(round(v)) // GRID) * GRID
    if down:
        return (int(round(v)) // GRID) * GRID
    return int(round(v / GRID)) * GRID


def node_rect(n):
    x, y = n["pos"]
    if n.get("flags", {}).get("collapsed"):
        w, h = COLLAPSED_W, COLLAPSED_H
    else:
        w, h = n["size"][0], n["size"][1]
    return (x, y - NODE_TITLE, w, h + NODE_TITLE)


def align(path):
    with open(path, encoding="utf-8") as f:
        wf = json.load(f)
    if "nodes" not in wf:
        return False

    changed = False
    for n in wf["nodes"]:
        px, py = n["pos"]
        sx, sy = snap(px), snap(py)
        if [sx, sy] != [px, py]:
            n["pos"] = [sx, sy]
            changed = True
        w, h = n["size"]
        rw, rh = int(round(w)), int(round(h))
        if [rw, rh] != [w, h]:
            n["size"] = [rw, rh]
            changed = True

    for g in wf.get("groups", []):
        # fit-to-members can pull new node centres inside the border; iterate
        # until the member set is stable so nothing overhangs the group edge
        bounding = list(g["bounding"])
        for _ in range(8):
            bx, by, bw, bh = bounding
            members = []
            for n in wf["nodes"]:
                rx, ry, rw, rh = node_rect(n)
                cx, cy = rx + rw / 2, ry + rh / 2
                if bx <= cx <= bx + bw and by <= cy <= by + bh:
                    members.append((rx, ry, rw, rh))
            if not members:
                break
            x0 = min(r[0] for r in members) - SIDE
            y0 = min(r[1] for r in members) - TOP
            x1 = max(r[0] + r[2] for r in members) + SIDE
            y1 = max(r[1] + r[3] for r in members) + SIDE
            new = [snap(x0, down=True), snap(y0, down=True),
                   snap(x1, up=True) - snap(x0, down=True),
                   snap(y1, up=True) - snap(y0, down=True)]
            if new == bounding:
                break
            bounding = new
        if bounding != list(g["bounding"]):
            g["bounding"] = bounding
            changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wf, f, indent=2, ensure_ascii=False)
    return changed


def main():
    for path in sorted(glob.glob(os.path.join(EX, "*.json"))):
        name = os.path.basename(path)
        print(f"{'aligned  ' if align(path) else 'unchanged'} {name}")


if __name__ == "__main__":
    main()
