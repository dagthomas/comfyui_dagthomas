# Generates h3_short_film_manual.json: the turbo short-film render graph with
# the Claude writer replaced by APNext H3 Manual Scenes - the film is a
# hand-authored script (the Lighthouse Letter example), no model is called.
import json
import os

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "h3")
SCRIPT_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lighthouse_letter_script.txt")

WRITER_ID = 163
CHARACTERS_ID = 157
NOTE_ID = 161
MANUAL_ID = 217  # last_node_id in the turbo file is 216

NOTE_TEXT = """# H3 Short Film - manual scenes

1. The film is written BY HAND: the **H3 Manual Scenes** node holds the whole script - one `=== SCENE NN | duration: S.S ===` envelope per clip, each a complete four-section H3 prompt (`subject_definitions`, `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`). Edit the text, change durations, add or remove scenes - no model is called, the script IS the film.
2. The node turns the script into the same matching lists the writers emit (`scenes`, `durations`, `lengths`), so the render side is identical to the turbo workflow: each scene renders as one clip of exactly its duration.
3. The default script is *The Lighthouse Letter*: 11 scenes, ~2 minutes - keep character and location anchor lines word-for-word identical across scenes (as the example does) so faces, wardrobe and sets stay consistent from clip to clip.
"""


def load(name):
    with open(os.path.join(EX, name), encoding="utf-8") as f:
        return json.load(f)


def save(wf, name):
    wf["last_node_id"] = max(n["id"] for n in wf["nodes"])
    wf["last_link_id"] = max((l[0] for l in wf["links"]), default=0)
    with open(os.path.join(EX, name), "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=2, ensure_ascii=False)
    print(f"wrote {name}")


def validate(wf, name):
    nodes = {n["id"]: n for n in wf["nodes"]}
    seen = set()
    for lid, src, sslot, dst, dslot, _t in wf["links"]:
        assert lid not in seen, f"{name}: duplicate link {lid}"
        seen.add(lid)
        s, d = nodes[src], nodes[dst]
        assert lid in (s["outputs"][sslot].get("links") or []), \
            f"{name}: link {lid} missing on {src}.outputs[{sslot}]"
        assert d["inputs"][dslot].get("link") == lid, \
            f"{name}: link {lid} mismatch on {dst}.inputs[{dslot}]"
    for n in wf["nodes"]:
        for i, inp in enumerate(n.get("inputs", [])):
            if inp.get("link") is not None:
                assert inp["link"] in seen, f"{name}: node {n['id']} input {i} dangling link {inp['link']}"
        for o, out in enumerate(n.get("outputs", [])):
            for lid in out.get("links") or []:
                assert lid in seen, f"{name}: node {n['id']} output {o} dangling link {lid}"
    print(f"validated {name}: {len(wf['nodes'])} nodes, {len(wf['links'])} links")


def main():
    with open(SCRIPT_TXT, encoding="utf-8") as f:
        script_text = f.read().strip()

    wf = load("h3_short_film_turbo.json")
    wf["id"] = "8f2a5f7e-4b1c-4b9e-9d3a-manualfilm01"
    nodes = {n["id"]: n for n in wf["nodes"]}
    writer = nodes[WRITER_ID]

    # links leaving the writer, keyed by output name; they are rewired below
    out_by_name = {o["name"]: o for o in writer["outputs"]}
    kept = {}  # link id -> new source slot on the manual node
    manual_slots = {"scenes": 0, "durations": 1, "lengths": 2, "scenes_text": 3}
    for name, slot in manual_slots.items():
        for lid in (out_by_name.get(name, {}).get("links") or []):
            kept[lid] = slot

    # drop the writer, the characters node and every link touching them that
    # is not rewired
    dropped_ids = {WRITER_ID, CHARACTERS_ID}
    dead_links = {
        l[0] for l in wf["links"]
        if (l[1] in dropped_ids or l[3] in dropped_ids) and l[0] not in kept
    }
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] not in dropped_ids]
    wf["links"] = [l for l in wf["links"] if l[0] not in dead_links]
    for n in wf["nodes"]:
        for inp in n.get("inputs", []):
            if inp.get("link") in dead_links:
                inp["link"] = None
        for out in n.get("outputs", []):
            if out.get("links"):
                out["links"] = [l for l in out["links"] if l not in dead_links]

    # the manual node takes the writer's place on the canvas
    manual = {
        "id": MANUAL_ID,
        "type": "H3ManualScenes",
        "pos": [writer["pos"][0], writer["pos"][1]],
        "size": [520, 780],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [
            {"name": "scenes", "type": "STRING", "links": [], "shape": 6},
            {"name": "durations", "type": "FLOAT", "links": [], "shape": 6},
            {"name": "lengths", "type": "INT", "links": [], "shape": 6},
            {"name": "scenes_text", "type": "STRING", "links": []},
            {"name": "scene_count", "type": "INT", "links": []},
            {"name": "total_seconds", "type": "FLOAT", "links": []},
        ],
        "properties": {"Node name for S&R": "H3ManualScenes"},
        "widgets_values": [script_text, 12.0, ""],
    }
    wf["nodes"].append(manual)

    # rewire the kept links onto the manual node's outputs
    for l in wf["links"]:
        if l[0] in kept:
            l[1] = MANUAL_ID
            l[2] = kept[l[0]]
            manual["outputs"][kept[l[0]]]["links"].append(l[0])

    nodes = {n["id"]: n for n in wf["nodes"]}
    nodes[NOTE_ID]["widgets_values"][0] = NOTE_TEXT
    nodes[NOTE_ID]["title"] = "How this works"

    save(wf, "h3_short_film_manual.json")
    validate(load("h3_short_film_manual.json"), "h3_short_film_manual.json")


if __name__ == "__main__":
    main()
