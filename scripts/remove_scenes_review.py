# Removes every H3ScenesReview node from the example workflows, wiring the
# review's `scenes` input source straight into whatever its `scenes` output
# fed (video prompt, preview, scene pick, chain plan). The node itself stays
# available in the pack for anyone who wants the gate.
import glob
import json
import os

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "h3")


def strip_reviews(path):
    with open(path, encoding="utf-8") as f:
        wf = json.load(f)
    if "nodes" not in wf:
        return False
    reviews = [n for n in wf["nodes"] if n["type"] == "H3ScenesReview"]
    if not reviews:
        return False

    nodes = {n["id"]: n for n in wf["nodes"]}
    links = {l[0]: l for l in wf["links"]}
    for review in reviews:
        rid = review["id"]
        # the single source feeding review.scenes
        in_link = next((l for l in wf["links"] if l[3] == rid), None)
        assert in_link is not None, f"{path}: review {rid} has no source"
        _lid, src, sslot = in_link[0], in_link[1], in_link[2]
        src_out = nodes[src]["outputs"][sslot]

        # repoint every outgoing link to the source
        for l in wf["links"]:
            if l[1] == rid:
                l[1], l[2] = src, sslot
                src_out["links"] = (src_out.get("links") or []) + [l[0]]

        # drop the incoming link and the node
        src_out["links"] = [x for x in (src_out.get("links") or []) if x != in_link[0]]
        wf["links"] = [l for l in wf["links"] if l[0] != in_link[0]]
        wf["nodes"] = [n for n in wf["nodes"] if n["id"] != rid]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=2, ensure_ascii=False)
    return True


def validate(path):
    with open(path, encoding="utf-8") as f:
        wf = json.load(f)
    if "nodes" not in wf:
        return
    nodes = {n["id"]: n for n in wf["nodes"]}
    seen = set()
    for lid, src, sslot, dst, dslot, _t in wf["links"]:
        assert lid not in seen
        seen.add(lid)
        assert lid in (nodes[src]["outputs"][sslot].get("links") or []), (path, lid)
        assert nodes[dst]["inputs"][dslot].get("link") == lid, (path, lid)
    for n in wf["nodes"]:
        for inp in n.get("inputs", []):
            if inp.get("link") is not None:
                assert inp["link"] in seen, (path, n["id"], inp)
        for out in n.get("outputs", []):
            for lid in out.get("links") or []:
                assert lid in seen, (path, n["id"], out["name"])


def main():
    for path in sorted(glob.glob(os.path.join(EX, "*.json"))):
        changed = strip_reviews(path)
        validate(path)
        print(f"{'stripped ' if changed else 'no review'} {os.path.basename(path)}")


if __name__ == "__main__":
    main()
