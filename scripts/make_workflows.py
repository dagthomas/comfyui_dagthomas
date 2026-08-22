# Generates the three new example workflows from the existing music-video
# examples, then validates every link table.
import copy
import json
import os

EX = r"X:\comfyui\comfyui\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui_dagthomas\examples\h3"


def load(name):
    with open(os.path.join(EX, name), encoding="utf-8") as f:
        return json.load(f)


def save(wf, name):
    # keep LiteGraph's id counters ahead of every node/link actually present,
    # so nodes added later in the UI can never collide
    wf["last_node_id"] = max(n["id"] for n in wf["nodes"])
    wf["last_link_id"] = max((l[0] for l in wf["links"]), default=0)
    with open(os.path.join(EX, name), "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=2, ensure_ascii=False)
    print(f"wrote {name}")


def node(wf, nid):
    return next(n for n in wf["nodes"] if n["id"] == nid)


def drop_links(wf, ids):
    ids = set(ids)
    wf["links"] = [l for l in wf["links"] if l[0] not in ids]
    for n in wf["nodes"]:
        for inp in n.get("inputs", []):
            if inp.get("link") in ids:
                inp["link"] = None
        for out in n.get("outputs", []):
            if out.get("links"):
                out["links"] = [l for l in out["links"] if l not in ids]


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


IMG_OUTS = [{"name": f"image_{i}", "type": "IMAGE", "links": None} for i in range(1, 10)]


# ======================================================================
# 0. Inject the H3 Song Analysis readout into the BASE music workflows
#    (idempotent). Every derived workflow below inherits it automatically.
#    High fixed ids (190+/400+) so the sections' hardcoded ids never collide.
# ======================================================================
def inject_song_analysis(base_name, nid, link_id, pos):
    wf = load(base_name)
    if any(n["type"] == "H3SongAnalysis" for n in wf["nodes"]):
        return
    la = node(wf, 164)  # LoadAudio
    la["outputs"][0]["links"] = list(la["outputs"][0].get("links") or []) + [link_id]
    wf["nodes"].append({
        "id": nid,
        "type": "H3SongAnalysis",
        "pos": pos,
        "size": [340, 160],
        "flags": {},
        "order": 2,
        "mode": 0,
        "inputs": [{"name": "audio", "type": "AUDIO", "link": link_id}],
        "outputs": [
            {"name": "audio", "type": "AUDIO", "links": None},
            {"name": "profile", "type": "STRING", "links": None},
            {"name": "bpm", "type": "FLOAT", "links": None},
            {"name": "intensity", "type": "INT", "links": None},
            {"name": "label", "type": "STRING", "links": None},
        ],
        "title": "Song Analysis — BPM / intensity",
        "properties": {"Node name for S&R": "H3SongAnalysis", "cnr_id": "comfyui_dagthomas"},
    })
    wf["links"].append([link_id, 164, 0, nid, 0, "AUDIO"])
    save(wf, base_name)
    validate(load(base_name), base_name)


def remove_song_analysis(wf):
    """For derived workflows with no audio (the presentation)."""
    an = next((n for n in wf["nodes"] if n["type"] == "H3SongAnalysis"), None)
    if an:
        drop_links(wf, [i["link"] for i in an.get("inputs", []) if i.get("link")])
        wf["nodes"] = [n for n in wf["nodes"] if n["id"] != an["id"]]


inject_song_analysis("h3_music_video.json", 190, 400, [-1140.0, 5470.3])
inject_song_analysis("h3_music_video_masked_audio.json", 191, 401, [-1140.0, 5403.8])

# The writers grew a trailing `prompt_mode` widget (REF/FL/Auto). Append /
# normalise the default on the BASE files' writer widget lists (idempotent).
PROMPT_MODE_DEFAULT = "Ref2VA (bind reference images)"
for base in ("h3_music_video.json", "h3_music_video_masked_audio.json"):
    wf = load(base)
    w = node(wf, 163)
    if len(w["widgets_values"]) == 32:
        w["widgets_values"].append(PROMPT_MODE_DEFAULT)
        save(wf, base)
    elif w["widgets_values"][32] != PROMPT_MODE_DEFAULT:
        w["widgets_values"][32] = PROMPT_MODE_DEFAULT
        save(wf, base)

# Every workflow defaults to seed -1 (randomize): a queue always writes a
# brand-new video, nothing is reused from earlier runs. The note explains the
# trade-off for the Review node's Continue flow.
REVIEW_SEED_NOTE = (
    "**Seed:** the writer's seed is **-1 (randomize)** - every queue writes a "
    "brand-new video; nothing is reused from earlier runs. To use the Review "
    "node's stop-edit-**Continue** flow, first set the writer's seed to a "
    "FIXED number: Continue relies on ComfyUI reusing the cached scenes, and "
    "with -1 each queue writes fresh scenes and re-reviews instead."
)
for base in ("h3_music_video.json", "h3_music_video_masked_audio.json"):
    wf = load(base)
    changed = False
    w = node(wf, 163)
    if w["widgets_values"][15] != -1:  # seed / control_after_generate
        w["widgets_values"][15] = -1
        w["widgets_values"][16] = "randomize"
        changed = True
    note = node(wf, 161)
    text = note["widgets_values"][0]
    if "Seed = cache" in text:  # replace the older fixed-seed wording
        text = text.split("\n\n**Seed = cache:")[0]
        changed = True
    if "**Seed:**" not in text:
        note["widgets_values"] = [text + "\n\n" + REVIEW_SEED_NOTE]
        changed = True
    if changed:
        save(wf, base)


# ======================================================================
# 0b. Direct save: drop the join + upscale tail in the BASE workflows -
#     VAE Decode feeds Create Video directly (audio from VAE Decode Audio)
#     and Save Video writes each scene's clip as its own file.
# ======================================================================
def direct_save(base_name):
    wf = load(base_name)
    doomed = [n for n in wf["nodes"]
              if n["type"] in ("H3ScenesJoin", "ImageScale", "RTXVideoSuperResolution")]
    if doomed:
        ids = {n["id"] for n in doomed}
        drop_links(wf, [l[0] for l in wf["links"] if l[1] in ids or l[3] in ids])
        wf["nodes"] = [n for n in wf["nodes"] if n["id"] not in ids]
        vd = next(n for n in wf["nodes"] if n["type"] == "VAEDecode")
        va = next(n for n in wf["nodes"] if n["type"] == "VAEDecodeAudio")
        cv = next(n for n in wf["nodes"] if n["type"] == "CreateVideo")
        l1 = max(l[0] for l in wf["links"]) + 1
        l2 = l1 + 1
        vd["outputs"][0]["links"] = [l1]
        va["outputs"][0]["links"] = [l2]
        cv["inputs"][0]["link"] = l1
        cv["inputs"][1]["link"] = l2
        wf["links"] += [
            [l1, vd["id"], 0, cv["id"], 0, "IMAGE"],
            [l2, va["id"], 0, cv["id"], 1, "AUDIO"],
        ]
    note = node(wf, 161)
    text = note["widgets_values"][0]
    old = ("The H3 video node runs once per piece; **H3 Scenes Join** stitches the clips "
           "and puts the song back under them.")
    new = ("The H3 video node runs once per piece and each clip is DECODED AND SAVED as "
           "its own video file (VAE Decode → Create Video → Save Video). Stitch "
           "externally, or re-add **H3 Scenes Join** before Create Video for one file "
           "with the original song.")
    if old in text:
        note["widgets_values"] = [text.replace(old, new)]
    if doomed or old in text:
        save(wf, base_name)
        validate(load(base_name), base_name)


direct_save("h3_music_video.json")
direct_save("h3_music_video_masked_audio.json")

# The examples ship EMPTY: no sample song filename, no sample direction, no
# sample lyrics - the user brings their own (idempotent).
for base in ("h3_music_video.json", "h3_music_video_masked_audio.json"):
    wf = load(base)
    w = node(wf, 163)
    la = node(wf, 164)  # LoadAudio
    if w["widgets_values"][0] or w["widgets_values"][1] or la["widgets_values"][0]:
        w["widgets_values"][0] = ""   # direction
        w["widgets_values"][1] = ""   # lyrics
        la["widgets_values"][0] = ""  # song filename
        save(wf, base)


# ======================================================================
# 0c. Original sound under every clip: Create Video takes its audio from the
#     writer's `audio_segments` (per-clip slices of the ORIGINAL full mix,
#     frame-aligned) instead of the decoded latent audio - which is the
#     model's re-rendering in the reference workflow and the vocals-only
#     stem in the masked one.
# ======================================================================
ORIGINAL_SOUND_NOTE = (
    "**Original sound:** every saved clip carries its ORIGINAL slice of the "
    "song (the writer's `audio_segments`, cut on the same frame grid) - not "
    "the model-generated audio, and in the masked workflow not the "
    "vocals-only stem. Wire Create Video's `audio` back to VAE Decode Audio "
    "if you want to hear what the model generated."
)


def original_audio(base_name):
    wf = load(base_name)
    cv = next(n for n in wf["nodes"] if n["type"] == "CreateVideo")
    al = next((l for l in wf["links"] if l[3] == cv["id"] and l[4] == 1), None)
    changed = False
    if al is not None and node(wf, al[1])["type"] == "VAEDecodeAudio":
        drop_links(wf, [al[0]])
        lid = max(l[0] for l in wf["links"]) + 1
        w = node(wf, 163)  # music writer: audio_segments = output 3
        w["outputs"][3]["links"] = list(w["outputs"][3].get("links") or []) + [lid]
        cv["inputs"][1]["link"] = lid
        wf["links"].append([lid, 163, 3, cv["id"], 1, "AUDIO"])
        changed = True
    note = node(wf, 161)
    if "Original sound" not in note["widgets_values"][0]:
        note["widgets_values"] = [note["widgets_values"][0] + "\n\n" + ORIGINAL_SOUND_NOTE]
        changed = True
    if changed:
        save(wf, base_name)
        validate(load(base_name), base_name)


original_audio("h3_music_video.json")
original_audio("h3_music_video_masked_audio.json")


# ======================================================================
# A. h3_presentation.json  (from h3_music_video.json)
# ======================================================================
wf = load("h3_music_video.json")
wf["id"] = "b7c4d1e2-31a5-4f6b-8d90-5a2c8e714f21"

# no source audio anywhere: drop LoadAudio (164) and its links (319, 325),
# the audio_segments -> ref_audio link (324), and the song-analysis readout
remove_song_analysis(wf)
drop_links(wf, [319, 324, 325])
wf["nodes"] = [n for n in wf["nodes"] if n["id"] != 164]

# the presentation has no source song: its sound IS the generated voice, so
# Create Video's audio comes from VAE Decode Audio (the base wires it from
# the music writer's audio_segments, which this writer does not have)
cv = next(n_ for n_ in wf["nodes"] if n_["type"] == "CreateVideo")
al = next((l for l in wf["links"] if l[3] == cv["id"] and l[4] == 1), None)
if al is not None and al[1] == 163:
    drop_links(wf, [al[0]])
    lid = max(l for l in (l_[0] for l_ in wf["links"])) + 1
    va = node(wf, 121)  # VAEDecodeAudio
    va["outputs"][0]["links"] = list(va["outputs"][0].get("links") or []) + [lid]
    cv["inputs"][1]["link"] = lid
    wf["links"].append([lid, 121, 0, cv["id"], 1, "AUDIO"])

# the writer becomes the Presentation Writer
w = node(wf, 163)
w["type"] = "H3ClaudeCodePresentationWriter"
w["title"] = "H3 Presentation Writer"
w["size"] = [520, 1080]
w["properties"] = {"Node name for S&R": "H3ClaudeCodePresentationWriter", "cnr_id": "comfyui_dagthomas"}
w["inputs"] = [
    {"name": "cast_1", "type": "STRING", "link": 320, "shape": 7},
    {"name": "cast_2", "type": "STRING", "link": None, "shape": 7},
    {"name": "cast_3", "type": "STRING", "link": None, "shape": 7},
    {"name": "cast_4", "type": "STRING", "link": None, "shape": 7},
    {"name": "llm", "type": "APNEXT_LLM", "link": None, "shape": 7},
    {"name": "image_1", "type": "IMAGE", "link": None, "shape": 7},
]
w["outputs"] = [
    {"name": "scenes", "type": "STRING", "links": [326]},
    {"name": "durations", "type": "FLOAT", "links": None},
    {"name": "lengths", "type": "INT", "links": [323]},
    {"name": "scenes_text", "type": "STRING", "links": [322]},
    {"name": "synopsis", "type": "STRING", "links": None},
    {"name": "script", "type": "STRING", "links": None},
    {"name": "cast", "type": "STRING", "links": None},
    {"name": "scene_count", "type": "INT", "links": None},
    {"name": "total_seconds", "type": "FLOAT", "links": None},
    {"name": "session_id", "type": "STRING", "links": None},
    {"name": "info", "type": "STRING", "links": None},
] + copy.deepcopy(IMG_OUTS)
w["widgets_values"] = [
    "Project Falcon benchmark results (v2.1 vs v1.4):\n"
    "- accuracy: 91.4% (up from 78.2%)\n"
    "- median latency: 95 ms (down from 220 ms)\n"
    "- memory: 1.9 GB (down from 3.4 GB)\n"
    "- supported languages: 14 (up from 6)\n"
    "Key finding: switching to the fused attention kernel accounts for 70% of the latency win.\n"
    "Known limitation: accuracy drops to 84% on inputs longer than 4k tokens.",
    "An enthusiastic keynote reveal. Dr. Maya Ellis presents on a dark stage with a giant "
    "LED screen behind her; confident, warm, a little playful. The audience is developers.",
    "Keynote stage (presenter + giant LED screen)",
    6,
    "Vary 5-15s (let Claude pace each scene)",
    12.0,
    "Auto (a graphic wherever it helps)",
    "Independent clips (hard cuts, T2V openers)",
    "Live-action, 35mm cinematic film aesthetic",
    "English",
    10,               # wildness
    "sonnet", False, True, True, 1800,
    -1, "randomize",  # seed + control: fresh every queue
    "",               # extra_cast
    "", "",           # custom_dialogue_language, custom_visual_style
    "", "",           # wardrobe, locations
    True,             # enforce_wardrobe
    "", "",           # extra_instructions, image_notes
    True, False,      # include_soundscape, include_non_diegetic_music
    "", "",           # resume_session_id, working_dir
    "Characters only (ignore picture backgrounds)",
    True, 4,          # save_scenes, scenes_per_call
    "Ref2VA (bind reference images)",  # prompt_mode
]
# link 322 now leaves output slot 3 (scenes_text); cast_1 is input slot 0 now
for l in wf["links"]:
    if l[0] == 322:
        l[2] = 3
    if l[0] == 320:
        l[4] = 0

# presenter instead of performer
c = node(wf, 157)
c["title"] = "H3 Characters — custom presenter + wardrobe"
c["widgets_values"] = [
    "\u270f\ufe0f custom (type in custom_character)", "(all)", 0, "fixed",
    "Dr. Maya Ellis: a research scientist in her early 40s, shoulder-length dark curls, rectangular glasses",
    "charcoal blazer over a white crew-neck tee, dark straight-leg jeans, thin silver watch on the left wrist",
]

# the preview cache belongs to the old example
p = node(wf, 153)
p["widgets_values"] = [""]
p.pop("widgets_values_named", None)

n = node(wf, 161)
n["widgets_values"] = [
    "# H3 Presentation\n\n"
    "1. Paste your **source material** (findings, benchmark numbers, a changelog, release "
    "notes) into the **H3 Presentation Writer** - it is the ground truth: every number, "
    "name and claim spoken or shown comes from it verbatim, nothing is invented.\n"
    "2. The writer plans the talk (hook \u2192 one point per scene \u2192 takeaway), writes one H3 "
    "prompt per scene with the presenter speaking generated dialogue and charts/graphics "
    "showing the real values, and emits matching lists: `scenes` \u2192 prompt, `lengths` \u2192 "
    "length.\n"
    "3. The H3 video node runs once per scene (voice and room tone are generated by the "
    "model) and each clip is saved as its own video file (VAE Decode → Create Video → "
    "Save Video); stitch externally, or re-add H3 Scenes Join for one file.\n\n"
    "The presenter comes from the H3 Characters node (\u270f\ufe0f custom); connect a face photo to "
    "the writer's `image_1..` to lock their identity. The `script` output is a teleprompter "
    "view of every spoken line - read it to check fact fidelity before rendering.\n\n"
    + REVIEW_SEED_NOTE
]

save(wf, "h3_presentation.json")
validate(load("h3_presentation.json"), "h3_presentation.json")


# ======================================================================
# B. h3_music_video_minimal.json  (from h3_music_video.json)
# ======================================================================
wf = load("h3_music_video.json")
wf["id"] = "9d5e2f80-6c17-4a3b-b2e4-0f8a94c6d357"

# minimal node has no cast sockets: drop H3Characters (157) and link 320
drop_links(wf, [320])
wf["nodes"] = [n for n in wf["nodes"] if n["id"] != 157]

w = node(wf, 163)
w["type"] = "H3MusicVideoMinimal"
w["title"] = "H3 Music Video (Minimal) — lyrics + look + 3 sliders"
w["size"] = [520, 620]
w["properties"] = {"Node name for S&R": "H3MusicVideoMinimal", "cnr_id": "comfyui_dagthomas"}
w["inputs"] = [
    {"name": "audio", "type": "AUDIO", "link": 319},
    {"name": "llm", "type": "APNEXT_LLM", "link": None, "shape": 7},
    {"name": "image_1", "type": "IMAGE", "link": 340, "shape": 7},
    {"name": "image_2", "type": "IMAGE", "link": None, "shape": 7},
]
w["outputs"] = [
    {"name": "scenes", "type": "STRING", "links": [326]},
    {"name": "durations", "type": "FLOAT", "links": None},
    {"name": "lengths", "type": "INT", "links": [323]},
    {"name": "audio_segments", "type": "AUDIO", "links": [324]},
    {"name": "scenes_text", "type": "STRING", "links": [322]},
    {"name": "session_id", "type": "STRING", "links": None},
    {"name": "info", "type": "STRING", "links": None},
] + copy.deepcopy(IMG_OUTS) + [
    {"name": "clip_starts", "type": "FLOAT", "links": None},
]
w["outputs"][7]["links"] = [341, 342]  # image_1 passthrough
# the rebuilt outputs must keep the base's audio_segments -> Create Video
# link (original song slice under every saved clip)
cv = next(n_ for n_ in wf["nodes"] if n_["type"] == "CreateVideo")
al = next((l for l in wf["links"] if l[3] == cv["id"] and l[4] == 1), None)
if al is not None and al[1] == 163:
    w["outputs"][3]["links"] = [324, al[0]]
w["widgets_values"] = [
    "",               # lyrics: the user brings their own
    "Live-action, neon noir: rain-slick night streets, cyan and magenta neon reflections, "
    "volumetric smoke and searchlights, anamorphic blue-streak flares, deep crushed blacks",
    80,   # performance
    30,   # pace
    45,   # wildness
    "sonnet",
    -1, "randomize",  # seed: fresh every queue
    "Ref2VA (bind reference images)",  # prompt_mode
]
# link 322 now leaves output slot 4 (scenes_text)
for l in wf["links"]:
    if l[0] == 322:
        l[2] = 4

# performer photo -> writer image_1 -> video node ref_image_0 (+ preview thumb)
wf["nodes"].append({
    "id": 168,
    "type": "LoadImage",
    "pos": [-1490.0, 5664.3],
    "size": [330, 314],
    "flags": {},
    "order": 7,
    "mode": 0,
    "inputs": [],
    "outputs": [
        {"name": "IMAGE", "type": "IMAGE", "links": [340]},
        {"name": "MASK", "type": "MASK", "links": None},
    ],
    "title": "Performer photo (reference)",
    "properties": {"Node name for S&R": "LoadImage"},
    "widgets_values": ["example.png", "image"],
})
v = node(wf, 136)
v["inputs"][3]["link"] = 343  # ref_images.ref_image_0
p = node(wf, 153)
p["inputs"][1]["link"] = 342
p["widgets_values"] = [""]
p.pop("widgets_values_named", None)
wf["links"] += [
    [340, 168, 0, 163, 2, "IMAGE"],
    [341, 163, 7, 136, 3, "IMAGE"],
    [342, 163, 7, 153, 1, "IMAGE"],
]
# 341 and 343 must match: use 341 on the video input
v["inputs"][3]["link"] = 341

n = node(wf, 161)
n["widgets_values"] = [
    "# H3 Music Video (Minimal)\n\n"
    "The one-box music video: **song + lyrics + a cinematic look + three sliders - go.**\n\n"
    "1. **Load the song** \u2192 `audio` (and `replace_audio` of H3 Scenes Join, so the finished "
    "video carries the original track). Paste the lyrics (`[0:12] line` timestamps = exact "
    "lip-sync).\n"
    "2. Pick a **visual_style** (the curated looks fix style, camera, lenses and colour) and "
    "set the sliders: `performance` (0 story \u2192 100 singer on camera), `pace` (0 slow \u2192 100 "
    "quick cuts), `wildness` (0 grounded \u2192 100 surreal).\n"
    "3. The model invents the concept and the performer; a photo on `image_1` locks the "
    "performer's face and passes through to the video node's `ref_image_0`.\n\n"
    "The full Music Video Writer runs underneath (Auto cutting on the music, lyric-driven "
    "imagery, wardrobe/location locks, saved scene bundles). Use the full writer's workflow "
    "when you need cast lines, locks, briefs or masked audio.\n\n"
    + REVIEW_SEED_NOTE
]

wf["last_node_id"] = 168
wf["last_link_id"] = 342
save(wf, "h3_music_video_minimal.json")
validate(load("h3_music_video_minimal.json"), "h3_music_video_minimal.json")


# ======================================================================
# C. h3_music_video_masked_audio_briefs.json  (from masked audio example)
# ======================================================================
wf = load("h3_music_video_masked_audio.json")
wf["id"] = "4f8a1b3c-9e26-4d75-a1c8-72e05b9d6e43"

BRIEFS = [
    (171, 0, 1,
     "Cold open: a rusted rooftop door bursts outward and Lena steps into the rain, "
     "red umbrella opening, the neon city sprawling behind her.",
     "the rooftop", "Lena", "", "one slow push-in, no cuts"),
    (172, 334, 0,
     "Lena walks the parapet edge like a tightrope, arms out, singing straight into the "
     "lens as rain streaks through the backlight.",
     "the rooftop", "Lena", "", "handheld, close"),
    (173, 335, 0,
     "First chorus: the giant pink billboard flickers on and floods the whole rooftop; "
     "wide shot, Lena tiny against the glow, umbrella tumbling away in the wind.",
     "the rooftop", "Lena", "", "wide, slow crane up"),
]
for i, (nid, in_link, num, desc, loc, cast, pics, cam) in enumerate(BRIEFS):
    wf["nodes"].append({
        "id": nid,
        "type": "H3SceneBrief",
        "pos": [-1851.2, 5560.0 + i * 230.0],
        "size": [330, 210],
        "flags": {},
        "order": 8 + i,
        "mode": 0,
        "inputs": [
            {"name": "brief_in", "type": "STRING", "link": in_link or None, "shape": 7},
        ],
        "outputs": [
            {"name": "briefs", "type": "STRING", "links": [334 + i]},
        ],
        "title": f"Scene Brief {i + 1}" + (" (pinned to scene 01)" if num else ""),
        "properties": {"Node name for S&R": "H3SceneBrief", "cnr_id": "comfyui_dagthomas"},
        "widgets_values": [desc, num, loc, cast, pics, cam],
    })

w = node(wf, 163)
w["inputs"].append({"name": "scene_briefs", "type": "STRING", "link": 336, "shape": 7})
wf["links"] += [
    [334, 171, 0, 172, 0, "STRING"],
    [335, 172, 0, 173, 0, "STRING"],
    [336, 173, 0, 163, 7, "STRING"],
]

n = node(wf, 161)
n["widgets_values"] = [
    n["widgets_values"][0]
    + "\n\n**Scene briefs (custom scenes):** the chained **H3 Scene Brief** nodes are YOUR "
    "plan - brief 1 is pinned to scene 01, the unpinned ones fill the following pieces in "
    "order; every remaining piece stays the model's to invent within the concept. Each "
    "brief is binding: its location, cast and camera wish are honoured, adapted to that "
    "piece's lyric lines, duration and energy."
]

# make room for the brief column next to the note
for g in wf["groups"]:
    if g["title"] == "User Inputs":
        g["bounding"][1] = min(g["bounding"][1], 5500.0)

wf["last_node_id"] = 173
wf["last_link_id"] = 336
save(wf, "h3_music_video_masked_audio_briefs.json")
validate(load("h3_music_video_masked_audio_briefs.json"), "h3_music_video_masked_audio_briefs.json")


# ======================================================================
# D. h3_music_video_dailies_gate.json  (from h3_music_video.json)
#    The queue-stopping Scenes Review is replaced by the LIVE Dailies Gate:
#    the run holds while the user prints / punches up / cuts in the browser.
# ======================================================================
wf = load("h3_music_video.json")
wf["id"] = "6a2d9c47-1e83-4b5f-9d02-c48f7a31b865"

w = node(wf, 163)
w["outputs"][1]["links"] = [329]    # durations -> gate
w["outputs"][10]["links"] = [330]   # session_id -> gate (enables punch-ups)

g = node(wf, 165)
g["type"] = "H3ScenesReviewGate"
g["title"] = "H3 Dailies Gate — print / punch up / cut"
g["size"] = [470, 560]
g["properties"] = {"Node name for S&R": "H3ScenesReviewGate", "cnr_id": "comfyui_dagthomas"}
g["inputs"] = [
    {"name": "scenes", "type": "STRING", "link": 326},
    {"name": "durations", "type": "FLOAT", "link": 329, "shape": 7},
    {"name": "session_id", "type": "STRING", "link": 330, "shape": 7},
    {"name": "llm", "type": "APNEXT_LLM", "link": None, "shape": 7},
]
g["outputs"] = [
    {"name": "scenes", "type": "STRING", "links": [321]},
    {"name": "scene_count", "type": "INT", "links": None},
    {"name": "status", "type": "STRING", "links": None},
]
g["widgets_values"] = [True, 0.0, "sonnet", True, 600, False]
w["widgets_values"][15] = -1            # seed: fresh video every queue
w["widgets_values"][16] = "randomize"   # (the gate never re-queues, so the
                                        # node cache buys nothing here)
wf["links"] += [
    [329, 163, 1, 165, 1, "FLOAT"],
    [330, 163, 10, 165, 2, "STRING"],
]

n = node(wf, 161)
n["widgets_values"] = [
    "# H3 Music Video with the Dailies Gate\n\n"
    "The music-video workflow with a LIVE screening stop: after the writer finishes, the "
    "run HOLDS at the **H3 Dailies Gate** (it does not stop - no re-queueing) and the "
    "takes appear on the node's desk in the browser.\n\n"
    "- **▶ Print it** - render exactly what is on the desk, hand edits included.\n"
    "- **✍ Punch-up** - type director's notes (and optionally which takes, e.g. `2, 4-5`), "
    "and the selected scenes are rewritten inside the writer's own model session - the "
    "synopsis, locks, lyrics and images are still in context. The new takes come back to "
    "the desk for another look; punch up as many rounds as you like.\n"
    "- **✋ Cut** - end the run, render nothing.\n\n"
    "Also on the desk: **🎲 New take** (a noticeably different rewrite, no notes needed), "
    "**↩ Undo** (server-side history of every rewrite, survives a reload), a per-take "
    "view (◀ Take NN ▶ - with the takes field empty, a rewrite targets the take being "
    "viewed), and an optional chime. `durations` keeps rewritten takes on their exact "
    "song-piece lengths; `session_id` is what enables the AI rewrites (without it the "
    "desk is edit-by-hand only). `auto_approve_minutes` > 0 prints automatically for "
    "unattended runs. The writer's seed is **-1 (randomize)**: every queue writes a "
    "brand-new video, nothing is reused from earlier runs."
]

wf["last_link_id"] = 330
save(wf, "h3_music_video_dailies_gate.json")
validate(load("h3_music_video_dailies_gate.json"), "h3_music_video_dailies_gate.json")


# ======================================================================
# E. Dailies Gate variants of the other examples (existing files untouched):
#    the H3ScenesReview node (id 165, scenes in = link 326, scenes out =
#    link 321) is swapped for the live gate, wired to the writer's durations
#    and session_id so punch-ups work.
# ======================================================================
GATE_NOTE = (
    "**Dailies Gate variant:** the run HOLDS live at the gate instead of "
    "stopping - **▶ Print it** renders with your hand edits, **✍ Punch-up** "
    "rewrites selected takes through the writer's own model session using your "
    "notes, **🎲 New take** asks for a different version, **↩ Undo** rolls a "
    "rewrite back, **✋ Cut** ends the run. No re-queueing; `session_id` wired "
    "into the gate is what enables the AI rewrites. The writer's seed is **-1 "
    "(randomize)** here, so EVERY queue writes a brand-new video - nothing is "
    "reused from earlier runs."
)


def swap_in_gate(src, dst, new_uuid, writer_id, durations_slot, session_slot,
                 seed_idx=None):
    wf = load(src)
    wf["id"] = new_uuid
    next_link = max(l[0] for l in wf["links"]) + 1
    dlink, slink = next_link, next_link + 1

    w = node(wf, writer_id)
    w["outputs"][durations_slot]["links"] = list(w["outputs"][durations_slot].get("links") or []) + [dlink]
    w["outputs"][session_slot]["links"] = list(w["outputs"][session_slot].get("links") or []) + [slink]

    g = node(wf, 165)
    g["type"] = "H3ScenesReviewGate"
    g["title"] = "H3 Dailies Gate — print / punch up / cut"
    g["size"] = [470, 560]
    g["properties"] = {"Node name for S&R": "H3ScenesReviewGate", "cnr_id": "comfyui_dagthomas"}
    g["inputs"] = [
        {"name": "scenes", "type": "STRING", "link": 326},
        {"name": "durations", "type": "FLOAT", "link": dlink, "shape": 7},
        {"name": "session_id", "type": "STRING", "link": slink, "shape": 7},
        {"name": "llm", "type": "APNEXT_LLM", "link": None, "shape": 7},
    ]
    g["outputs"] = [
        {"name": "scenes", "type": "STRING", "links": [321]},
        {"name": "scene_count", "type": "INT", "links": None},
        {"name": "status", "type": "STRING", "links": None},
    ]
    g["widgets_values"] = [True, 0.0, "sonnet", True, 600, False]
    wf["links"] += [
        [dlink, writer_id, durations_slot, 165, 1, "FLOAT"],
        [slink, writer_id, session_slot, 165, 2, "STRING"],
    ]
    wf["last_link_id"] = slink

    # the gate never re-queues, so the ComfyUI node cache buys nothing here:
    # randomize the writer's seed so every queue is a fresh video
    if seed_idx is not None:
        w["widgets_values"][seed_idx] = -1
        w["widgets_values"][seed_idx + 1] = "randomize"

    note = node(wf, 161)
    note["widgets_values"] = [note["widgets_values"][0] + "\n\n" + GATE_NOTE]

    save(wf, dst)
    validate(load(dst), dst)


# minimal writer: durations = output 1, session_id = output 5, seed widget 6
swap_in_gate(
    "h3_music_video_minimal.json", "h3_music_video_minimal_dailies_gate.json",
    "0c7e5f92-4ab1-4d38-9e67-21b84cd5a9f3", 163, 1, 5, seed_idx=6,
)
# full music video writer (masked + briefs): durations 1, session 10, seed 15
swap_in_gate(
    "h3_music_video_masked_audio_briefs.json",
    "h3_music_video_masked_audio_briefs_dailies_gate.json",
    "8b1f3a64-72c9-4e05-b3d8-f96a02e47c15", 163, 1, 10, seed_idx=15,
)
# presentation writer: durations 1, session 9, seed 16
swap_in_gate(
    "h3_presentation.json", "h3_presentation_dailies_gate.json",
    "3e9d0b28-56f4-4c71-a2e6-84d17fb0c62a", 163, 1, 9, seed_idx=16,
)

# ======================================================================
# F. Turbo render chain: the speed setup copied VERBATIM from the user's
#    reference render graph (X:\fl2v.json) - turbo LoRA, chunked
#    feed-forward, Sage + LowVRAM + SoL attention, EasyCache, Spectrum,
#    euler @ 4 steps. apply_turbo(wf) swaps it into any workflow that
#    carries the standard render ids (UNET 127, patches 143/141, scheduler
#    124, guider 126, sampler 123). Skipped quietly when the reference file
#    is not on this machine.
# ======================================================================
FL2V_REF = r"X:\fl2v.json"
HAVE_TURBO = os.path.exists(FL2V_REF)
if HAVE_TURBO:
    with open(FL2V_REF, encoding="utf-8") as f:
        _ref = json.load(f)

    def ref_node(type_name):
        return next(n for n in _ref["nodes"] if n["type"] == type_name)

    TURBO_CHAIN = [
        "LoraLoaderModelOnly",
        "MiniMaxChunkFeedForward",
        "MiniMaxH3MemoryEfficientSageAttentionPatch",
        "MiniMaxLowVRAMAttention",
        "SolAttnPatch",
        "EasyCache",
        "SpectrumApplyMiniMaxH3",
    ]

    def apply_turbo(wf):
        drop_links(wf, [287, 288, 284, 285])
        wf["nodes"] = [n for n in wf["nodes"] if n["id"] not in (141, 143)]
        base_id, base_link = 210, 510
        prev = (127, 0)  # UNETLoader MODEL output
        for i, type_name in enumerate(TURBO_CHAIN):
            src = ref_node(type_name)
            nid, lid = base_id + i, base_link + i
            inputs = []
            for inp in src.get("inputs", []):
                entry = {k: inp[k] for k in ("name", "type", "shape") if k in inp}
                entry["link"] = lid if inp.get("name") == "model" else None
                inputs.append(entry)
            new_node = {
                "id": nid,
                "type": type_name,
                "pos": [-1500.0 + (i % 4) * 340, 4270.0 + (i // 4) * 190],
                "size": src.get("size", [300, 130]),
                "flags": {},
                "order": 10 + i,
                "mode": 0,
                "inputs": inputs,
                "outputs": [{"name": "MODEL", "type": "MODEL", "links": []}],
                "properties": src.get("properties", {"Node name for S&R": type_name}),
            }
            for key in ("widgets_values", "widgets_values_named"):
                if key in src:
                    new_node[key] = copy.deepcopy(src[key])
            wf["nodes"].append(new_node)
            node(wf, prev[0])["outputs"][prev[1]]["links"] = (
                list(node(wf, prev[0])["outputs"][prev[1]].get("links") or []) + [lid]
            )
            wf["links"].append([lid, prev[0], prev[1], nid, 0, "MODEL"])
            prev = (nid, 0)

        tail = node(wf, prev[0])
        l_sched, l_guide = base_link + len(TURBO_CHAIN), base_link + len(TURBO_CHAIN) + 1
        tail["outputs"][0]["links"] = [l_sched, l_guide]
        node(wf, 124)["inputs"][0]["link"] = l_sched   # BasicScheduler.model
        node(wf, 126)["inputs"][0]["link"] = l_guide   # BasicGuider.model
        wf["links"] += [
            [l_sched, prev[0], 0, 124, 0, "MODEL"],
            [l_guide, prev[0], 0, 126, 0, "MODEL"],
        ]
        ref_sched = ref_node("BasicScheduler")["widgets_values"]
        node(wf, 124)["widgets_values"] = list(ref_sched)
        node(wf, 124).pop("widgets_values_named", None)
        node(wf, 123)["widgets_values"] = list(ref_node("KSamplerSelect")["widgets_values"])
        node(wf, 123).pop("widgets_values_named", None)
        lora = ref_node("LoraLoaderModelOnly")["widgets_values"][0]
        note = node(wf, 161)
        note["widgets_values"] = [note["widgets_values"][0] + (
            "\n\n**Turbo variant:** the render chain carries the speed setup from the "
            f"reference graph - `{lora}` LoRA, chunked feed-forward, Sage + LowVRAM + SoL "
            f"attention, EasyCache and Spectrum, sampled with euler at {ref_sched[1]} steps. "
            "Needs the packs providing those nodes (Spectrum-MiniMax-H3, the SoL/turbo "
            "patch packs) and the turbo LoRA in models/loras."
        )]

    wf = load("h3_music_video_masked_audio.json")
    wf["id"] = "d4b7e2a9-63f1-48c5-8b2a-97e04c15af38"
    apply_turbo(wf)
    save(wf, "h3_music_video_masked_audio_turbo.json")
    validate(load("h3_music_video_masked_audio_turbo.json"), "h3_music_video_masked_audio_turbo.json")
else:
    print(f"skipped turbo variants: {FL2V_REF} not found")


# ======================================================================
# G. Short film: the Presentation workflow reshaped around the Short Film
#    Writer (identical output layout, so all wiring holds) - manuscript in,
#    scene count OR target length, Claude/Codex adapts the whole film.
#    Plus a turbo-render variant when the reference graph is available.
# ======================================================================
FILM_MANUSCRIPT = (
    "THE LAST DELIVERY\n\n"
    "Night rain over a small harbour town. MAJA (60s, retired postwoman, steel-grey "
    "bun, yellow oilskin coat) finds one undelivered letter from 1987 behind a loose "
    "panel in her old mail van. The address: the lighthouse. She drives the coughing "
    "van up the coast road, headlights cutting the rain. At the lighthouse she meets "
    "ESPEN (70s, the keeper, white beard, coarse wool sweater), who never got the "
    "letter his late wife wrote him the week they argued. Maja hands it over. He "
    "reads it in the lamp room while the beam turns. It says: 'I was never angry. "
    "Come home.' Espen laughs and cries at once. Maja pours coffee from a thermos. "
    "Dawn breaks; the rain stops; the lamp goes dark as the sun takes over.\n"
    "ESPEN: 'Forty years late.'\n"
    "MAJA: 'Post's like that.'"
)

wf = load("h3_presentation.json")
wf["id"] = "5c8f2d71-9a44-4e06-b7d3-18e6a29c04f5"
w = node(wf, 163)
w["type"] = "H3ClaudeCodeShortFilmWriter"
w["title"] = "H3 Short Film Writer"
w["properties"] = {"Node name for S&R": "H3ClaudeCodeShortFilmWriter", "cnr_id": "comfyui_dagthomas"}
w["widgets_values"] = [
    FILM_MANUSCRIPT,
    "Target length (use target_minutes)",
    8,                # scene_count (unused in target mode)
    2.0,              # target_minutes
    "Independent clips (hard cuts, T2V openers)",
    "Live-action, 35mm cinematic film aesthetic",
    "English",
    25,               # wildness
    "sonnet", False, True, True, 1800,
    -1, "randomize",  # seed: fresh every queue
    "",               # extra_cast
    "", "",           # custom_dialogue_language, custom_visual_style
    "", "",           # wardrobe, locations
    True,             # enforce_wardrobe
    "", "",           # extra_instructions, image_notes
    False,            # include_on_screen_text
    True, True,       # include_soundscape, include_non_diegetic_music
    "", "",           # resume_session_id, working_dir
    "Characters only (ignore picture backgrounds)",
    True, 4,          # save_scenes, scenes_per_call
    "Ref2VA (bind reference images)",  # prompt_mode
]
c = node(wf, 157)
c["title"] = "H3 Characters — the lead + wardrobe"
c["widgets_values"] = [
    "\u270f\ufe0f custom (type in custom_character)", "(all)", 0, "fixed",
    "Maja: a retired postwoman in her 60s, weathered kind face, steel-grey hair in a tight bun",
    "bright yellow oilskin coat, chunky knitted mustard scarf, dark rubber boots",
]
n = node(wf, 161)
n["widgets_values"] = [
    "# H3 Short Film\n\n"
    "1. Paste your **manuscript** - a story, treatment, script or synopsis - into the "
    "**H3 Short Film Writer**. It is adapted faithfully: named characters, events and "
    "written dialogue survive verbatim into the scenes.\n"
    "2. Size the film either by **scene count** or by **target length** "
    "(`length_mode`) - in target mode the node derives the scene count (~11 s per "
    "scene) and the model paces each scene's duration so the total lands close to "
    "the target. Long films are written in chunks continuing one session, with a "
    "film-so-far recap, wardrobe/location locks and a full Beats plan in the "
    "synopsis.\n"
    "3. The H3 video node runs once per scene (dialogue, ambience and the score are "
    "generated by the model - that IS the film's sound) and each clip is saved as "
    "its own file; stitch externally or re-add H3 Scenes Join for one file.\n\n"
    "Characters come from H3 Characters nodes or the manuscript itself; a face photo "
    "on `image_1..` locks a lead's identity (REF prompt mode). The `script` output "
    "is the film's dialogue as a script - read it before rendering.\n\n"
    + REVIEW_SEED_NOTE
]
save(wf, "h3_short_film.json")
validate(load("h3_short_film.json"), "h3_short_film.json")

if HAVE_TURBO:
    wf = load("h3_short_film.json")
    wf["id"] = "7e3a9c50-2b16-4f88-a4d1-c95e60b823d7"
    apply_turbo(wf)
    save(wf, "h3_short_film_turbo.json")
    validate(load("h3_short_film_turbo.json"), "h3_short_film_turbo.json")

print("ALL-WORKFLOWS-OK")
