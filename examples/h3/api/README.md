# H3 workflows as APIs

Everything in this folder turns the example workflows into HTTP calls, so you can
build your own front-end on top of a running ComfyUI server.

## Files

| File | What it is |
|---|---|
| `*.api.json` | The example workflows in ComfyUI **API format** — the exact body for `POST /prompt` (`{"prompt": <this>, "client_id": ...}`). Every widget is a named input you can patch before queueing. |
| `h3_client.ts` | Dependency-free TypeScript reference client (browser or Node ≥ 22): upload → patch → queue → stream progress → answer the Dailies Gate → collect clips. |

Exports included: `h3_music_video`, `h3_music_video_minimal`, `h3_presentation`,
plus their `_dailies_gate` variants. (The masked-audio workflows use third-party
nodes whose input names we can't verify offline — export those from the ComfyUI
UI with dev-mode **Save (API Format)** if you need them.)

## The two flavours

- **Plain exports** ship with the Scenes Review node forced to **Bypass** — a
  headless run goes writer → render → save with no stops. What you patch is what
  you get.
- **`_dailies_gate` exports** pause server-side and emit a
  `apnext.h3.review_gate` event on the same `/ws` socket, carrying the scenes as
  text. Your front-end shows its own review screen and POSTs the decision to
  `/apnext/h3/review_gate` (`approve` / `reroll` (+`feedback`, `scenes`,
  `variant`) / `undo` / `stop`); `GET /apnext/h3/review_gate/pending` re-attaches
  after a reload. Set `auto_approve_minutes` on the gate as a safety net for
  unattended runs.

## Typical patches

```ts
H3Client.patchByClass(wf, "LoadAudio",  { audio: await client.upload(song) });
H3Client.patchByClass(wf, "LoadImage",  { image: await client.upload(photo) }); // minimal wf
H3Client.patchByClass(wf, "H3Characters", {
  character: "✏️ custom (type in custom_character)",
  custom_character: "Lena: a singer in her early 30s, platinum pixie cut",
  wardrobe: "red leather biker jacket, white ribbed tank top",
});
H3Client.patchByClass(wf, "H3ClaudeCodeMusicVideoWriter", {
  lyrics: "[0:12] ...", visual_style: "Live-action, neon noir: ...",
  wildness: 45, seed: -1, prompt_mode: "Ref2VA (bind reference images)",
});
```

Reference images by node id instead of class when you use several: the writers'
`image_1..image_9` inputs take `["<LoadImage node id>", 0]` pairs — or wire an
**H3 Characters** node's new `image` pass-through (`image` in → `image` out), so
each character and their face photo travel as one unit: `cast` → `cast_N`,
`image` → `image_N`.

## Outputs

Each scene's clip is saved as its own file as it renders (direct save — no join
step). `GET /history/{prompt_id}` lists them; `GET /view?filename=...` downloads.
The `h3_client.ts` `run()` helper returns them as ready URLs. Progress, the Song
Analysis readout and node errors all arrive on `/ws`.

Regenerate the exports after changing the canvas examples with
[`scripts/make_api_exports.py`](../../../scripts/make_api_exports.py) — it maps
widgets to named inputs using the live node definitions
(`scripts/make_workflows.py` regenerates the canvas examples themselves, and
`scripts/check_widgets.py` validates widget layouts against the nodes).
