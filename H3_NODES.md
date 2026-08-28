# APNext H3 nodes — the complete guide

Everything in this pack that touches **MiniMax‑H3** video prompting: how each node works, what plugs into what, and which example workflow shows it. All nodes live under **APNext → H3** in the node search and share one colour language on the canvas (see [Canvas helpers](#canvas-helpers-and-the-graphgen-theme)).

> Workflows referenced below are in [`examples/h3/`](examples/h3/). Drag a `.json` onto the ComfyUI canvas to load it.

---

## Table of contents

1. [The big picture](#the-big-picture)
2. [Shared concepts](#shared-concepts)
   - [Which model writes: Claude Code, the `llm` socket, the router](#which-model-writes)
   - [Template variables `{character1}` …](#template-variables)
   - [Wardrobe and location locks](#wardrobe-and-location-locks)
   - [Context sockets `context_1..8`](#context-sockets)
   - [Reference images](#reference-images)
   - [Sessions: `session_id` / `resume_session_id`](#sessions)
   - [Lists: one element per scene](#lists-one-element-per-scene)
3. [Writers (single prompt)](#writers-single-prompt)
   - [APNext H3 Prompt Writer](#apnext-h3-prompt-writer)
   - [APNext H3 Reference Prompt Writer](#apnext-h3-reference-prompt-writer)
   - [APNext H3 Claude Code Writer](#apnext-h3-claude-code-writer)
   - [APNext H3 Claude Code Reference Writer](#apnext-h3-claude-code-reference-writer)
   - [APNext H3 Claude Code Continue Writer](#apnext-h3-claude-code-continue-writer)
   - [APNext H3 Claude Code Refiner](#apnext-h3-claude-code-refiner)
4. [Writers (many scenes)](#writers-many-scenes)
   - [APNext H3 Claude Code Scenes Writer](#apnext-h3-claude-code-scenes-writer)
   - [APNext H3 Crossover Writer](#apnext-h3-crossover-writer)
   - [APNext H3 Music Video Writer](#apnext-h3-music-video-writer)
   - [APNext H3 Music Video (Minimal)](#apnext-h3-music-video-minimal)
   - [APNext H3 Presentation Writer](#apnext-h3-presentation-writer)
   - [APNext H3 Short Film Writer](#apnext-h3-short-film-writer)
5. [Cast and model](#cast-and-model)
   - [APNext H3 Characters](#apnext-h3-characters)
   - [APNext H3 LLM Backend](#apnext-h3-llm-backend-ollama--local--api)
6. [Scene utilities](#scene-utilities)
   - [APNext H3 Scene Pick](#apnext-h3-scene-pick)
   - [APNext H3 Scene Counter](#apnext-h3-scene-counter)
   - [APNext H3 Scenes Join](#apnext-h3-scenes-join)
   - [APNext H3 Scenes → Contex Loop Plan](#apnext-h3-scenes--contex-loop-plan)
   - [APNext H3 Refine Encode + Mouth Guard (face‑refine second pass)](#apnext-h3-refine-encode--mouth-guard-facerefine-second-pass)
   - [APNext H3 Manual Scenes (script → lists)](#apnext-h3-manual-scenes-script--lists)
   - [APNext H3 Dailies Gate (print / punch up / cut)](#apnext-h3-dailies-gate-print--punch-up--cut)
   - [APNext H3 Song Analysis (BPM / intensity)](#apnext-h3-song-analysis-bpm--intensity)
   - [APNext H3 Sound Events (bass hits / drops / stops)](#apnext-h3-sound-events-bass-hits--drops--stops)
   - [APNext H3 Cut Plan (scenes from the music)](#apnext-h3-cut-plan-scenes-from-the-music)
   - [APNext H3 Sync Check (does the picture hit the beat?)](#apnext-h3-sync-check-does-the-picture-hit-the-beat)
   - [APNext H3 Music Video Chain Render (carry frames between scenes)](#apnext-h3-music-video-chain-render-carry-frames-between-scenes)
   - [APNext H3 Short Film Chain Render (carry picture + sound between scenes)](#apnext-h3-short-film-chain-render-carry-picture--sound-between-scenes)
   - [APNext H3 Scene Retake (render one scene again)](#apnext-h3-scene-retake-render-one-scene-again)
   - [APNext H3 Resolution Planner (Crop Only)](#apnext-h3-resolution-planner-crop-only)
7. [Viewing](#viewing)
   - [APNext H3 Prompt Preview](#apnext-h3-prompt-preview)
8. [Canvas helpers and the APNext theme](#canvas-helpers-and-the-apnext-theme)
9. [Workflow index](#workflow-index)
10. [Files and where to tune things](#files-and-where-to-tune-things)

---

## The big picture

```
                       ┌──────────────────────┐
   idea / images ─────►│  a WRITER            │── h3_prompt ──────────────► MiniMax H3 video node (prompt)
   H3 Characters ─────►│  (single or scenes)  │── scenes / durations ─────► … runs once per scene …
   context nodes ─────►│                      │── session_id ─────────────► Refiner / Continue Writer
   H3 LLM Backend ────►│  llm                 │── image_1..9 passthrough ─► video node ref_image_N
                       └──────────────────────┘
                                 │
                                 └── scenes_text / h3_prompt ───► H3 Prompt Preview (colour-coded, thumbnails)

   per-scene clips ──► H3 Scenes Join ──► Create Video ──► Save Video   (one file)
```

Every writer turns a short **idea** (plus optional images, a cast, context and locks) into a complete, spec‑compliant MiniMax‑H3 prompt — the official MiniMax writing guides in `data/h3/` are the system prompt, so the model follows the real spec. The *Claude Code* writers do it through your local Claude Code CLI (or any model through the `llm` socket); the two plain writers go straight to an API / local server through the model dropdown.

---

## Shared concepts

### Simple form vs. advanced inputs

Every H3 node opens in a **short form** — the inputs you touch on every run (idea / direction, duration, style, dialogue language, model, seed, the cast / wardrobe / location boxes, images, `llm` and context sockets). Everything else (shot plan, wildness, camera motion, soundscape / music toggles, `research`, `director`, `use_subscription`, `timeout_seconds`, custom style / language, `resume_session_id`, `working_dir`, typed reference sockets, backend tuning …) is flagged as an *advanced* input and sits behind the node's **Show advanced inputs** toggle (ComfyUI's own mechanism, `Settings → Lite Graph → Node → Always show advanced widgets` shows them everywhere). The lists live in `nodes/h3/__init__.py` (`with_advanced_inputs`).

### Which model writes

| Node family | Default engine | Switch to another model |
|---|---|---|
| **Claude Code** writers (Writer, Reference Writer, Continue, Refiner, Scenes, Crossover, Music Video, Presentation) | Claude Code CLI, your own login (`model` = `sonnet` / `opus` / `haiku` / `fable` / `default`) | Pick **`codex`** in the `model` dropdown to write with the **OpenAI Codex CLI** instead (shown when installed — `npm i -g @openai/codex`, `codex login`; `codex:<model-id>` picks a specific model via an LLM Backend / `model_override`). Or connect an **[H3 LLM Backend](#apnext-h3-llm-backend-ollama--local--api)** to the `llm` socket, or pick a discovered `ollama:` / `lmstudio:` / `local:` entry in the dropdown |
| Plain writers (Prompt Writer, Reference Prompt Writer) | `auto-detect` (first API key of Claude → GPT → Gemini → Grok → Groq, then Claude Code CLI, then a running local server) | `model` dropdown, `model_override` (`ollama:qwen3:8b`), `local_base_url` |

The Claude Code block on every Claude Code node: `model`, `research` (web research before writing — agent CLIs only; on Codex it enables the web-search tool), `director` (loads the H3 director skills from `data/h3/skills` with their reference libraries — both CLIs read them from disk), `use_subscription` (hide the API key so the CLI bills your seat: `ANTHROPIC_API_KEY` for Claude Code, `OPENAI_API_KEY` for Codex), `timeout_seconds`, `seed`, and optional `resume_session_id` / `working_dir`. Sessions stick to their backend: Claude Code ids resume with Claude Code, `codex-` ids with Codex, `local-` ids with the same local model — feeding one to the wrong backend gives a clear error.

When a node runs **off‑CLI** (Ollama etc.): `research` is ignored; the director skills are pasted into the system prompt (turn on the backend's `inline_skill_references` to paste the whole reference library too — much better prompts, but it pushes the system prompt past 45k tokens, so raise the backend's `num_ctx` to match); `session_id` still works through a text‑only local session kept under ComfyUI's temp folder. A Claude Code session id cannot be resumed with a local model and vice versa — the error says so.

### Template variables

Every free‑text box of every writer (`idea`, `direction`, `lyrics`, `extra_instructions`, `wardrobe`, `locations`, `image_notes`, `extra_cast`, `instruction`, custom style / language) expands `{variables}` from what is wired into the node **before** the text is sent to the model:

| Variable | Value |
|---|---|
| `{character1}` `{actor1}` `{franchise1}` / `{show1}` `{cast1}` `{wardrobe1}` | the 1st [H3 Characters](#apnext-h3-characters) node feeding this node |
| `{character2}` … | the 2nd, and so on — numbered in socket order (`cast_1..4` first, then `context_1..8`); a chain `A → cast_in → B → cast_1` gives A = 1, B = 2 |
| `{characters}` | “A, B and C” |
| `{cast}` | every cast line, one per line |
| `{context_1}` / `{context1}`, `{cast_1}` … | the raw text on that socket |

A read‑only **`{vars}`** strip under each writer lists what is currently available (click a chip to copy); typing `{` in any multiline box opens an **autocomplete** of those variables (filter, ↑/↓, Enter/Tab/click, Esc). Unknown `{names}` are left alone. Implementation: `nodes/h3/template_vars.py`, `web/js/h3_template_vars.js`.

### Wardrobe and location locks

Multi‑scene writers (Scenes, Crossover, Music Video, Presentation) fix continuity mechanically:

- **Wardrobe lock** — one line per character, `Name: anchor, anchor, anchor` (exact colour + material + garment, accessories with their side). The writer copies the anchors word‑for‑word into *every shot* the character is on screen in. Empty = the model fixes an outfit per character in the synopsis’s `Wardrobe:` lines itself. A wardrobe typed on an **H3 Characters** node travels with its cast line and is merged in automatically (writer‑typed lines win for the same name).
- **Location lock** — one line per recurring place, `Name: anchor, anchor …` (fixed features with colour/material and LEFT/RIGHT position, openings, surfaces, practical light). Restated in the first shot of every scene set there, so the room looks the same in every scene. Empty = the model locks every place used in more than one scene in the synopsis’s `Locations:` lines.
- **`enforce_wardrobe`** — after writing, the node parses the synopsis locks and checks every shot (wardrobe) and every scene (locations); any miss triggers **one combined repair turn** in the same session. The result lands in `info` (`wardrobe: ok (2 locked) | locations: repaired 3 -> 0`) and the console lists each miss. Code: `nodes/h3/scenes_support.py` (`enforce_continuity`).

### Context sockets

`context_1..8` accept the `prompt` output of other APNext nodes (Time, Scene, Poses, Plots, Feelings, Cinematic, Science, Geography, Architecture, Fashion …) or any STRING. The writer detects the kind (through the graph) and tells the model how to use it — emotional tone is expressed through performance and light, time of day through the sky, and so on. Sockets grow as you connect them. Guidance text: `utils/apnext_context.py` → `CATEGORY_GUIDANCE`.

### Reference images

- `image` (single writers): the keyframe(s) the video model actually gets — I2VA first frame, L2VA last frame, FL2VA both; handed back out as `first_frame` / `last_frame`.
- `subject_1..3`, `scenery_1..3`, `object_1..3` (Claude Code Writer / Scenes Writer): **typed references** the model describes in words; the video model never sees them.
- `image_1..9` (Reference writers, Crossover, Music Video, Presentation, Short Film): `<Picture 1>..<Picture N>` in connection order; downscaled copies go to the writing model, and the tensors on the matching `image_N` outputs are fitted **the way the released reference pipeline fits them: short edge to 2048, sides rounded to /32 — upscaling included**. ComfyUI's own video node only ever *downscales* (`min(1.0, 2048/short_edge)`), so a small reference (e.g. 512 px) would reach the DiT with **16× fewer reference latent rows** than the model was trained to see — identity fidelity is the whole job of a reference image. Our pass‑through does the resize first, making the stock node's own scaling a no‑op. Note the cost: reference rows ride through every sampling step, so upscaled references render slower — the console line shows the final size before you commit. Aspects outside the pipeline's 1:4..4:1 limit get a warning. `image_notes` (`Image 1: the singer`) tells the model what each picture is.
- **`prompt_mode` REF / FL** (Music Video, Minimal, Presentation): which official guide the scenes are written against. **Ref2VA** (the default, and the behaviour these writers always had) uses `guide_ref_en.md` and binds the attached pictures as `<Picture N>` — for reference images of people. **FL / T2VA** uses `guide_base_en.md` and creates everything from scratch in words — pictures are ignored, no `<Picture N>` labels (the music writer keeps the ref guide only when the song pieces ride as `<Audio 1>` reference audio). **Auto** picks Ref when images are connected, FL otherwise. An empty value (a workflow saved before the switch existed) runs as Ref.

### Sessions

Every Claude Code node returns a `session_id`. Feed it to the **Refiner** (`session_id`) or the **Continue Writer** / any writer (`resume_session_id`) and the next turn continues the same conversation — the guide, the images and the previous prompt are already in context, so a revision costs a short turn instead of a re‑send. Works for Claude Code sessions and (text‑only) for local‑model sessions.

### Lists: one element per scene

The multi‑scene writers output `scenes` and `durations` (and, for the music video, `lengths` and `audio_segments`) as **ComfyUI lists**. A video node downstream therefore runs once per element — one queue renders every scene. Use **Scene Pick** to collapse the list to one scene, **Scenes Join** to stitch the rendered clips into one video, or **Scenes → Contex Loop Plan** to hand the scenes to the Contex‑Loop chain for true scene‑to‑scene continuity.

### Project names

Every multi‑scene writer (Scenes, Crossover, Music Video, Music Video Minimal, Presentation, Short Film) carries a `project_name` widget — auto‑filled with a cinematography‑flavoured tag like `NeonDollyFoley-7k3q` when the node is created, and freely editable. The tag follows the seed: whenever the `seed` widget changes (a hand edit, or `control_after_generate: randomize` after a queue) an auto‑generated name is swapped for a new one, so every seed gets its own project folder; a name you typed yourself is left alone. A seed ≥ 0 always maps to the same name, in the UI and in headless/API runs alike. The name comes back out on the writer's `project_name` output; the example workflows wire it into **Save Video**'s `filename_prefix`, so every clip a run produces is named after its project and the output folder shows at a glance which videos belong together. Saved scene bundles (`save_scenes` → `output/apnext_scenes/`) carry the name too — in the bundle and in its filename — and **Scenes Load** returns it on its own `project_name` output so a re‑render keeps the tag. An empty widget generates a name each run from the seed (random when `seed` is -1; fixed seeds re‑run into the same project). The name is three words from pools of 40 × 30 × 30 plus a 4‑character base‑36 tail drawn from the same seed stream (~60 billion combinations), so two seeds never share a folder; the front‑end and the Python side derive the same name from the same seed.

---

## Writers (single prompt)

### APNext H3 Prompt Writer
`H3BasePromptWriter` · provider‑agnostic (dropdown) · workflow: [`h3_context_ollama_qwen.json`](examples/h3/h3_context_ollama_qwen.json)

Expands an idea (and/or an image) into a full **base‑format** H3 prompt: `integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music`.

- `task_type`: T2VA (text only), I2VA (first frame), FL2VA (first + last), L2VA (last frame) — decides what the `image` batch means and writes the required image‑alignment instruction line.
- `duration_seconds`, `shot_plan` (Auto or an exact count), `visual_style` (guide styles + the APNext Cinematic vocabulary; *Custom* uses `custom_visual_style`), `wildness` 0–100 (above 40 seeds surreal events; `seed` picks them), camera `motion` / `amplitude` / `speed`, dialogue on/off + `dialogue_language` (*Custom* → `custom_dialogue_language`), on‑screen text, soundscape, music toggles.
- `model` / `temperature` / `seed`; optional `model_override`, `local_base_url`; `context_1..8`.
- Outputs the whole prompt plus each section, `model_used`, and the `first_frame` / `last_frame` tensors for the video node.

### APNext H3 Reference Prompt Writer
`H3RefPromptWriter` · provider‑agnostic

Same as above but in the **Ref2VA** format (`subject_definitions` / `summary` / `retention_analysis` / `detailed_description` / `overall_soundscape` / `non_diegetic_music`) for *MiniMax H3 Reference to Video*. Extra inputs: `reference_role` (Auto / Subject / Picture / Style reference / Storyboard) — how the attached pictures are to be used; `word_target`; `reference_notes`; `image_1..9` in → `image_1..9` passthrough out.

### APNext H3 Claude Code Writer
`H3ClaudeCodeBaseWriter` · Claude Code / `llm` · workflows: [`h3_context_claude_code.json`](examples/h3/h3_context_claude_code.json), [`h3_llm_backend_writer_refiner.json`](examples/h3/h3_llm_backend_writer_refiner.json)

The base‑format writer on the Claude Code CLI: same creative inputs as the Prompt Writer, plus the Claude Code block (`research`, `director`, `use_subscription`, `timeout_seconds`), typed references (`subject_*`, `scenery_*`, `object_*`), `resume_session_id`, `working_dir` (a folder the CLI may read — a script, a shot list), `llm`. Returns `session_id` for the Refiner and `info` (model, time, turns, cost, session).

### APNext H3 Claude Code Reference Writer
`H3ClaudeCodeRefWriter` · Claude Code / `llm`

The Ref2VA writer on Claude Code: `reference_role`, `word_target`, `reference_notes`, `image_1..9` in/out, plus the Claude Code block and `llm`. Its `director` skills are `h3-prompt-director`, `h3-ref2va`, `h3-style-craft`.

### APNext H3 Claude Code Continue Writer
`H3ClaudeCodeContinueWriter` · Claude Code / `llm`

Writes the **next clip** from the frames of the previous one. Feed it the rendered `frames`; it samples `frame_count` of them every `frame_stride` frames, shows them to the model and writes a continuation:

- `continuation_mode`: **I2VA** — the last frame becomes the new first frame (hand `first_frame` out to the video node); **T2VA** — a new scene, the frames are context only.
- `previous_prompt` (or `resume_session_id` so the writer already knows it), `idea` for what happens next, the usual creative controls, `context_1..8`, `llm`.
- Outputs the prompt and sections, `session_id`, `first_frame` (the frame to start from) and `context_frames` (the sampled strip, for a preview).

### APNext H3 Claude Code Refiner
`H3ClaudeCodeRefiner` · Claude Code / `llm` · workflow: [`h3_llm_backend_writer_refiner.json`](examples/h3/h3_llm_backend_writer_refiner.json)

Revises an existing prompt (base or Ref2VA — it detects the format) with a plain‑language `instruction` (“make the dialogue sharper, keep every timestamp”). With the writer’s `session_id` the model still has the guide, the images and the draft in context; without it the prompt is re‑sent. Optional `image` to introduce a new reference. Outputs the revised prompt and every section label either format can produce, plus the (new or continued) `session_id`.

---

## Writers (many scenes)

### APNext H3 Claude Code Scenes Writer
`H3ClaudeCodeScenesWriter` · Claude Code / `llm` · workflows: [`h3_scenes_batch.json`](examples/h3/h3_scenes_batch.json), [`h3_scenes_pick_one.json`](examples/h3/h3_scenes_pick_one.json)

One idea → **1–10 consecutive scenes** forming one story, each a complete base‑format T2VA prompt, with a hand‑off between adjacent scenes.

- `scene_count`, `duration_mode` (*Fixed* = every scene `scene_duration`; *Vary 5–15 s* lets the model pace each scene) , `continuity_mode` (*Independent clips* — hard cuts, `already speaking` openers; *Continuous chain* — written for C2V / motion‑context chaining: each scene opens on the previous last frame, one continuous take, a 2 s silent hand‑off before a new speaker), the single‑writer creative controls, typed references, `context_1..8`.
- `wardrobe`, `locations`, `enforce_wardrobe` — see [locks](#wardrobe-and-location-locks).
- Outputs `scenes` **(list)**, `durations` **(list)**, `scenes_text` (all scenes in `=== SCENE NN | duration: S.S ===` envelopes — wire to the Preview), `synopsis`, `scene_count`, `session_id`, `info`.

### APNext H3 Crossover Writer
`H3ClaudeCodeCrossoverWriter` · Claude Code / `llm` · workflows: [`h3_crossover_batch.json`](examples/h3/h3_crossover_batch.json), [`h3_crossover_pick_one.json`](examples/h3/h3_crossover_pick_one.json), [`h3_crossover_contex_chain.json`](examples/h3/h3_crossover_contex_chain.json), [`h3_llm_backend_crossover.json`](examples/h3/h3_llm_backend_crossover.json)

A **cast** (from H3 Characters nodes on `cast_1..4`, or typed in `extra_cast`) + your `direction` → 1–10 crossover scenes in which characters from different shows share one story, each a four‑section T2VA prompt (`subject_definitions` / `integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music`). The rules come from `data/h3/guide_crossover_en.md`, distilled from rendered crossover productions: actor pinning, `<Subject 1>` speaker binding, silence mandates, `not in frame` isolation, positioned two‑shots, no dead air, grounded entrances, hand‑offs between scenes. No title cards.

- Cast lines are used verbatim in `subject_definitions`; `{character1}` … work in every text box.
- `shots_per_scene`, `visual_style`, `dialogue_language`, `wildness`, `duration_mode` / `scene_duration`, `continuity_mode` as on the Scenes Writer.
- `image_1..9` reference pictures (`image_notes` names them) — pictured characters are bound to their `<Picture k>` and their wardrobe lock is taken from the picture; the originals pass through to `image_1..9` outputs.
- Locks and `enforce_wardrobe`; `session_id` / `info`.
- Outputs `scenes` **(list)**, `durations` **(list)**, `scenes_text`, `synopsis`, `cast` (merged), `scene_count`, `session_id`, `info`, `image_1..9`.

### APNext H3 Music Video Writer
`H3ClaudeCodeMusicVideoWriter` · Claude Code / `llm` · workflows: [`h3_music_video.json`](examples/h3/h3_music_video.json), [`h3_music_video_masked_audio.json`](examples/h3/h3_music_video_masked_audio.json), [`h3_music_video_masked_audio_briefs.json`](examples/h3/h3_music_video_masked_audio_briefs.json) (masked latent + Scene Briefs), [`h3_music_video_masked_audio_ollama.json`](examples/h3/h3_music_video_masked_audio_ollama.json) / [`h3_music_video_masked_audio_ollama_blindref.json`](examples/h3/h3_music_video_masked_audio_ollama_blindref.json) / [`h3_music_video_masked_audio_ollama_textonly.json`](examples/h3/h3_music_video_masked_audio_ollama_textonly.json) (written locally by Ollama)

**`Lyric map` + imagination.** Before anything is staged the writer reads the *whole* lyric: the synopsis block now opens with a `Lyric map:` — one line per section (`Verse 1 (0:00-0:18): what it literally says | what it is really about | the strongest image it hands you | how it feels`), read as a whole song (who speaks to whom, what changes between the first verse and the last chorus, where the turn is) — and the Concept and the SCENE PLAN have to grow out of that map, so every piece stages *its* lines' meaning. A `BE IMAGINATIVE` directive then demands one specific, striking image per scene built from this lyric — stock music‑video staging (a singer standing in a room, walking down a street, sitting on a bed, rain on a window, a car at night, staring into the middle distance) is banned unless the lyric literally says so, scale and element vary across the plan, and no two scenes share their image. Under *Literal* the same demand arrives as `LYRICS TO THE LETTER, IMAGINATIVELY`: the map's image column is the picture of the words, every scene plan row quotes the line it stages, no metaphor, no substitute image — but the Concept still decides the *world* those words happen in (era, place, people, scale, palette), the same stock staging is banned, and every line has to land in a place, at a scale and in a light the lyric did not hand you. *Literal* means the sung noun is in frame; it never means a singer in a room. The lock parser stops at `Lyric map:` / `Concept:` / `Motifs:` headers, so the new section can never be read as a wardrobe or location lock.

**`lyric_interpretation`** — how far the pictures may stray from the words. The lyric is always audible and sung as `performance_mode` says; this decides what the *picture* does with it. **Auto** (default) leaves it to the writer, who picks a reading for the song. **Literal** shows what the words say. **Loose** stages the feeling and the situation behind each line, never its nouns. **Metaphor** builds the whole video on one image system that is not the lyric (named in the synopsis Concept so every chunk writer stages the same one). **Counterpoint** tells a different story — other people, place, events — that rhymes with the song's structure. **Reframe** keeps what the song is about but moves it into another world, era or genre. **Surreal** is escalating dream logic, described physically, never with the word "dreamlike". **Surprise me** picks one of the non‑literal readings by seed, so re‑running the same song gives a different video (the log says `🎭 lyric interpretation: Metaphor (picked by seed)`). With `scenes_from_lyrics` on, the lyrics still decide the cuts; the interpretation decides the pictures. The user's `direction` always wins on anything it states outright.

Turns a **song** into a whole music video:

1. **Cuts the audio** into consecutive pieces no longer than H3 renders in one clip (`min_segment_seconds`–`max_segment_seconds`, 5–15 s). `segment_mode` *Auto* cuts on the music (spectral‑flux onsets, energy steps, section changes — and, with timed lyrics, right before a lyric line), *Fixed* takes the longest allowed piece every time, *Lyric lines* tries hardest to cut before a line. Every piece length is snapped to H3’s frame grid (5 + 17k frames at 24 fps) so each rendered clip is exactly as long as its audio slice, and the clip timeline **is** the frame counts: each scene starts exactly where the previous scene’s frames end and its length is chosen from that start towards the planned cut, so the audio slices tile the song with no skip or repeat at the seams, `clip_starts` is the stitched video’s own timeline (the masked latent lines up with the picture), and a plan whose times were rounded to 10 ms in the cut‑plan text cannot drift — the stitched video never drifts.
2. **Measures the sound** — tempo by autocorrelating the onset envelope (`~124 BPM`), plus an aggression profile from transient spikes and loudness (intensity 0–100: gentle → laid‑back → mid‑energy → driving → aggressive, and the dynamics spread in dB). The profile is shown in the console / `segment_table` and steers the writing: a soft song gets long tender takes and breathing cuts, an aggressive one gets hard on‑the‑beat cuts, bold camera and stark light — and cuts, gestures and choreography are timed to the measured bar length.
3. **Writes one scene per piece** (four‑section prompt) — with a word budget for the **picture** (place, framing, action; ~30 words per second) and the timed sync sentences for the piece's beats, hits and drops written **on top** of it, never instead of it (the `📏` length report counts picture words only): the piece is `<Audio 1>`, reused 1:1 as the clip’s soundtrack; `performance_mode` *Performance* has the singer lip‑sync the piece’s lyric lines on camera (`<Subject 1> sings <d>[English] exact line</d> in sync with <Audio 1>`), *Narrative* answers the lyric with pictures, *Mixed* alternates; quiet pieces get long intimate shots, loud/peak pieces more cuts and the chorus look. The story, setting and ending come from the lyrics and your `direction` only - the node has no `wildness` dial and injects no suggestion pools of its own (workflows saved with the old `wildness` widget still load: the stale value is dropped on load and gone on the next save) - and stock clichés; the song's **form is measured** (`nodes/h3/song_structure.py`: beats, tempo checked against half/double/triplet, bars where a downbeat is clear, sections from chroma + timbre self‑similarity, and which sections *repeat* - the most‑repeated, hardest‑hitting group is the chorus) so every piece is tagged `section chorus 2` / `verse 1` / `bridge` even when the lyrics carry no tags, the cuts prefer section starts and downbeats, and the plan sees `SONG STRUCTURE: intro 0:00 · chorus 1 0:29 · verse 1 0:57 …`. Anything the measurement is not sure of is withheld rather than guessed: unclear repeats give unnamed sections, an unclear accent gives no bars (outer‑space imagery, camera‑into‑the‑mouth body journeys, walking‑away endings) are banned unless the concept asks. Long songs are written in chunks — by default **in parallel** (`parallel_chunks`, on): one planning call by `model` fixes the synopsis, the locks and a per‑scene plan, then up to 4 chunks at a time are drafted from that plan by `draft_model` (default **haiku** — several times faster than sonnet at drafting), and one continuity pass by `model` repairs any drift. Turn `parallel_chunks` off for the classic serial run where every chunk continues one session (`draft_model` still drafts chunks 2+ there; `same as model` disables the split entirely). In the serial run every chunk after the first is written against a **story‑so‑far ledger** — one line per finished scene (its place, staging and opening framing, with the style sentence stripped), the previous scene called out — and the prompt treats that ledger as a do‑not‑repeat list, so each new scene has to move somewhere else, open differently and advance the story; parallel chunks get the same rule against the other rows of the scene plan.
4. Emits matching **lists**: `scenes` → the video node’s `prompt`; `lengths` (frame counts) → `length`; `audio_segments` → `ref_audio_1`; `durations`; plus `segment_table` (`01  0:00.00 – 0:15.08  (15.08s, 362 frames)  energy: peak  lyrics: …`), `scenes_text`, `synopsis`, `cast`, `scene_count`, `song_seconds`, `session_id`, `info`, `image_1..9`.

Inputs: `continuity` (last widget — the same three choices as Chain Render: *cut plan decides* continues the take over soft cuts, *flow everywhere* is one take, **cut everywhere** writes every scene as its own clip on a hard cut, a fresh setup each time; set Chain Render's `continuity` to match), `audio` (Load Audio), `direction` (the concept), `lyrics` (`[0:15] line`, `0:15 line`, LRC `[00:15.20] line` or a range `0:15 - 0:18 line` for exact sync; `[Chorus]` tags kept; untimed lines spread evenly), `beat_grid` (a socket for [Beat Grid](#apnext-h3-beat-grid-bpm--every-beat)'s `grid_json`: its phase‑fitted beats, downbeats and BPM then replace the writer's own measurement in every piece's `[beat]` line, so the briefs, the Cut Plan and Beat Emphasis all work from one grid), `lyrics_in` (a socket for lyrics from another node — [Lyrics Transcribe](#apnext-h3-lyrics-transcribe-timed-lyrics-from-the-song)'s `lyrics`; used only while the `lyrics` box is empty, so typed corrections always win and the box never locks; both empty = transcribe with Whisper (`transcribe_lyrics`), else instrumental), `dialogue_language` (lyric language), cast (`cast_1..4` / `extra_cast` — an H3 Characters node in ✏️ custom mode with a `wardrobe` is made for the performer), locks, images, the Claude Code block, `llm`, and `sound_events` (an [H3 Sound Events](#apnext-h3-sound-events-bass-hits--drops--stops) node — every piece's brief then lists the bass hits, drops and stops inside that clip, timed from its own start, and the scenes are staged on them). Finish with **Scenes Join** (`replace_audio` = the song) → Create Video → Save Video. Code: `nodes/h3/claude_code_music_video_writer.py`, `nodes/h3/music_support.py`.

**On the beat.** Each piece's brief now carries a `[beat] every 0.469 s at +0.00 +0.47 +0.94 ... s` line (from the measured structure) next to its `[bars] downbeats` line, so the model times the action to the pulse from the piece's own first frame; when the piece opens on a beat it says so — *the scene OPENS on the beat: its first frame is the beat, start the action there*. With a beat‑exact cut plan (Cut Plan `beat_snap`) every piece does, and the writer keeps the scene lengths frame‑exact (it logs this; render with Chain Render).

### APNext H3 Music Video (Minimal)
`H3MusicVideoMinimal` · Claude Code / Codex / `llm` · workflow: [`h3_music_video_minimal.json`](examples/h3/h3_music_video_minimal.json)

The one‑box music video: **song + lyrics + a cinematic look + two sliders — go.** The full [Music Video Writer](#apnext-h3-music-video-writer) runs underneath with opinionated defaults: the model invents the concept and the performer, the song is cut **on the music** (Auto segmentation), the imagery of every scene is staged from its lyric lines, dialogue language matches the lyrics, and the director skills are on.

- `visual_style` — the curated cinematic looks (35mm, Wes Anderson, neon noir, …); *Auto* lets the model pick one for the song.
- `performance` slider — 0–33 Narrative (nobody sings on camera), 34–66 Mixed, 67–100 Performance (lip‑synced on camera).
- `pace` slider — 0 = long slow pieces (~15 s), 100 = quick cuts (~6 s); cuts still land on the music.
- `model` + `llm` socket — Claude Code aliases, `codex`, or any local/API model; `seed`; `image_1..9` reference pictures fix the performer's face and pass through to the video node.

Outputs the rendering essentials: `scenes` / `durations` / `lengths` / `audio_segments` **(lists)**, `scenes_text`, `session_id`, `info`, `image_1..9`, `clip_starts`. Runs save to `output/apnext_scenes/` like the full writer. Reach for the full writer when you need a written concept, cast lines, wardrobe/location locks, scene briefs, or the masked‑audio path. Code: `nodes/h3/music_video_minimal.py`.

### APNext H3 Presentation Writer
`H3ClaudeCodePresentationWriter` · Claude Code / `llm` · workflow: [`h3_presentation.json`](examples/h3/h3_presentation.json)

Turns **source material** — scientific findings, benchmark numbers, a paper abstract, a changelog, code, release notes — into a **presented video**: a presenter walks through the material to camera, scene by scene, with charts and graphics that display the real values. No audio input; the spoken script, the visual aids and the pacing are all generated.

1. **Facts are sacred**: `source_material` is the ground truth. Every number, unit, date, name and claim spoken or shown on screen must come **verbatim** from it — nothing invented, rounded or “improved”; where the material gives no number the presenter speaks qualitatively instead.
2. **Plans the talk in the synopsis**: an `Outline:` line per scene (`NN: the point covered + its visual aid`) covering all the material’s key points in teaching order — scene 01 hooks and names the topic, each middle scene covers one point, the last scene lands the takeaway. Long talks are written in chunks (`scenes_per_call`) — by default **in parallel** (`parallel_chunks`): one planning call by `model` fixes the synopsis, Outline and locks, then up to 4 chunks at a time are drafted by `draft_model` (default haiku) and one continuity pass repairs drift; turn it off for the classic serial run continuing one session with a talk‑so‑far recap.
3. **Stages the graphics as objects in the scene** per `presentation_format` (keynote stage + LED screen, whiteboard drawn by hand, news studio insets, lab demo, boardroom pitch, documentary, tech screencast — *Auto* picks one): chart type named, title and axis labels in verbatim double quotes, 2–5 labeled values from the material, and the visual relationship stated (which bar is taller, where the line rises) so the picture reads even where small text renders imperfectly. `visual_aids` sets how often (*Auto* / every scene / key data moments / none).
4. **Paces the script to the clock** (~2.3 words per second; in *Vary* duration mode a heavier point gets a longer scene, never a faster read) and emits the same matching **lists** as the other multi‑scene writers: `scenes` → the video node’s `prompt`, `lengths` (frame counts, snapped to H3’s grid) → `length`, `durations`; plus `scenes_text`, `synopsis`, `script` (teleprompter view of every spoken `<d>` line per scene), `cast`, `scene_count`, `total_seconds`, `session_id`, `info`, `image_1..9`.

Inputs: `source_material` (the ground truth), `direction` (who presents, where, the tone), `presentation_format`, `scene_count` (1–24), `duration_mode` / `scene_duration`, `visual_aids`, `continuity_mode`, cast (`cast_1..4` / `extra_cast`; empty = the model invents a presenter), locks + `enforce_wardrobe`, `image_1..9` reference pictures (`image_notes`), scene briefs, the Claude Code block, `llm`, `save_scenes` (JSON bundle for **Scenes Load**).

### APNext H3 Short Film Writer
`H3ClaudeCodeShortFilmWriter` · Claude Code / Codex / `llm` · workflows: [`h3_short_film.json`](examples/h3/h3_short_film.json), `h3_short_film_turbo.json` (generated when the turbo reference graph is present)

Turns a **manuscript** — a story, treatment, script or synopsis of any length — into a whole short film, the way the Music Video Writer turns a song into a music video, but paced by **story** instead of audio:

1. **Size it either way**: `length_mode` = *Scene count* (exactly `scene_count` scenes, 1–80) or *Target length* (type `target_minutes` and the node derives the scene count at ~11 s per scene, then tells the model to pace each scene's 5–15 s `duration:` so the finished film lands close to the target).
2. **Faithful adaptation**: the manuscript is the source — named characters, events, places and any written dialogue survive **verbatim** into the scenes; the model invents only connective tissue. The synopsis carries a full `Beats:` plan (one story beat per scene) plus the wardrobe/location locks. Long films are written in chunks — by default **in parallel** (`parallel_chunks`): one planning call by `model` fixes the synopsis, Beats and locks, then up to 4 chunks at a time are drafted by `draft_model` (default haiku) and one continuity pass repairs drift; turn it off for the classic serial run continuing one session with a film‑so‑far recap.
3. **Cinematic craft baked in**: three‑act structure directives (act turns at ~¼ and ~¾ unless the manuscript says otherwise), coverage variety, motivated light, scene‑to‑scene hand‑offs, literal camera language, and a score (`non_diegetic_music`) holding one musical identity across the film. The model's generated dialogue, ambience and score **are** the film's sound.
4. Everything else works like the other multi‑scene writers: cast sockets + `extra_cast`, wardrobe/location locks with the repair pass, `image_1..9` reference pictures (normalised) with **REF/FL `prompt_mode`**, scene briefs, wildness, context sockets, template variables, sessions, `save_scenes`, and the same `scenes` / `durations` / `lengths` list outputs (+ `script` — the film's dialogue as a script).

The example workflow adapts a two‑minute harbour‑town story; the **turbo variant** renders it with the speed chain (turbo LoRA + SoL/Sage/EasyCache/Spectrum at 4 steps). Code: `nodes/h3/claude_code_short_film_writer.py`. `wildness` here is a pure scale (0 = sober, 100 = totally unhinged staging): only the level and its band label reach the model — no specific surreal elements are injected — and the facts, chart values and on‑screen text stay verbatim at every level. Finish with **Scenes Join** → Create Video → Save Video. Code: `nodes/h3/claude_code_presentation_writer.py`.

---

## Cast and model

**`handoff_pass`** (Continuous chain only, default **on**) — after the scenes are written (and the wardrobe / location repair), one more LLM turn reads every scene's *ending* against the *opening* that follows it and rewrites the openings that drift — a prop that vanished or changed hands, a person who moved or changed pose, a re‑established wide shot, a changed light — so the first frame of each scene *is* the last frame of the previous one, exactly as the [Short Film Chain Render](#apnext-h3-short-film-chain-render-carry-picture--sound-between-scenes) pins it. Only the rewritten scenes are re‑emitted and spliced in by number (a reply that touches scene 01 or anything outside the chain is discarded, `OK` means nothing needed changing); the log shows `🔗 hand-off pass: N scene opening(s) rewritten`.

### `interpretation` — how far the pictures may stray from the source

Every multi‑scene writer (Short Film, Crossover, Presentation, Scenes; the Music Video Writer has the lyric‑flavoured `lyric_interpretation`) ends with an `interpretation` dropdown. The source — the manuscript, the idea, the direction, the material — is always honoured for what it states outright; this decides what the *picture* does with it:

- **Auto** (default) — the writer's own judgement, as before.
- **Literal** — a faithful adaptation; imagination goes into *how* it is shown.
- **Loose** — keeps the people and the events, stages the feeling behind each beat rather than its literal props and places.
- **Metaphor** — one image system that is not the source (named in the synopsis Concept so every chunk writer stages the same one); every beat maps onto a development of it.
- **Counterpoint** — a different story, other people and places, that rhymes with the source's structure; written dialogue may survive as voice‑over.
- **Reframe** — keeps what the source is about, retold in another era, genre or scale; nouns translated into that world, dialogue adapted only where the world makes a word impossible.
- **Surreal** — escalating dream logic, described physically.
- **Surprise me** — one of the non‑literal readings picked by seed, so re‑running the same source gives a different film (the log says `🎭 interpretation: Reframe (picked by seed)`).

The Presentation Writer adds a hard rule: every fact, number, name and spoken line stays exactly as the material gives it — only the staging, the world and the visual aids are reinterpreted.

### `transition_style` — what a continuing scene may do with its place

The chain carry hands H3 the previous scene's last frames — the actor's identity, pose, direction and speed continue exactly — but not the previous *place*, so a change of place inside a continuing take has to be written, and left unsaid the writers keep continuing scenes in the room they started in. The Short Film, Crossover and Scenes writers (in *Continuous chain* mode) and the Music Video Writer (for pieces marked `CONTINUES the previous take`) end with a `transition_style` dropdown: **Auto** — the writer picks walk‑through or reveal per scene where the plan moves; **Stay** — same place, as before; **Walk‑through** — the actor carries the take through a door, corridor or corner into the new place, the camera following without cutting, the new place's anchors and light named as they come into frame behind and around the moving actor; **Reveal** (default) — the actor holds while the camera moves and a new space is disclosed around them, completed within the first half of the scene; **New place** — every scene is its own place: the plan moves the story somewhere new in every scene and continuing scenes get there inside the take (walk‑through or reveal, alternating), so no two consecutive scenes share a location. Hard cuts are unaffected, and — except under *New place* — a scene whose plan stays put stays put.

### Prompt rules measured from ostris/minimax_h3_1k

`scripts/h3_dataset_survey.py` profiles a corpus of 1,000 H3 prompts that are known to render well ([ostris/minimax_h3_1k](https://huggingface.co/datasets/ostris/minimax_h3_1k), 5‑second clips + the txt that made them). The report is in [`data/h3/h3_1k_survey.md`](data/h3/h3_1k_survey.md); the writers' directives and `data/h3/guide_base_en.md` follow what it measured:

- **Length scales with the clip** — ~30 words per second of `integrated_multimodal_description`: 120–190 for 5 s, 160–250 for 10 s, 200–310 for 15 s (the corpus: median 152, never above 250; the old 350–500 was two to three times too long). The Music Video Writer prints each piece's budget on its `PIECE` line; the `📏 description:` log line judges every scene against its own duration.
- **Shots** — one up to ~8 s, two beyond, three only past 10 s, never four (half the corpus is a single shot). `[Shot 2] At 00:0X.XXX, the shot cuts to …` for every cut.
- **Sound fields** — `overall_soundscape` is one sentence, a comma list of concrete sounds; `non_diegetic_music` is `N/A` unless the scene calls for score (56 % of the corpus), then one sentence with instrumentation, tempo and one cue tied to a visible moment.
- **Dialogue** — the voice quality rides on the speaker id the first time (`(S1), her voice sharp, breathless, and quick-paced,`), silent on‑screen characters are marked `(no ID, non-vocalizing)`, a line that runs past the end closes with `<cutoff>`.
- **Style** — the default `visual_style` is now `Live-action, cinematic` (the corpus's most common opener; `35mm` appears in 2.5 % of it), followed in the scene's own words by the light and grade, then the framing. The `visual_style` dropdown is **looks only** — medium, film stock, grading, hues, lens, camera, era — never a place: corpus openers and aesthetics that name a setting (`mint-green kitchen`, `suburban backyard`, `Beach Day`, `Diner` …) are filtered out at build time (`_PLACE_WORDS_RE` in `nodes/h3/common.py`; places belong in `direction` and the location locks), which leaves 1,330 entries. It leads with the openers taken verbatim from the corpus, grouped by medium — device footage (phone, doorbell, bodycam, night‑vision, screen recording), broadcast / commercial, 2D and 3D animation, clay / puppets / tabletop, game engine, silent era — regenerated with `python scripts/h3_dataset_survey.py --style-list data/h3/dataset_visual_styles.json`.

### APNext H3 Sampler + Save Clip (each scene to disk as it finishes)

`H3SampleAndSave` · workflows: every single‑clip music video, presentation, short film and scenes example (it replaced the SamplerCustomAdvanced → Save Clip pair)

SamplerCustomAdvanced and Save Clip in **one node**. ComfyUI maps a scene list over each node before moving to the next, so with the two separate nodes all 20 latents sample first and nothing reaches disk until the last one — and a crash at scene 19 loses everything. This node takes the sampler's inputs (`noise`, `guider`, `sampler`, `sigmas`, `latent_image`) plus Save Clip's (`vae`, `audio`, `filename_prefix`, `fps`, `format`) and, per list item, samples → decodes → muxes the audio slice → writes the clip **before the next scene starts sampling**: the output folder fills up while the run is still going, exactly like Chain Render. `output` is the sampled latent, so anything that hung off the sampler (VAE Decode Audio, an upscaler) still does. Code: `nodes/h3/sample_and_save.py`.

### APNext H3 Save Clip / De‑Rope + Save Clip / AWQ Encoder Loader

- **`H3SaveClip`** — *APNext H3 Save Clip (decode → disk, low memory)*. Decodes one H3 latent and writes the clip straight to disk (replaces VAE Decode + Create Video + Save Video). Each list item is decoded, saved and freed before the next, so an 18‑scene run needs the RAM of one clip, and clips already saved survive a mid‑run OOM. Inputs `samples` + `vae` (or `images`), optional `audio` (the writer's `audio_segments`), `filename_prefix` (wire a writer's `project_name`), `fps`, `format` (mp4/mkv H.264, webm AV1). Output `file_path`.
- **`H3DeRopeSave`** — *APNext H3 De‑Rope + Save Clip (Motion Lab by matlowai)*. One‑node temporal de‑rope + save: finds where a clip's motion is too fast for H3, holds those frames, regenerates the clip v2v at partial denoise with the song stretched onto the held clock, recovers exact real time by dropping the held frames, and writes the mp4 — per clip, so memory stays at one clip. `enabled` off makes it a plain Save Clip. Outputs `file_path`, `report`.
- **`MiniMaxH3AWQEncoderLoader`** — loads a compressed‑tensors **AWQ (W4A16) Qwen3‑VL** checkpoint as the H3 text encoder (vendored from fbjr's `qwen3-vl-32b-W4A16-AWQ-H3`, unmodified, with attribution). A drop‑in for `CLIPLoader` when you run the 4‑bit encoder: ComfyUI supplies the H3 architecture and tokenizer, comfy‑kitchen the CUDA W4A16 operator, this node the compressed‑tensors adaptation (in memory, view‑based — no second checkpoint on disk). See [`h3_music_video_masked_audio_awq.json`](examples/h3/h3_music_video_masked_audio_awq.json).

### APNext H3 Characters
`H3Characters` · workflows: all crossover workflows, [`h3_music_video.json`](examples/h3/h3_music_video.json)

Character / actor / franchise lookup from `data/h3/characters.tsv`, or your own character.

- `character`: a `Character — Actor (Show)` entry, `🎲 random` (seeded, optionally narrowed by `franchise_filter`), or `✏️ custom` → describe your own in `custom_character` (`Lena: a middle‑aged woman with a limp and a silver bob` keeps `Lena` as the name; `Name (played by Actor) from Show` also works).
- `wardrobe`: this character’s wardrobe lock (3–5 exact anchors). It rides along on the cast line (`… | wardrobe: …`) into the Crossover / Music Video writers and is merged into their lock automatically, so the outfit lives with the character.
- `cast_in`: chain several Characters nodes into one cast list (A → `cast_in` of B → B’s `cast` → `cast_1`).
- `image` in / `image` out (pass‑through): the character's reference picture travels WITH the character — wire `cast` → a writer's `cast_N` and `image` → the matching `image_N`, so one node bundles the identity and its face photo.
- Outputs `character`, `actor`, `franchise`, `file_path` (the reference clip), `cast` (`Character (played by Actor) from Show`, + wardrobe), `wardrobe`.
- On the canvas the node is **rose**; it feeds the writers’ **rose** `cast_N` sockets and also any `context_N` socket (then `{character1}` works too).
- Shoutout to [malcolmrey](https://huggingface.co/malcolmrey) for the crossover ideas and for finding valid characters — his [various dataset](https://huggingface.co/datasets/malcolmrey/various) is a great place to mine cast entries that the video model actually knows.

### APNext H3 LLM Backend (Ollama / local / API)
`H3LLMBackend` · workflows: [`h3_llm_backend_crossover.json`](examples/h3/h3_llm_backend_crossover.json), [`h3_llm_backend_writer_refiner.json`](examples/h3/h3_llm_backend_writer_refiner.json), [`h3_music_video_masked_audio_ollama.json`](examples/h3/h3_music_video_masked_audio_ollama.json), [`h3_music_video_masked_audio_ollama_textonly.json`](examples/h3/h3_music_video_masked_audio_ollama_textonly.json)

**OpenRouter** — one key for every model. The dropdown lists OpenRouter's **whole text‑model catalogue** (fetched from its public `/models` endpoint when the page loads, cached 10 minutes; ~350 ids, curated ones first then A–Z, filtered to models that read and write text and to the interactive endpoints — no `:batch` variants; type in the dropdown to filter). Offline, or with `APNEXT_OPENROUTER_LIST=0`, it falls back to the curated list, and *custom* + `openrouter:<vendor/model>` always works for any id on [openrouter.ai/models](https://openrouter.ai/models). The key comes from `OPENROUTER_API_KEY` in the environment or from the node's **`api_key`** field (appended last; saved in plain text inside the workflow, so use the environment on a shared box). `base_url` may point at a gateway in front of OpenRouter; otherwise leave it empty. Open models routed this way may think out loud — their `<think>` blocks are stripped like a local model's — and get the local request timeout (15 min) rather than the cloud one. Example: [`h3_music_video_masked_audio_openrouter.json`](examples/h3/h3_music_video_masked_audio_openrouter.json).

One node that says *“write with THIS model”* for every Claude Code H3 node. Drag its **pink** `llm` output into any number of writers’ `llm` sockets. While it is connected the writer’s own `model` / `draft_model` widgets are greyed out and relabelled with the backend’s model (`model → ollama:qwen3.8:27b`) — the backend writes **everything** (plan, scene chunks, continuity repair); nothing goes through Claude Code, and the console confirms it with `✅ H3 via ollama:…`.

- `model`: every `ollama:` / `lmstudio:` / `local:` model your servers were serving at page load, the cloud API models, `auto-detect`, or **custom** + `model_name` (`ollama:qwen3:14b`, `lmstudio:qwen/qwen3-8b`, `local:my-model`, `claude:claude-sonnet-5`, `gpt:gpt-5.6`; a bare Ollama tag like `qwen3:8b` is understood).
- **`num_ctx`** (Ollama only, default **65536**) — the context window the model is loaded with. **This is the setting that decides whether a local run works at all.** An H3 system prompt is ≈9k tokens text‑only and ≈15k with reference images, before your song, cast and lyrics; Ollama picks its own default from free VRAM — as little as **4k** — and silently truncates everything past it, starting with the rules the model is meant to follow. The result is not an error, it is a bad prompt. `0` leaves the server default alone. Ollama’s OpenAI‑compatible endpoint cannot set this, so a non‑zero `num_ctx` (or a `thinking` choice) sends the call through Ollama’s own `/api/chat` instead. **Size it for the job:** a 20‑scene music video chunk call carries ~30k tokens of prompt (system prompt, director skills, the whole scene plan with per‑piece beats and lyrics, a reference picture), so **32768 leaves ~1k tokens for the reply** and Ollama stops mid‑scene — the Ollama examples ship with **65536**. The writer now warns before such a call (`⚠️ H3: this prompt is ~31,000 tokens against num_ctx 32,768 …`) and, if Ollama still cuts the reply off (`done_reason=length`), the run **fails with the numbers** instead of quietly padding the video with a repeated scene; a cut‑off JSON reply from any provider yields the complete scenes it did contain (`salvaged N complete scene(s)`) rather than one scene of raw JSON.
- **`thinking`** (Ollama only, default **off**) — whether a hybrid reasoning model (Qwen3, DeepSeek‑R1, gpt‑oss…) reasons before answering. Off is much faster and the H3 rules are already in the system prompt. Any `<think>` block that arrives inline is stripped before parsing either way, on every local provider.
- **`unload_after`** (Ollama only, default **on**) — once a writer has all its scenes (repair pass included), Ollama is asked to drop the model from VRAM immediately instead of holding it for its `keep_alive` window, so the memory is back for the video render that follows. Turn it off when another writer runs right after (a reload costs ~10–30 s).
- **`structured_output`** (default **auto**) — how multi‑scene answers travel back. *auto*: on Ollama the reply is **JSON constrained by a schema** (`{"synopsis", "scenes": [{scene, duration, prompt}]}`, valid by construction — the sampler cannot produce a broken envelope), other backends keep the `=== SCENE NN ===` text envelopes; *on*: every backend is asked for JSON (not enforceable through the CLIs, but Claude/Codex write it reliably); *off*: text envelopes everywhere. Parsing is JSON‑first with the text envelopes as fallback (and the envelopes no longer require the `=== END SCENE ===` closer), so nothing that worked before can break. The scene text itself — the labelled fields the video node reads — is identical either way.
- **📏 Show 1024 tokens** — a button on the node. Opens a modal with (1) a real block of exactly 1024 tokens, cut from the H3 guide and **counted with your own model’s tokenizer**, (2) what an H3 run actually spends against this node’s `num_ctx`, and (3) every pulled model’s KV‑cache cost per token, how much context fits in **this** machine’s VRAM and RAM, its `vision` / `thinking` / `tools` capabilities, and a **benchmark** button that measures real tokens/sec and the GPU/CPU split on the spot.
- `base_url` (default `http://0.0.0.0:11434` — Ollama on this machine, a wildcard host is read as localhost; a LAN box / other port goes here; empty = the prefix default or `OLLAMA_BASE_URL` / `LMSTUDIO_BASE_URL` / `LOCAL_LLM_BASE_URL`), `temperature`, `max_tokens` (multi‑scene runs need room), `inline_skill_references`.
- Outputs `llm` and `model_used`. Unplug the socket and the writer is back on Claude Code.

**Vision, and how to do without it.** The writers’ `prompt_mode` decides two separate things — whether the pictures are bound as `<Picture N>`, and whether the *writing* model is shown them:

| `prompt_mode` | Writer sees the picture | Video model gets it | Needs a vision model |
|---|---|---|---|
| **Ref2VA** | yes | yes | **yes** |
| **Ref2VA blind** | no | yes | no |
| **FL / T2VA** | no | no | no |

**Ref2VA blind** is the one to reach for on a local model. Face consistency across clips comes from the picture reaching the *video* model, not from the writer having looked at it — so a text‑only model (a small quantised one, an uncensored fine‑tune, anything without the `vision` capability) still gets full reference‑image rendering. The writer is told explicitly that it cannot see the pictures, must never guess at their contents, and must take who is in each one from the cast lines and `image_notes` (`Image 1: Mara, the lead singer`); an unnoted picture binds to the cast member in the same position. Blind mode still uses the reference guide, so the system prompt is ≈15k rather than ≈9k tokens — but it sends no image data at all. The token modal tags every pulled model with `vision` / `thinking` / `tools`, read from its own metadata.

---

## Scene utilities

### APNext H3 Scene Pick
`H3ScenePick` · workflows: [`h3_crossover_pick_one.json`](examples/h3/h3_crossover_pick_one.json), [`h3_scenes_pick_one.json`](examples/h3/h3_scenes_pick_one.json)

Collapses the `scenes` / `durations` lists to **one** scene by `index` (clamped), so one scene can be rendered or refined on its own. Outputs `scene`, `duration`, `index`, `count`. Put an incrementing primitive on `index` to step through a run one queue at a time.

### APNext H3 Scene Counter
`H3SceneCounter` · workflows: [`h3_crossover_pick_one.json`](examples/h3/h3_crossover_pick_one.json), [`h3_scenes_pick_one.json`](examples/h3/h3_scenes_pick_one.json)

Progress readout for a pick‑one run: wire **Scene Pick**'s `index` and `count` outputs in and the node shows a big **X / N** with a progress bar and "M scenes remaining", updated every queue. Also outputs the `status` text and the `remaining` count for filenames or notes.

### APNext H3 Scenes Join
`H3ScenesJoin` · workflows: [`h3_scenes_batch.json`](examples/h3/h3_scenes_batch.json), [`h3_crossover_batch.json`](examples/h3/h3_crossover_batch.json), [`h3_music_video.json`](examples/h3/h3_music_video.json)

Sits between the per‑scene `VAE Decode` / `VAE Decode Audio` and a single `Create Video` → `Save Video`: gathers the per‑scene `IMAGE` (and optional `AUDIO`) lists back into **one** frame batch and one audio track so one queue run writes one whole file.

- Frames concatenated in order; `crossfade_frames` blends the cut (0 = hard cut) and trims audio to keep A/V aligned; `size_mismatch` resizes stray scenes to the first scene’s resolution or errors.
- Audio joined in order (sample rate / channels unified); `replace_audio` swaps the joined track for one of your own (the original song from the Music Video Writer).
- Outputs `images`, `audio`, `frame_count`, `scene_count`.

### APNext H3 Scenes → Contex Loop Plan
`H3ScenesToChainPlan` · workflow: [`h3_crossover_contex_chain.json`](examples/h3/h3_crossover_contex_chain.json)

For **continuity across scenes** (the last frames and audio of scene N carried into scene N+1). Converts the `scenes` / `durations` lists into the plan JSON that [ComfyUI‑MiniMaxH3‑Contex‑Loop](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Contex-Loop)’s *MiniMax H3 Contex Loop Plan* node accepts on `plan_json_input` (`shots[]` with `id`, `prompt`, `duration_seconds`, `seed`, optional `prompt_prefix`, `defaults.steps`). That pack renders every scene in order with the previous tail as context and assembles one MP4. Inputs `id_prefix`, `base_seed`, `seed_mode`, `steps`, `prompt_prefix`; outputs `plan_json`, `shot_count`.

### APNext H3 Scene Brief (manual scene)
`H3SceneBrief` · feeds the Music Video, Crossover and Scenes writers

Manual scene planning: one node = one scene YOU design — `description` (what happens), `location`, `cast` (names from your cast), `pictures` (which reference images apply, e.g. `Picture 2 = the rooftop, use as the location`) and an optional `camera` wish. Chain several through `brief_in` (like Characters chain through `cast_in`) and wire the final `briefs` output into a writer's **`scene_briefs`** socket. Each brief becomes the **binding plan** for its scene — the writer still produces the full production‑ready H3 envelope (grammar, timestamps, wardrobe locks, lyric sync), but the content follows your plan. `scene_number` pins a brief to a specific scene/piece; `0` fills scenes in chaining order, skipping pinned ones; scenes without a brief stay the model's to invent within the concept. Template vars (`{character1}` …) work inside every field.

### APNext H3 Manual Scenes (script → lists)
`H3ManualScenes` · workflow: [`h3_short_film_manual.json`](examples/h3/h3_short_film_manual.json)

The fully hand‑authored counterpart of the multi‑scene writers: paste the **finished H3 prompts yourself** and get the same matching lists the writers emit — `scenes`, `durations`, `lengths` (snapped to H3's 5 + 17k frame grid), `scenes_text`, `scene_count`, `total_seconds` — so the render side of any writer workflow plugs in unchanged and **no model is called**. The `script` accepts the writers' `=== SCENE NN | duration: S.S ===` envelopes (duration optional per scene) or bare prompts split wherever a line starts with `subject_definitions:`; per‑scene durations come from the envelope header, then the `durations` box (comma/space separated), then `default_duration`. Unlike **Scene Brief** (a plan a writer expands), Manual Scenes renders your text verbatim.

### APNext H3 Refine Encode + Mouth Guard (face‑refine second pass)
`H3RefineEncode` + `H3MouthGuard` · workflow: [`h3_face_refine_mouthguard.json`](examples/h3/h3_face_refine_mouthguard.json)

The **mouth‑guarded face‑refine v2v pass**: upscale a rendered pass‑1 clip, re‑encode it, and re‑render it toward reference face images — while the mouth region (and the soundtrack) are **protected from denoising**, so the pass‑1 lip‑sync survives the likeness restore. Without the guard, a Ref2VA refine rewrites the mouth and the character stops talking.

- **Refine Encode** builds the clean AV latent: the upscaled pass‑1 frames go through the H3 video VAE (frame count trimmed to the 17k+5 grid, dims must be /32), and the pass‑1 audio is copied verbatim from the pass‑1 latent (`source_latent`, preferred) or encoded from an `audio` track. Its `frame_count`/`width`/`height` outputs wire straight into **MiniMax H3 Reference to Video** (whose own LATENT output is discarded — only its conditioning is used).
- **Mouth Guard** takes a per‑frame **lips mask** — core ComfyUI's `Detect Face Landmarks (MediaPipe)` → `Draw Face Mask (MediaPipe)` with `regions=custom`, lips only (one‑time setup: download `mediapipe_face_fp32.safetensors` from `https://huggingface.co/Comfy-Org/mediapipe/resolve/main/detection/mediapipe_face_fp32.safetensors` into `models/detection/`) — reduces it onto the H3 latent grid (16× spatial, the cyclic 1,4,4,4,4 temporal groups), grows and feathers it, and installs it as the latent's nested noise mask (lips = preserve, `protect_audio` locks the soundtrack). `mask_preview` shows the protected region per frame; `report` prints the geometry.
- **Sampling**: `RandomNoise` + `BasicScheduler` with `denoise` ≈ 0.35–0.5 (the refine strength). **Never a DisableNoise / pre‑noised flow** (e.g. the latent‑upscaler's combined re‑noise): the sampler restores protected regions from the *input* latent, so a pre‑noised input would preserve noise instead of lips.
- **Quality**: when `ComfyUI-H3-Motion-Context-MultiRef` is installed the guard also enables its H3 mask engine (protected tokens run at the model's clean‑conditioning timestep — same mechanism as keyframes; clean boundary). Without it the mouth is still preserved by the sampler's inpaint blend, just seamier. Mask granularity is ~32 source pixels, so refine at **2×** (the workflow default) — small faces then give the mouth enough latent cells. The example also carries a muted same‑seed **A/B branch** (no guard) and a muted **pixel composite** branch that pastes the pass‑1 mouth pixels back after decode as a belt‑and‑braces finish.

### APNext H3 Scenes Load (from disk)
`H3ScenesLoad` · pairs with the writers' `save_scenes` toggle

Re‑render a saved run **without any LLM call**. The Music Video Writer's `save_scenes` toggle (on by default) stores every successful generation as a JSON bundle in `output/apnext_scenes/` — scenes, synopsis, segment times, durations, frame lengths, clip starts, cast, tables. This node's file picker lists the bundles (newest first; refresh the browser to re‑scan) and outputs mirror the writer's core outputs, so it drops into the same graph: `scenes` → review/render, `lengths` → `length`, `clip_starts` → the masked‑audio context node. Connect the same song to `audio` and the per‑clip `audio_segments` (`ref_audio_1`) are re‑sliced from the saved segment times (a duration mismatch is warned about). Cached by file mtime.

**Continue from the scenes you already have — `run_mode` on the Music Video Writer.** The quicker way when it is the *last* run you want again: the writer's last widget, `run_mode`, switches between **write** (call the LLM, as always) and **reuse last run** — the node skips the LLM (and the lyric transcription) and returns the newest bundle it saved itself, with the same scenes, lengths, clip starts, cast and `project_name`, so everything downstream re‑renders unchanged with other sampler settings, seeds or references and no rewiring. It is the scenes‑stage equivalent of bypass: while it is on, edits to the direction, lyrics or cast are ignored (the log says `♻️ … reused <file> … no LLM call`); set it back to *write* for a new script. Cached by the bundle's mtime, so re‑queues stay cheap; an older run is a job for this node's file picker.

### APNext H3 Scenes Review (edit before render)
`H3ScenesReview` · optional gate — wire it between a writer (or refiner) and the render when you want to proof scenes first (the example workflows render directly)

The opt‑in commit gate. In **Review** mode (the default) a queue run fills the node's editor with the incoming `scenes` (or a single `h3_prompt`) and stops cleanly — the editor shows the same colour‑coded tags as the Prompt Preview but is fully editable, all scenes at once or one scene at a time (scope selector / ◀ ▶). Edit what you like, then queue again: the mode has flipped to **Continue** automatically, so the next run renders exactly the editor text. **▶ Continue** and **🎲 Recreate** buttons do it in one click — Recreate bumps the writer's seed and reviews a fresh draft. **Bypass** passes through untouched. A `source-fingerprint` line in the editor header ties the edits to the scenes they came from: if the incoming scenes change (new cast, direction, lyrics or seed upstream), the **fresh scenes render** and the editor is refilled — stale edits are never rendered and the node never touches the writer's seed. A new run therefore always makes a new video; to edit before rendering, give the writer a **fixed seed** yourself so Continue runs reuse its cached answer. Keep the `=== SCENE NN ===` markers and the scene count (the writer's `durations` / `audio_segments` stay aligned; a mismatch is padded/trimmed with a warning). Outputs `scenes`, `scene_count`.

### APNext H3 Dailies Gate (print / punch up / cut)
`H3ScenesReviewGate` · workflow: [`h3_music_video_dailies_gate.json`](examples/h3/h3_music_video_dailies_gate.json)

The **live** sibling of Scenes Review, styled as a screening‑room "dailies desk": instead of stopping the queue, the run **holds open** at this node while the freshly written scenes wait in the browser, and it moves again the moment a button is pressed — no re‑queueing, no seed juggling. On the desk:

- **▶ Print it** — render exactly what is on the desk, hand edits included (the editor is the same colour‑coded overlay as the Prompt Preview, viewable **all takes at once or one take at a time** — ◀ Take NN ▶).
- **✍ Punch‑up** — type director's notes (and optionally which takes: `2, 4-5`; empty = the take being viewed, or all in the all‑takes view) and the selected scenes are rewritten **inside the writer's own model session** — synopsis, locks, lyrics, reference images still in context. Hand edits are folded in *first*, so "fix a line by hand, then have the model rebuild the scene around it" is one round trip. Works with Claude Code, Codex (`codex-`) and local‑model (`local-`) sessions; the gate auto‑switches its `model` to codex for a `codex-` session id.
- **🎲 New take** — the same rewrite with the notes dropped: the model is asked for a noticeably different version of the selected takes. Roll as many as you like.
- **↩ Undo** — every rewrite is kept in a server‑side history for the life of the gate, so a bad punch‑up rolls back instantly (and the history survives a browser reload).
- **✋ Cut** — end the run cleanly, render nothing.

Wire the writer's `scenes` in, plus `durations` (keeps rewritten takes on their exact lengths — matters for music videos) and `session_id` (what enables the AI rewrites; without it the desk is edit‑by‑hand only). `auto_approve_minutes` > 0 prints automatically with a **live countdown** in the header so unattended runs still render; `chime` plays a soft browser tone when takes land; ComfyUI's Stop button also releases the gate, and a browser reload re‑attaches to a waiting desk. Because the gate sits *before* the render, no video model weights are held while it waits. Choose this gate for hands‑on sessions; choose **Scenes Review** when you want the free stop‑edit‑requeue flow with the writer's cached seed. Outputs `scenes`, `scene_count`, `status`. Code: `nodes/h3/scenes_review_gate.py`, `web/js/h3_review_gate.js`.

### APNext H3 Song Analysis (BPM / intensity)
`H3SongAnalysis` · a **readout**, not part of the pipeline: its outputs feed nothing, and the writer measures the same structure itself. It was dropped from the examples once Beat Grid, Sound Events and Cut Plan showed the same numbers *and* drove the run; add it from the node list when you want the BPM / intensity / form on the canvas for a song.

The Music Video Writer's sound measurement as a **visible readout**: wire the song in and the node shows `~124 BPM | driving (intensity 68/100) | moderate dynamics (9 dB) | 2.7 onset spikes/s` right on the canvas — tempo from the autocorrelated onset envelope (0 = no steady pulse), aggression 0–100 from transient spikes + sustained loudness + punch, and the dynamics spread. Use it to sanity‑check a song before a long run or to pick `wildness` / `performance` settings; outputs `audio` (pass‑through), `profile` (the one‑liner, wireable into any STRING/context input), `bpm`, `intensity`, `label`. The writers measure this themselves either way — this node just makes it visible. Code: `nodes/h3/song_analysis.py`, maths in `nodes/h3/music_support.py`. A second line shows the measured **form** - `form (97% sure): intro 0:00 · chorus 1 0:16 · verse 2 0:30 …` plus the meter when a downbeat is clear - and the new `structure` output carries the same text.

### APNext H3 Masked Song Latent (song frozen in the audio latent)

`H3MaskedSongLatent` · code: `nodes/h3/masked_song.py` · a **drop‑in** for `MiniMaxH3SongMaskedAVContext` (ComfyUI‑H3‑Motion‑Context‑MultiRef): the same inputs (`latent`, `audio_vae`, `master_audio`, `clip_start_seconds`, `context_length`, `source_fps`, `crop`, optional `vae` / `source_frames` / `source_latent`), the same three outputs (`latent`, `trim_frames`, `clip_audio`), no third‑party pack. Writes the exact master‑song slice this clip covers into the H3 audio latent, protects it with a nested noise mask (video generated, audio preserved), and optionally pins the previous clip's tail as a visual prefix (latent copy or decoded frames) for chaining. **Chain Render** and **Scene Retake** pick it first and fall back to the MultiRef node when it is missing; every masked‑audio example now runs on it (swapped in place with `scripts/swap_masked_song_node.py`, which is also how a workflow of your own is converted), with AudioSeparation's vocal stem wired to `voice` for the voice gate.

What it does differently comes from measuring the real H3 audio VAE (`scripts/h3_audio_vae_probe.py`, run from the ComfyUI root with ComfyUI's python):

- **Round‑trip delay is 0 ms** — nothing to compensate; the slice starts on the sample `clip_start_seconds` names.
- **The encoder needs a past.** It is a same‑padded DAC conv stack under a *causal* attention block, so a slice cut hard at the clip start gives the first tokens zero‑padded history and nothing to attend to: token 0 comes out ~57 % off what a continuous encode of the song gives, and the error takes ~0.5 s to decay. `preroll_seconds` (default **1.0 s**, whole 25 ms ticks) encodes that much song *before* the clip and drops those tokens: 1 s brings the head to ~1 %, 2 s to the floor. **0 reproduces the MultiRef node's hard cut** for an A/B — the decoded audio is intelligible either way, so which the *model* lip‑syncs better is a render test, not a codec fact.
- **The convs also look ahead** ~200 ms, so the last tokens of a hard‑cut slice differ too; `lookahead_seconds` (default **0.2 s**) encodes that much real song after the cut (silence once the song has ended) and cuts it back. The MultiRef node did this only as an error‑recovery retry.
- **Voice gate** (`voice` input — the vocal stem from AudioSeparation): H3 has one audio latent and it is a mix, so the voice cannot go to the lips and the music to the motion separately — but the freeze can follow the voice in *time*. With `voice` connected the audio rows are held at `audio_denoise` (0 = frozen) wherever the stem is sounding (Voice Over Music's sidechain detector, averaged onto the 40 Hz grid, grown by `gate_hold_seconds` = 0.2 s on both sides) and at **`gap_denoise`** (0.15) between phrases, so the model has a little freedom over the music bed exactly where no lip‑sync is at stake. Core turns the per‑tick mask into per‑row timesteps natively. The log line reports the share of sung ticks.
- **Lips before the word?** Nothing on this node moves the sound against the picture — its slice is sample‑exact to the clip timeline. A consistent lead is the model anticipating the vocal; trim it with **`sync_offset_ms`** on [Stitch Clips](#apnext-h3-stitch-clips-gapless-from-the-saved-files) (negative = sound earlier). `gate_hold_seconds` only widens the frozen region around each phrase; it does not shift anything.
- **`audio_denoise`** (default 0 = fully frozen) is the audio rows' noise‑mask value (the sung ticks' value when the gate is on): core turns a fractional mask into a per‑row timestep, so 0.05–0.15 lets the model re‑touch the audio at that noise level while the vocal stays — an A/B knob; the finished video still gets the real song via Scenes Join `replace_audio`.

Everything the mask does is ComfyUI core (`comfy.model_base.MiniMaxH3`: per‑row timesteps from the nested mask, preserved rows re‑injected from the input latent every step) — nothing is patched. Tests: `tests/test_masked_song.py` (CPU, fake VAE).

### APNext H3 Stitch Clips (gapless, from the saved files)

`H3StitchClips` · code: `nodes/h3/stitch_clips.py` · workflow: [`h3_music_video_masked_audio_ollama_masked_song.json`](examples/h3/h3_music_video_masked_audio_ollama_masked_song.json)

H3 Sample + Save writes every scene as its own mp4 as it finishes — that is what keeps a 20‑scene run at the memory of one clip. Putting those files back together is where seams come from: every AAC stream carries ~23 ms of encoder priming and tail padding that a player only trims at the *start and end of a file*, so a stream‑copy concatenation (ffmpeg `-c copy`, an editor's "append") leaves a 2–14 ms hole at every join. The clips are sample‑exact; the container is not.

This node runs once after the last scene (wire Sample + Save's `file_path` list into `file_paths`) and writes `<project>_full.mp4` next to the clips: the H.264 packets are **copied** with their timestamps shifted onto one timeline — no re‑encode, no quality loss, no frames in RAM — and the sound is either the **original song** (`song` = Load Audio) muxed once from 0:00, or, with `song` unplugged, the clips' own audio decoded, cut to each clip's exact frame count and encoded once. Either way there is nothing at the joins for a player to trim. `suffix` names the file, `enabled` off passes the clips through. **`sync_offset_ms`** slides the whole soundtrack against the picture: **negative = the sound comes earlier** — the control to reach for when the lips move a touch before you hear the word (try −40 to −80; one frame is 41.7 ms, and audio leading the picture is noticed from ~45 ms, lagging from ~100 ms) — positive = later. It belongs here and not on the latent node: the masked latent is sample‑exact to the picture timeline, so a consistent lead is the model anticipating the sound, and the mux is where a constant is trimmed away without changing what the model heard. Outputs `file_path` and a `report`. Clips must share resolution and codec (they do, from one run). Verified on a five‑clip run: 773 frames, song at +0.00 ms, no seam silences.

### APNext H3 Sync Check (does the picture hit the beat?)
`H3SyncCheck` · decoded frames + the clip's audio → report

Timestamps in a prompt are a request, not a guarantee. This node **measures** a render: it finds the hits in the clip's audio, finds the frames where the picture changes, and counts how often a hit has a picture change within ±2 frames — against the same count for the same hits shifted to random times. A **lift over chance** is real in‑clip sync; no lift means the sync you feel is coming from the cuts between clips, not from inside them. It also reports the onset‑envelope / picture‑change correlation and its best lag, and lists every hit with its offset to the nearest picture change.

Measured on 18 H3 clips of one video (26 Aug 2026): 58 % of hits had a picture change within ±80 ms, chance 49 % — a +9 % lift; correlation +0.05. That is why the Cut Plan puts the drops on the cuts and the writer is told that **the cut is the only exact hit**: a listed moment inside the first 0.4 s of a piece becomes that piece's opening image, already at the peak of the move.

Inputs: `images` (the decoded clip or a joined video), `audio` (its own audio piece, or the whole song for a joined video), `fps`, `tolerance_frames`, `min_strength`; optional `sound_events` for that audio. Outputs: `report`, `hit_rate`, `chance`, `lift`.

### APNext H3 Scene Retake (render one scene again)
`H3SceneRetake` · model/clip/VAEs + sampler + a saved bundle → one clip

A finished video has one scene that did not land. Re‑queuing the graph writes the whole film again (a randomising writer seed) and renders every scene to get to the one you wanted. This node reads the run's **saved scene bundle** (`output/apnext_scenes/`, written by every writer's `save_scenes`; `bundle` = *latest* or a specific file), takes one `scene_number`, and renders **just that scene** with a new `seed` — and, if you like, a `prompt_override` (paste the scene from the Prompt Preview, edit, retake). Its **🎬 Retake this scene** button bumps the seed and queues **only this node** (ComfyUI partial execution), so the writer, the sound nodes and the other scenes never run again.

Continuity survives the retake: both chain renders (and this node) write every rendered scene's **sampled latent** to `output/apnext_latents/<project>_sNN.pt` (`save_latents`, on by default), so with `continuity` = *continue*, a retake of scene 07 pins scene 06's real tail — picture and sound — to its head exactly as the chain render did (`context_frames` 39 / 90 / 141, `audio_feather_ticks` for films), and the retaken 07 becomes the tail a later retake of 08 continues from. No saved take → hard cut, with a note in the log. Music videos: connect `master_audio` and the scene's piece is cut at the bundle's segment and masked in, as the render did; films: leave it unconnected and H3 generates the sound. Wire the same `ref_image` pictures the run used. The clip lands at `filename_prefix` + `_sNN_retake`; outputs `file_path`, the `prompt` used, and a `report`.

### APNext H3 Short Film Chain Render (carry picture + sound between scenes)
`H3ShortFilmChainRender` · writer `scenes` + `lengths` + model/clip/VAEs + sampler → clips on disk

The short‑film pipeline renders every scene as an independent clip, and H3 generates each clip's sound from nothing — so at every scene change the room tone, the score and whatever was mid‑air at the cut all **start over**. This node renders the scenes **one after another inside one execution** and, for every scene after the first, pins the last `context_frames` of the previous scene's **sampled latent — picture and sound together** — to the head of the new latent, protected by a noise mask, with the audio mask feathered over its last `audio_feather_ticks` (8 = 0.2 s) so the join is a release rather than a wall. The model reads the pinned run as "this clip's picture and sound so far" and continues both; the delivered clip has the head trimmed off, picture and sound by the same duration, and its audio cut to exactly its frame count (H3 rounds its audio grid up ~8 ms per clip, which would otherwise accumulate into a click down the chain). No decode / re‑encode round trip: the latent is copied straight across (ComfyUI‑H3‑Motion‑Context‑MultiRef's `MiniMaxH3GeneratedAVMaskedContext`, the node its own AV‑extension workflows use). Without that pack the core `MiniMaxH3AddGuide` still carries the previous picture; the sound is then generated per scene and the report says so.

`context_frames` is how much of the previous scene is sent into the next, and `context_mode` is *how*: **masked AV latent** copies both streams into the new latent under a noise mask — the surest carry, but the two grids only share a boundary every 51 frames, so it takes **39** (~1.6 s), **90** (~3.75 s) or **141** (~5.9 s); **motion context guide** pins the previous frames as never‑denoised conditioning rows at frame 0 and the tail of the previous *sound* on the same timeline from its latent — any H3 run, **5, 22, 39, 56, 73, 90, 107, 124, 141** — so you can send as little as a fifth of a second; **core guide clip** carries the picture only and needs no pack. The log says what the frames snapped to; more context is a longer, surer continuation and fewer frames left for the scene. The continuing scene renders on a longer grid (scene + context, rounded up to H3's 5 + 17k) and the grid padding stays **in** the delivered clip on purpose: the next scene continues from the latent's real tail, and a tail the film never showed would be a jump at the join (the report lists any `+N grid pad`). A scene too long to fit the context under H3's 362‑frame range opens on a hard cut with a warning. `continuity`: *flow everywhere* / *cut everywhere*. Up to four `ref_image` pictures (Ref2VA) apply to every scene.

Pair it with the **Short Film Writer**'s `continuity_mode` = *Continuous chain* and its `handoff_pass` (below), so the words and the pixels agree at every join. Outputs `file_paths` (list), a `report`, and **`audio`** — one continuous track for the whole film. Every clip's own audio is a hard splice at its seam (and an AAC encode of its own), which is what you hear as "a new clip". Because a continuing scene's pinned head *is* a second take of the previous scene's tail, the node blends the two across that overlap with an equal‑power crossfade (`seam_crossfade_ms`, 400 ms by default; hard cuts get a 5 ms declick), so the track has no splice at all and its length is unchanged. Wire `audio` into **Scenes Join**'s `replace_audio` (or a Save Audio); it is also written as `<prefix>_audio.wav` next to the clips.

### APNext H3 Music Video Chain Render (carry frames between scenes)
`H3MusicVideoChainRender` · writer lists + model/clip/VAEs + sampler → clips on disk

ComfyUI's list processing renders the writer's scenes as **independent clips** — scene N+1 never sees scene N's frames. This node renders the scenes **one after another inside one execution** and carries the last `context_frames` (22 ≈ 0.9 s; 5 / 22 / 39 / 56 … are H3's valid runs) of each delivered clip into the next one where the take should continue: per scene it builds the **Ref2VA** conditioning (up to four `ref_image` pictures, the same for every scene), masks the **song piece** into the audio latent (`master_audio`, the masked‑audio path), pins the previous tail to the head of the new latent, samples with your `sampler` / `sigmas` (`seed + k`), decodes, trims the pinned head and the grid padding off, saves the clip with its audio piece, and keeps its tail for the next scene. The continuing scene is rendered on a longer H3 grid length (piece + prefix, e.g. 124 + 22 → 158) and cut back to the piece's own frames, so the audio piece still matches the video to the frame; a piece too long to fit the prefix under H3's 362‑frame range opens on a hard cut with a warning.

**Which boundaries continue** is the Cut Plan's call (`continuity` = *cut plan decides*): a cut placed on a **drop, section start, stop or one of your taps stays a hard cut** — that is where the sync lives — while a cut placed on a mere onset / downbeat / lyric line **continues the take**. *flow everywhere* and *cut everywhere* force it. The writer, given the same `cut_plan`, marks those pieces `CONTINUES the previous take` and writes them as one continuing shot that opens on the previous piece's last frame (a `CONTINUING PIECES` directive), so prompt and pixels agree. The node's **`audio`** output is the song's pieces butted together sample‑exactly — the song itself over the rendered span — for **Scenes Join**'s `replace_audio`: joining the clips' own AAC‑encoded slices instead puts a small gap or click at every cut.

`carry` (appended last) decides how a continuing scene gets the previous one on the masked‑song path: **previous latent** (default) copies the sampled video latent tail straight into the new latent — the pack's `source_latent` input, no decode / re‑encode, nothing lost at each link, the same carry the Short Film Chain Render uses — while the song stays authoritative for the sound; **previous frames** decodes the tail and encodes it again (the original path). Mechanisms, in order of preference: this pack's **`H3MaskedSongLatent`** (see [Masked Song Latent](#apnext-h3-masked-song-latent-song-frozen-in-the-audio-latent)), else **`MiniMaxH3SongMaskedAVContext`** (ComfyUI‑H3‑Motion‑Context‑MultiRef — the node the masked‑audio workflows ran on before the swap; its `source_frames` + `context_length` path is a masked AV prefix protected by a nested denoise mask, with the song offset so the pinned head carries the previous piece's tail audio too), else the core **`MiniMaxH3AddGuide`** (previous tail anchored as a guide clip at frame 0; audio then generated). Outputs: `file_paths` (list) and a `report`. Ref2VA + carried frames is the same combination the Contex Loop pack's own Ref2V examples use.

**`conditioning_audio`** (optional): what H3 *listens* to instead of `master_audio` — Beat Emphasis' mix, or a drum stem — masked into the latent at the same song offsets — for a music video feed it the **vocal stem** (Beat Emphasis fed by AudioSeparation's vocals, or the stem itself), since the lips follow whatever the model listens to. The `audio` output and every clip's own audio still carry the original song pieces. Beat‑exact cut plans (Cut Plan `beat_snap`) need this node: it renders the next 17‑frame length up and delivers exactly the scene's frames.

### APNext H3 Cut Plan (scenes from the music)
`H3CutPlan` · Load Audio → **Cut Plan** → writer

Decide the scenes *before* anything is written. Give it the song and how long a scene may be (`min_seconds` / `max_seconds`, 5.2–15 s by default) and it returns every scene the video needs — how many, and each one's start and end on the H3 frame grid — placed where the music wants a cut: a measured **section start** (chorus in, verse in), a **drop** landing at the top of the new scene, a **stop**, a **downbeat**, a **lyric line** start, or failing all that the strongest onset. It is the same cutter the Music Video Writer runs internally, with the song's measured form and (optionally) the Sound Events list folded in, exposed as a node so the plan can be **seen and hand‑edited** first.

- `segment_mode` — Auto (cut on the music) / Fixed (as long as allowed) / Lyric lines (cut just before lyric phrases; needs timed `lyrics`).
- `lyrics` (optional, timed — typed) or `lyrics_in` (the same from a socket, e.g. Lyrics Transcribe; the typed box wins while it holds text) and `sound_events` (optional, from the Sound Events node) make the cuts lyric‑ and drop‑aware.
- `cut_placement` — which side of a beat the cut sits on. H3's frame grid moves a cut in 0.71 s steps, so a cut is never exactly *on* a hit: **Before the beat** puts the cut just ahead of the hit, so the hit is the first thing in the new scene (the classic music‑video cut — drops, downbeats and onsets all open the new scene; tapped cuts round *down* onto the grid); **After the beat** keeps the hit as the last thing in the outgoing scene and opens the new one on the release (tapped cuts round *up*); **Auto** is the cutter's usual mix (onsets and downbeats from either side, drops opening the new scene). The plan's header says `cuts before the beat` / `cuts after the beat`, and a drop that closes a scene reads `cut: drop closes the scene`.
- Outputs: `audio` (pass‑through), **`cut_plan`** (the text), `count`, `summary`. The node shows the plan on the canvas:

```
CUT PLAN: 17 scenes | 6-12 s per scene | 2:53.22 total
01  0:00.00 - 0:07.29   7.29s  quiet  intro        cut: drop lands here
02  0:07.29 - 0:13.88   6.58s  quiet  intro        cut: drop lands here
03  0:13.88 - 0:25.42  11.54s  quiet  chorus 1     cut: onset
```

**🎮 Tap the cuts** — a button on the node: it plays the upstream Load Audio file in the browser; press **space** (or the big button) where the video should cut, undo with backspace, and *Apply* writes the taps into `manual_cuts`. A tap lands a little after the sound, so each one is snapped to the nearest onset (±150 ms) and becomes a hard scene boundary (`cut: your tap`); stretches between taps longer than `max_seconds` are still cut on the music.

Wire `cut_plan` into the **Music Video Writer**'s `cut_plan` socket and the video is cut into **exactly these scenes**: the writer's own `segment_mode` / `max_segment_seconds` / `min_segment_seconds` are ignored (and greyed out on the canvas while the socket is linked). Only the scene number and the two clock times matter to the parser — everything after them is commentary — so a hand edit is a normal text edit: change an end time, delete a line, and the writer follows; the seams are repaired (each scene starts where the previous ends, the last one ends with the song).

**Beat‑exact cuts.** The frame grid moves a cut in 0.71 s steps, so on the grid a scene boundary is never exactly on a beat — but the grid is a *render* constraint, not a delivery one: Chain Render renders the next grid length up and trims the pad, so a scene may be any whole number of frames. `beat_snap` (*nearest beat* / *nearest downbeat*) takes the cutter's grid‑placed cuts and moves each one onto the nearest beat, rounded to the frame — every scene then **opens on the pulse to within half a frame (21 ms)**. The beats come from `beat_grid` (Beat Grid's `grid_json`, with your BPM override and offset) when wired, else from the measured structure. The plan's header carries `beat-exact` and each line says how far its cut moved (`cut: onset -> on the beat (-83 ms)`); the writer reads the marker and keeps the frame counts exact instead of snapping them back to the grid. **Chain Render only** — on the single‑clip path (Save Clip) the render pad would stay in the clip.

### APNext H3 Sound Events (bass hits / drops / stops)
`H3SoundEvents` · in every masked‑audio music‑video example, between Load Audio and the writer

**Where the hits are**, so the picture can be staged ON the music rather than near it. Wire the song in and the node finds and labels every moment worth cutting on, with the second it lands at:

```
# ~156.5 BPM | mid-energy (intensity 57/100) | compressed dynamics (5.3 dB)
[0:00.01] DROP      | heavy | the beat enters - full energy
[0:00.30] BASS HIT  | heavy
[0:03.63] BASS HIT  | solid
[0:20.14] BUILD     | heavy | a riser into the next drop
[0:27.99] STOP      | heavy | the music cuts out
```

Seven labels: **BASS HIT** (kick / low hit), **IMPACT** (crash, slam, snare that lands like a punch), **DROP** (the beat arriving), **STOP** (the music cutting out), **BUILD** (a riser into a drop), **SECTION** (the arrangement turning), **ACCENT** (bright top‑end — off by default, it floods).

Wire `events` into the **Music Video Writer**'s `sound_events` socket and each piece's brief carries only the hits inside *that clip*, timed **from the clip's own start** — `[+2.10s] BASS HIT (heavy)` — which is something a director can stage; `[1:47.3]` is not. The writer is told to land a cut, a camera hit, a light change or a move on them, to open the frame on a DROP, freeze or empty it on a STOP, tighten through a BUILD, and never to invent a hit that is not listed. With a list connected, the listed moments (and lyric phrases) are the **only** sync targets: the writer drops its own BPM / bar‑grid instruction and the “cut on the beat, everything hits” wording, so what the type toggles select is what the video syncs to.

**🎚 Preview & edit events** — a button on the node that opens a **full‑screen editor**: the whole waveform (decoded in the browser; ctrl+wheel or the buttons zoom, click seeks, space plays, a playhead follows), and under it every event as a **block** — its wind‑up, the instant it lands (the white line) and its settle, i.e. the exact `[+cue ->+land ->+settle s]` window the writer's brief lists. **Drag** a block to move it, drag its **left / right edge** to make the wind‑up or the settle longer or shorter, **click** it to edit type / landing time / strength / wind‑up / settle in the inspector (`▶ from here` plays the second before it), **Delete** strikes it out (again keeps it), **double‑click** the lane to add an event of the type chosen in the toolbar. Hit blocks are as **tall as they are strong**, and behind them, translucent teal, runs the **shaped signal** — the waveform after the Signal section's gain → EQ → curve, i.e. exactly what the detectors score — with a **beat grid** at the track's measured BPM, phase‑fitted to the kept hits, so a hit off the pulse shows at a glance (the toolbar reads `128.0 BPM · 83% of hits on it`). The side panel keeps the detector controls: `sensitivity` / `min_gap` (Recompute), a **Signal** section — `gain` (dB), a `curve` (dynamics: above 1 expands so the big hits stand proud, below 1 compresses so the soft ones come up), and a five‑band **EQ** (sub 60 Hz shelf, bass 150 Hz, low 400 Hz, mid 1.5 kHz, high 5 kHz shelf, ±24 dB; double‑click a slider to reset, `flat` resets all) that redraws the lane waveform live, can be **🎧 monitored** through the same chain, and lights `Recompute ●` until the detectors have been re‑run on it — the kind toggles, the `min_strength` / `max_strength` band with the density readout and the strength histogram (click a bar: min; shift‑click: max). Faded blocks are removed by the settings, striped ones are struck out, ✎ marks a hand edit, + an added event. **Apply to node** writes the settings, the struck‑out times (`rejected`) and the hand edits (`edits`, JSON) into the node; **Discard hand edits** clears them.

Widgets: `sensitivity` (0.25 hair‑trigger … 2.0 strict — it divides the median factor every detector scores against), `min_gap_seconds` (refractory gap, 0.18 s keeps a kick and its trailing snare apart), `max_events` (default **600** — a song has a kick on every beat, 300–400 in three minutes, and the writer already limits how many hits each scene's brief lists, so the cap only needs to catch runaway tracks; the old 120 threw two thirds of a normal song's kicks away and the survivors looked random against the waveform. Structure is always kept; when the cap bites, the hit stream is thinned **evenly** — the strongest hits of every 4 s survive, not the strongest of the whole song), `min_strength` / `max_strength` (a **strength band**, 0–1 with 1 = the loudest hit in the track — a hit's strength is 60 % how loud the low band *is* at the hit and 40 % how much it *rises*, so *light / solid / heavy* follow what you hear, not just the onset: the floor alone thins a busy track, the ceiling as well targets one layer of the mix — `0.35`–`0.45` keeps only the hits around 0.4, leaving the soft kicks *and* the big slams out; `max_strength` 1.0 = no ceiling; drops / stops / sections / builds are never dropped by either), one toggle per detector, `time_offset_ms` (a global nudge, positive = later — events are already **snapped onto the waveform's own peaks**: the detectors time the frame where the spectral flux rises, which is where the attack *starts*, 20–50 ms before the sound is actually there, so every hit and drop is moved to the envelope peak in a short window after it; 0 is where the sound is), `on_beat_weight` (prefer hits **on the beat**: the tempo is measured, a grid is phase‑fitted to the hits, a hit within 45 ms of a line keeps its strength and one a quarter‑beat or more away is scaled by `1 − weight` — 0.5 lets off‑beat hits survive only where nothing on the beat competes, 1.0 removes them given any `min_strength`; timing and structure untouched; the editor's `on-beat weight` slider is the same value and its grid is the same fit), `gain_db` / `dynamics_curve` / `eq_sub_db` / `eq_bass_db` / `eq_low_db` / `eq_mid_db` / `eq_high_db` (the **signal shaping** in front of every detector: gain → five RBJ biquads applied zero‑phase in the frequency domain → `|x|^curve` relative to full scale; the detectors score peaks against the surrounding audio, so on a compressed master a kick and a hat clear the floor by the same margin — cut the 5 kHz shelf, lift the 150 Hz bell and push the curve to 1.5 and the kicks become the loudest thing they see; all neutral by default), `rejected` (struck‑out times) and `edits` (the editor's hand edits as JSON — events moved, stretched, retyped, re‑weighted or added, each keyed by the time it was *detected* at so it survives a re‑detection with the same settings; clear it to return to the detector's list). A hand‑set window shows in the table as `[0:12.55] BASS HIT <- [0:12.05] -> [0:13.40]` and travels through both outputs to the writer. Outputs `audio` (pass‑through), `events` (the table), `events_json`, `summary`, `count`; the writer lists at most one hit per 1.5 s of each scene's length (6 for a 9 s clip), drops / stops / builds first.

**Drops and stops are confirmed, not just detected.** The signed 1.5 s loudness novelty still gives the instant, but each peak has to coincide with a change in an **energy state machine** (after scheb/sound‑to‑light‑osc): the 1 s level median against the track's 30 s median, three states — calm / neutral / intense — with **hysteresis** (entering a state takes +2.5 / −3 dB, staying only +0.8 / −0.9 dB) and a 0.5 s **hold** before any flip. A DROP must be a state rise between 0.75 s before the peak and 2.5 s after it, a STOP a state fall; a two‑bar fill or a dip‑and‑return moves the novelty but never flips the state, so it no longer reads as DROP, STOP, DROP.

**How it works.** The detectors are the ones fitted against real tracks in [graphgen](https://github.com/dagthomas)'s `AudioEngine`, ported from real‑time Web Audio to a whole‑song torch pass — band‑limited **spectral flux** (positive per‑bin rises over 20–250 Hz) against an **adaptive median floor** for bass hits, since a kick lifts every low bin at once while a bassline note change moves only a few; time‑domain **RMS rise** for impacts, because a slam is a broadband transient the FFT smears out; and **signed loudness novelty** (the 1.5 s after minus the 1.5 s before) for drops and stops. Offline improves on the original twice over: the median window is *centred* on each frame instead of trailing it, and thresholds are relative to the whole track, so the first bar scores as accurately as the last. Where both hit detectors fire on one drum the label is settled by **which band dominates** in linear energy — low is a BASS HIT, bright is an IMPACT — so one hit never appears twice under two names. Pure torch, no librosa; ~0.5 s for a 200 s song. Code: `nodes/h3/sound_events.py`.

### APNext H3 Beat Grid (BPM + every beat)

`H3BeatGrid` · workflows: [`h3_music_video_masked_audio_ollama_beatsync.json`](examples/h3/h3_music_video_masked_audio_ollama_beatsync.json)

A straight BPM detector. The tempo comes from the autocorrelation of the onset envelope (the same `estimate_bpm` Song Analysis reports), the **phase** is fitted to the song's own onsets — or, with Sound Events' `events` wired in, to the detected hits (brute force over 64 phases, the table then says `81/83 detected hits sit on the grid`) — and the **downbeat** is guessed from which beat carries the most energy (labelled a guess: a backbeat can put the snare on 2 and 4). Widgets: `bpm_override` (0 = measure), `beats_per_bar`, `offset_ms`, `subdivision` (beats / eighths / sixteenths / half notes). Outputs `audio` (pass‑through), `bpm`, `beat_times` (`[0:00.23] BEAT 1 | bar 1 | ONE` …), `grid_json` (bpm, period, phase, downbeat, `beats[]`, `bars[]`, `lines[]` — wire into Cut Plan's `beat_grid` and Beat Emphasis' `grid_json`), `summary`, `count`. It is the same grid the Sound Events editor draws.

### APNext H3 Resolution (1344x768 = 1.0 MP)
`H3ResolutionSelector` · drop‑in for core's *Resolution Selector* (same `width` / `height` outputs)

Frame sizes with the **trained frame as the unit**. H3 is trained at 1344×768 — 16:9, 1,032,192 pixels — and core's Resolution Selector counts megapixels from 1 MP = 1024², so its "1.0" is 3% under the trained frame and 16:9 lands on 1344×768 only by rounding luck. Here **`megapixels` 1.0 *is* 1344×768** (the slider says 1.0; the `info` output says 1.03 real MP), every other `aspect_ratio` gets the same pixel count at its shape — 9:16 → 768×1344, 1:1 → 1024×1024, 3:2 → 1248×832, 2:3 → 832×1248, 4:3 → 1152×864, 3:4 → 864×1152 — and the slider (0.25–2.0) scales them all together while keeping the aspect: 16:9 at 0.5 → 960×544, at 2.0 → 1920×1088. Every side is a multiple of 32 (`multiple`, the H3 nodes' step). The size is *chosen* on that grid rather than rounded onto it: of the four grid corners around the ideal size the one with the smallest error wins, the aspect weighing twice the pixel count, so a 4:3 request gives real 4:3 framing rather than the nearest pixel count at a slightly wrong shape. Outputs `width`, `height`, `megapixels` (the real count) and `info` (`1344x768 | 16:9 | 1.03 MP (1.00 of the trained frame)`). Code: `nodes/h3/resolution.py`.

### APNext H3 Lyrics Transcribe (timed lyrics from the song)
`H3LyricsTranscribe` · workflows: [`h3_music_video_masked_audio_ollama_beatsync.json`](examples/h3/h3_music_video_masked_audio_ollama_beatsync.json)

Whisper (large‑v3‑turbo, downloaded on first use, released after each run, cached per waveform) turns the song — or better its vocal stem — into timed `[m:ss] line` lyrics as a **node**, so the words exist *before* the cuts are placed. The Music Video Writer can transcribe on its own, but it runs last; wire this node's `lyrics` into **both** the Cut Plan's `lyrics_in` (`segment_mode` = *Lyric lines* then cuts right before the lines) and the writer's `lyrics_in`, and the whole chain follows the same words. The `lyrics` text boxes on both nodes stay editable next to the socket — whatever is typed there wins, so a misheard line is fixed by hand without unplugging anything. Outputs `lyrics`, `count`, `info`. Code: `nodes/h3/lyrics_transcribe.py`.

### APNext H3 Voice Over Music (conditioning mix)

`H3VoiceOverMusic` · workflows: every masked‑audio and Chain Render music video (`h3_music_video_masked_audio*.json`, `h3_music_video_latent_chain.json`) — it sits between AudioSeparation and the masked‑audio node's `master_audio` / Chain Render's `conditioning_audio`

The conditioning mix a mix engineer would make. H3 lip‑syncs to the audio in its latent and moves the picture to the rhythm it hears there, and every extreme loses something: the full song buries the consonants (mouths drift), a bare vocal stem has no beat in it (nothing to cut on), a synthetic thump track is not the song. This node puts the **voice on top at full level** (`voice`, AudioSeparation's Vocals) and the **real music under it** at `music_db` (default **‑9 dB**; `music_1..3` are summed — the whole song, or Drums + Bass + Other for the instrumental without a second copy of the voice), and with `duck_db` (default **‑6 dB**) a sidechain dips the music a little more only while the voice is actually sounding (5 ms in / 150 ms out), so words stay readable and the drums come back up between phrases. Peak‑normalised. Wire `audio` into Chain Render's **`conditioning_audio`** (or the masked‑audio node's `master_audio`); the original song still goes to the output. Code: `nodes/h3/voice_mix.py`.

### APNext H3 Beat Emphasis (conditioning audio)

`H3BeatEmphasis` · workflows: [`h3_music_video_masked_audio_ollama_beatsync.json`](examples/h3/h3_music_video_masked_audio_ollama_beatsync.json)

A **conditioning mix** for the masked‑audio render. H3 generates the picture to match the audio it is given, and Sync Check measured that on plain song audio only ~58% of hits get a picture change within 80 ms against 49% by chance — the model nods along but does not land on the beat. This node makes the beats impossible to miss in the audio the model *listens* to: a gain spike on every hit (`boost_db`, peaking exactly on the hit, `attack_ms` / `hold_ms` / `decay_ms`), everything between ducked (`duck_db`), an optional synthetic **click** / **thump** layered on each one (`layer`, `layer_level`), `scale_by_strength` so a light hit gets a light nudge, a `dynamics_curve`, and peak normalisation. Hits come from Sound Events' `events` (bass hits, impacts, accents, drops), from Beat Grid's `grid_json` (every beat; downbeats a little harder; a beat within 60 ms of a hit defers to the hit), or — with neither wired — from its own default detection. Wire its `audio` into the masked‑audio node's `master_audio` or Chain Render's **`conditioning_audio`**; never into Save Clip — the original song still goes to the output. **Feed it the vocal stem, not the song**: H3 lip‑syncs to whatever it is conditioned on, and the full mix spiked +9 dB and ducked −6 dB buries the voice (the beatsync example lost its lip‑sync exactly this way). With AudioSeparation's vocals as `audio`, `boost_db` 3, `duck_db` 0 and the thump layer, the voice stays intact and the thumps carry the beat. Whether the picture follows more tightly is an **experiment**: render with and without and run Sync Check on both.

### APNext H3 Resolution Planner (Crop Only)
`H3ResolutionPlannerCropOnly` · by gabbo

Plans a two‑stage generate‑then‑upscale resolution pair and centre‑crops the input image to that exact aspect ratio (no resampling or padding). `resolution_mode`, `stage1_megapixels`, `upscale_mode` (2× → stage‑1 steps of 32 / stage‑2 of 64; 1.5× → 64 / 96), `max_crop_percent`. Outputs `cropped_image`, `stage1_width/height`, `stage2_width/height`, `upscale_factor`, `plan_info`.

---

## Viewing

### APNext H3 Prompt Preview
`H3PromptPreview` · in every example workflow

Output node that renders any H3 prompt colour‑coded — `<Subject N>` sage, `<Picture N>` slate, `<Video N>` mauve, `<Audio N>` terracotta, `[Shot N]` gold, `(S1)` speaker ids teal, `<d>` dialogue honey, markers rose, section headers, timestamps, camera vocabulary — with a **Copy** button, a stats line (shots, subjects, pictures, lines of dialogue, chars) and pass‑through `text`. Connect the reference images to `image_1..9` and a thumbnail of each appears in a strip above the prompt and inline next to every `<Picture N>` tag (click to enlarge; unreferenced pictures are marked *(unused)*); the **Thumbs** button toggles them and is saved with the node. Wire a writer’s `h3_prompt` or a scenes writer’s `scenes_text` into `text`.

---

## Canvas helpers and the APNext theme

All in `Settings → APNext` (and the *APNext* top‑menu / canvas right‑click), implemented in `web/js/apnext_theme.js`:

| Setting | What it does |
|---|---|
| **APNext theme** (On/Off) | The *Dark Botanical* palette (installed as a normal custom ComfyUI palette “APNext”), graphgen’s 22 px dot grid on the plain near-black canvas, IBM Plex Sans / Cormorant / JetBrains Mono, rounded‑lg nodes, bold white headers. Off restores the previous palette, font, radius and colours. |
| **APNext node look** | Header tinted with the node’s hue + hued bottom border (body always dark), only the header corners rounded, 1 px panel border, port **tabs** pinned to the node edge that extend outward when connected or hovered. Works for the classic canvas renderer; with ComfyUI’s *Nodes 2.0* (Vue) renderer the same look is applied through CSS (dark body, coloured header, white bold title, square inputs). |
| **Recolour coloured nodes & groups** | Nodes/groups with their own colour (right‑click → Colors, or packs that pre‑colour) are drawn in the nearest botanical hue — header only, body neutral; stored colours untouched. |
| **Wire style** | ComfyUI default / Bezier / Smooth step / Step / Straight / Cable (springy sag + wobble). |
| **Gravity wires** (On/Off) | Hanging verlet ropes (graphgen `rope.svelte.ts`); `Wire slack` / `weight` / `segments` tune it. Off = physics fully stopped. |
| **Highlight drop targets** | While dragging a link, every slot that can take it pulses (tan; pink = already connected, would be replaced) — sized in screen pixels so it reads zoomed out. |
| **Connect sparks** | A particle burst + expanding ring in the link’s colour at the input when a connection is made. |
| **Layout tools** (`web/js/apnext_layout.js`) | Canvas right‑click → *APNext: layout* (also in the command palette): **Auto‑layout** (vendored dagre, left→right along the data flow, spacing in Settings → APNext → Layout), **Space out** (pushes overlapping nodes apart just enough to stop touching, keeps the arrangement, respects pinned nodes), **Align** left/right/top/bottom/row/column and **Distribute** horizontally/vertically on the selection. 2+ selected nodes = the selection, otherwise the whole graph; groups are re‑fitted around their nodes. |
| **Colour‑code APNext nodes** | Family colours in the palette hues: **sage** writers, **rose** Characters, **pink** LLM Backend (same as the `llm` link), **slate** previews, **gold** scene utilities, **mauve** context / prompt generators, **teal** vision nodes, **terracotta** the rest. Header only; only nodes you have not coloured yourself. |

---

## Workflow index

**Model per workflow:** every example that loads a picture (`LoadImage`) runs the **ref2va** checkpoint (`minimax_h3_ref2va_pruned_int8_convrot.safetensors`) with the writer in **Ref2VA** prompt mode, and the picture is wired to the **H3 Characters** node (`image` → the writer's `image_1`), the video node / Chain Render reference input and the Prompt Preview. Examples without a picture run **fl2va** with `prompt_mode = FL / T2VA`.

| File | Shows |
|---|---|
| [`h3_context_claude_code.json`](examples/h3/h3_context_claude_code.json) | Claude Code Writer steered by Time / Scene / Feelings / Cinematic context nodes → Reference‑to‑Video → Save Video; Preview. |
| [`h3_context_ollama_qwen.json`](examples/h3/h3_context_ollama_qwen.json) | The same with the plain Prompt Writer on `ollama:qwen3:8b` (`model_override`). |
| [`h3_llm_backend_writer_refiner.json`](examples/h3/h3_llm_backend_writer_refiner.json) | One **LLM Backend** feeding the Claude Code Writer *and* the Refiner; `session_id` chained; draft vs refined previews; the refined prompt is rendered. |
| [`h3_scenes_batch.json`](examples/h3/h3_scenes_batch.json) | Scenes Writer → every scene rendered in one queue → **Scenes Join** → one video. |
| [`h3_scenes_pick_one.json`](examples/h3/h3_scenes_pick_one.json) | Scenes Writer → **Scene Pick** (one scene per queue). |
| [`h3_crossover_batch.json`](examples/h3/h3_crossover_batch.json) | Three chained **Characters** → Crossover Writer → all scenes rendered → Scenes Join → one video. |
| [`h3_crossover_pick_one.json`](examples/h3/h3_crossover_pick_one.json) | Crossover Writer → Scene Pick. |
| [`h3_crossover_contex_chain.json`](examples/h3/h3_crossover_contex_chain.json) | Crossover Writer (continuous‑chain mode) → **Scenes → Contex Loop Plan** → the Contex‑Loop chain (previous tail carried forward, one assembled MP4). |
| [`h3_llm_backend_crossover.json`](examples/h3/h3_llm_backend_crossover.json) | The crossover batch written by a local model through an **LLM Backend** (`ollama:qwen3:14b`). |
| [`h3_music_video.json`](examples/h3/h3_music_video.json) | Load Audio → **Music Video Writer** (custom performer + wardrobe from a Characters node) → clips (`lengths` → `length`, `audio_segments` → `ref_audio_1`) → each clip saved directly (VAE Decode → Create Video → Save Video). |
| [`h3_music_video_masked_audio.json`](examples/h3/h3_music_video_masked_audio.json) | The strongest lip‑sync: writer in `audio_mode` = *Masked latent* → **AudioSeparation** vocal stem + `clip_starts` → **H3 Song Audio + Masked Video Context** (the song slice written into the H3 audio latent, protected from denoising) → sampler → each clip saved directly. The song is deliberately *not* wired to `ref_audio`. Needs ComfyUI‑H3‑Motion‑Context‑MultiRef + audio‑separation‑nodes. |
| [`h3_music_video_masked_audio_briefs.json`](examples/h3/h3_music_video_masked_audio_briefs.json) | The masked‑latent music video **plus custom scenes**: three chained **H3 Scene Brief** nodes → the writer's `scene_briefs` — brief 1 pinned to scene 01, the rest filling in order, every unplanned piece still the model's to invent. |
| [`h3_music_video_masked_audio_chain.json`](examples/h3/h3_music_video_masked_audio_chain.json) | The cut‑plan music video rendered by **Chain Render**: one node replaces Ref2VA → masked audio → guider → sampler → Save Clip, renders the scenes in order and carries the last 22 frames across every boundary the Cut Plan marks as a soft cut. The vocal stem goes into `conditioning_audio` so H3 lip‑syncs to the voice while `master_audio` keeps the song's timing. |
| [`h3_music_video_latent_chain.json`](examples/h3/h3_music_video_latent_chain.json) | The same cut‑plan music video with the previous scene carried as a **latent** (`carry` = *previous latent*, the Short Film Chain Render's carry: the sampled video latent tail copied straight into the next scene, no decode / re‑encode) while the song stays masked in; `save_latents` on for Scene Retake, and the node's `audio` output for the join. Vocal stem into `conditioning_audio` for the lip‑sync, as in the chain workflow. |
| [`h3_short_film_chain.json`](examples/h3/h3_short_film_chain.json) | The short film rendered by the **Short Film Chain Render**: the writer in *Continuous chain* mode with the hand‑off pass on, and one node replacing Ref2VA → guider → sampler → decode → Save Clip that carries the last 39 frames **and their sound** of every scene into the next, so the room tone and score do not restart at each cut. |
| [`h3_scene_retake.json`](examples/h3/h3_scene_retake.json) | **Scene Retake**: the short‑film model setup with one Scene Retake node reading the newest saved bundle — set the scene number, optionally rewrite the prompt, click **🎬 Retake this scene** (queues only that node), and the take continues from the previous scene's saved latent when the run was chain‑rendered. |
| [`h3_music_video_retake.json`](examples/h3/h3_music_video_retake.json) | **Music‑video Scene Retake**: the masked‑audio model setup with Load Audio → `master_audio`, so the retaken scene gets its slice of the song masked into the audio latent as the run did, and continues from the previous scene's saved take (chain‑rendered runs save one per scene). |
| [`h3_music_video_masked_audio_cut_plan.json`](examples/h3/h3_music_video_masked_audio_cut_plan.json) | The masked‑latent music video with the scenes **decided by the music first**: Load Audio → Sound Events → **Cut Plan** → the writer's `cut_plan` socket (its own segment widgets grey out). Tap the cuts on the Cut Plan node or edit the plan text. A muted **Decode → Sync Check** branch after the sampler measures, on a test run, whether the picture actually hits the beat. |
| [`h3_music_video_masked_audio_ollama.json`](examples/h3/h3_music_video_masked_audio_ollama.json) | The masked‑latent music video **written entirely locally**: an **H3 LLM Backend** on `ollama:qwen3.8:27b` (`num_ctx` 32768, thinking off) drives the writer — no API key, nothing leaves the machine. The model is multimodal, so a performer photo goes to both the writer’s `image_1` and the video node’s `ref_image_0` and `prompt_mode` is **Ref2VA**. Notes on the canvas cover the Ollama setup and the local‑model settings. |
| [`h3_music_video_masked_audio_ollama_beatsync.json`](examples/h3/h3_music_video_masked_audio_ollama_beatsync.json) | **Everything on the beat**, written locally: the Chain Render masked‑audio video with the Ollama `qwen3.8:27b` backend and a performer photo (writer `image_1`, Chain Render `ref_image_1`, Preview), plus the beat nodes — **Beat Grid** (BPM + every beat, phase‑fitted to Sound Events' hits) feeding **Cut Plan** `beat_grid` with `beat_snap` = nearest beat, so every scene opens ON the beat frame‑exact and the writer's briefs list each piece's `[beat]` offsets; **Lyrics Transcribe** on the vocal stem feeding both the Cut Plan (*Lyric lines*: a cut before every sung line, snapped on the beat) and the writer (Literal reading: every scene stages its lines to the letter, the performer lip‑syncs them); **Beat Emphasis** fed the **vocal stem** (a thump on every beat, `boost_db` 3 / `duck_db` 0 so the voice survives) into Chain Render's `conditioning_audio` — what H3 listens to: the lips follow the voice, the thumps carry the beat, while the original song still goes to the output. A/B it with Sync Check. |
| [`h3_music_video_masked_audio_ollama_beatsync_cuts.json`](examples/h3/h3_music_video_masked_audio_ollama_beatsync_cuts.json) | **Everything on the beat, every scene its own clip**: the beat‑sync workflow above with `continuity` = **cut everywhere** on both the Music Video Writer and Chain Render — every cut is a hard cut and every scene a fresh setup (own framing, own opening image, free to change place), nothing carried from the previous take. Same Lyrics Transcribe / Beat Grid / Beat Emphasis chain. |
| [`h3_music_video_masked_audio_ollama_blindref.json`](examples/h3/h3_music_video_masked_audio_ollama_blindref.json) | **Blind Ref2VA** — the performer photo reaches the video node as `<Picture 1>` and is rendered normally, but the writer never sees a pixel and takes who is in it from `image_notes`. Reference‑image quality on a model with **no vision at all**; the usual pick for a local or uncensored model. |
| [`h3_music_video_masked_audio_ollama_masked_song.json`](examples/h3/h3_music_video_masked_audio_ollama_masked_song.json) | The Ollama masked‑audio video on **this pack's own H3 Masked Song Latent** instead of the MultiRef node (no ComfyUI‑H3‑Motion‑Context‑MultiRef needed): Ref2VA `positive` → guider as before, Ref2VA `LATENT` → Masked Song Latent → Sample + Save. `preroll_seconds` 1.0 / `lookahead_seconds` 0.2 encode the song slice in context; set both to 0 for an A/B against the old hard‑cut behaviour. The vocal stem is also wired to `voice` (the **voice gate**): sung ticks frozen, gaps at `gap_denoise` 0.15 — unplug `voice` for one value everywhere. After the last scene **H3 Stitch Clips** joins the saved clips into `<project>_full.mp4` with the song muxed once — no AAC priming gaps at the seams. |
| [`h3_music_video_masked_audio_ollama_textonly.json`](examples/h3/h3_music_video_masked_audio_ollama_textonly.json) | The same local run with **no images anywhere** — `prompt_mode` = **FL / T2VA**, so it works on any text model. You write the performer, the wardrobe, the locations and the concept; whatever you leave out, the model invents differently in every scene. |
| [`h3_music_video_masked_audio_turbo.json`](examples/h3/h3_music_video_masked_audio_turbo.json) | The masked‑latent music video with a **speed render chain**: turbo LoRA → chunked feed‑forward → Sage + LowVRAM + SoL attention → EasyCache → Spectrum, sampled with **euler at 4 steps**. Needs the Spectrum / SoL / turbo patch packs and the turbo LoRA. |
| [`h3_short_film.json`](examples/h3/h3_short_film.json) | **Short Film Writer**: a manuscript adapted into a whole film — scene count or target length, verbatim dialogue, Beats plan, generated dialogue/score as the film's sound, each clip saved directly. `h3_short_film_turbo.json` renders the same with the 4‑step turbo chain. |
| [`h3_short_film_manual.json`](examples/h3/h3_short_film_manual.json) | The turbo short film with **hand‑authored scenes**: an **H3 Manual Scenes** node holds the whole script (the 11‑scene *Lighthouse Letter* example) — edit the envelopes, no LLM call, render directly with the 4‑step turbo chain. |
| [`h3_music_video_minimal.json`](examples/h3/h3_music_video_minimal.json) | **Music Video (Minimal)**: song + lyrics + a curated look + three sliders; a performer photo on `image_1` passes through to `ref_image_0`; each clip saved directly. |
| [`h3_presentation.json`](examples/h3/h3_presentation.json) | **Presentation Writer**: source material (benchmark numbers) presented by a custom presenter on a keynote stage — `scenes` → prompt, `lengths` → length, generated voice, each clip saved directly. |
| [`h3_music_video_dailies_gate.json`](examples/h3/h3_music_video_dailies_gate.json) | The music video with the **Dailies Gate**: the run holds live in the browser — print (render with hand edits), punch up selected takes through the writer's own session with director's notes, roll a new take, undo, or cut. |
| [`h3_music_video_minimal_dailies_gate.json`](examples/h3/h3_music_video_minimal_dailies_gate.json) | The **Minimal** music video with the Dailies Gate instead of Scenes Review — the fastest hands‑on loop: sliders → live desk → print/punch‑up. |
| [`h3_music_video_masked_audio_briefs_dailies_gate.json`](examples/h3/h3_music_video_masked_audio_briefs_dailies_gate.json) | The masked‑latent + Scene Briefs workflow with the Dailies Gate — plan takes by hand, then punch them up live before the expensive render. |
| [`h3_presentation_dailies_gate.json`](examples/h3/h3_presentation_dailies_gate.json) | The Presentation Writer with the Dailies Gate — check the `script`'s fact fidelity on the desk and punch up takes before rendering the talk. |
| [`h3_face_refine_mouthguard.json`](examples/h3/h3_face_refine_mouthguard.json) | **Mouth‑guarded face refine**: a rendered pass‑1 clip 2×‑upscaled → **Refine Encode** → Ref2VA re‑render toward face reference images, with the **Mouth Guard** protecting the lips + soundtrack so the lip‑sync survives; MediaPipe lips mask, muted same‑seed A/B and pixel‑composite branches. |

Every music‑video / presentation example defaults to **seed -1 (randomize)** — a queue always writes a brand‑new video (pin the seed to use the Review node's cached Continue flow) — and **saves each scene's clip directly** (VAE Decode → Create Video → Save Video, one file per clip); re‑add **H3 Scenes Join** before Create Video if you want one stitched file. In the music workflows every saved clip carries **its original slice of the song** (the writer's `audio_segments`, frame‑aligned) — never the model's re‑rendered audio, and in the masked workflows never the vocals‑only stem; the presentations keep the generated voice, which is their real soundtrack. The music examples also carry an **H3 Song Analysis** readout next to Load Audio, showing the measured BPM / intensity the writer steers by, and every masked‑audio (latent‑audio) example now also carries an **H3 Sound Events** node feeding the writer's `sound_events` socket, so each scene is written against the bass hits, drops and stops that actually land inside its own clip.

Every example has an **H3 Prompt Preview** wired to the writer’s prompt / scenes‑text output.

### Your own front‑end (API) and H3 Studio

[`examples/h3/api/`](examples/h3/api/) ships the main workflows in ComfyUI **API format** (`*.api.json` — the body for `POST /prompt`, every widget a named patchable input) plus [`h3_client.ts`](examples/h3/api/h3_client.ts), a dependency‑free TypeScript client: upload the song / reference images, patch inputs, queue, stream progress over `/ws`, answer the **Dailies Gate** from your own UI (`/apnext/h3/review_gate`), and collect the saved clips. See the folder's [README](examples/h3/api/README.md).

**H3 Studio** — a built‑in dashboard on that same API: the glowing clapper logo in the **bottom‑left corner** of the ComfyUI canvas opens it (`/extensions/comfyui_dagthomas/h3_dashboard.html`). Pick a workflow, build a **cast of up to four characters** (roster pick from the full character list or a custom description + wardrobe, each with their own uploaded **reference image** — synthesized into `H3Characters` / `LoadImage` nodes and wired to `cast_N`, `image_N` and the video node's ref slots at queue time), edit the remaining inputs (song upload included), queue, watch live progress, answer the Dailies Gate (print / punch‑up / new take / undo / cut), and play the finished clips — all without the node canvas. The logo breathes teal whenever a gate is holding a run for review. Code: `web/h3_dashboard.html`, `web/js/h3_dashboard_button.js`, `nodes/h3/dashboard.py` (the `/apnext/h3/api_workflows` + `/apnext/h3/characters` routes).

---

## Files and where to tune things

| Want to change… | Edit |
|---|---|
| The writing rules themselves | `data/h3/guide_base_en.md`, `guide_ref_en.md`, `guide_crossover_en.md`, `guide_chain_en.md` (the official guides, used verbatim as system prompts) |
| Director craft / gold examples (`director` on) | `data/h3/skills/*/SKILL.md` + `references/*.md` |
| The character list | `data/h3/characters.tsv` |
| How a context kind steers the scene | `utils/apnext_context.py` → `CATEGORY_GUIDANCE` |
| Wardrobe / location lock wording and the repair turn | `nodes/h3/scenes_support.py` |
| Scene envelope parsing | `nodes/h3/scenes_support.py` (`parse_scenes`, `envelope_contract`) |
| Which model runs an H3 node / local sessions | `nodes/h3/claude_code_support.py`, `nodes/h3/llm_backend.py`, `utils/llm_router.py` |
| Song segmentation / lyrics parsing / BPM + aggression profile | `nodes/h3/music_support.py` (`estimate_bpm`, `song_profile`) |
| Music‑video plot / ending variety, cliché bans | `nodes/h3/claude_code_music_video_writer.py` → `PLOT_ARCHETYPES`, `ENDING_MOVES`, `_ANTI_CLICHE_DIRECTIVE`, `_WILD_FUN_DIRECTIVE` |
| Template variables | `nodes/h3/template_vars.py`, `web/js/h3_template_vars.js` |
| Preview rendering / colours | `web/js/h3_prompt_preview.js` |
| Theme, wires, highlights, sparks, node colours | `web/js/apnext_theme.js` |
| Autogrow sockets (`context_N`, `image_N`) | `web/js/apnext_context_inputs.js`, `web/js/h3_reference_images.js` |

### TODO — watching upstream

- **[ComfyUI PR #15789](https://github.com/Comfy-Org/ComfyUI/pull/15789)** (`--disable-subgraph-caching` + per‑node `"no_cache": true`, draft as of 2026‑08): compute‑and‑release for intermediate tensors. Our render chains are the target case — a full music video keeps per‑scene latents, decoded frames, the joined batch **and** the upscaled batch in the output cache after the run. When it merges: wrap the render section of the example workflows (conditioning → sampler → decode → join → upscale) in a subgraph, or set `no_cache` on those heavy nodes in the workflow generator — while **keeping the writers cached**: the Scenes Review stop‑edit‑requeue flow and fixed‑seed re‑runs rely on the writer's cached output to avoid paying for the LLM twice. (The Dailies Gate holds one run open and doesn't need the cache at all, so it pairs cleanly with aggressive no‑caching.)
