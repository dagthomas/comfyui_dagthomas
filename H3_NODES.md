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
5. [Cast and model](#cast-and-model)
   - [APNext H3 Characters](#apnext-h3-characters)
   - [APNext H3 LLM Backend](#apnext-h3-llm-backend-ollama--local--api)
6. [Scene utilities](#scene-utilities)
   - [APNext H3 Scene Pick](#apnext-h3-scene-pick)
   - [APNext H3 Scene Counter](#apnext-h3-scene-counter)
   - [APNext H3 Scenes Join](#apnext-h3-scenes-join)
   - [APNext H3 Scenes → Contex Loop Plan](#apnext-h3-scenes--contex-loop-plan)
   - [APNext H3 Dailies Gate (print / punch up / cut)](#apnext-h3-dailies-gate-print--punch-up--cut)
   - [APNext H3 Resolution Planner (Crop Only)](#apnext-h3-resolution-planner-crop-only)
7. [Viewing](#viewing)
   - [APNext H3 Prompt Preview](#apnext-h3-prompt-preview)
8. [Canvas helpers and the Graphgen theme](#canvas-helpers-and-the-graphgen-theme)
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

When a node runs **off‑CLI** (Ollama etc.): `research` is ignored; the director skills are pasted into the system prompt (turn on the backend's `inline_skill_references` to paste the whole reference library too — much better prompts, needs a large context window: raise Ollama's `num_ctx`); `session_id` still works through a text‑only local session kept under ComfyUI's temp folder. A Claude Code session id cannot be resumed with a local model and vice versa — the error says so.

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
- `image_1..9` (Reference writers, Crossover, Music Video, Presentation): `<Picture 1>..<Picture N>` in connection order; downscaled copies go to the model, the originals come back out on the matching `image_N` outputs — wire those to the same slots on *MiniMax H3 Reference to Video*. `image_notes` (`Image 1: the singer`) tells the model what each picture is.

### Sessions

Every Claude Code node returns a `session_id`. Feed it to the **Refiner** (`session_id`) or the **Continue Writer** / any writer (`resume_session_id`) and the next turn continues the same conversation — the guide, the images and the previous prompt are already in context, so a revision costs a short turn instead of a re‑send. Works for Claude Code sessions and (text‑only) for local‑model sessions.

### Lists: one element per scene

The multi‑scene writers output `scenes` and `durations` (and, for the music video, `lengths` and `audio_segments`) as **ComfyUI lists**. A video node downstream therefore runs once per element — one queue renders every scene. Use **Scene Pick** to collapse the list to one scene, **Scenes Join** to stitch the rendered clips into one video, or **Scenes → Contex Loop Plan** to hand the scenes to the Contex‑Loop chain for true scene‑to‑scene continuity.

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
`H3ClaudeCodeMusicVideoWriter` · Claude Code / `llm` · workflows: [`h3_music_video.json`](examples/h3/h3_music_video.json), [`h3_music_video_masked_audio.json`](examples/h3/h3_music_video_masked_audio.json), [`h3_music_video_masked_audio_briefs.json`](examples/h3/h3_music_video_masked_audio_briefs.json) (masked latent + Scene Briefs)

Turns a **song** into a whole music video:

1. **Cuts the audio** into consecutive pieces no longer than H3 renders in one clip (`min_segment_seconds`–`max_segment_seconds`, 5–15 s). `segment_mode` *Auto* cuts on the music (spectral‑flux onsets, energy steps, section changes — and, with timed lyrics, right before a lyric line), *Fixed* takes the longest allowed piece every time, *Lyric lines* tries hardest to cut before a line. Every piece length is snapped to H3’s frame grid (5 + 17k frames at 24 fps) so each rendered clip is exactly as long as its audio slice — the stitched video never drifts.
2. **Writes one scene per piece** (four‑section prompt): the piece is `<Audio 1>`, reused 1:1 as the clip’s soundtrack; `performance_mode` *Performance* has the singer lip‑sync the piece’s lyric lines on camera (`<Subject 1> sings <d>[English] exact line</d> in sync with <Audio 1>`), *Narrative* answers the lyric with pictures, *Mixed* alternates; quiet pieces get long intimate shots, loud/peak pieces more cuts and the chorus look. Long songs are written in chunks of 6 scenes continuing one session (same synopsis and locks).
3. Emits matching **lists**: `scenes` → the video node’s `prompt`; `lengths` (frame counts) → `length`; `audio_segments` → `ref_audio_1`; `durations`; plus `segment_table` (`01  0:00.00 – 0:15.08  (15.08s, 362 frames)  energy: peak  lyrics: …`), `scenes_text`, `synopsis`, `cast`, `scene_count`, `song_seconds`, `session_id`, `info`, `image_1..9`.

Inputs: `audio` (Load Audio), `direction` (the concept), `lyrics` (`[0:15] line`, `0:15 line` or LRC `[00:15.20] line` for exact sync; `[Chorus]` tags kept; untimed lines spread evenly; empty = instrumental), `dialogue_language` (lyric language), cast (`cast_1..4` / `extra_cast` — an H3 Characters node in ✏️ custom mode with a `wardrobe` is made for the performer), locks, images, the Claude Code block, `llm`. Finish with **Scenes Join** (`replace_audio` = the song) → Create Video → Save Video. Code: `nodes/h3/claude_code_music_video_writer.py`, `nodes/h3/music_support.py`.

### APNext H3 Music Video (Minimal)
`H3MusicVideoMinimal` · Claude Code / Codex / `llm` · workflow: [`h3_music_video_minimal.json`](examples/h3/h3_music_video_minimal.json)

The one‑box music video: **song + lyrics + a cinematic look + three sliders — go.** The full [Music Video Writer](#apnext-h3-music-video-writer) runs underneath with opinionated defaults: the model invents the concept and the performer, the song is cut **on the music** (Auto segmentation), the imagery of every scene is staged from its lyric lines, dialogue language matches the lyrics, and the director skills are on.

- `visual_style` — the curated cinematic looks (35mm, Wes Anderson, neon noir, …); *Auto* lets the model pick one for the song.
- `performance` slider — 0–33 Narrative (nobody sings on camera), 34–66 Mixed, 67–100 Performance (lip‑synced on camera).
- `pace` slider — 0 = long slow pieces (~15 s), 100 = quick cuts (~6 s); cuts still land on the music.
- `wildness` slider — 0 grounded → 100 fully surreal (same bands as the full writer).
- `model` + `llm` socket — Claude Code aliases, `codex`, or any local/API model; `seed`; `image_1..9` reference pictures fix the performer's face and pass through to the video node.

Outputs the rendering essentials: `scenes` / `durations` / `lengths` / `audio_segments` **(lists)**, `scenes_text`, `session_id`, `info`, `image_1..9`, `clip_starts`. Runs save to `output/apnext_scenes/` like the full writer. Reach for the full writer when you need a written concept, cast lines, wardrobe/location locks, scene briefs, or the masked‑audio path. Code: `nodes/h3/music_video_minimal.py`.

### APNext H3 Presentation Writer
`H3ClaudeCodePresentationWriter` · Claude Code / `llm` · workflow: [`h3_presentation.json`](examples/h3/h3_presentation.json)

Turns **source material** — scientific findings, benchmark numbers, a paper abstract, a changelog, code, release notes — into a **presented video**: a presenter walks through the material to camera, scene by scene, with charts and graphics that display the real values. No audio input; the spoken script, the visual aids and the pacing are all generated.

1. **Facts are sacred**: `source_material` is the ground truth. Every number, unit, date, name and claim spoken or shown on screen must come **verbatim** from it — nothing invented, rounded or “improved”; where the material gives no number the presenter speaks qualitatively instead.
2. **Plans the talk in the synopsis**: an `Outline:` line per scene (`NN: the point covered + its visual aid`) covering all the material’s key points in teaching order — scene 01 hooks and names the topic, each middle scene covers one point, the last scene lands the takeaway. Long talks are written in chunks (`scenes_per_call`) continuing one session, with a talk‑so‑far recap.
3. **Stages the graphics as objects in the scene** per `presentation_format` (keynote stage + LED screen, whiteboard drawn by hand, news studio insets, lab demo, boardroom pitch, documentary, tech screencast — *Auto* picks one): chart type named, title and axis labels in verbatim double quotes, 2–5 labeled values from the material, and the visual relationship stated (which bar is taller, where the line rises) so the picture reads even where small text renders imperfectly. `visual_aids` sets how often (*Auto* / every scene / key data moments / none).
4. **Paces the script to the clock** (~2.3 words per second; in *Vary* duration mode a heavier point gets a longer scene, never a faster read) and emits the same matching **lists** as the other multi‑scene writers: `scenes` → the video node’s `prompt`, `lengths` (frame counts, snapped to H3’s grid) → `length`, `durations`; plus `scenes_text`, `synopsis`, `script` (teleprompter view of every spoken `<d>` line per scene), `cast`, `scene_count`, `total_seconds`, `session_id`, `info`, `image_1..9`.

Inputs: `source_material` (the ground truth), `direction` (who presents, where, the tone), `presentation_format`, `scene_count` (1–24), `duration_mode` / `scene_duration`, `visual_aids`, `continuity_mode`, cast (`cast_1..4` / `extra_cast`; empty = the model invents a presenter), locks + `enforce_wardrobe`, `image_1..9` reference pictures (`image_notes`), scene briefs, the Claude Code block, `llm`, `save_scenes` (JSON bundle for **Scenes Load**). `wildness` here is a pure scale (0 = sober, 100 = totally unhinged staging): only the level and its band label reach the model — no specific surreal elements are injected — and the facts, chart values and on‑screen text stay verbatim at every level. Finish with **Scenes Join** → Create Video → Save Video. Code: `nodes/h3/claude_code_presentation_writer.py`.

---

## Cast and model

### APNext H3 Characters
`H3Characters` · workflows: all crossover workflows, [`h3_music_video.json`](examples/h3/h3_music_video.json)

Character / actor / franchise lookup from `data/h3/characters.tsv`, or your own character.

- `character`: a `Character — Actor (Show)` entry, `🎲 random` (seeded, optionally narrowed by `franchise_filter`), or `✏️ custom` → describe your own in `custom_character` (`Lena: a middle‑aged woman with a limp and a silver bob` keeps `Lena` as the name; `Name (played by Actor) from Show` also works).
- `wardrobe`: this character’s wardrobe lock (3–5 exact anchors). It rides along on the cast line (`… | wardrobe: …`) into the Crossover / Music Video writers and is merged into their lock automatically, so the outfit lives with the character.
- `cast_in`: chain several Characters nodes into one cast list (A → `cast_in` of B → B’s `cast` → `cast_1`).
- Outputs `character`, `actor`, `franchise`, `file_path` (the reference clip), `cast` (`Character (played by Actor) from Show`, + wardrobe), `wardrobe`.
- On the canvas the node is **rose**; it feeds the writers’ **rose** `cast_N` sockets and also any `context_N` socket (then `{character1}` works too).
- Shoutout to [malcolmrey](https://huggingface.co/malcolmrey) for the crossover ideas and for finding valid characters — his [various dataset](https://huggingface.co/datasets/malcolmrey/various) is a great place to mine cast entries that the video model actually knows.

### APNext H3 LLM Backend (Ollama / local / API)
`H3LLMBackend` · workflows: [`h3_llm_backend_crossover.json`](examples/h3/h3_llm_backend_crossover.json), [`h3_llm_backend_writer_refiner.json`](examples/h3/h3_llm_backend_writer_refiner.json)

One node that says *“write with THIS model”* for every Claude Code H3 node. Drag its **pink** `llm` output into any number of writers’ `llm` sockets.

- `model`: every `ollama:` / `lmstudio:` / `local:` model your servers were serving at page load, the cloud API models, `auto-detect`, or **custom** + `model_name` (`ollama:qwen3:14b`, `lmstudio:qwen/qwen3-8b`, `local:my-model`, `claude:claude-sonnet-5`, `gpt:gpt-5.6`; a bare Ollama tag like `qwen3:8b` is understood).
- `base_url` (LAN box / other port; empty = the prefix default or `OLLAMA_BASE_URL` / `LMSTUDIO_BASE_URL` / `LOCAL_LLM_BASE_URL`), `temperature`, `max_tokens` (multi‑scene runs need room), `inline_skill_references`.
- Outputs `llm` and `model_used`. Unplug the socket and the writer is back on Claude Code.

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

### APNext H3 Scenes Load (from disk)
`H3ScenesLoad` · pairs with the writers' `save_scenes` toggle

Re‑render a saved run **without any LLM call**. The Music Video Writer's `save_scenes` toggle (on by default) stores every successful generation as a JSON bundle in `output/apnext_scenes/` — scenes, synopsis, segment times, durations, frame lengths, clip starts, cast, tables. This node's file picker lists the bundles (newest first; refresh the browser to re‑scan) and outputs mirror the writer's core outputs, so it drops into the same graph: `scenes` → review/render, `lengths` → `length`, `clip_starts` → the masked‑audio context node. Connect the same song to `audio` and the per‑clip `audio_segments` (`ref_audio_1`) are re‑sliced from the saved segment times (a duration mismatch is warned about). Cached by file mtime.

### APNext H3 Scenes Review (edit before render)
`H3ScenesReview` · in every example workflow, between the writer (or refiner) and the render

The commit gate: nothing renders until the scene text has been through it. In **Review** mode (the default) a queue run fills the node's editor with the incoming `scenes` (or a single `h3_prompt`) and stops cleanly — the editor shows the same colour‑coded tags as the Prompt Preview but is fully editable, all scenes at once or one scene at a time (scope selector / ◀ ▶). Edit what you like, then queue again: the mode has flipped to **Continue** automatically, so the next run renders exactly the editor text. **▶ Continue** and **🎲 Recreate** buttons do it in one click — Recreate bumps the writer's seed and reviews a fresh draft. **Bypass** passes through untouched. A `source-fingerprint` line in the editor header ties the edits to the scenes they came from: change the cast, direction, lyrics or seed upstream and the next queue **re-reviews with the fresh scenes automatically** instead of rendering the stale editor text. Keep the `=== SCENE NN ===` markers and the scene count (the writer's `durations` / `audio_segments` stay aligned; a mismatch is padded/trimmed with a warning), and give the writer a **fixed seed** so Continue runs reuse its cached answer instead of paying for new scenes. Outputs `scenes`, `scene_count`.

### APNext H3 Dailies Gate (print / punch up / cut)
`H3ScenesReviewGate` · workflow: [`h3_music_video_dailies_gate.json`](examples/h3/h3_music_video_dailies_gate.json)

The **live** sibling of Scenes Review, styled as a screening‑room "dailies desk": instead of stopping the queue, the run **holds open** at this node while the freshly written scenes wait in the browser, and it moves again the moment a button is pressed — no re‑queueing, no seed juggling. On the desk:

- **▶ Print it** — render exactly what is on the desk, hand edits included (the editor is the same colour‑coded overlay as the Prompt Preview, viewable **all takes at once or one take at a time** — ◀ Take NN ▶).
- **✍ Punch‑up** — type director's notes (and optionally which takes: `2, 4-5`; empty = the take being viewed, or all in the all‑takes view) and the selected scenes are rewritten **inside the writer's own model session** — synopsis, locks, lyrics, reference images still in context. Hand edits are folded in *first*, so "fix a line by hand, then have the model rebuild the scene around it" is one round trip. Works with Claude Code, Codex (`codex-`) and local‑model (`local-`) sessions; the gate auto‑switches its `model` to codex for a `codex-` session id.
- **🎲 New take** — the same rewrite with the notes dropped: the model is asked for a noticeably different version of the selected takes. Roll as many as you like.
- **↩ Undo** — every rewrite is kept in a server‑side history for the life of the gate, so a bad punch‑up rolls back instantly (and the history survives a browser reload).
- **✋ Cut** — end the run cleanly, render nothing.

Wire the writer's `scenes` in, plus `durations` (keeps rewritten takes on their exact lengths — matters for music videos) and `session_id` (what enables the AI rewrites; without it the desk is edit‑by‑hand only). `auto_approve_minutes` > 0 prints automatically with a **live countdown** in the header so unattended runs still render; `chime` plays a soft browser tone when takes land; ComfyUI's Stop button also releases the gate, and a browser reload re‑attaches to a waiting desk. Because the gate sits *before* the render, no video model weights are held while it waits. Choose this gate for hands‑on sessions; choose **Scenes Review** when you want the free stop‑edit‑requeue flow with the writer's cached seed. Outputs `scenes`, `scene_count`, `status`. Code: `nodes/h3/scenes_review_gate.py`, `web/js/h3_review_gate.js`.

### APNext H3 Resolution Planner (Crop Only)
`H3ResolutionPlannerCropOnly` · by gabbo

Plans a two‑stage generate‑then‑upscale resolution pair and centre‑crops the input image to that exact aspect ratio (no resampling or padding). `resolution_mode`, `stage1_megapixels`, `upscale_mode` (2× → stage‑1 steps of 32 / stage‑2 of 64; 1.5× → 64 / 96), `max_crop_percent`. Outputs `cropped_image`, `stage1_width/height`, `stage2_width/height`, `upscale_factor`, `plan_info`.

---

## Viewing

### APNext H3 Prompt Preview
`H3PromptPreview` · in every example workflow

Output node that renders any H3 prompt colour‑coded — `<Subject N>` sage, `<Picture N>` slate, `<Video N>` mauve, `<Audio N>` terracotta, `[Shot N]` gold, `(S1)` speaker ids teal, `<d>` dialogue honey, markers rose, section headers, timestamps, camera vocabulary — with a **Copy** button, a stats line (shots, subjects, pictures, lines of dialogue, chars) and pass‑through `text`. Connect the reference images to `image_1..9` and a thumbnail of each appears in a strip above the prompt and inline next to every `<Picture N>` tag (click to enlarge; unreferenced pictures are marked *(unused)*); the **Thumbs** button toggles them and is saved with the node. Wire a writer’s `h3_prompt` or a scenes writer’s `scenes_text` into `text`.

---

## Canvas helpers and the Graphgen theme

All in `Settings → APNext` (and the *APNext* top‑menu / canvas right‑click), implemented in `web/js/apnext_theme.js`:

| Setting | What it does |
|---|---|
| **Graphgen theme** (On/Off) | The *Dark Botanical* palette (installed as a normal custom ComfyUI palette “APNext Graphgen”), graphgen’s 22 px dot grid on the plain near-black canvas, IBM Plex Sans / Cormorant / JetBrains Mono, rounded‑lg nodes, bold white headers. Off restores the previous palette, font, radius and colours. |
| **Graphgen node look** | Header tinted with the node’s hue + hued bottom border (body always dark), only the header corners rounded, 1 px panel border, port **tabs** pinned to the node edge that extend outward when connected or hovered. Works for the classic canvas renderer; with ComfyUI’s *Nodes 2.0* (Vue) renderer the same look is applied through CSS (dark body, coloured header, white bold title, square inputs). |
| **Recolour coloured nodes & groups** | Nodes/groups with their own colour (right‑click → Colors, or packs that pre‑colour) are drawn in the nearest botanical hue — header only, body neutral; stored colours untouched. |
| **Wire style** | ComfyUI default / Bezier / Smooth step / Step / Straight / Cable (springy sag + wobble). |
| **Gravity wires** (On/Off) | Hanging verlet ropes (graphgen `rope.svelte.ts`); `Wire slack` / `weight` / `segments` tune it. Off = physics fully stopped. |
| **Highlight drop targets** | While dragging a link, every slot that can take it pulses (tan; pink = already connected, would be replaced) — sized in screen pixels so it reads zoomed out. |
| **Connect sparks** | A particle burst + expanding ring in the link’s colour at the input when a connection is made. |
| **Layout tools** (`web/js/apnext_layout.js`) | Canvas right‑click → *APNext: layout* (also in the command palette): **Auto‑layout** (vendored dagre, left→right along the data flow, spacing in Settings → APNext → Layout), **Space out** (pushes overlapping nodes apart just enough to stop touching, keeps the arrangement, respects pinned nodes), **Align** left/right/top/bottom/row/column and **Distribute** horizontally/vertically on the selection. 2+ selected nodes = the selection, otherwise the whole graph; groups are re‑fitted around their nodes. |
| **Colour‑code APNext nodes** | Family colours in the palette hues: **sage** writers, **rose** Characters, **pink** LLM Backend (same as the `llm` link), **slate** previews, **gold** scene utilities, **mauve** context / prompt generators, **teal** vision nodes, **terracotta** the rest. Header only; only nodes you have not coloured yourself. |

---

## Workflow index

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
| [`h3_music_video.json`](examples/h3/h3_music_video.json) | Load Audio → **Music Video Writer** (custom performer + wardrobe from a Characters node) → clips (`lengths` → `length`, `audio_segments` → `ref_audio_1`) → Scenes Join with `replace_audio` = the song → one music video. |
| [`h3_music_video_masked_audio.json`](examples/h3/h3_music_video_masked_audio.json) | The strongest lip‑sync: writer in `audio_mode` = *Masked latent* → **AudioSeparation** vocal stem + `clip_starts` → **H3 Song Audio + Masked Video Context** (the song slice written into the H3 audio latent, protected from denoising) → sampler → Scenes Join with `replace_audio` = the full mix. The song is deliberately *not* wired to `ref_audio`. Needs ComfyUI‑H3‑Motion‑Context‑MultiRef + audio‑separation‑nodes. |
| [`h3_music_video_masked_audio_briefs.json`](examples/h3/h3_music_video_masked_audio_briefs.json) | The masked‑latent music video **plus custom scenes**: three chained **H3 Scene Brief** nodes → the writer's `scene_briefs` — brief 1 pinned to scene 01, the rest filling in order, every unplanned piece still the model's to invent. |
| [`h3_music_video_minimal.json`](examples/h3/h3_music_video_minimal.json) | **Music Video (Minimal)**: song + lyrics + a curated look + three sliders; a performer photo on `image_1` passes through to `ref_image_0`; Scenes Join with `replace_audio` = the song. |
| [`h3_presentation.json`](examples/h3/h3_presentation.json) | **Presentation Writer**: source material (benchmark numbers) presented by a custom presenter on a keynote stage — `scenes` → prompt, `lengths` → length, generated voice, Scenes Join → one talk. |
| [`h3_music_video_dailies_gate.json`](examples/h3/h3_music_video_dailies_gate.json) | The music video with the **Dailies Gate**: the run holds live in the browser — print (render with hand edits), punch up selected takes through the writer's own session with director's notes, or cut. |

Every example has an **H3 Prompt Preview** wired to the writer’s prompt / scenes‑text output.

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
| Song segmentation / lyrics parsing | `nodes/h3/music_support.py` |
| Music‑video plot / ending variety, cliché bans | `nodes/h3/claude_code_music_video_writer.py` → `PLOT_ARCHETYPES`, `ENDING_MOVES`, `_ANTI_CLICHE_DIRECTIVE`, `_WILD_FUN_DIRECTIVE` |
| Template variables | `nodes/h3/template_vars.py`, `web/js/h3_template_vars.js` |
| Preview rendering / colours | `web/js/h3_prompt_preview.js` |
| Theme, wires, highlights, sparks, node colours | `web/js/apnext_theme.js` |
| Autogrow sockets (`context_N`, `image_N`) | `web/js/apnext_context_inputs.js`, `web/js/h3_reference_images.js` |

### TODO — watching upstream

- **[ComfyUI PR #15789](https://github.com/Comfy-Org/ComfyUI/pull/15789)** (`--disable-subgraph-caching` + per‑node `"no_cache": true`, draft as of 2026‑08): compute‑and‑release for intermediate tensors. Our render chains are the target case — a full music video keeps per‑scene latents, decoded frames, the joined batch **and** the upscaled batch in the output cache after the run. When it merges: wrap the render section of the example workflows (conditioning → sampler → decode → join → upscale) in a subgraph, or set `no_cache` on those heavy nodes in the workflow generator — while **keeping the writers cached**: the Scenes Review stop‑edit‑requeue flow and fixed‑seed re‑runs rely on the writer's cached output to avoid paying for the LLM twice. (The Dailies Gate holds one run open and doesn't need the cache at all, so it pairs cleanly with aggressive no‑caching.)
