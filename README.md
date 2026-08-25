# comfyui_dagthomas

**Advanced Prompt Generation & Multi-Model AI Integration for ComfyUI**

A comprehensive suite of nodes for ComfyUI featuring multi-provider LLM support (OpenAI, Gemini, Claude, Grok, Groq, QwenVL), local model inference (Phi, MiniCPM, Ollama), professional image effects, and advanced prompt generation tools.

---

## 📦 Installation

### Method 1: ComfyUI Manager (Recommended)
Search for "comfyui_dagthomas" in ComfyUI Manager and click Install.

### Method 2: Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dagthomas/comfyui_dagthomas
cd comfyui_dagthomas
pip install -r requirements.txt
```

---

## 🔑 API Key Configuration

Set your API keys as environment variables:

```bash
# OpenAI GPT
set OPENAI_API_KEY=sk-your-key-here

# Google Gemini
set GEMINI_API_KEY=your-key-here

# Anthropic Claude
set ANTHROPIC_API_KEY=your-key-here
# or
set CLAUDE_API_KEY=your-key-here

# xAI Grok
set XAI_API_KEY=your-key-here
# or
set GROK_API_KEY=your-key-here

# Groq
set GROQ_API_KEY=your-key-here
```

---

## 🧩 Node Categories

### 📝 Universal Nodes (Model-Agnostic)

#### APNext Universal Generator
**Display Name:** `APNext Universal Generator`

A model-agnostic prompt generator that automatically detects available API keys and supports all major LLM providers.

| Input | Description |
|-------|-------------|
| `input_text` | Base text to enhance |
| `model` | Select provider:model or "auto-detect" |
| `generation_mode` | Creative, Balanced, Focused, or Custom |
| `seed` | Seed for reproducible variations |
| `style_preference` | Cinematic, Photorealistic, Artistic, etc. |
| `detail_level` | Brief to Very Detailed output |

**Supported Models:**
- `gpt:gpt-4o`, `gpt:gpt-4o-mini`, `gpt:gpt-4-turbo`
- `gemini:gemini-2.5-flash`, `gemini:gemini-2.5-pro`
- `claude:claude-sonnet-4.5`, `claude:claude-3-5-sonnet`
- `grok:grok-beta`, `grok:grok-2-vision`
- `groq:llama-3.3-70b-versatile`

**Returns:** `(generated_prompt, model_used, seed_used)`

---

#### APNext Universal Vision Cloner
**Display Name:** `APNext Universal Vision Cloner`

Analyze images with any supported vision model to generate detailed descriptions or clone image styles.

| Input | Description |
|-------|-------------|
| `images` | One or more images to analyze |
| `model` | Vision model to use (auto-detect available) |
| `fade_percentage` | Blend percentage for multiple images |
| `analysis_mode` | Detailed Analysis, Style Cloning, Scene Description, Creative Interpretation |
| `output_format` | Text Only, JSON Structure, or Formatted Prompt |

**Returns:** `(formatted_output, raw_response, faded_image, model_used)`

---

### 🤖 Google Gemini Nodes

#### Gemini Prompt Enhancer
**Display Name:** `APNext Gemini Prompt Enhancer`

Enhances prompts with cinematic terminology and LLM refinement for video/image generation.

| Input | Description |
|-------|-------------|
| `base_prompt` | Original prompt to enhance |
| `enhancement_mode` | Random Mix, Cinematic/Lighting/Camera/Motion/Style Focus, Full Enhancement, or LLM Only |
| `use_llm` | Enable Gemini LLM enhancement |
| `intensity` | Enhancement intensity (0.1-2.0) |
| Optional dropdowns | visual_style, lighting_type, camera_angle, shot_size, lens_type, color_tone, etc. |

**Returns:** `(enhanced_prompt, random_enhanced, llm_enhanced)`

---

#### Gemini Custom Vision
**Display Name:** `APNext Gemini Custom Vision`

Analyze multiple images with custom prompts. Supports dynamic prompt templates with variable substitution.

| Input | Description |
|-------|-------------|
| `images` | Input images |
| `custom_prompt` | Custom analysis prompt |
| `dynamic_prompt` | Enable ##TAG##, ##SEX##, ##PRONOUNS##, ##WORDS## substitution |
| `fade_percentage` | Blend multiple images together |

**Returns:** `(output, clip_l, faded_image)`

---

#### Gemini Text Only
**Display Name:** `APNext Gemini Text Only`

Pure text generation with Gemini models. Supports dynamic prompt templates.

**Returns:** `(output, clip_l)`

---

#### Gemini Next Scene
**Display Name:** `APNext Gemini Next Scene`

Generate cinematic transitions for visual narratives. Creates the "next scene" based on a previous prompt and current frame.

| Input | Description |
|-------|-------------|
| `image` | Current frame image |
| `original_prompt` | Previous scene description |
| `focus_on` | Camera Movement, Framing Evolution, Environmental Reveals, Atmospheric Shifts |
| `transition_intensity` | Subtle, Moderate, or Dramatic |

**Returns:** `(next_scene_prompt, short_description)`

---

### 💬 OpenAI GPT Nodes

#### GPT Mini Generator
**Display Name:** `APNext GPT Mini Generator`

Efficient text generation using GPT-4o-mini.

| Input | Description |
|-------|-------------|
| `input_text` | Text to enhance |
| `happy_talk` | Enthusiastic vs professional tone |
| `compress` | Enable output compression |
| `poster` | Movie poster style formatting |

---

#### GPT Vision Cloner
**Display Name:** `APNext GPT Vision Cloner`

Clone image styles using GPT-4o vision capabilities with custom prompts.

---

#### GPT Custom Vision
**Display Name:** `APNext GPT Custom Vision`

Full custom vision analysis with GPT-4o.

---

### 🧠 Anthropic Claude Nodes

#### Claude Text Generator
**Display Name:** `APNext Claude Text Generator`

Text generation with Claude models (Claude 3.5 Sonnet, Claude Sonnet 4.5).

| Input | Description |
|-------|-------------|
| `input_text` | Text to process |
| `claude_model` | Model selection |
| `happy_talk`, `compress`, `poster` | Output style controls |
| `variation_instruction` | Custom instruction for creative variations |

---

#### Claude Vision Analyzer
**Display Name:** `APNext Claude Vision Analyzer`

Image analysis with Claude's multimodal capabilities.

---

### ⚡ xAI Grok Nodes

#### Grok Text Generator
**Display Name:** `APNext Grok Text Generator`

Text generation using xAI's Grok models.

---

#### Grok Vision Analyzer
**Display Name:** `APNext Grok Vision Analyzer`

Image analysis with Grok vision models.

---

### 🚀 Groq Nodes (Ultra-Fast Inference)

#### Groq Text Generator
**Display Name:** `APNext Groq Text Generator`

Lightning-fast text generation using Groq's optimized infrastructure with Llama and Mixtral models.

| Input | Description |
|-------|-------------|
| `groq_model` | llama-3.3-70b-versatile, llama-3.1-8b-instant, etc. |
| Other standard LLM inputs |

---

#### Groq Vision Analyzer
**Display Name:** `APNext Groq Vision Analyzer`

Fast image analysis with Groq vision models.

---

### 🔍 QwenVL Nodes (Local Vision)

#### QwenVL Vision Analyzer
**Display Name:** `APNext QwenVL Vision Analyzer`

Local vision analysis using Qwen-VL models. Downloads models automatically.

| Input | Description |
|-------|-------------|
| `images` | Input images |
| `qwen_model` | Qwen3-VL-4B-Instruct, etc. |
| `max_tokens` | Maximum response length |
| `keep_model_loaded` | Cache model in memory |

---

#### QwenVL Vision Cloner
**Display Name:** `APNext QwenVL Vision Cloner`

Clone image styles locally without API calls.

---

#### QwenVL Video Analyzer
**Display Name:** `APNext QwenVL Video Analyzer`

Analyze video content frame-by-frame.

---

#### QwenVL Next Scene
**Display Name:** `APNext QwenVL Next Scene`

Generate cinematic scene transitions locally using QwenVL models. Takes a previous scene description and **1-5 frame images**, then creates natural camera movements, framing evolution, and atmospheric shifts. Multiple frames help the model understand motion/progression.

| Input | Description |
|-------|-------------|
| `images` | 1-5 frame images (batch) |
| `original_prompt` | Previous scene description |
| `qwen_model` | QwenVL model to use |
| `prompt_file` | Custom prompt template file |
| `custom_prompt` | Override with inline prompt (optional) |
| `max_frames` | Max frames to use from batch (1-5) |
| `focus_on` | Camera Movement, Framing Evolution, Environmental Reveals, Atmospheric Shifts |
| `transition_intensity` | Subtle, Moderate, or Dramatic |
| `keep_model_loaded` | Cache model in memory |

**Returns:** `(next_scene_prompt, short_description)`

**Custom Prompts:** Create your own prompt templates in `data/custom_prompts/`. Use `##ORIGINAL_PROMPT##` as placeholder for the previous scene description. Included templates:
- `next_scene.txt` - Default detailed cinematography prompt
- `qwen_next_scene_simple.txt` - Simplified version
- `qwen_next_scene_video.txt` - Optimized for AI video generation

---

#### QwenVL Frame Prep
**Display Name:** `APNext QwenVL Frame Prep`

Utility node to prepare multiple images for QwenVL Next Scene. Accepts up to 5 individual images or a batch, scales them to max dimensions, and outputs a batched tensor.

| Input | Description |
|-------|-------------|
| `max_width` | Maximum width (default 1024) |
| `max_height` | Maximum height (default 1024) |
| `image_1` - `image_5` | Individual image inputs |
| `image_batch` | Pre-batched images (optional) |

**Returns:** `(images, frame_count)`

---

#### QwenVL Z-Image Vision
**Display Name:** `APNext QwenVL Z-Image Vision`

Analyzes images and outputs in Z-Image TurnBuilder chat format with `<|im_start|>/<|im_end|>` tokens.

---

### 🦙 Ollama Nodes (Local LLM)

#### Ollama Node
**Display Name:** `APNext OllamaNode`

Local LLM inference using Ollama. Supports any model installed in your Ollama instance.

| Input | Description |
|-------|-------------|
| `input_text` | Text to process |
| `model_name` | Any Ollama model (llama3, mistral, etc.) |
| `happy_talk`, `compress` | Output controls |

---

#### Ollama Vision
**Display Name:** `APNext OllamaVision`

Local vision analysis with Ollama multimodal models (llava, bakllava, etc.).

---

### 📸 MiniCPM Nodes (Local Vision)

#### MiniCPM Image Node
**Display Name:** `APNext MiniCPM Image`

Image understanding with MiniCPM-V 4.5 (OpenBMB). Supports thinking mode for complex reasoning.

| Input | Description |
|-------|-------------|
| `images` | Input images |
| `question` | Question about the image |
| `enable_thinking` | Deep reasoning mode |
| `precision` | bfloat16 or float16 |
| `unload_after_inference` | Free memory after use |

---

#### MiniCPM Video Node
**Display Name:** `APNext MiniCPM Video`

Video understanding and analysis.

---

### 🔬 Phi Nodes (Microsoft Vision)

#### Phi Model Loader
**Display Name:** `APNext Phi Model Loader`

Load Microsoft Phi-3.5-vision-instruct model.

| Input | Description |
|-------|-------------|
| `model_version` | Phi-3.5-vision-instruct |
| `image_crops` | 4 or 16 crops for detail |
| `attention_mechanism` | flash_attention_2, sdpa, or eager |

---

#### Phi Model Inference / Custom Inference
**Display Name:** `APNext Phi Model Inference`

Run inference with loaded Phi model.

---

### 🎨 Image FX Nodes

Professional image effects using optimized tensor operations.

#### APNext Bloom FX
Creates a bloom/glow effect on bright areas.

| Input | Description |
|-------|-------------|
| `intensity` | Bloom strength (0-5) |
| `threshold` | Brightness threshold (0-1) |
| `blur_radius` | Glow spread (1-50) |
| `blend_mode` | additive, screen, or overlay |

---

#### APNext Color Grading FX
Professional color grading with LUT support or manual controls.

| Input | Description |
|-------|-------------|
| `method` | manual or lut_file |
| `lut_file` | .cube, .3dl, or image LUT |
| `exposure` | -3 to +3 stops |
| `contrast`, `saturation` | Standard adjustments |
| `highlights`, `shadows` | Tone controls |
| `temperature`, `tint` | White balance |

**Supported LUT Formats:** .cube (Adobe/Blackmagic), .3dl (Autodesk/Flame), Image LUTs (.png, .jpg)

---

#### APNext Sharpen FX
Intelligent image sharpening.

---

#### APNext Noise FX
Add film grain and noise effects.

---

#### APNext Rough FX
Add texture and roughness.

---

#### APNext Cross Processing FX
Film cross-processing color effects.

---

#### APNext Split Toning FX
Separate color toning for highlights and shadows.

---

#### APNext HDR Tone Mapping FX
HDR-style tone mapping.

---

#### APNext Glitch Art FX
Digital glitch and databending effects.

---

#### APNext Film Halation FX
Classic film halation (light bleeding) effect.

---

### 📐 Latent Generators

#### APNext Latent Generator
**Display Name:** `APNext Latent Generator`

Generate latent tensors with intelligent dimension calculation.

| Input | Description |
|-------|-------------|
| `width`, `height` | Base dimensions (0 = auto-calculate) |
| `megapixel_scale` | Target megapixels (0.1-2.0) |
| `aspect_ratio` | 1:1, 3:2, 4:3, 16:9, 21:9 |
| `is_portrait` | Portrait orientation |

**Returns:** `(LATENT, width, height)`

---

#### PGSD3 Latent Generator
**Display Name:** `APNext PGSD3LatentGenerator`

Optimized latent generation for Stable Diffusion 3 pipelines.

---

### 📏 Resolution Planning

#### H3 Resolution Planner (Crop Only)
**Display Name:** `APNext H3 Resolution Planner (Crop Only) - by gabbo`

> Original node and algorithm by **gabbo**. Ported into this pack with the planning logic unchanged.

Plans a two-stage *generate → upscale* resolution pair and center-crops the input image to the **exact** aspect ratio of that plan, so nothing in the chain has to resample or pad. Step sizes are chosen so both stages always land on clean multiples of 32:

| Upscale | Stage 1 steps | Stage 2 steps |
|---------|---------------|---------------|
| `2x`    | 32            | 64            |
| `1.5x`  | 64            | 96            |

| Input | Description |
|-------|-------------|
| `image` | Source image; only its dimensions drive the plan |
| `resolution_mode` | `target_megapixels`, `max_stage1_from_input`, `max_final_from_input` |
| `stage1_megapixels` | Target stage 1 size in MP (0.05–4.00). `target_megapixels` mode only |
| `upscale_mode` | `2x` or `1.5x` |
| `max_crop_percent` | Max share of input area croppable (0–25%). The two `max_*` modes only; falls back to the least-lossy candidate if nothing fits |

**Modes**
- `target_megapixels` — hits the requested stage 1 megapixels while staying as close as possible to the input aspect ratio.
- `max_stage1_from_input` — largest stage 1 the input can feed natively within the crop budget.
- `max_final_from_input` — largest stage 2 (final) the input can feed natively within the crop budget.

**Returns:** `(cropped_image, stage1_width, stage1_height, stage2_width, stage2_height, upscale_factor, plan_info)`

`plan_info` is a human-readable summary of the chosen plan:

```
mode: target_megapixels @ 2x
input: 1920x1080
crop: 1917x1065 at (1,7) - 1.54% of area removed
aspect: 9:5
stage 1: 864x480 (0.40 MP)
stage 2: 1728x960 (1.58 MP)
```

---

### 🤖 Claude Code Node

Runs prompts through the [Claude Code](https://claude.com/claude-code) CLI installed on the machine, using **its** login. Nothing to configure in ComfyUI: if `claude` is on PATH and you have run it once to log in, it works. Install it, or point `CLAUDE_CODE_PATH` at the binary. No API key, no key field, no per-node authorisation step.

Why bother when there is already a `claude:` provider? Because this one uses your **subscription seat instead of API billing**, and because it is agentic — it can search the web and read files while it writes.

**Subscription, not API.** The node hides `ANTHROPIC_API_KEY` from the CLI (`use_subscription`, on by default) so the call authenticates with your Claude Code login. Usage then counts against your plan's rolling window — the CLI reports it as `rateLimitType: five_hour` — rather than being metered as API spend. The `cost=$…` figure in the `info` output is the CLI's estimate of *equivalent* API cost, not a charge. Heavy batches can still exhaust the window, and the node prints a clear warning when the limit is reached.

Under the hood it speaks Claude Code's bidirectional streaming protocol (`--input-format stream-json --output-format stream-json`), so images go inline as base64 blocks, progress appears in the ComfyUI console while the model works, and the process is killed the moment you cancel the queue.

#### APNext Claude Code
**Display Name:** `APNext Claude Code`

| Input | Description |
|-------|-------------|
| `prompt` | What to ask |
| `model` | `sonnet`, `opus`, `haiku`, `fable`, or `default` (whatever the CLI is set to) |
| `enable_research` | Lets it use WebSearch, WebFetch, Glob, Grep and Read while answering. Off = answers from the prompt alone. **Reaches the internet when on** |
| `use_subscription` | Hides `ANTHROPIC_API_KEY` from the CLI so your login is used. Off bills the API key instead |
| `timeout_seconds` | Give research runs room — 600s default |
| `seed` | Claude Code has no seed; this only controls ComfyUI caching. `-1` re-runs every queue |
| `image` *(optional)* | Reference frame(s), written to a scratch folder and read by the CLI |
| `system_prompt` *(optional)* | Replaces Claude Code's own system prompt |
| `resume_session_id` *(optional)* | Feed a previous node's `session_id` to continue that conversation |
| `working_dir` *(optional)* | Run inside a real folder so it can read those files. Empty = throwaway scratch folder |

**Returns:** `(text, session_id, info)` — `info` carries model, duration, turns and cost.

**Refinement chains.** Wire `session_id` into a second node's `resume_session_id` and it keeps the whole conversation, images included: node 1 writes the prompt, node 2 says *"make shot 2 wilder and cut the dialogue"* without re-sending any of it.

**Notes and limits**

- **No `temperature`, `seed` or `max_tokens`.** The CLI owns sampling. Vary the prompt to vary the output.
- **Slower than an API call** — 5–30s typical, longer with research on, because it is a full agent loop.
- **More tokens per call.** Claude Code carries its own tool preamble (~35k cached tokens), so a call costs more than hitting the API directly with the same prompt. The subscription seat is the trade.
- **Your `~/.claude/CLAUDE.md`, skills and hooks are deliberately ignored** (`--setting-sources ""`). A personal "always apply Go best practices" rule has no business rewriting a video prompt.
- **No tools at all by default.** Images need none — they are sent inline. Only `enable_research` grants any, and only the read-only set. `Bash`, `Write` and `Edit` are never granted.
- **Cancelling the queue kills the CLI** instead of leaving it running against your quota.

---

### 🎥 MiniMax-H3 Prompt Nodes

Both nodes take a **short idea, an image, or both** and expand it into a complete, spec-compliant MiniMax-H3 video prompt. The official MiniMax writing guides ship verbatim in `data/h3/` and are used as the system prompt, so the model follows the real spec rather than a paraphrase — edit those files to tune behaviour globally.

Any provider works — cloud, local, or the Claude Code CLI. `auto-detect` picks the first of Claude → GPT → Gemini → Grok → Groq that has an API key set, then the Claude Code CLI, then a running local server. When an image is connected it is sent as vision input, so the model describes the frame itself instead of you writing the description.

**Claude Code.** Pick `claudecode:sonnet` (or `opus` / `haiku`) in the `model` dropdown and the prompt is written by your locally installed Claude Code CLI, using its own login — no API key in ComfyUI, and the work counts against your Claude Code subscription seat rather than API billing. Images are handed over as real files for the CLI to read, so vision works exactly as it does with the other providers. The entries only appear when the CLI is installed. See [Claude Code Node](#-claude-code-node) for the details and the standalone node.

**Local models.** Any OpenAI-compatible server works: Ollama, LM Studio, vLLM, llama.cpp server, LocalAI, TabbyAPI, text-generation-webui. Whatever is running when the ComfyUI page loads is listed at the bottom of the `model` dropdown as `ollama:…`, `lmstudio:…` or `local:…` — start a server or pull a new model, refresh the page, and it appears (no ComfyUI restart). Use a vision-capable model, e.g. `ollama:qwen3-vl:8b`, if you want to connect images.

| Prefix | Default URL | Env override |
|--------|-------------|--------------|
| `ollama:` | `http://localhost:11434/v1` | `OLLAMA_BASE_URL`, `OLLAMA_HOST` |
| `lmstudio:` | `http://localhost:1234/v1` | `LMSTUDIO_BASE_URL` |
| `local:` | `http://localhost:8000/v1` | `LOCAL_LLM_BASE_URL` |

The optional `local_base_url` input points a single node somewhere else (a LAN box, a different port) — `192.168.1.10:11434` is enough, the scheme and `/v1` are filled in. The optional `model_override` input takes an exact `provider:model` string and wins over the dropdown, which is how you reach a model the dropdown has not discovered, e.g. `ollama:qwen3:8b`. Set `APNEXT_LOCAL_LLM_DISCOVERY=0` to skip the local probe entirely.

**Writing the Claude Code H3 nodes with Ollama instead (APNext H3 LLM Backend).** The six Claude Code H3 nodes (Claude Code Writer, Reference Writer, Refiner, Continue Writer, Crossover Writer, Scenes Writer) default to the Claude Code CLI, but each has an optional `llm` socket. Drop an **`APNext H3 LLM Backend (Ollama / local / API)`** node, pick a model (the dropdown lists every `ollama:` / `lmstudio:` / `local:` model your servers were serving at page load, the cloud API models, or `custom` with a free `model_name` such as `qwen3:8b`), optionally set `base_url`, `temperature`, `max_tokens`, and drag its `llm` output into any number of H3 nodes — they then write with that model through the shared LLM router. The node's own `model` dropdown also lists the discovered local models directly, so a quick switch needs no extra node. What changes off-CLI: `research` is ignored (no web tools), the `director` skills are pasted into the system prompt (turn on `inline_skill_references` on the backend node to paste their whole reference library too — much better prompts, but the system prompt passes 45k tokens, so raise `num_ctx` to match), and `session_id` / `resume_session_id` still work through a text-only local session kept under ComfyUI's temp folder (a Claude Code session id cannot be resumed with a local model and vice versa — the error says so). Everything else — wardrobe/location locks and their repair turn, template variables, outputs — is identical.

**Sizing an Ollama run (`num_ctx`, `thinking`, and the 📏 button).** An H3 system prompt is ≈9k tokens for a text-only run and ≈15k with reference images, *before* your song, cast and lyrics — and Ollama picks its own context window from free VRAM, as little as 4k, then silently truncates everything past it. Nothing errors; the model simply never sees the rules it is supposed to follow. So the backend node carries **`num_ctx`** (default 32768 — set it, don't leave it to the server) and **`thinking`** (default off: a reasoning pass costs wall-clock time and the H3 rules are already in the prompt; any `<think>` block that does arrive is stripped before parsing). Press **📏 Show 1024 tokens** on the node to see what a token actually is — a real 1024-token block cut from the H3 guide and counted with *your* model's tokenizer — what an H3 run spends against your `num_ctx`, and, for every model you have pulled, its KV-cache cost per token, how much context fits in this machine's VRAM versus its RAM, whether it has `vision` / `thinking` / `tools`, and a benchmark button that measures real tokens/sec and the GPU/CPU split. Three ready workflows, one per answer to *does the writing model have eyes*: [`h3_music_video_masked_audio_ollama.json`](examples/h3/h3_music_video_masked_audio_ollama.json) (`prompt_mode` = **Ref2VA** — the photo is bound as `<Picture 1>` and shown to the writer, so it needs a vision model), [`h3_music_video_masked_audio_ollama_blindref.json`](examples/h3/h3_music_video_masked_audio_ollama_blindref.json) (**Ref2VA blind** — the photo still reaches the video model and renders identically, but the writer never sees it and takes who is in it from `image_notes`, so **any text model works and you keep face consistency**), and [`h3_music_video_masked_audio_ollama_textonly.json`](examples/h3/h3_music_video_masked_audio_ollama_textonly.json) (**FL / T2VA** — no images anywhere, you write the wardrobe and locations yourself).

**Cutting on the music (APNext H3 Sound Events).** A new node next to Load Audio finds *where the hits are* and labels them with the second they land at — `BASS HIT`, `IMPACT`, `DROP`, `STOP`, `BUILD`, `SECTION` — using the detectors fitted against real tracks in graphgen's `AudioEngine`, ported from real-time Web Audio to a whole-song torch pass (band-limited spectral flux against an adaptive median for kicks, waveform RMS-rise for slams, signed loudness novelty for drops and stops). Wire its `events` output into the Music Video Writer's `sound_events` socket — already done in every masked-audio example — and each piece's brief carries only the hits inside that clip, timed **from the clip's own start** (`[+2.10s] BASS HIT (heavy)`), with a directive to land the cut, the camera hit or the light change on them. Roughly half a second to analyse a 200-second song, pure torch, no librosa.


**Simple vs. advanced form.** Every H3 node shows a short form by default; the rarely‑touched inputs are marked *advanced* and appear behind the node's **Show advanced inputs** toggle (ComfyUI's native mechanism). See `H3_NODES.md` for the per-node split.

**Template variables in the text boxes.** Every H3 writer expands `{variables}` in its free-text inputs (`idea`, `direction`, `extra_instructions`, `wardrobe`, `locations`, `image_notes`, `extra_cast`, `instruction`, custom style/language) from what is wired into the node, before the text is sent to the model: `{character1}` `{actor1}` `{franchise1}` `{cast1}` for the first H3 Characters node feeding the node (numbered in socket order — `cast_1` sockets first, then `context_1..8`; chained `cast_in` characters count oldest-first), `{characters}` ("A, B and C"), `{cast}` (every cast line), and `{context_1}` / `{cast_1}` for the raw text on a socket. So `{character1} barges into {character2}'s kitchen` works with two Characters nodes connected. A read-only **`{vars}`** strip under each writer's inputs lists what is currently available (click a chip to copy it), typing `{` in any multiline text box pops up an autocomplete of those variables (keep typing to filter, ↑/↓, Enter/Tab or click to insert, Esc to dismiss), and the console prints what was available and used on every run. Chained Characters nodes (A → `cast_in` of B → `cast_1`) give `{character1}` = A and `{character2}` = B; Characters nodes on separate sockets number in socket order. Unknown `{names}` are left as they are.

**Graphgen look + wire styles (optional).** `Settings → APNext → Graphgen theme` (also in the top menu under *APNext* and the canvas right-click menu) restyles ComfyUI like [graphgen](X:/KODE/graphgen): the *Dark Botanical* palette (warm near-black, tan accent, dusty-pink highlight, muted botanical port colours — installed as a normal custom colour palette named *APNext Graphgen*), graphgen's 22 px dot grid on the plain near-black canvas (opaque, no glows), IBM Plex Sans / Cormorant / JetBrains Mono, and — with *Graphgen node look* on — graphgen's node shell: rounded header corners on an otherwise square box with a 1 px panel border, a header that carries the node's hue as a faint tint plus a hued bottom border (body stays neutral), white semibold title text, and port tabs pinned to the node edge that extend outward when connected or hovered instead of circles; with *Recolour coloured nodes & groups* (on by default) every node or group that carries its own colour — right-click → Colors, or packs that pre-colour their nodes — is drawn in the nearest botanical hue and the node-colour menu swatches become botanical, so the whole canvas matches; *Off* restores the previous palette, font, radius and the original node colours (they are never modified, only drawn differently). **Wire style** is a separate setting with graphgen's edge styles: *ComfyUI default*, *Bezier*, *Smooth step*, *Step*, *Straight* and *Cable* (a springy wire that sags and wobbles after a drag — `cable.svelte.ts`); **Gravity wires** (a hanging verlet rope — `rope.svelte.ts`; `Wire slack` / `weight` / `segments` tune it) is its own on/off toggle that overrides the wire style, off by default. When a physics style is not selected its simulation is fully stopped (no loop, no extra redraws); when selected it sleeps as soon as nothing moves. The APNext panels (H3 Prompt Preview, `{vars}` strip, autocomplete, thumbnails) use the Dark Botanical colours by default, theme on or off. Three canvas helpers live under `Settings → APNext → Canvas helpers`, all on by default and independent of the theme: **Highlight drop targets** (while you drag a link, every slot that can take it pulses with a ring — pink when it is already connected and would be replaced; works from outputs and from inputs), **Connect sparks** (a particle burst in the link's colour at the input when a connection is made — graphgen's Sparks), and **Colour-code APNext nodes** (every node of this pack gets a family colour in the palette hues: sage writers, rose Characters, pink LLM Backend — the same pink as the `llm` link —, slate previews, gold scene utilities, mauve context generators, teal vision/caption nodes, terracotta the rest; only nodes you have not coloured yourself, and turning it off removes them again).

#### APNext H3 Prompt Writer
**Display Name:** `APNext H3 Prompt Writer`

Writes the base format — `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music` — per [VIDEO_PROMPT_WRITING_GUIDE_base_en.md](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).

| Input | Description |
|-------|-------------|
| `idea` | Your short prompt or image description — the thing being expanded |
| `task_type` | `T2VA` (text only), `I2VA` (first frame), `FL2VA` (first + last), `L2VA` (last frame). Non-T2VA emits the exact reference-alignment instruction line |
| `duration_seconds` | Drives cut times and the `S.SS` value in the alignment line |
| `shot_plan` | Auto, or force 1–4 shots |
| `visual_style` | Auto, one of the guide's styles (`Cinematic`, `live-action`, `2D-animated`, `3D CG`, `claymation`, `watercolor`, `vintage film`), or anything from the APNext Cinematic vocabulary (film stock/format, colour grading, the aesthetics list). Pick **Custom** and type your own in `custom_visual_style` - a filled-in custom box always wins over the dropdown |
| `wildness` | **0 = literal, 100 = fully unhinged.** See below |
| `camera_motion` / `camera_amplitude` / `camera_speed` | The guide's full camera vocabulary. Medium amplitude and normal speed are omitted from the output, as the spec requires |
| `include_dialogue` | Off ⇒ no `(Sx)` IDs and no `<d>` blocks at all |
| `dialogue_language` | What the characters actually speak — 36 languages, or `Auto` to let the model pick one that fits the setting |
| `custom_dialogue_language` *(optional)* | Anything not in the list — `Norwegian (Bergen dialect)`, `Latin`. Overrides the dropdown |
| `include_on_screen_text` | Whether readable signs/banners/subtitles appear |
| `include_soundscape` / `include_non_diegetic_music` | Off writes `N/A` into that field |
| `model`, `temperature`, `seed` | Provider selection and sampling — cloud models plus any local server that answered |
| `image` *(optional)* | Reference frame(s), sent as vision input |
| `extra_instructions` *(optional)* | Free-form extra direction |
| `model_override` *(optional)* | Exact `provider:model` string, beats the dropdown — e.g. `ollama:qwen3:8b` |
| `local_base_url` *(optional)* | Where the local server lives, e.g. `192.168.1.10:11434`. Empty = the default for the prefix |

**Returns:** `(h3_prompt, integrated_multimodal_description, overall_soundscape, non_diegetic_music, model_used)` — the full prompt plus each field split out for separate wiring.

---

#### APNext H3 Reference Prompt Writer
**Display Name:** `APNext H3 Reference Prompt Writer`

Writes the six-section full-reference format per [VIDEO_PROMPT_WRITING_GUIDE_ref_en.md](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md). Shares every option above, plus:

| Input | Description |
|-------|-------------|
| `task_type` | The `[bracketed]` summary prefix: `keyframe completion`, `reference generation`, `video editing`, `video continuation`, `audio reuse`, `audio reference`. Auto lets the model combine them with ` + ` |
| `reference_role` | How attached images get labelled: auto, `<Subject N>`, standalone `<Picture N>`, style-only, or storyboard |
| `word_target` | Target length of `detailed_description` (guide recommends 350–500) |
| `image_1` … `image_9` *(optional)* | Up to nine reference images — the same limit as ComfyUI's *MiniMax H3 Reference to Video* node. Sockets grow as you connect them: plug in `image_1` and `image_2` appears |
| `reference_notes` *(optional)* | Per-reference notes, one per line — also how you describe video/audio references you can't attach |

**Returns:** `(h3_prompt, subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music, model_used, image_1 … image_9)`

Attached image *k* **is** `<Picture k>` in the prompt, and it comes straight back out on the `image_k` output. Wire those outputs into the H3 video node's `image_1 … image_9` and the numbering in the prompt and the numbering the video model sees can never drift apart. The base writers do the same with `first_frame` / `last_frame`: frame 0 and the last frame of whatever you connected to `image` come back out for the video node's first/last-frame sockets.

---

#### H3 × Claude Code

Four nodes that write H3 through the local Claude Code CLI instead of an API key. They inherit every rule from the writers above — same guides, same camera vocabulary, same wildness bands — and swap `model`/`temperature` for the CLI's own controls. Use these instead of picking `claudecode:` in the dropdown when you want research or refinement; the dropdown is fine for a plain one-shot.

| Node | Display name | Writes |
|------|--------------|--------|
| Base | `APNext H3 Claude Code Writer` | The base format (T2VA / I2VA / FL2VA / L2VA) |
| Reference | `APNext H3 Claude Code Reference Writer` | The six-section full-reference rewrite |
| Refiner | `APNext H3 Claude Code Refiner` | A revision of an existing prompt |
| Continue | `APNext H3 Claude Code Continue Writer` | The prompt for the *next* clip, from the last frames of the previous one |

Shared inputs on all four: `model` (sonnet/opus/haiku/fable/default), `research`, `director`, `use_subscription`, `timeout_seconds`, `seed`, plus optional `working_dir`. The writers also take `resume_session_id`. Both writers return `session_id` and `info` in place of `model_used`.

**`director`** (on by default) loads the **H3 director skills** that ship in `data/h3/skills/`. Each is a Claude Code-style `SKILL.md` — short instructions always in context — plus a `references/` library opened on demand with the Read tool. They are written around how these nodes work (the node's task type, duration, shot plan, camera, dialogue toggles and wildness band are treated as decisions already made; attached image *k* is `<Picture k>`; the exact field labels the node parses), so they complement the numbered directives instead of second-guessing them.

| Skill | Loaded by | Always in context | Read on demand |
|-------|-----------|-------------------|----------------|
| `h3-prompt-director` | all three | obeying the node's directives, output boundary, timeline & continuity, speech / `<d>` tags, soundscape vs music, resumed-session revisions, silent validation | prompt grammar, edge cases, quick examples, H3/ComfyUI facts |
| `h3-base-format` | Writer, Refiner (base) | the three-field contract, verbatim I2VA / FL2VA / L2VA alignment lines, how the `image` batch maps to first/last frame | T2VA gold examples, keyframe gold examples, condensed official guide |
| `h3-ref2va` | Reference Writer, Refiner (ref) | six-section contract, `image_1..9` → `<Picture 1..9>`, what each `reference_role` means, roles, retention vocabulary, HOW-vs-WHAT | Ref2VA gold examples, full-reference guide, video style-transfer lab |
| `h3-style-craft` | all three | expanding `visual_style` into observable craft, one pack per layer, translating named references, animation timing without frame-rate claims, wildness scaling | style picker, pack catalogue, style anchors, temporal animation techniques |

The node grants read access to `data/h3/skills` with `--add-dir`, so it works with an empty `working_dir` and without touching your own Claude settings. The official MiniMax guide stays authoritative for format; the skills add craft. Edit a `SKILL.md` or drop new `.md` files into a `references/` folder to tune behaviour, and any of the four folders can be symlinked into `~/.claude/skills/` for interactive use.

**`research`** sends Claude Code to the web before it writes — how the real location looks, period-correct wardrobe, how the light behaves there, how the physical event actually unfolds — and folds what it finds in as concrete visual detail. It is instructed never to cite anything or add commentary, so the output stays a clean H3 prompt.

**Images on the Claude Code Writer.** The `image` socket carries the **keyframes** — the picture(s) the video model will actually get, per `task_type`: **I2VA** → `<Picture 1>` first frame; **L2VA** → `<Picture 1>` last frame; **FL2VA** → batch two, frame 0 = `<Picture 1>`, last = `<Picture 2>`; **T2VA** → context only, no `<Picture N>` (the node prints a warning). `first_frame` / `last_frame` hand them back for the H3 video node.

Nine **typed reference sockets** — `subject_1..3`, `scenery_1..3`, `object_1..3` — take pictures that should only be *described*. The video model never sees them; Claude puts what matters into words and ignores the rest:

| Socket | Carries over | Ignored |
|--------|--------------|---------|
| `subject_N` | who they are — face, hair, build, wardrobe, marks; described precisely in `[Shot 1]` and kept identical across shots | the photo's backdrop, light, framing, mood — the scene comes from your idea |
| `scenery_N` | the place — architecture/terrain, light, weather, palette, layout, as the setting | any people in the picture |
| `object_N` | a prop/product — shape, colour, material, markings, scale | where it sits in the photo |

Any sizes mix freely; numbering follows connection order per kind. With references but nothing on `image`, the prompt is written as T2VA whatever `task_type` says (there is nothing to align), and the node says so. Typical use: a character sheet on `subject_1`, a location photo on `scenery_1`, *“she walks into the bar and orders”* as the idea.

**The refiner** is the reason sessions matter. Wire a writer's `session_id` into it and describe the change in plain language — *"use two shots instead of one, and set it at night"*. Resuming means the guide, the reference images and the model's own reasoning are still in context, so it edits surgically rather than rewriting from scratch, and you send only the instruction. Leave `session_id` empty and it still works, re-sending the prompt with the matching guide; it auto-detects which format it is looking at. Its 10 outputs cover both formats, and fields the format does not use come back empty.

```
[Load Image] ──► [H3 Claude Code Writer] ──h3_prompt──► [Preview as Text]
                          └──session_id──► [H3 Claude Code Refiner] ──► [Preview as Text]
                                             instruction: "make shot 2 wilder"
```

**The continue writer** chains clips. Wire the decoded frames of a generated clip into `frames`; it keeps the last `frame_count` frames, `frame_stride` apart (default 4 frames, one every 6 — the final frame is always included), and shows them to Claude Code together with the `previous_prompt` and your `idea` for what happens next. It writes an **I2VA** prompt for a new clip of `duration_seconds` where the last frame is `<Picture 1>` at 0.00 s, and hands that frame back out as `first_frame` for the video node; the sampled frames come out as `context_frames` so you can preview what it saw. The earlier frames are context only — they show which way the camera and the action were moving, so the continuation does not reverse a pan or freeze mid-swing. Switch `continuation_mode` to **T2VA** to cut to a new scene of the same story instead of continuing seamlessly. Feed it the previous node's `session_id` (the writer's, or the previous Continue Writer's) and Claude Code still has every prompt and frame so far in context, so characters, speaker IDs and the soundscape stay consistent over a long chain; without it, paste every prompt so far into `previous_prompt`, oldest first. Same directive set as the base writer (shot plan, style, camera, dialogue toggles, wildness), with `visual_style` Auto meaning *keep the previous clip's look*.

```
[H3 Claude Code Writer] ──h3_prompt──► [MiniMax H3 video] ──► [Decode] ──frames──► [H3 Claude Code Continue Writer] ──h3_prompt──► [MiniMax H3 video]
          └──h3_prompt───────────────────────────────────────────────previous_prompt──┘        └──first_frame──────────────────────────────┘
          └──session_id──────────────────────────────────────────────resume_session_id┘
```

#### APNext context sockets (all writer nodes)

Every H3 writer — `APNext H3 Prompt Writer`, `APNext H3 Reference Prompt Writer`, all the Claude Code H3 nodes (Writer, Reference Writer, Refiner, Continue Writer, Crossover Writer, Scenes Writer) and the generic `APNext Claude Code` node — has `context_1..context_8` sockets (they grow as you connect them). Wire any other APNext node into them: **Time, Scene, Poses, Plots, Feelings, Cinematic, Photography, Science, Geography, Architecture, Fashion, People, Interaction, Stuff, Vehicle, Typography, Brands, Art/Artist, Keywords, Video Game, Human, Character**, or `H3 Characters` (`cast`) and another writer's `h3_prompt`.

The node looks up which node type feeds each socket (via ComfyUI's hidden graph inputs), labels the socket in the UI (`context: time`), and hands the model a block like:

```
CONTEXT FROM CONNECTED APNEXT NODES
1. [time] from APNext Time (fields: eras=Victorian, time (Random))
   Input: Victorian era, dusk,
   Use it as: Period, era, decade or time of day. Make it visible: period-correct wardrobe, props, ... State the time of day in every shot's opening line ...
2. [feelings] from APNext Feelings (fields: creepy (Multiple Random))
   Input: a creeping dread, flickering candles,
   Use it as: Emotional tone. Express it through performance, delivery, body language, lighting ... not by naming the emotion.
```

The per-kind instructions live in `utils/apnext_context.py` (`CATEGORY_GUIDANCE`) — edit them to change how a category steers the scene. Unknown sources are passed through as generic context. Example workflows: `examples/h3/h3_context_claude_code.json` and `h3_context_ollama_qwen.json` (same sockets on the plain writer with `model_override = ollama:qwen3:8b`).

#### APNext H3 Characters
**Display Name:** `APNext H3 Characters`

A lookup node for the character reference set. It reads `data/h3/characters.tsv` (`Relative File Path`, `Character / Subject Name`, `Real Actor / Actress`, `Franchise / Show`) and outputs the pieces as separate strings plus a ready-made cast line.

| Input | Description |
|-------|-------------|
| `character` | Dropdown of every unique `Character — Actor (Franchise)` entry, `🎲 random`, or `✏️ custom` (describe your own in `custom_character`) |
| `franchise_filter` | Only used with random: restrict the pool to one franchise/show |
| `seed` | Drives the random pick, so it stays stable per queue |
| `custom_character` *(optional)* | Your own character, free text; used with `✏️ custom` (or whenever filled in). `Lena: a middle-aged woman with a limp and a silver bob` keeps `Lena` as the name the writers and `{characterN}` use; `Name (played by Actor) from Show` also works |
| `wardrobe` *(optional)* | This character's wardrobe lock: 3–5 exact anchors, comma separated. It travels with the cast line (`… | wardrobe: …`) into the Crossover / Music Video writers, which merge it into their wardrobe lock and copy it word-for-word into every shot — so the outfit lives with the character, not the writer. A line typed into the writer's own `wardrobe` box for the same name wins |
| `cast_in` *(optional)* | Cast lines from an upstream Characters node; this node's line is appended, so several chain into one cast list |

| Output | Description |
|--------|-------------|
| `character` / `actor` / `franchise` | The three columns (franchise verbatim, including any appended look description) |
| `file_path` | The reference clip's relative path |
| `cast` | `Character (played by Actor) from Show` (+ ` | wardrobe: …` when set) — the `subject_definitions:` form, with anything chained through `cast_in` above it |
| `wardrobe` | The wardrobe text, passed through |

Edit the TSV to add or remove entries; duplicates (same character, actor and franchise) are collapsed automatically.

#### APNext H3 Crossover Writer
**Display Name:** `APNext H3 Crossover Writer`

Takes a cast (from `H3 Characters` nodes, or typed by hand) and your steer, and has the local Claude Code CLI write **1–10 crossover scenes** — characters from different shows sharing one story — each a complete four-section T2VA prompt (`subject_definitions:` / `integrated_multimodal_description:` / `overall_soundscape:` / `non_diegetic_music:`). The rules come from `data/h3/guide_crossover_en.md`, distilled from rendered crossover productions: actor pinning, `<Subject 1>` speaker binding, silence mandates, `not in frame` isolation, positioned two-shots, no dead air, grounded entrances, hand-offs between scenes. No title cards are written unless you ask.

| Input | Description |
|-------|-------------|
| `direction` | Your creative brief: premise, tone, where it happens, what must happen, who should clash |
| `extra_cast` | Extra characters typed by hand, one per line |
| `cast_1..cast_4` *(optional)* | `cast` outputs from Characters nodes (chain several through `cast_in`, or use several sockets) |
| `scene_count` | 1–10 scenes |
| `duration_mode` / `scene_duration` | Fixed length per scene, or let Claude pace each scene 5–15 s (`scene_duration` is the fallback) |
| `continuity_mode` | **Independent clips** (each scene its own T2V clip, hard cuts, `already speaking` openers) or **Continuous chain** — scenes written for C2V / motion-context chaining per `data/h3/guide_chain_en.md`: scene N+1 opens on scene N's last frame, one continuous take with no `the shot cuts to`, a 2 s silent hand-off before a *new* speaker's first line, outgoing person kept as a tagged subject, rotating moving closers, one lighting string |
| `shots_per_scene`, `visual_style`, `dialogue_language`, `wildness` | Same meaning as on the other writers |
| `wardrobe` *(optional)* | Wardrobe lock, one line per character (`Sheldon: dark-brown corduroy jacket, forest-green cotton T-shirt, small silver ring in the LEFT nostril`), copied word-for-word into **every shot** the character is on screen in. Empty = Claude fixes one outfit per character in the synopsis's `Wardrobe:` lines and repeats it. Anchors are exact phrases — precise colour + material + garment, accessories and marks with their side — no synonyms, nothing extra, nothing dropped. The Scenes Writer has the same input |
| `locations` *(optional)* | Location lock, one line per recurring place (`Sheldon's living room: beige three-seat sofa facing a wall-mounted TV on the LEFT, tall bookshelf of comics behind it, bay window with white blinds on the RIGHT, warm tungsten floor lamp in the far corner`), copied word-for-word into the first shot of **every scene set there**, so the same room looks the same in every scene. Empty = the model names each place used in more than one scene and fixes 3–6 anchors in the synopsis's `Locations:` lines, then repeats them. Anchors are fixed features with colour/material and position (LEFT/RIGHT from the main camera side), openings, surfaces and practical light — no synonyms (`sofa` never becomes `couch`), nothing dropped or moved. The Scenes Writer has the same input |
| `enforce_wardrobe` *(optional, default on)* | After writing, the node parses the synopsis `Wardrobe:` and `Locations:` lines and checks that every shot a character is on screen in restates all of their anchors verbatim (off-screen mentions don't count), and that every scene set in a locked place restates all of that place's anchors. Any miss triggers one combined repair turn in the same session; the result is in `info` (`wardrobe: ok (2 locked) | locations: ok (1 locked)` / `repaired 6 -> 0`) and the console lists each miss |
| `image_1..image_9` *(optional)* | Reference images, `<Picture 1>..<Picture N>` in connection order. Downscaled copies go to Claude so it can recognise who/what each picture is (add `image_notes` like `Image 1: Sheldon`, `Image 3: the diner`); pictured characters are bound to their picture in `subject_definitions` and the wardrobe lock is taken from the picture. The originals come back out on the matching `image_N` outputs — wire them to the same slots on *MiniMax H3 Reference to Video* |
| `model`, `research`, `director`, `use_subscription`, `timeout_seconds`, `seed` | The Claude Code block; `director` loads the `h3-crossover` skill with verified gold examples |

| Output | Description |
|--------|-------------|
| `scenes` **(list)** | One prompt per scene — a downstream video node runs once per element |
| `durations` **(list)** | The matching seconds per scene; wire into your frame-count math |
| `scenes_text` | All scenes with `=== SCENE NN | duration: S.S ===` envelopes, for preview/saving |
| `synopsis`, `cast`, `scene_count`, `session_id`, `info` | Story summary, the merged cast, how many scenes parsed, and the Claude Code session for the refiner |
| `image_1..image_9` | The reference images passed straight through, same order as the inputs — wire to *MiniMax H3 Reference to Video* |

#### APNext H3 Claude Code Scenes Writer
**Display Name:** `APNext H3 Claude Code Scenes Writer`

The Claude Code prompt writer, but for a run of scenes: one idea in, **1–10 consecutive T2VA prompts** out in the base three-field format, each with its own duration, forming one continuous story. Same director skills, camera vocabulary, dialogue toggles and wildness bands as `APNext H3 Claude Code Writer`. Optional `image` and `subject_/scenery_/object_` sockets are described into every scene for consistency (nothing becomes `<Picture N>`). Has the same `continuity_mode` switch (independent clips vs continuous chain). Outputs mirror the crossover writer: `scenes` and `durations` are lists.

#### APNext H3 Music Video Writer
**Display Name:** `APNext H3 Music Video Writer`

Turns a **song** into a whole music video. The node cuts the audio into consecutive pieces no longer than H3 renders in one clip (5–15 s), choosing the cut points on the music — onsets, energy steps, section changes, and (with timed lyrics) right before a lyric line — with every piece length snapped to H3's frame grid (5 + 17k frames at 24 fps) so each rendered clip is exactly as long as its audio slice and the stitched video never drifts. It then writes **one scene per piece** (four-section H3 prompt): the piece is `<Audio 1>`, reused 1:1 as the clip's soundtrack; in *Performance* mode the singer lip-syncs the piece's lyric lines on camera (`<Subject 1> sings <d>[English] exact line</d> in sync with <Audio 1>`), *Narrative* mode answers the lyric with pictures, *Mixed* alternates; quiet pieces get long intimate shots, loud/peak pieces more cuts and the chorus look. Long songs are written in chunks of 6 scenes that continue one session (same synopsis, wardrobe and location locks).

| Input | Description |
|-------|-------------|
| `audio` | The song (Load Audio) |
| `direction` | The concept: performer, place, look, arc, motifs, what the chorus looks like |
| `lyrics` | One line per line. `[0:15] line`, `0:15 line` or LRC `[00:15.20] line` make the sync exact; `[Chorus]`-style tags are kept; untimed lines are spread evenly (approximate). Empty = instrumental |
| `performance_mode` | Performance / Narrative / Mixed |
| `segment_mode`, `max_segment_seconds`, `min_segment_seconds` | *Auto* cuts on the music inside the allowed range, *Fixed* takes the longest allowed piece each time, *Lyric lines* tries hardest to cut before a line |
| `shots_per_scene`, `visual_style`, `dialogue_language` (lyric language), `wildness` | As on the other writers |
| `cast_1..4`, `extra_cast` | The performer(s): H3 Characters (`✏️ custom` + `wardrobe` is made for this) or typed lines such as `Lena: a singer in her 30s with a platinum pixie cut` |
| `wardrobe`, `locations`, `enforce_wardrobe` | Locks as on the Crossover Writer (cast-carried wardrobe is merged in). For runs of more than 6 scenes the locks are checked and reported but not re-emitted |
| `image_1..9`, `image_notes` | Reference pictures (the performer's face, the place); passed through to the video node |
| Claude Code block, `llm` | As on the other writers; an `APNext H3 LLM Backend` makes it run on Ollama |

| Output | Description |
|--------|-------------|
| `scenes` **(list)** | One prompt per piece → the H3 video node's `prompt` |
| `durations` **(list)** / `lengths` **(list)** | Seconds / H3 frame counts per piece → `length` (no math node needed) |
| `audio_segments` **(list)** | The matching AUDIO slice per piece → `ref_audio_1` (`<Audio 1>`) |
| `segment_table` | The cut list: `01  0:00.00 – 0:15.08  (15.08s, 362 frames)  energy: peak  lyrics: …` |
| `scenes_text`, `synopsis`, `cast`, `scene_count`, `song_seconds`, `session_id`, `info`, `image_N` | As on the Crossover Writer |

```
[Load Audio] ─AUDIO─┬─► [H3 Music Video Writer] ─scenes/lengths/audio_segments─► [MiniMax H3 Reference to Video] … [VAE Decode]
                    │                                                                                                │
                    └─────────────────────────── replace_audio ──► [H3 Scenes Join] ◄─ IMAGE list ───────────────────┘
                                                                        └─► [Create Video] ─► [Save Video]   (one music video)
```

`examples/h3/h3_music_video.json` is this graph end to end (song → pieces → clips → one video with the original track).

#### APNext H3 Scene Pick
**Display Name:** `APNext H3 Scene Pick`

Collapses a `scenes` list to one element by `index` (0-based, clamped) and returns that scene, its `duration`, the resolved `index` and the list `count`. Fix the writer's seed, then step the index to render scenes one at a time.

```
[H3 Characters] ─cast─► [H3 Characters] ─cast─► [H3 Crossover Writer] ─scenes (list)──► [MiniMax H3 video]   (renders every scene)
                                                                      └─durations (list)─► [duration → frames]
                                                                      └─scenes_text──────► [H3 Prompt Preview]

[H3 Crossover Writer] ─scenes/durations─► [H3 Scene Pick index=2] ─scene/duration─► [MiniMax H3 video]   (renders one)
```

#### APNext H3 Prompt Preview
**Display Name:** `APNext H3 Prompt Preview`

Output node that renders any H3 prompt colour-coded (`<Subject N>`, `<Picture N>`, `<Video N>`, `<Audio N>`, `<d>` dialogue, `[Shot N]`, speaker IDs, section headers, camera vocabulary) with a Copy button, and passes the text through. Connect the reference images to `image_1..image_9` and a small thumbnail of each appears in a strip above the prompt and inline next to every `<Picture N>` tag (click to enlarge; pictures not referenced in the text are marked *(unused)*). The **Thumbs** button in the panel bar toggles the thumbnails on/off; the choice is saved with the node.

#### APNext H3 Scenes Join
**Display Name:** `APNext H3 Scenes Join`

In the batch workflows ComfyUI renders every scene as its own clip (one `Save Video` per list element). Drop this node between the per-scene `VAE Decode` / `VAE Decode Audio` and a single `Create Video` → `Save Video` to get **one continuous video** of all scenes instead. Takes the per-scene `IMAGE` (and optional `AUDIO`) lists, concatenates frames in order and joins the audio tracks (sample rate / channel count unified to the first scene), outputs one `images` batch + one `audio` track plus `frame_count` / `scene_count`. `crossfade_frames` blends the cut between scenes (0 = hard cut); `size_mismatch` resizes stray scenes to the first scene's resolution or errors; `replace_audio` swaps the joined per-scene audio for one track of your own (the original song from the Music Video Writer). `h3_scenes_batch.json` and `h3_crossover_batch.json` ship with it wired in; for true scene-to-scene continuity use the Contex Loop route below.

```
[per-scene VAE Decode] ─IMAGE list─► [H3 Scenes Join] ─images/audio─► [Create Video] ─► [Save Video]   (one file)
[per-scene VAE Decode Audio] ─AUDIO list─┘
```

#### APNext H3 Scenes → Contex Loop Plan
**Display Name:** `APNext H3 Scenes → Contex Loop Plan`

For **continuity across scenes** (the last frames and audio of scene N carried into scene N+1). Converts the `scenes` / `durations` lists into the plan JSON that [ComfyUI-MiniMaxH3-Contex-Loop](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Contex-Loop)'s `MiniMax H3 Contex Loop Plan` node accepts on `plan_json_input` (`shots[]` with `id`, `prompt`, `duration_seconds`, `seed`, plus optional `prompt_prefix` and `defaults.steps`). That pack's loop body then renders every scene in order with the previous tail as motion/audio context, checkpoints and final assembly. Alternatives for one-scene-per-run chains: core `MiniMaxH3AddGuide`, `ComfyUI-H3-Motion-Context`, or this repo's `APNext H3 Claude Code Continue Writer` (last frame → I2VA first frame).

Example workflows in `examples/h3/`: `h3_crossover_batch.json` (render every scene in one queue, stitched by `H3 Scenes Join` into one video), `h3_crossover_pick_one.json` (Scene Pick + incrementing index, one scene per run), `h3_scenes_batch.json` (every scene rendered then stitched by `H3 Scenes Join` into one video), `h3_scenes_pick_one.json`, `h3_music_video.json` (song → H3 Music Video Writer → clips → one video with the original track), `h3_llm_backend_crossover.json` (the crossover batch written by a local model: an **H3 LLM Backend** set to `ollama:qwen3:14b` on the writer's `llm` socket), `h3_llm_backend_writer_refiner.json` (one LLM Backend feeding both the Claude Code Writer and the Refiner, `session_id` chained so the refiner resumes the local session; draft and refined prompt previewed side by side, the refined one rendered), and `h3_crossover_contex_chain.json` — the Contex-Loop *T2V – Normal* example with the crossover writer in continuous-chain mode feeding `Scenes → Contex Loop Plan` into `plan_json_input`, so the whole run renders scene after scene with the previous tail carried forward. Every example has an **H3 Prompt Preview** wired to the writer's prompt / scenes text output so the generated prompt is visible on the canvas.

---

#### Dialogue, `<d>` tags and language

`<d>[English] …</d>` in the output is not stray markup — it is how H3 marks **spoken audio**, and the guide requires it. The speaker's identifying phrase, `(S1)` ID, action and delivery stay *outside* the tag; only the language tag and the words actually said go inside:

```
the young, gravel-voiced warrior (S1) grits his teeth and shouts:
<d>[English] Stand still, you overgrown worm!</d>
```

Strip the tags before generating and H3 will treat those words as narration to depict rather than speech to voice.

- **Don't want speech at all?** `include_dialogue` → off. No `<d>` blocks, no `(Sx)` IDs.
- **Want another language?** Pick it in `dialogue_language`. The characters then genuinely speak it — the node instructs the model to write the lines in that language rather than write English and label it otherwise, which matters for smaller local models that would happily do the latter.
- **`Auto`** picks a language that fits the scene: *"two market traders haggle over spices in old Cairo"* comes back in Egyptian colloquial Arabic.
- **`custom_dialogue_language`** takes anything the dropdown lacks, dialects included — `Norwegian (Bergen dialect)` yields `<d>[Norwegian (Bergen dialect)] Det kjem til å regne heile dagen, du.</d>`.

Narration, action and camera language always stay English; only the spoken words change.

---

#### The `wildness` slider

One dial from conservative to unhinged. Above 40 it also injects concrete surreal **events** (not mood words) drawn from a 40-entry pool — selection is driven by `seed`, so the same seed gives the same weirdness.

| Range | Band | Behaviour | Random elements |
|-------|------|-----------|-----------------|
| 0–15 | Conservative | Strictly literal, no invented events | 0 |
| 16–40 | Grounded | Believable, well-directed embellishment | 0 |
| 41–65 | Bold | Strong authorial choices, physics still holds | 1 |
| 66–85 | Wild | Surreal juxtapositions, dreamlike logic | 2 |
| 86–100 | Unhinged | Scale, gravity and continuity all negotiable | 3 |

Injected elements are filmable, e.g. *"the subject's shadow moves a beat out of sync"*, *"a doorway opens onto a completely different biome"*, *"rain falls upward into the sky"*.

---

### 🎲 Prompt Generators

#### Auto Prompter
**Display Name:** `Auto Prompter`

Generate random prompts from extensive category databases.

| Input | Description |
|-------|-------------|
| `subject` | Main subject (can include LoRA triggers) |
| `custom` | Prefix text for styling |
| `artform` | Photography, digital art, etc. |
| Various category selections | Random or specific choices |

---

#### APNext Node
**Display Name:** `APNext Node`

Advanced prompt building with category-based enhancements.

### Overview

![Node Family Overview](https://github.com/user-attachments/assets/89c23e6f-44f5-4d2f-bb37-abf8cbd797c4)

The system includes numerous nodes that can be chained together to create complex workflows:

![Node Chaining Example](https://github.com/user-attachments/assets/bf402844-ffdc-4dcf-bc6c-28d40e125011)

Supports **24 main categories** with subcategories:
- **Architecture:** styles, buildings, interiors, materials
- **Art:** painting, sculpture, techniques, palettes
- **Artist:** concept artists, illustrators, painters
- **Character:** anime, fantasy, sci-fi, superheroes
- **Cinematic:** directors, genres, effects, color grading
- **Fashion:** designers, outfits, accessories
- **Feelings:** emotional modifiers
- **Geography:** countries, nationalities
- **Human:** jobs, hobbies, groups
- **Interaction:** individual, couple, group, crowd interactions
- **Keywords:** modifiers, genres, trending terms
- **People:** archetypes, body types, expressions
- **Photography:** cameras, lenses, lighting, film types
- **Plots:** action, romance, horror, sci-fi scenarios
- **Poses:** portrait and action poses
- **Scene:** weather, textures, environments
- **Science:** astronomy, mathematics, medical
- **Stuff:** seasonal objects, gadgets, fantasy items
- **Time:** eras, decades, centuries
- **Typography:** fonts, word art styles
- **Vehicle:** cars, classic cars, vehicle types
- **Video Game:** games, engines, actions

---

### 🔧 Utility Nodes

#### String Merger
**Display Name:** `APNext String Merger`

Combine multiple strings with separators.

---

#### Flexible String Merger
**Display Name:** `APNext Flexible String Merger`

Advanced string combining with custom formatting.

---

#### Sentence Mixer
**Display Name:** `APNext Sentence Mixer`

Shuffle and mix sentences from multiple inputs for creative variations.

---

#### Custom Prompt Loader
**Display Name:** `APNext Custom Prompts`

Load prompt templates from the `data/custom_prompts/` directory.

Included templates:
- `promptcreator.txt` - Full creative prompt generation
- `image_analyze.txt` - Image analysis prompts
- `gemini_video.txt` - Video generation prompts
- `cloner.txt` - Style cloning prompts
- Various LoRA-specific templates (ohwx, t5xxl, etc.)

---

#### Local Random Prompt
**Display Name:** `APNext Local random prompt`

Load random prompts from local text files.

---

#### Random Integer Generator
**Display Name:** `APNext Random Integer Generator`

Generate random integers with min/max range.

---

## 📁 Adding Custom Categories

Create your own categories for APNextNode:

1. Create a folder in `data/next/` (e.g., `data/next/mycategory/`)
2. Add JSON files for each field

### Simple Format
```json
["item1", "item2", "item3"]
```

### Advanced Format
```json
{
  "preprompt": "with",
  "separator": " and ",
  "endprompt": "visual effects",
  "items": ["motion blur", "lens flare", "particle effects"],
  "attributes": {
    "motion blur": ["dynamic", "cinematic"],
    "lens flare": ["bright", "atmospheric"]
  }
}
```

---

## 📝 Custom Prompt Templates

Create your own prompt templates for use with the **Custom Prompt Loader** node.

### Location
Place `.txt` files in: `data/custom_prompts/`

### Creating a Template

Templates are plain text files containing instructions for LLM nodes. They support dynamic variable substitution:

| Variable | Description |
|----------|-------------|
| `##TAG##` | Replaced with the `tag` input (e.g., "ohwx man") |
| `##SEX##` | Replaced with the `sex` input (e.g., "male", "female") |
| `##PRONOUNS##` | Replaced with pronouns (e.g., "him, his") |
| `##WORDS##` | Replaced with target word count |

### Example Template

Create a file `data/custom_prompts/my_style.txt`:

```
As a professional art critic, describe the provided image in detail.
Focus on creating a cohesive scene as if describing a movie still.

If the subject is ##TAG##, use ##PRONOUNS## pronouns appropriately.
The subject is ##SEX##.

Include:
- Main subject description with clothing, accessories, position
- Setting and environment details
- Lighting type, direction, and atmosphere
- Color palette and emotional tone
- Camera angle and composition

Output approximately ##WORDS## words.
Do not use JSON format. Provide a single cohesive paragraph.
```

### Included Templates

| Template | Purpose |
|----------|---------|
| `promptcreator.txt` | Detailed image analysis (~150 words) |
| `promptcreator_small.txt` | Concise image analysis |
| `image_analyze.txt` | General image description |
| `cloner.txt` | Style cloning prompts |
| `gemini_video.txt` | Video generation prompts |
| `gemini_ohwx.txt` | LoRA trigger-aware prompts |
| `t5xxl.txt` | T5-XXL optimized prompts |
| `ltxv.txt` | LTX Video model prompts |
| `next_scene.txt` | Cinematic scene transitions |

---

## ⚙️ Configuring LLM Models

Customize available models by editing JSON configuration files in the `data/` folder.

### Model Configuration Files

| File | Provider | Description |
|------|----------|-------------|
| `gemini_models.json` | Google Gemini | Gemini model list |
| `gpt_models.json` | OpenAI | GPT model list |
| `claude_models.json` | Anthropic | Claude model list |
| `grok_models.json` | xAI | Grok model list |
| `groq_models.json` | Groq | Groq model list (text + vision) |
| `qwenvl_models.json` | QwenVL | Local Qwen vision models |

### QwenVL Models - Adding Private/Custom Models

QwenVL nodes support loading additional models from private configuration files. This allows you to add custom or uncensored models without modifying the main configuration.

**How to add private models:**

1. Create a JSON file in `data/` with a name matching `private_*qwenvl*.json`
   - Examples: `private_qwenvl_models.json`, `private_uncensored.qwenvl_models.json`

2. Use the same format as `qwenvl_models.json`:

```json
{
    "models": [
        "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated",
        "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated",
        "another-namespace/custom-model"
    ]
}
```

3. Restart ComfyUI - the models will appear in the QwenVL node dropdowns

**Notes:**
- Private files are loaded in addition to the main `qwenvl_models.json`
- Duplicate models are automatically filtered out
- Supports full HuggingFace repo paths (`namespace/model-name`)
- Models are downloaded to `ComfyUI/models/LLM/Qwen-VL/` on first use

### Basic Format

Most model files use a simple array format:

```json
{
    "models": [
        "model-name-1",
        "model-name-2",
        "model-name-3"
    ]
}
```

### Example: Adding New Gemini Models

Edit `data/gemini_models.json`:

```json
{
    "models": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash-lite",
        "gemini-exp-1206"
    ]
}
```

### Example: Adding New Claude Models

Edit `data/claude_models.json`:

```json
{
    "models": [
        "claude-sonnet-4.5",
        "claude-sonnet-4",
        "claude-sonnet-3.7",
        "claude-opus-4.1",
        "claude-opus-4",
        "claude-haiku-3.5",
        "claude-haiku-3"
    ]
}
```

### Groq Models (Advanced Format)

Groq supports separate text and vision model lists:

```json
{
    "text_models": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "groq/compound",
        "qwen/qwen3-32b"
    ],
    "vision_models": [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct"
    ],
    "note": "Edit this file to add/remove models"
}
```

### Notes

- **Restart ComfyUI** after editing model configuration files
- For Groq, the system will first try to fetch models from the API, then fall back to the JSON file
- Model names must match exactly what the provider's API expects
- Invalid model names will cause API errors at runtime

---

## 🖼️ Example Workflows

Example workflows are available in the `examples/` directory:

- **APNext workflows:** `examples/flux/apnext/`
- **Florence2 local:** `examples/flux/florence2/`
- **GPT-4o Vision:** `examples/flux/gpt-4o_vision/`
- **Ollama local:** `examples/flux/ollama_local_llm/`
- **MiniCPM:** `examples/minicpm/`

---

## 📋 Requirements

```
openai>=2.54.0,<3.0.0
anthropic>=0.121.0
google-genai>=2.18.0
httpx>=0.28.1
huggingface_hub[hf_xet]>=0.34.0
chardet>=5.2.0
```

Anything ComfyUI already ships in its own `requirements.txt` — `Pillow`, `requests`, `transformers`, `scipy`, `tqdm`, `numpy`, `torch` — is deliberately **not** repeated, since re-pinning it only risks downgrading the base install.

Two constraints worth knowing about:

- **`openai` is capped below 3.0.** v3 switched to HTTPX2 and stopped shipping `httpx`; the GPT/Grok/Groq nodes pass an `httpx.Client` as `http_client=`, which v3 rejects.
- **Gemini uses `google-genai`, not `google-generativeai`.** The legacy SDK hard-pinned `google-ai-generativelanguage==0.6.15`, which forced `protobuf<6` and dragged grpcio into the ComfyUI environment. The current SDK needs neither.

`decord` is listed but commented out: it is unmaintained and not numpy-2 safe, and the QwenVL/MiniCPM video nodes fall back to OpenCV automatically. Uncomment it in `requirements.txt` if you specifically want decord-based frame decoding.

---

## 🔄 Model Support Matrix

| Provider | Text | Vision | Video | Local |
|----------|------|--------|-------|-------|
| OpenAI GPT | ✅ | ✅ | ❌ | ❌ |
| Google Gemini | ✅ | ✅ | ✅ | ❌ |
| Anthropic Claude | ✅ | ✅ | ❌ | ❌ |
| Claude Code CLI | ✅ | ✅ | ❌ | ⚙️ local CLI, own login |
| xAI Grok | ✅ | ✅ | ❌ | ❌ |
| Groq | ✅ | ✅ | ❌ | ❌ |
| QwenVL | ✅ | ✅ | ✅ | ✅ |
| Ollama | ✅ | ✅ | ❌ | ✅ |
| LM Studio / vLLM / llama.cpp | ✅ | ✅ | ❌ | ✅ |
| MiniCPM | ✅ | ✅ | ✅ | ✅ |
| Phi-3.5 | ✅ | ✅ | ❌ | ✅ |

Ollama and the other OpenAI-compatible servers are selectable directly in the H3 prompt writers via the `ollama:` / `lmstudio:` / `local:` prefixes — see [MiniMax-H3 Prompt Nodes](#-minimax-h3-prompt-nodes). Vision depends on the loaded model being multimodal.

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

Built for the ComfyUI community. Special thanks to all contributors and users providing feedback.

Shoutout to [**malcolmrey**](https://huggingface.co/malcolmrey) for the crossover ideas and for finding valid characters — see his [various dataset](https://huggingface.co/datasets/malcolmrey/various) on Hugging Face. The H3 Crossover Writer and the character casting owe him a beer. 🍻

The example workflows stand on these excellent node packs:

- [**ComfyUI-H3-Motion-Context-MultiRef**](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef) by **seitanism** — the masked-audio latent technique behind the `h3_music_video_masked_audio` workflow (the song written into the H3 audio latent and protected from denoising, for structural lip-sync).
- [**audio-separation-nodes-comfyui**](https://github.com/christian-byrne/audio-separation-nodes-comfyui) by **christian-byrne** — the vocal-stem separation feeding that same workflow.
- [**Nvidia RTX Nodes**](https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI) by **Comfy-Org / NVIDIA** — the RTX Video Super Resolution finishing pass used across the example workflows.
- [**ComfyUI-MiniMaxH3-Contex-Loop**](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Contex-Loop) by **ethanfel** — the continuity-chain renderer targeted by the `h3_crossover_contex_chain` workflow.
- [**dagre**](https://github.com/dagrejs/dagre) (MIT) — vendored for the canvas auto-layout tools.
