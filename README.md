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

#### APNext H3 Prompt Writer
**Display Name:** `APNext H3 Prompt Writer`

Writes the base format — `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music` — per [VIDEO_PROMPT_WRITING_GUIDE_base_en.md](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).

| Input | Description |
|-------|-------------|
| `idea` | Your short prompt or image description — the thing being expanded |
| `task_type` | `T2VA` (text only), `I2VA` (first frame), `FL2VA` (first + last), `L2VA` (last frame). Non-T2VA emits the exact reference-alignment instruction line |
| `duration_seconds` | Drives cut times and the `S.SS` value in the alignment line |
| `shot_plan` | Auto, or force 1–4 shots |
| `visual_style` | Auto, or one of the guide's styles (`Cinematic`, `live-action`, `2D-animated`, `3D CG`, `claymation`, `watercolor`, `vintage film`) |
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
