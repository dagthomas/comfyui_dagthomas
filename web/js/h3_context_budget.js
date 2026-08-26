// APNext H3 context budget - the "📏 Show 1024 tokens" modal
//
// `num_ctx` is the single setting that decides whether an Ollama run writes
// usable H3 scenes or noise: the system prompt alone is 9-15k tokens, and
// Ollama picks its default context from free VRAM - as little as 4k - so on a
// small default the writing rules are silently truncated away. Nobody has an
// intuition for what a token is, so this button shows:
//
//   1. a real block of exactly 1024 tokens, cut and counted with the user's
//      own model (binary search on `prompt_eval_count`),
//   2. where an H3 run's context actually goes, against this node's num_ctx,
//   3. what each pulled model costs per token of context, how much context
//      fits in THIS machine's VRAM and RAM, and - on demand - the real
//      tokens/sec and GPU/CPU split measured on the spot.
//
// Backed by nodes/h3/context_budget.py. Everything degrades: with no Ollama
// the modal still opens on an estimated sample and the static guidance.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_TYPE = "H3LLMBackend";
const STYLE_ID = "apnext-h3-ctx-style";

// Measured on the shipped guides + skills (characters / 4). They are what the
// writers actually send, so they set the floor for a workable num_ctx.
const H3_COSTS = [
  { label: "System prompt - text-only run (FL / T2VA)", tokens: 8900 },
  { label: "System prompt - with reference images (Ref2VA)", tokens: 15200 },
  { label: "System prompt - inline_skill_references ON", tokens: 45000 },
  { label: "User prompt - song table, cast, wardrobe, plan", tokens: 3000 },
];

const CSS = `
.apnext-ctx-wrap {
  position: fixed; inset: 0; z-index: 10000; background: rgba(0,0,0,.72);
  display: flex; align-items: center; justify-content: center; padding: 24px;
  color-scheme: dark;
}
.apnext-ctx {
  background: #14120e; border: 1px solid #463f33; border-radius: 10px;
  width: min(1080px, 96vw); max-height: 92vh; display: flex; flex-direction: column;
  box-shadow: 0 18px 60px rgba(0,0,0,.7);
  font: 13px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; color: #e8e4df;
}
.apnext-ctx header {
  display: flex; align-items: center; gap: 12px; padding: 12px 16px;
  border-bottom: 1px solid #2c2820; flex: 0 0 auto;
}
.apnext-ctx header h2 { margin: 0; font-size: 15px; font-weight: 650; letter-spacing: .01em; }
.apnext-ctx header .sub { color: #9a9590; font-size: 12px; margin-left: auto; }
.apnext-ctx header button.close {
  background: none; border: 0; color: #9a9590; font-size: 20px; line-height: 1;
  cursor: pointer; padding: 0 4px;
}
.apnext-ctx header button.close:hover { color: #e8e4df; }
.apnext-ctx .body { overflow: auto; padding: 14px 16px 18px; overscroll-behavior: contain; }
.apnext-ctx h3 {
  margin: 20px 0 8px; font-size: 12px; font-weight: 650; text-transform: uppercase;
  letter-spacing: .1em; color: #d4a574;
}
.apnext-ctx h3:first-child { margin-top: 0; }
.apnext-ctx p { margin: 0 0 8px; color: #bdb8b1; }
.apnext-ctx .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }
.apnext-ctx button.pill, .apnext-ctx select {
  background: #1e1c17; color: #e8e4df; border: 1px solid #463f33; border-radius: 6px;
  padding: 4px 10px; font: inherit; font-size: 12px; cursor: pointer;
}
.apnext-ctx button.pill:hover, .apnext-ctx select:hover { border-color: #d4a574; }
.apnext-ctx button.pill[aria-pressed="true"] { background: #3a2f22; border-color: #d4a574; color: #f0c99a; }
.apnext-ctx button.pill:disabled { opacity: .5; cursor: default; }
.apnext-ctx .sample {
  border: 1px solid #463f33; border-radius: 8px; background: #0e0d0a;
  padding: 10px 12px; max-height: 340px; overflow: auto; white-space: pre-wrap;
  word-break: break-word; font: 12px/1.5 ui-monospace, "Cascadia Mono", Consolas, monospace;
  color: #cfcac3; user-select: text;
}
.apnext-ctx .caption { color: #9a9590; font-size: 11.5px; margin: 6px 0 0; }
.apnext-ctx table { border-collapse: collapse; width: 100%; font-size: 12px; }
.apnext-ctx th, .apnext-ctx td {
  text-align: right; padding: 5px 8px; border-bottom: 1px solid #2c2820; white-space: nowrap;
}
.apnext-ctx th:first-child, .apnext-ctx td:first-child { text-align: left; }
.apnext-ctx th { color: #9a9590; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
.apnext-ctx td.name { font-family: ui-monospace, Consolas, monospace; color: #8fe3cd; }
.apnext-ctx .warn { color: #e0a45c; }
.apnext-ctx .bad { color: #e07a5c; }
.apnext-ctx .good { color: #8fe3cd; }
.apnext-ctx .tag {
  display: inline-block; border: 1px solid #463f33; border-radius: 4px; padding: 0 5px;
  font-size: 10.5px; color: #bdb8b1; margin-left: 4px;
}
.apnext-ctx .tag.vision { border-color: #56c8ad; color: #8fe3cd; }
`;

const GB = 1024 ** 3;
const MB = 1024 ** 2;
const fmtGB = (b) => `${(b / GB).toFixed(1)} GB`;
const fmtInt = (n) => Math.round(n).toLocaleString();
const fmtK = (n) => (n >= 1000 ? `${(n / 1024).toFixed(n >= 10240 ? 0 : 1)}k` : String(Math.round(n)));

function ensureStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const s = document.createElement("style");
  s.id = STYLE_ID;
  s.textContent = CSS;
  document.head.appendChild(s);
}

function widget(node, name) {
  return node?.widgets?.find((w) => w.name === name);
}

// The model this node will actually call: the dropdown, or model_name when it
// is set to "custom".
function nodeModel(node) {
  const picked = String(widget(node, "model")?.value ?? "");
  const custom = String(widget(node, "model_name")?.value ?? "");
  const chosen = picked.startsWith("custom") ? custom : picked;
  return chosen.trim();
}

function el(tag, props = {}, ...children) {
  const n = Object.assign(document.createElement(tag), props);
  for (const c of children) n.append(c);
  return n;
}

// ---------------------------------------------------------------------------
// How much context fits: weights + KV cache against the memory that exists
// ---------------------------------------------------------------------------

// Ollama needs headroom on top of weights and KV for the compute graph and
// (on the GPU) the display. A flat reserve is crude but honest enough to keep
// the answer on the safe side.
const GPU_RESERVE = 1.2 * GB;
const CPU_RESERVE = 2 * GB;

function fitsTokens(model, budget, reserve) {
  const perToken = model.kv_bytes_per_token || 0;
  if (!perToken || !budget) return null;
  const spare = budget - (model.size || 0) - reserve;
  return spare > 0 ? Math.floor(spare / perToken) : 0;
}

function capabilityTags(model) {
  const caps = model.capabilities || [];
  const out = [];
  if (caps.includes("vision")) out.push(["vision", "vision"]);
  if (caps.includes("thinking")) out.push(["thinking", ""]);
  if (caps.includes("tools")) out.push(["tools", ""]);
  return out;
}

// ---------------------------------------------------------------------------
// The modal
// ---------------------------------------------------------------------------

async function openModal(node) {
  ensureStyle();

  const wrap = el("div", { className: "apnext-ctx-wrap" });
  const panel = el("div", { className: "apnext-ctx" });
  const body = el("div", { className: "body" });

  const close = () => wrap.remove();
  wrap.addEventListener("click", (e) => { if (e.target === wrap) close(); });
  const onKey = (e) => { if (e.key === "Escape") { close(); document.removeEventListener("keydown", onKey); } };
  document.addEventListener("keydown", onKey);

  const head = el("header");
  head.append(
    el("h2", { textContent: "What fits in the context window" }),
    el("span", { className: "sub", textContent: nodeModel(node) || "no model picked" }),
    el("button", { className: "close", textContent: "×", onclick: close }),
  );
  panel.append(head, body);
  wrap.append(panel);
  document.body.append(wrap);

  // ---- section 1: the sample ---------------------------------------------
  body.append(el("h3", { textContent: "1. This is what N tokens looks like" }));
  body.append(el("p", {
    textContent:
      "Cut from the H3 prompt guide the writer actually sends, so it tokenises like a " +
      "real run. With Ollama reachable the length is measured with your own model's " +
      "tokenizer, not estimated.",
  }));

  const sizeRow = el("div", { className: "row" });
  const sample = el("div", { className: "sample", textContent: "loading…" });
  const caption = el("p", { className: "caption" });
  let target = 1024;

  async function loadSample(n) {
    target = n;
    for (const b of sizeRow.querySelectorAll("button.pill")) {
      b.setAttribute("aria-pressed", String(Number(b.dataset.n) === n));
    }
    sample.textContent = "measuring with your model…";
    caption.textContent = "";
    try {
      const res = await api.fetchApi("/apnext/h3/token_sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target: n,
          model: nodeModel(node),
          base_url: String(widget(node, "base_url")?.value ?? ""),
        }),
      });
      const data = await res.json();
      const chars = (data.text || "").length;
      sample.textContent = data.text || "";
      caption.textContent =
        `${fmtInt(data.tokens)} tokens · ${fmtInt(chars)} characters · ` +
        `${(chars / Math.max(1, data.tokens)).toFixed(2)} chars per token · ${data.note || ""}`;
      renderBudget();
    } catch (err) {
      sample.textContent = "";
      caption.textContent = `could not load the sample: ${err}`;
    }
  }

  for (const n of [256, 1024, 4096, 8192]) {
    const b = el("button", { className: "pill", textContent: `${n} tokens` });
    b.dataset.n = String(n);
    b.onclick = () => loadSample(n);
    sizeRow.append(b);
  }
  body.append(sizeRow, sample, caption);

  // ---- section 2: this node's budget --------------------------------------
  body.append(el("h3", { textContent: "2. Where an H3 run's context goes" }));
  const budgetBox = el("div");
  body.append(budgetBox);

  function renderBudget() {
    const numCtx = Number(widget(node, "num_ctx")?.value ?? 0);
    const maxTokens = Number(widget(node, "max_tokens")?.value ?? 8000);
    const inlineRefs = Boolean(widget(node, "inline_skill_references")?.value);
    budgetBox.replaceChildren();

    if (!numCtx) {
      budgetBox.append(el("p", {
        className: "warn",
        textContent:
          "num_ctx is 0, so Ollama's own default is used - it is chosen from free VRAM " +
          "and can be as small as 4k, which is not enough for any H3 run. Set it to 32768.",
      }));
    }

    const rows = H3_COSTS.filter((c) => inlineRefs || c.tokens !== 45000);
    const table = el("table");
    table.append(el("tr", {}, el("th", { textContent: "What is sent" }), el("th", { textContent: "Tokens" }),
      el("th", { textContent: "Blocks of " + target })));
    for (const cost of rows) {
      table.append(el("tr", {},
        el("td", { textContent: cost.label }),
        el("td", { textContent: `≈ ${fmtInt(cost.tokens)}` }),
        el("td", { textContent: (cost.tokens / target).toFixed(1) + " ×" }),
      ));
    }
    table.append(el("tr", {},
      el("td", { textContent: "The answer (max_tokens on this node)" }),
      el("td", { textContent: fmtInt(maxTokens) }),
      el("td", { textContent: (maxTokens / target).toFixed(1) + " ×" }),
    ));

    const worst = (inlineRefs ? 45000 : 15200) + 3000 + maxTokens;
    const ok = numCtx >= worst;
    table.append(el("tr", {},
      el("td", {}, el("b", { textContent: "Worst case for one call" })),
      el("td", {}, el("b", { textContent: fmtInt(worst) })),
      el("td", { textContent: (worst / target).toFixed(1) + " ×" }),
    ));
    budgetBox.append(table);

    budgetBox.append(el("p", {
      className: ok ? "good" : "bad",
      textContent: numCtx
        ? ok
          ? `num_ctx = ${fmtInt(numCtx)} - room for the worst case with ` +
            `${fmtInt(numCtx - worst)} tokens to spare.`
          : `num_ctx = ${fmtInt(numCtx)} is SHORT of the ${fmtInt(worst)} this run can need. ` +
            `Everything past the limit is dropped silently, starting with the rules the ` +
            `model is supposed to follow. Raise num_ctx to at least ${fmtInt(worst)}.`
        : "",
    }));
    budgetBox.append(el("p", {
      className: "caption",
      textContent:
        "Text-only runs (prompt_mode = FL / T2VA) send the smaller system prompt; " +
        "reference images add the Ref2VA guide on top. Both figures are measured from " +
        "the shipped guides and skills.",
    }));
  }
  renderBudget();

  // ---- section 3: this machine -------------------------------------------
  body.append(el("h3", { textContent: "3. What this machine can hold, and how fast" }));
  const hw = el("div", {}, el("p", { textContent: "querying Ollama…" }));
  body.append(hw);

  loadSample(1024);

  try {
    const res = await api.fetchApi(
      "/apnext/h3/context_budget?base_url=" +
      encodeURIComponent(String(widget(node, "base_url")?.value ?? "")),
    );
    const data = await res.json();
    hw.replaceChildren();

    const gpu = data.hardware?.gpu;
    const ram = data.hardware?.ram;
    hw.append(el("p", {
      textContent:
        (gpu ? `GPU: ${gpu.name} - ${fmtGB(gpu.total)} total, ${fmtGB(gpu.free)} free right now. ` : "No CUDA GPU visible. ") +
        (ram ? `System RAM: ${fmtGB(ram.total)} total, ${fmtGB(ram.available)} free. ` : "") +
        `KV cache type: ${data.hardware?.kv_cache_type || "f16"}.`,
    }));

    if (!data.running) {
      hw.append(el("p", { className: "warn", textContent: data.error || "Ollama is not reachable." }));
    } else if (!(data.models || []).length) {
      hw.append(el("p", {
        className: "warn",
        textContent: `Ollama is running at ${data.base_url} but no model is pulled. ` +
          "Pull one first, e.g. `ollama pull qwen3:8b`.",
      }));
    } else {
      const table = el("table");
      table.append(el("tr", {},
        el("th", { textContent: "Model" }),
        el("th", { textContent: "Weights" }),
        el("th", { textContent: "KV / 1k tok" }),
        el("th", { textContent: "Max ctx on GPU" }),
        el("th", { textContent: "Max ctx on CPU" }),
        el("th", { textContent: "Trained ctx" }),
        el("th", { textContent: "" }),
      ));

      for (const m of data.models) {
        const perToken = m.kv_bytes_per_token || 0;
        const onGpu = fitsTokens(m, gpu?.total, GPU_RESERVE);
        const onCpu = fitsTokens(m, ram?.total, CPU_RESERVE);
        const trained = m.geometry?.context_length || 0;

        const nameCell = el("td", { className: "name" }, m.name);
        for (const [label, cls] of capabilityTags(m)) {
          nameCell.append(el("span", { className: `tag ${cls}`, textContent: label }));
        }

        const bench = el("button", { className: "pill", textContent: "benchmark" });
        const benchCell = el("td", {}, bench);
        bench.onclick = async () => {
          bench.disabled = true;
          bench.textContent = "running…";
          try {
            const r = await api.fetchApi("/apnext/h3/token_benchmark", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                model: m.name,
                base_url: String(widget(node, "base_url")?.value ?? ""),
                num_ctx: Number(widget(node, "num_ctx")?.value ?? 0),
                num_predict: 64,
              }),
            });
            const b = await r.json();
            if (b.error) throw new Error(b.error);
            const share = b.on_gpu_fraction ?? null;
            const where = share === null ? ""
              : share >= 0.999 ? "fully on GPU"
              : share <= 0.001 ? "fully on CPU"
              : `${Math.round(share * 100)}% on GPU`;
            benchCell.replaceChildren(el("span", {
              className: share === null || share > 0.9 ? "good" : "warn",
              title:
                `writes ${b.tokens_per_second.toFixed(1)} tok/s, reads ` +
                `${fmtInt(b.prompt_tokens_per_second)} tok/s (measured over a ` +
                `${fmtInt(b.prompt_eval_count)}-token prompt)` +
                (b.context_length ? `, loaded at ${fmtInt(b.context_length)} context` : ""),
              textContent:
                `${b.tokens_per_second.toFixed(0)} tok/s out · ` +
                `${fmtK(b.prompt_tokens_per_second)} tok/s in` +
                (b.h3_prompt_seconds ? ` · ${b.h3_prompt_seconds.toFixed(0)}s to read an H3 prompt` : "") +
                (where ? ` · ${where}` : ""),
            }));
          } catch (err) {
            benchCell.replaceChildren(el("span", { className: "bad", textContent: String(err).slice(0, 90) }));
          }
        };

        table.append(el("tr", {},
          nameCell,
          el("td", { textContent: m.size ? fmtGB(m.size) : "?" }),
          el("td", { textContent: perToken ? `${(perToken * 1024 / MB).toFixed(0)} MB` : "?" }),
          el("td", {
            className: onGpu === null ? "" : onGpu >= 32768 ? "good" : onGpu > 0 ? "warn" : "bad",
            textContent: onGpu === null ? "-" : onGpu ? fmtInt(onGpu) : "does not fit",
          }),
          el("td", {
            className: onCpu === null ? "" : onCpu >= 32768 ? "good" : onCpu > 0 ? "warn" : "bad",
            textContent: onCpu === null ? "-" : onCpu ? fmtInt(onCpu) : "does not fit",
          }),
          el("td", { textContent: trained ? fmtInt(trained) : "?" }),
          benchCell,
        ));
      }
      hw.append(table);
      hw.append(el("p", {
        className: "caption",
        textContent:
          "Max ctx = (total memory − weights − headroom) ÷ KV cache per token, where KV per " +
          "token = layers × KV heads × (key + value dims) × 2 bytes, read from each model's " +
          "own metadata. Never set num_ctx above the trained context. GPU is the number that " +
          "matters: once weights and KV stop fitting in VRAM, Ollama spills layers to system " +
          "RAM and generation drops from GPU memory bandwidth to CPU memory bandwidth - an " +
          "order of magnitude, which turns a two-minute scene chunk into twenty. CPU-only can " +
          "hold far more context than any consumer GPU; it is the speed, not the room, that " +
          "makes it impractical. Press benchmark for the real number on this box.",
      }));
      hw.append(el("p", {
        className: "caption",
        textContent:
          "Set OLLAMA_KV_CACHE_TYPE=q8_0 before starting Ollama to roughly halve the KV " +
          "cache and double the context that fits, at a small quality cost.",
      }));
    }
  } catch (err) {
    hw.replaceChildren(el("p", { className: "bad", textContent: `could not query the server: ${err}` }));
  }
}

app.registerExtension({
  name: "apnext.h3.context_budget",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_TYPE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const __btn = this.addWidget("button", "📏 Show 1024 tokens", null, () => {
        openModal(this).catch((err) => console.error("[APNext ctx]", err));
      });
      __btn.serialize = false;   // a button has no value to save; a saved null would shift later widgets
      return r;
    };
  },
});
