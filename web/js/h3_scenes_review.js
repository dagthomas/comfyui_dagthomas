// APNext H3 Scenes Review - the editable gate between a writer and the render.
//
// A Review run fills this editor with the scenes (colour-coded with the same
// token highlighting as the Prompt Preview, but editable) and stops the queue
// before anything renders. Edit all scenes at once or one scene at a time
// (scope selector), then hit "Continue" (or just queue again - the mode has
// already flipped to Continue). "Recreate" bumps the writer's seed and
// reviews a fresh draft.
//
// The highlighting works as an overlay: a transparent-text <textarea> sits on
// top of a <pre> that renders the highlighted HTML; extra CSS flattens the
// token chips (no padding/border/inline-block) so both layers share exact
// glyph metrics and never drift apart.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ensureStyle, highlight, LEGEND } from "./h3_prompt_preview.js";

const NODE_CLASS = "H3ScenesReview";
const MODE_REVIEW = "Review (stop here and edit)";
const MODE_CONTINUE = "Continue (render the editor text)";
const MODE_BYPASS = "Bypass (pass scenes through)";
const STYLE_ID = "apnext-h3-review-style";

const ENV_RE = /===\s*SCENE\s+(\d+)\s*===\s*([\s\S]*?)\s*===\s*END\s+SCENE\s*\1\s*===/gi;

const CSS = `
.apnext-h3-edit .apnext-h3-editwrap { position: relative; flex: 1 1 0; min-height: 0; }
.apnext-h3-edit .apnext-h3-hl,
.apnext-h3-edit .apnext-h3-input {
  position: absolute; inset: 0; margin: 0; box-sizing: border-box;
  padding: 8px 10px; overflow: auto; overscroll-behavior: contain;
  font-family: ui-monospace, "Cascadia Code", "JetBrains Mono", Menlo, Consolas, monospace;
  font-size: 12px; line-height: 1.55;
  white-space: pre-wrap; word-break: break-word; tab-size: 4;
}
.apnext-h3-edit .apnext-h3-hl { pointer-events: none; color: #e8e4df; }
.apnext-h3-edit .apnext-h3-input {
  background: transparent; color: transparent; caret-color: #e8e4df;
  border: 0; outline: none; resize: none; width: 100%; height: 100%;
}
.apnext-h3-edit .apnext-h3-input::selection { background: rgba(212,165,116,0.35); color: transparent; }
/* flatten the chips inside the editor: colours and tints only - any padding,
   border, margin or inline-block would desync the overlay metrics */
.apnext-h3-edit .apnext-h3-hl .h3-tag,
.apnext-h3-edit .apnext-h3-hl .h3-d,
.apnext-h3-edit .apnext-h3-hl .h3-header {
  display: inline; padding: 0; border: 0; margin: 0; border-radius: 2px;
}
.apnext-h3-edit .apnext-h3-hl .h3-header { background: rgba(212,165,116,0.14); }
.apnext-h3-edit select.apnext-h3-scope {
  background: #1e1c17; color: #e8e4df; border: 1px solid #463f33;
  border-radius: 4px; font-size: 10.5px; padding: 1px 4px; max-width: 130px;
}
.apnext-h3-edit .apnext-h3-bar button.apnext-h3-nav { padding: 2px 6px; }
/* pulsing glow while the gate has stopped the run and waits for edits */
.apnext-h3-edit.apnext-h3-waiting {
  border-color: #d4a574;
  animation: apnext-h3-pulse 1.15s ease-in-out infinite;
}
.apnext-h3-edit.apnext-h3-waiting .apnext-h3-title { color: #d4a574; }
@keyframes apnext-h3-pulse {
  0%, 100% { box-shadow: 0 0 0 2px rgba(212,165,116,0.85), 0 0 18px 4px rgba(212,165,116,0.35); }
  50%      { box-shadow: 0 0 0 3px rgba(232,180,184,0.95), 0 0 34px 10px rgba(212,165,116,0.65); }
}
`;

function ensureReviewStyle() {
  ensureStyle();
  if (document.getElementById(STYLE_ID)) return;
  const st = document.createElement("style");
  st.id = STYLE_ID;
  st.textContent = CSS;
  document.head.appendChild(st);
}

function parseEnvelopes(text) {
  const out = [];
  ENV_RE.lastIndex = 0;
  let m;
  while ((m = ENV_RE.exec(text || "")) !== null) out.push({ no: Number(m[1]), body: m[2].trim() });
  out.sort((a, b) => a.no - b.no);
  return out;
}

function headerOf(text) {
  const lines = [];
  for (const line of (text || "").split("\n")) {
    if (line.startsWith("#")) lines.push(line);
    else if (line.trim() === "" && lines.length) lines.push(line);
    else break;
  }
  while (lines.length && lines[lines.length - 1].trim() === "") lines.pop();
  return lines.length ? lines.join("\n") + "\n" : "";
}

function serializeEnvelopes(header, envs) {
  const parts = header ? [header] : [];
  for (const { no, body } of envs) {
    const nn = String(no).padStart(2, "0");
    parts.push(`=== SCENE ${nn} ===\n${body}\n=== END SCENE ${nn} ===\n`);
  }
  return parts.join("\n");
}

function widgetOf(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function buildEditor(node) {
  ensureReviewStyle();

  const root = document.createElement("div");
  root.className = "apnext-h3 apnext-h3-edit";

  const bar = document.createElement("div");
  bar.className = "apnext-h3-bar";
  const title = document.createElement("span");
  title.className = "apnext-h3-title";
  title.textContent = "Scenes review";
  bar.appendChild(title);

  const legend = document.createElement("div");
  legend.className = "apnext-h3-legend";
  for (const [cls, label] of LEGEND.slice(0, 5)) {
    const s = document.createElement("span");
    s.className = `h3-tag ${cls}`;
    s.textContent = label;
    legend.appendChild(s);
  }
  bar.appendChild(legend);

  const prev = document.createElement("button");
  prev.className = "apnext-h3-nav";
  prev.textContent = "◀";
  prev.title = "Previous scene";
  const scope = document.createElement("select");
  scope.className = "apnext-h3-scope";
  scope.title = "Edit all scenes at once, or one scene at a time";
  const next = document.createElement("button");
  next.className = "apnext-h3-nav";
  next.textContent = "▶";
  next.title = "Next scene";
  bar.appendChild(prev);
  bar.appendChild(scope);
  bar.appendChild(next);

  const wrap = document.createElement("div");
  wrap.className = "apnext-h3-editwrap";
  const hl = document.createElement("pre");
  hl.className = "apnext-h3-hl";
  hl.setAttribute("aria-hidden", "true");
  const ta = document.createElement("textarea");
  ta.className = "apnext-h3-input";
  ta.spellcheck = false;
  ta.placeholder = "Queue the graph in Review mode and the scenes appear here.";
  wrap.appendChild(hl);
  wrap.appendChild(ta);

  root.appendChild(bar);
  root.appendChild(wrap);

  for (const ev of ["pointerdown", "mousedown", "wheel", "dblclick", "contextmenu"]) {
    bar.addEventListener(ev, (e) => e.stopPropagation());
    wrap.addEventListener(ev, (e) => e.stopPropagation());
  }
  root.addEventListener("keydown", (e) => e.stopPropagation());

  // state: fullText is the widget value; view is "all" or a scene number
  const state = { fullText: "", view: "all", raf: 0 };

  const paint = () => {
    state.raf = 0;
    hl.innerHTML = highlight(ta.value, null) + "\n";
    hl.scrollTop = ta.scrollTop;
    hl.scrollLeft = ta.scrollLeft;
  };
  const schedulePaint = () => {
    if (!state.raf) state.raf = requestAnimationFrame(paint);
  };
  ta.addEventListener("scroll", () => {
    hl.scrollTop = ta.scrollTop;
    hl.scrollLeft = ta.scrollLeft;
  });

  const rebuildScope = () => {
    const envs = parseEnvelopes(state.fullText);
    const want = state.view;
    scope.innerHTML = "";
    const all = document.createElement("option");
    all.value = "all";
    all.textContent = `All scenes (${envs.length || "-"})`;
    scope.appendChild(all);
    for (const { no } of envs) {
      const o = document.createElement("option");
      o.value = String(no);
      o.textContent = `Scene ${String(no).padStart(2, "0")}`;
      scope.appendChild(o);
    }
    scope.value = want === "all" || !envs.some((e) => e.no === want) ? "all" : String(want);
    const single = scope.value !== "all";
    prev.style.display = next.style.display = envs.length > 1 ? "" : "none";
    return envs;
  };

  const showView = () => {
    const envs = rebuildScope();
    if (state.view !== "all") {
      const env = envs.find((e) => e.no === state.view);
      if (!env) state.view = "all";
      else {
        ta.value = env.body;
        schedulePaint();
        return;
      }
    }
    ta.value = state.fullText;
    schedulePaint();
  };

  ta.addEventListener("input", () => {
    if (state.view === "all") {
      state.fullText = ta.value;
    } else {
      const envs = parseEnvelopes(state.fullText);
      const env = envs.find((e) => e.no === state.view);
      if (env) {
        env.body = ta.value;
        state.fullText = serializeEnvelopes(headerOf(state.fullText), envs);
      } else {
        state.view = "all";
        state.fullText = ta.value;
      }
    }
    schedulePaint();
  });

  const switchView = (v) => {
    state.view = v;
    showView();
    ta.scrollTop = 0;
  };
  scope.addEventListener("change", () => {
    switchView(scope.value === "all" ? "all" : Number(scope.value));
  });
  const step = (d) => {
    const envs = parseEnvelopes(state.fullText);
    if (!envs.length) return;
    if (state.view === "all") return switchView(envs[d > 0 ? 0 : envs.length - 1].no);
    const i = envs.findIndex((e) => e.no === state.view);
    const j = i < 0 ? 0 : (i + d + envs.length) % envs.length;
    switchView(envs[j].no);
  };
  prev.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); step(-1); });
  next.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); step(1); });

  const setValue = (text) => {
    state.fullText = text ?? "";
    showView();
  };

  // glow + title change while the gate waits for the user's edits
  const setWaiting = (on) => {
    root.classList.toggle("apnext-h3-waiting", !!on);
    title.textContent = on ? "Scenes review — waiting for your edits" : "Scenes review";
  };

  setValue("");
  return { root, getValue: () => state.fullText, setValue, setWaiting };
}

app.registerExtension({
  name: "apnext.h3.scenesReview",

  setup() {
    // a Review run pushes the serialized scenes here, then stops the queue
    api.addEventListener("apnext.h3.scenes_review", (ev) => {
      const { node: id, text } = ev.detail || {};
      const node =
        app.graph?.getNodeById?.(Number(id)) ?? app.graph?.getNodeById?.(id);
      if (!node || (node.comfyClass !== NODE_CLASS && node.type !== NODE_CLASS)) return;
      node._h3Review?.setValue(text || "");
      const w = widgetOf(node, "edited");
      if (w) w.value = text || "";
      const mode = widgetOf(node, "mode");
      if (mode) mode.value = MODE_CONTINUE; // the next queue renders the editor text
      node._h3Review?.setWaiting(true); // glow until the user acts
      app.canvas?.setDirty?.(true, true);
    });
    // any new queue means the user has acted: stop the glow everywhere
    api.addEventListener("execution_start", () => {
      for (const n of app.graph?._nodes || []) {
        if (n?.type === NODE_CLASS || n?.comfyClass === NODE_CLASS) n._h3Review?.setWaiting(false);
      }
    });
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_CLASS) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const node = this;

      // swap the plain multiline widget for the highlighted overlay editor,
      // keeping the name and the widget position so (de)serialization holds
      const plainIdx = this.widgets?.findIndex((w) => w.name === "edited") ?? -1;
      if (plainIdx >= 0) this.widgets.splice(plainIdx, 1);
      const editor = buildEditor(this);
      this._h3Review = editor;
      const w = this.addDOMWidget("edited", "H3_SCENES_EDIT", editor.root, {
        getValue: () => editor.getValue(),
        setValue: (v) => editor.setValue(typeof v === "string" ? v : ""),
        serialize: true,
        hideOnZoom: false,
        getMinHeight: () => 260,
      });
      if (plainIdx >= 0) {
        const i = this.widgets.indexOf(w);
        if (i >= 0 && i !== plainIdx) {
          this.widgets.splice(i, 1);
          this.widgets.splice(plainIdx, 0, w);
        }
      }

      const btnContinue = this.addWidget("button", "▶ Continue - render this text", null, () => {
        const mode = widgetOf(node, "mode");
        if (mode) mode.value = MODE_CONTINUE;
        node._h3Review?.setWaiting(false);
        app.queuePrompt(0, 1);
      });
      const btnRecreate = this.addWidget("button", "🎲 Recreate - new draft from the writer", null, () => {
        // bump the seed of the node feeding `scenes`, then review again
        const inp = (node.inputs || []).find((i) => i.name === "scenes");
        const links = node.graph?.links;
        const link =
          links && inp?.link != null
            ? (typeof links.get === "function" ? links.get(inp.link) : links[inp.link])
            : null;
        const src = link ? node.graph?.getNodeById?.(link.origin_id) : null;
        const seed = src?.widgets?.find((x) => x.name === "seed");
        if (seed && typeof seed.value === "number" && seed.value >= 0) {
          seed.value = Math.floor(Math.random() * 2 ** 48);
        }
        const mode = widgetOf(node, "mode");
        if (mode) mode.value = MODE_REVIEW;
        node._h3Review?.setWaiting(false);
        app.queuePrompt(0, 1);
      });
      for (const b of [btnContinue, btnRecreate]) if (b) b.options = { ...(b.options || {}), serialize: false };

      const size = this.computeSize();
      this.setSize([Math.max(size[0], 460), Math.max(size[1], 380)]);
      return r;
    };

    // restore the editor content when a saved workflow loads
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const r = onConfigure?.apply(this, arguments);
      const w = widgetOf(this, "edited");
      if (w && typeof w.value === "string") this._h3Review?.setValue(w.value);
      return r;
    };
  },
});
