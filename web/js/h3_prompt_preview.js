// APNext H3 Prompt Preview
//
// Renders a MiniMax-H3 prompt inside the node with colour-coded tags so the
// structure (subjects, pictures, videos, audio, dialogue, shots, speaker IDs,
// section headers) can be read at a glance. The raw text is selectable and a
// "Copy" button puts the untouched prompt on the clipboard.

import { app } from "../../../scripts/app.js";

const NODE_CLASS = "H3PromptPreview";
const STYLE_ID = "apnext-h3-preview-style";

// ---------------------------------------------------------------------------
// Styling
// ---------------------------------------------------------------------------

const CSS = `
.apnext-h3 {
  position: relative;
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
  max-height: 100%;
  min-height: 0;
  box-sizing: border-box;
  background: #161512;
  border: 1px solid #2c2820;
  border-radius: 6px;
  color: #e8e4df;
  font-family: ui-monospace, "Cascadia Code", "JetBrains Mono", Menlo, Consolas, monospace;
  font-size: 12px;
  line-height: 1.55;
  overflow: hidden;
}
.apnext-h3-bar {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  padding: 5px 8px;
  border-bottom: 1px solid #2c2820;
  background: #1e1c17;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 10.5px;
  user-select: none;
}
.apnext-h3-bar .apnext-h3-title {
  font-weight: 600;
  color: #9a9590;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  margin-right: 4px;
}
.apnext-h3-legend {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
  flex: 1;
}
.apnext-h3-legend span {
  padding: 0 6px;
  border-radius: 999px;
  border: 1px solid transparent;
  line-height: 16px;
}
.apnext-h3-bar button {
  border: 1px solid #463f33;
  background: #1e1c17;
  color: #e8e4df;
  border-radius: 4px;
  padding: 2px 9px;
  font-size: 11px;
  cursor: pointer;
  font-family: inherit;
}
.apnext-h3-bar button:hover { background: #2c2820; }
.apnext-h3-bar button.apnext-ok { background: #3a4a2e; border-color: #a7bd84; }
.apnext-h3-bar button.apnext-on { background: #3d3222; border-color: #d4a574; color: #f2e6d6; }
.apnext-h3-thumbs {
  flex: 0 0 auto;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  padding: 6px 10px;
  border-bottom: 1px dashed #2c2820;
  background: #131210;
  user-select: none;
}
.apnext-h3-thumbs figure { margin: 0; display: flex; flex-direction: column; align-items: center; gap: 3px; }
.apnext-h3-thumbs img {
  display: block; max-height: 72px; max-width: 120px; border-radius: 4px;
  border: 1px solid rgba(136,169,192,0.55); background: #000; cursor: zoom-in;
}
.apnext-h3-thumbs figcaption { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 10px; }
.h3-pic-inline {
  display: inline-block; vertical-align: middle; height: 22px; width: auto; max-width: 40px;
  border-radius: 3px; border: 1px solid rgba(136,169,192,0.55); margin: -3px 2px -3px 1px; cursor: zoom-in;
}
.apnext-h3-zoom {
  position: fixed; inset: 0; z-index: 10000; background: rgba(0,0,0,0.8);
  display: flex; align-items: center; justify-content: center; cursor: zoom-out;
}
.apnext-h3-zoom img { max-width: 92vw; max-height: 92vh; border-radius: 6px; box-shadow: 0 0 40px #000; }
.apnext-h3-zoom span { position: fixed; bottom: 16px; color: #ddd; font: 12px system-ui, sans-serif; }
.apnext-h3-body {
  /* min-height: 0 lets the flex item shrink below its content height, so a
     short node scrolls the prompt instead of clipping it. */
  flex: 1 1 0;
  min-height: 0;
  overflow: auto;
  overscroll-behavior: contain;
  padding: 8px 10px 10px;
  white-space: pre-wrap;
  word-break: break-word;
  user-select: text;
  cursor: text;
}
.apnext-h3-body::selection, .apnext-h3-body *::selection { background: rgba(212,165,116,0.35); color: #fff; }
.apnext-h3-empty { color: #7a756e; font-style: italic; }
.apnext-h3-meta {
  flex: 0 0 auto;
  margin-top: 0;
  padding-top: 5px;
  border-top: 1px dashed #2c2820;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 10.5px;
  color: #9a9590;
  padding: 4px 10px;
  user-select: none;
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

/* --- token colours ------------------------------------------------------ */
.h3-tag {
  display: inline-block;
  padding: 0 4px;
  border-radius: 3px;
  border: 1px solid;
  font-weight: 600;
  line-height: 1.3;
  margin: 0 1px;
}
.h3-subject  { color: #a7bd84; background: rgba(167,189,132,0.16);  border-color: rgba(167,189,132,0.55); }
.h3-picture  { color: #88a9c0; background: rgba(136,169,192,0.16); border-color: rgba(136,169,192,0.55); }
.h3-video    { color: #b58fc2; background: rgba(181,143,194,0.16); border-color: rgba(181,143,194,0.55); }
.h3-audio    { color: #d08a5e; background: rgba(208,138,94,0.16); border-color: rgba(208,138,94,0.55); }
.h3-shot     { color: #ccb777; background: rgba(204,183,119,0.14); border-color: rgba(204,183,119,0.55); }
.h3-speaker  { color: #84b3a6; background: rgba(132,179,166,0.14); border-color: rgba(132,179,166,0.5); font-weight: 600; }
.h3-marker   { color: #d49aa6; background: rgba(212,154,166,0.14);  border-color: rgba(212,154,166,0.55); }

.h3-d {
  background: rgba(205, 160, 106, 0.12);
  border-bottom: 1px solid rgba(205, 160, 106, 0.5);
  border-radius: 2px;
  padding: 0 2px;
}
.h3-d-tag  { color: #a88a55; font-weight: 700; }
.h3-d-lang { color: #cda06a; font-weight: 600; }
.h3-d-text { color: #f0e3cc; font-style: italic; }

.h3-time   { color: #9a9590; font-weight: 600; }
.h3-quote  { color: #e8b4b8; }
.h3-header {
  display: inline-block;
  color: #e8e4df;
  font-weight: 700;
  background: #1e1c17;
  border-left: 3px solid #d4a574;
  padding: 1px 6px;
  border-radius: 3px;
  margin: 2px 0;
}
.h3-instr  { color: #9a9590; font-style: italic; }
.h3-na     { color: #7a756e; }
.h3-cam    { color: #a7bcc9; }
`;

function ensureStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = CSS;
  document.head.appendChild(style);
}

// ---------------------------------------------------------------------------
// Highlighter
// ---------------------------------------------------------------------------

function esc(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

const SECTION_HEADERS =
  "integrated_multimodal_description|overall_soundscape|non_diegetic_music|" +
  "subject_definitions|summary|retention_analysis|detailed_description";

const CAMERA_TERMS =
  "zoom(?:s|ing)? (?:in|out)|push(?:es|ing)? in|pull(?:s|ing)? out|pan(?:s|ning)? (?:left|right)|" +
  "truck(?:s|ing)? (?:left|right)|tilt(?:s|ing)? (?:up|down)|pedestal(?:s|ing)? (?:up|down)|" +
  "arc shot|tracking shot|static shot|shake(?:s|ing)? (?:slightly|strongly)|" +
  "roll(?:s|ing)? (?:clockwise|counterclockwise)|" +
  "with (?:small|large) amplitude|at (?:slow|fast) speed|" +
  "(?:camera|shot) (?:cuts|transitions|changes|switches) to|cross-dissolve|fade|wipe";

// Highlight the inner content of a <d>...</d> block: [Language] tag + spoken text.
function renderDialogue(inner) {
  const m = inner.match(/^(\s*)\[([^\]]*)\](.*)$/s);
  if (!m) return `<span class="h3-d-text">${esc(inner)}</span>`;
  return (
    esc(m[1]) +
    `<span class="h3-d-lang">[${esc(m[2])}]</span>` +
    `<span class="h3-d-text">${esc(m[3])}</span>`
  );
}

// Highlight everything that is NOT dialogue.
function renderProse(text) {
  // Tokenise so already-emitted HTML never gets re-matched.
  const re = new RegExp(
    [
      `(<(?:Subject|Picture|Video|Audio)\\s*(?:\\d+|N)>)`, // 1 reference tags
      `(<scenetrans>|<cutoff>|\\[unclear\\])`,           // 2 markers
      `(\\[Shot\\s*(?:\\d+|N)\\])`,                        // 3 shot
      `(\\(S\\d+(?:\\s*,\\s*S\\d+)*\\))`,                  // 4 speaker id
      `(\\b\\d{2}:\\d{2}(?:\\.\\d{1,3})?\\b|\\b\\d+(?:\\.\\d+)?-second\\b|\\b\\d+\\.\\d{2} seconds?\\b)`, // 5 time
      `("[^"\\n]{1,120}")`,                                // 6 on-screen text
      `(\\bN/A\\b)`,                                        // 7 N/A
      `(\\b(?:${CAMERA_TERMS})\\b)`,                        // 8 camera vocabulary
    ].join("|"),
    "gi"
  );

  let out = "";
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    out += esc(text.slice(last, m.index));
    last = re.lastIndex;
    const raw = m[0];
    if (m[1]) {
      const kind = raw.replace(/[<>\d\sN]/g, "").toLowerCase();
      out += `<span class="h3-tag h3-${kind}">${esc(raw)}</span>`;
    } else if (m[2]) {
      out += `<span class="h3-tag h3-marker">${esc(raw)}</span>`;
    } else if (m[3]) {
      out += `<span class="h3-tag h3-shot">${esc(raw)}</span>`;
    } else if (m[4]) {
      out += `<span class="h3-tag h3-speaker">${esc(raw)}</span>`;
    } else if (m[5]) {
      out += `<span class="h3-time">${esc(raw)}</span>`;
    } else if (m[6]) {
      out += `<span class="h3-quote">${esc(raw)}</span>`;
    } else if (m[7]) {
      out += `<span class="h3-na">${esc(raw)}</span>`;
    } else if (m[8]) {
      out += `<span class="h3-cam">${esc(raw)}</span>`;
    } else {
      out += esc(raw);
    }
  }
  out += esc(text.slice(last));
  return out;
}

function renderLine(line) {
  // Section header lines: "field_name: rest"
  const hm = line.match(new RegExp(`^(\\s*)(${SECTION_HEADERS})(:)(.*)$`, "s"));
  if (hm) {
    return (
      esc(hm[1]) +
      `<span class="h3-header">${esc(hm[2])}${esc(hm[3])}</span>` +
      renderSegment(hm[4])
    );
  }
  // Keyframe alignment instruction lines
  if (/^\s*(For the target video|How the reference pictures align)/i.test(line)) {
    return `<span class="h3-instr">${renderSegment(line)}</span>`;
  }
  return renderSegment(line);
}

// Split a line into dialogue / non-dialogue segments.
function renderSegment(text) {
  const re = /<d>([\s\S]*?)(<\/d>|$)/gi;
  let out = "";
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    out += renderProse(text.slice(last, m.index));
    out +=
      `<span class="h3-d"><span class="h3-d-tag">&lt;d&gt;</span>` +
      renderDialogue(m[1]) +
      (m[2] ? `<span class="h3-d-tag">&lt;/d&gt;</span>` : "") +
      `</span>`;
    last = re.lastIndex;
    if (!m[2]) break; // unterminated <d>, swallow the rest
  }
  out += renderProse(text.slice(last));
  return out;
}

function highlight(text, thumbs) {
  let html = text.split("\n").map(renderLine).join("\n");
  if (thumbs && thumbs.size) {
    html = html.replace(
      /<span class="h3-tag h3-picture">(&lt;Picture\s*(\d+)&gt;)<\/span>/gi,
      (all, inner, n) => {
        const t = thumbs.get(Number(n));
        if (!t) return all;
        return `<span class="h3-tag h3-picture">${inner}</span><img class="h3-pic-inline" src="${t.url}" title="Picture ${n} — click to enlarge" data-h3-pic="${n}" draggable="false">`;
      }
    );
  }
  return html;
}

function thumbUrl(t) {
  const q = new URLSearchParams({ filename: t.filename, subfolder: t.subfolder || "", type: t.type || "temp" });
  q.set("t", String(Date.now())); // defeat the cache when a slot re-renders
  return `/view?${q.toString()}`;
}

function showZoom(src, label) {
  const z = document.createElement("div");
  z.className = "apnext-h3-zoom";
  const img = document.createElement("img");
  img.src = src;
  const cap = document.createElement("span");
  cap.textContent = label || "";
  z.appendChild(img);
  z.appendChild(cap);
  z.addEventListener("click", () => z.remove());
  document.body.appendChild(z);
}

function stats(text) {
  const count = (re) => (text.match(re) || []).length;
  const uniq = (re) => new Set((text.match(re) || []).map((s) => s.toLowerCase())).size;
  const parts = [];
  const shots = uniq(/\[Shot\s*\d+\]/gi);
  if (shots) parts.push(`${shots} shot${shots === 1 ? "" : "s"}`);
  const subj = uniq(/<Subject\s*\d+>/gi);
  if (subj) parts.push(`${subj} subject${subj === 1 ? "" : "s"}`);
  const pics = uniq(/<Picture\s*\d+>/gi);
  if (pics) parts.push(`${pics} picture${pics === 1 ? "" : "s"}`);
  const vids = uniq(/<Video\s*\d+>/gi);
  if (vids) parts.push(`${vids} video${vids === 1 ? "" : "s"}`);
  const aud = uniq(/<Audio\s*\d+>/gi);
  if (aud) parts.push(`${aud} audio ref${aud === 1 ? "" : "s"}`);
  const spk = uniq(/\(S\d+\)/g);
  if (spk) parts.push(`${spk} speaker${spk === 1 ? "" : "s"}`);
  const d = count(/<d>/gi);
  if (d) parts.push(`${d} line${d === 1 ? "" : "s"} of dialogue`);
  parts.push(`${text.length} chars`);
  return parts;
}

// ---------------------------------------------------------------------------
// Clipboard
// ---------------------------------------------------------------------------

async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (_) {
    /* fall through */
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch (_) {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Widget
// ---------------------------------------------------------------------------

const LEGEND = [
  ["h3-subject", "<Subject N>"],
  ["h3-picture", "<Picture N>"],
  ["h3-video", "<Video N>"],
  ["h3-audio", "<Audio N>"],
  ["h3-shot", "[Shot N]"],
  ["h3-speaker", "(S1)"],
  ["h3-d h3-d-lang", "<d>[Lang] ...</d>"],
  ["h3-marker", "<scenetrans>"],
];

function buildPanel(node) {
  ensureStyle();

  const root = document.createElement("div");
  root.className = "apnext-h3";

  const bar = document.createElement("div");
  bar.className = "apnext-h3-bar";

  const title = document.createElement("span");
  title.className = "apnext-h3-title";
  title.textContent = "H3 prompt";
  bar.appendChild(title);

  const legend = document.createElement("div");
  legend.className = "apnext-h3-legend";
  for (const [cls, label] of LEGEND) {
    const s = document.createElement("span");
    s.className = `h3-tag ${cls}`;
    s.textContent = label;
    legend.appendChild(s);
  }
  bar.appendChild(legend);

  const thumbBtn = document.createElement("button");
  thumbBtn.textContent = "Thumbs";
  thumbBtn.title = "Show / hide reference-image thumbnails (connect image_1..image_9)";
  bar.appendChild(thumbBtn);

  const btn = document.createElement("button");
  btn.textContent = "Copy";
  btn.title = "Copy the raw prompt text to the clipboard";
  bar.appendChild(btn);

  const strip = document.createElement("div");
  strip.className = "apnext-h3-thumbs";
  strip.style.display = "none";

  const body = document.createElement("div");
  body.className = "apnext-h3-body";

  const meta = document.createElement("div");
  meta.className = "apnext-h3-meta";

  root.appendChild(bar);
  root.appendChild(strip);
  root.appendChild(body);
  root.appendChild(meta);

  // Keep canvas from stealing pointer events (drag / zoom) while selecting text.
  for (const ev of ["pointerdown", "mousedown", "wheel", "dblclick", "contextmenu"]) {
    body.addEventListener(ev, (e) => e.stopPropagation());
    bar.addEventListener(ev, (e) => e.stopPropagation());
    strip.addEventListener(ev, (e) => e.stopPropagation());
  }
  // click-to-enlarge for inline and strip thumbnails
  root.addEventListener("click", (e) => {
    const img = e.target?.closest?.("img[data-h3-pic]");
    if (!img) return;
    e.preventDefault();
    e.stopPropagation();
    showZoom(img.src, `Picture ${img.dataset.h3Pic}`);
  });
  root.addEventListener("keydown", (e) => e.stopPropagation());

  let current = "";
  let thumbs = new Map(); // picture index -> {url, ...}
  let showThumbs = true;

  const renderStrip = () => {
    strip.innerHTML = "";
    const on = showThumbs && thumbs.size > 0;
    strip.style.display = on ? "" : "none";
    thumbBtn.classList.toggle("apnext-on", showThumbs);
    thumbBtn.textContent = showThumbs ? "Thumbs ✓" : "Thumbs";
    if (!on) return;
    const used = new Set((current.match(/<Picture\s*(\d+)>/gi) || []).map((m) => Number(m.replace(/\D/g, ""))));
    for (const [n, t] of [...thumbs.entries()].sort((a, b) => a[0] - b[0])) {
      const fig = document.createElement("figure");
      const img = document.createElement("img");
      img.src = t.url;
      img.dataset.h3Pic = String(n);
      img.draggable = false;
      img.title = `Picture ${n} — click to enlarge`;
      const cap = document.createElement("figcaption");
      cap.className = "h3-tag h3-picture";
      cap.textContent = `<Picture ${n}>` + (used.has(n) ? "" : " (unused)");
      if (!used.has(n)) cap.style.opacity = "0.55";
      fig.appendChild(img);
      fig.appendChild(cap);
      strip.appendChild(fig);
    }
  };

  const render = () => {
    if (!current.trim()) {
      body.innerHTML = `<span class="apnext-h3-empty">Run the graph to preview the H3 prompt here.</span>`;
      meta.textContent = "";
      renderStrip();
      return;
    }
    body.innerHTML = highlight(current, showThumbs ? thumbs : null);
    meta.innerHTML = stats(current)
      .map((s) => `<span>${esc(s)}</span>`)
      .join("");
    renderStrip();
  };

  const setText = (text) => {
    current = text ?? "";
    render();
  };
  const setThumbs = (list) => {
    thumbs = new Map();
    for (const t of list || []) {
      if (t && t.filename) thumbs.set(Number(t.index), { ...t, url: thumbUrl(t) });
    }
    render();
  };
  const setShowThumbs = (on) => {
    showThumbs = !!on;
    render();
  };

  thumbBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    setShowThumbs(!showThumbs);
    if (node) {
      node.properties = node.properties || {};
      node.properties.h3_show_thumbs = showThumbs;
    }
  });

  btn.addEventListener("click", async (e) => {
    e.preventDefault();
    e.stopPropagation();
    const ok = await copyText(current);
    btn.textContent = ok ? "Copied ✓" : "Copy failed";
    btn.classList.toggle("apnext-ok", ok);
    setTimeout(() => {
      btn.textContent = "Copy";
      btn.classList.remove("apnext-ok");
    }, 1400);
  });

  setText("");
  return { root, setText, getText: () => current, setThumbs, setShowThumbs, getShowThumbs: () => showThumbs };
}

app.registerExtension({
  name: "apnext.h3.promptPreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_CLASS) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);

      const panel = buildPanel(this);
      const widget = this.addDOMWidget("h3_preview", "H3_PREVIEW", panel.root, {
        getValue: () => panel.getText(),
        setValue: (v) => panel.setText(typeof v === "string" ? v : ""),
        serialize: true,
        hideOnZoom: false,
        // No fixed height: a DOM widget with only a minimum grows to fill
        // whatever space the node is resized to, like a multiline text box.
        getMinHeight: () => 220,
      });
      this._h3Panel = panel;
      if (this.properties && typeof this.properties.h3_show_thumbs === "boolean") {
        panel.setShowThumbs(this.properties.h3_show_thumbs);
      }

      const size = this.computeSize();
      this.setSize([Math.max(size[0], 460), Math.max(size[1], 320)]);
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const t = message?.text;
      const text = Array.isArray(t) ? t.join("\n\n") : t ?? "";
      this._h3Panel?.setThumbs(Array.isArray(message?.thumbs) ? message.thumbs : []);
      this._h3Panel?.setText(text);
      // keep the serialized widget value in sync so it survives reload
      const w = this.widgets?.find((x) => x.name === "h3_preview");
      if (w) w.value = text;
    };

    // Restore last preview when a saved workflow is loaded.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      onConfigure?.apply(this, arguments);
      const w = this.widgets?.find((x) => x.name === "h3_preview");
      if (this.properties && typeof this.properties.h3_show_thumbs === "boolean") {
        this._h3Panel?.setShowThumbs(this.properties.h3_show_thumbs);
      }
      if (w && typeof w.value === "string") this._h3Panel?.setText(w.value);
    };
  },
});
