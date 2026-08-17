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
  box-sizing: border-box;
  background: #16181d;
  border: 1px solid #2c3038;
  border-radius: 6px;
  color: #d7dae0;
  font-family: ui-monospace, "Cascadia Code", "JetBrains Mono", Menlo, Consolas, monospace;
  font-size: 12px;
  line-height: 1.55;
  overflow: hidden;
}
.apnext-h3-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  padding: 5px 8px;
  border-bottom: 1px solid #2c3038;
  background: #1c1f26;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 10.5px;
  user-select: none;
}
.apnext-h3-bar .apnext-h3-title {
  font-weight: 600;
  color: #9aa3b2;
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
  border: 1px solid #3a4150;
  background: #262b35;
  color: #d7dae0;
  border-radius: 4px;
  padding: 2px 9px;
  font-size: 11px;
  cursor: pointer;
  font-family: inherit;
}
.apnext-h3-bar button:hover { background: #313847; }
.apnext-h3-bar button.apnext-ok { background: #1f4d34; border-color: #2f7a52; }
.apnext-h3-body {
  flex: 1;
  overflow: auto;
  padding: 8px 10px 10px;
  white-space: pre-wrap;
  word-break: break-word;
  user-select: text;
  cursor: text;
}
.apnext-h3-body::selection, .apnext-h3-body *::selection { background: #3d5a99; color: #fff; }
.apnext-h3-empty { color: #6b7280; font-style: italic; }
.apnext-h3-meta {
  margin-top: 0;
  padding-top: 5px;
  border-top: 1px dashed #2c3038;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 10.5px;
  color: #7c8594;
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
.h3-subject  { color: #7ee2a8; background: rgba(46,160,96,0.16);  border-color: rgba(46,160,96,0.55); }
.h3-picture  { color: #7dc4ff; background: rgba(56,139,253,0.16); border-color: rgba(56,139,253,0.55); }
.h3-video    { color: #c9a5ff; background: rgba(150,90,255,0.16); border-color: rgba(150,90,255,0.55); }
.h3-audio    { color: #ffb070; background: rgba(255,140,50,0.16); border-color: rgba(255,140,50,0.55); }
.h3-shot     { color: #ffd166; background: rgba(255,190,50,0.14); border-color: rgba(255,190,50,0.55); }
.h3-speaker  { color: #5fe0d8; background: rgba(40,190,180,0.14); border-color: rgba(40,190,180,0.5); font-weight: 600; }
.h3-marker   { color: #ff7b7b; background: rgba(255,80,80,0.14);  border-color: rgba(255,80,80,0.55); }

.h3-d {
  background: rgba(255, 224, 130, 0.10);
  border-bottom: 1px solid rgba(255, 224, 130, 0.45);
  border-radius: 2px;
  padding: 0 2px;
}
.h3-d-tag  { color: #b8a04a; font-weight: 700; }
.h3-d-lang { color: #ffe08a; font-weight: 600; }
.h3-d-text { color: #fff3c4; font-style: italic; }

.h3-time   { color: #a3b1c6; font-weight: 600; }
.h3-quote  { color: #f5b7d5; }
.h3-header {
  display: inline-block;
  color: #ffffff;
  font-weight: 700;
  background: #2b3140;
  border-left: 3px solid #6c8cff;
  padding: 1px 6px;
  border-radius: 3px;
  margin: 2px 0;
}
.h3-instr  { color: #8f9bb0; font-style: italic; }
.h3-na     { color: #6b7280; }
.h3-cam    { color: #b8c4ff; }
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

function highlight(text) {
  return text.split("\n").map(renderLine).join("\n");
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

  const btn = document.createElement("button");
  btn.textContent = "Copy";
  btn.title = "Copy the raw prompt text to the clipboard";
  bar.appendChild(btn);

  const body = document.createElement("div");
  body.className = "apnext-h3-body";

  const meta = document.createElement("div");
  meta.className = "apnext-h3-meta";

  root.appendChild(bar);
  root.appendChild(body);
  root.appendChild(meta);

  // Keep canvas from stealing pointer events (drag / zoom) while selecting text.
  for (const ev of ["pointerdown", "mousedown", "wheel", "dblclick", "contextmenu"]) {
    body.addEventListener(ev, (e) => e.stopPropagation());
    bar.addEventListener(ev, (e) => e.stopPropagation());
  }
  root.addEventListener("keydown", (e) => e.stopPropagation());

  let current = "";
  const setText = (text) => {
    current = text ?? "";
    if (!current.trim()) {
      body.innerHTML = `<span class="apnext-h3-empty">Run the graph to preview the H3 prompt here.</span>`;
      meta.textContent = "";
      return;
    }
    body.innerHTML = highlight(current);
    meta.innerHTML = stats(current)
      .map((s) => `<span>${esc(s)}</span>`)
      .join("");
  };

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
  return { root, setText, getText: () => current };
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

      const size = this.computeSize();
      this.setSize([Math.max(size[0], 460), Math.max(size[1], 320)]);
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const t = message?.text;
      const text = Array.isArray(t) ? t.join("\n\n") : t ?? "";
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
      if (w && typeof w.value === "string") this._h3Panel?.setText(w.value);
    };
  },
});
