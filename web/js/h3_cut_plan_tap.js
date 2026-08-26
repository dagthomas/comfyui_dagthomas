// APNext H3 Cut Plan - "🎮 Tap the cuts"
//
// The fastest way to decide where a video cuts is to listen and tap. This
// button on the Cut Plan node plays the upstream Load Audio file in the
// browser and records a cut every time you press SPACE (or click the big
// button). Taps are shown on a strip, the last one can be undone, and Apply
// writes them into the node's `manual_cuts` widget - the node then snaps
// each tap to the nearest onset (a tap always lands a little after the
// sound) and uses them as hard scene boundaries, cutting the stretches
// between them on the music as usual.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_TYPE = "H3CutPlan";
const STYLE_ID = "apnext-h3-tap-style";

const CSS = `
.apnext-tap-wrap { position: fixed; inset: 0; z-index: 10000; background: rgba(0,0,0,.78);
  display: flex; align-items: center; justify-content: center; padding: 24px; color-scheme: dark; }
.apnext-tap { background: #14120e; border: 1px solid #463f33; border-radius: 10px; width: min(860px, 96vw);
  display: flex; flex-direction: column; box-shadow: 0 18px 60px rgba(0,0,0,.7);
  font: 13px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: #e8e4df; }
.apnext-tap * { box-sizing: border-box; }
.apnext-tap :is(button, input):focus, .apnext-tap :is(button, input):focus-visible { outline: none; }
.apnext-tap header { display: flex; align-items: center; gap: 12px; padding: 12px 16px; border-bottom: 1px solid #2c2820; }
.apnext-tap header h2 { margin: 0; font-size: 15px; font-weight: 650; }
.apnext-tap header .sub { color: #9a9590; font-size: 12px; margin-left: auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 50%; }
.apnext-tap header button.close { background: none; border: 0; color: #9a9590; font-size: 20px; cursor: pointer; }
.apnext-tap .body { padding: 14px 16px 18px; display: flex; flex-direction: column; gap: 12px; }
.apnext-tap p { margin: 0; color: #c9c4bd; }
.apnext-tap .clock { font: 600 28px/1 "JetBrains Mono", Consolas, monospace; color: #e8e4df; }
.apnext-tap .clock small { font-size: 13px; color: #9a9590; margin-left: 8px; font-weight: 400; }
.apnext-tap .strip { position: relative; height: 56px; background: #1e1c17; border: 1px solid #2c2820; border-radius: 6px; overflow: hidden; cursor: pointer; }
.apnext-tap .strip .head { position: absolute; top: 0; bottom: 0; width: 2px; background: #e8e4df; }
.apnext-tap .strip i { position: absolute; top: 6px; bottom: 6px; width: 3px; border-radius: 2px; background: #d4a574; }
.apnext-tap .strip i.snap { background: #84b3a6; }
.apnext-tap .strip .beat { position: absolute; top: 44px; bottom: 4px; width: 1px; background: rgba(154,149,144,.35); }
.apnext-tap .strip .section { position: absolute; top: 0; bottom: 0; width: 1px; background: rgba(181,143,194,.7); }
.apnext-tap button.big { width: 100%; padding: 28px 12px; border-radius: 10px; border: 0; cursor: pointer;
  font-size: 22px; font-weight: 700; letter-spacing: .04em; background: #d4a574; color: #1a1510; user-select: none; }
.apnext-tap button.big:active { background: #e8b4b8; }
.apnext-tap .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.apnext-tap button.pill { background: #1e1c17; color: #e8e4df; border: 1px solid #463f33; border-radius: 999px; padding: 5px 14px; cursor: pointer; font: inherit; }
.apnext-tap button.pill:hover { border-color: #d4a574; }
.apnext-tap button.pill.primary { background: #d4a574; color: #1a1510; border-color: #d4a574; font-weight: 600; }
.apnext-tap .foot { display: flex; gap: 10px; justify-content: flex-end; align-items: center; padding: 12px 16px; border-top: 1px solid #2c2820; }
.apnext-tap .foot .note { margin-right: auto; color: #9a9590; font-size: 12px; }
.apnext-tap .error { color: #d07070; }
`;

function ensureStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = CSS;
  document.head.append(style);
}

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  node.append(...children.filter((c) => c != null));
  return node;
}

const widget = (node, name) => node.widgets?.find((w) => w.name === name);

function fmt(sec) {
  const m = Math.floor(sec / 60), s = sec - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
}

function upstreamAudioFile(node) {
  const graph = node.graph ?? app.graph;
  let cur = node;
  for (let hops = 0; cur && hops < 8; hops++) {
    const input = cur.inputs?.find((i) => i.name === "audio") ?? cur.inputs?.find((i) => i.type === "AUDIO");
    if (input?.link == null) return null;
    const link = graph.links?.[input.link];
    const src = link && graph.getNodeById(link.origin_id);
    if (!src) return null;
    const w = src.widgets?.find((x) => x.name === "audio" && typeof x.value === "string");
    if (src.type === "LoadAudio" || w) return w?.value || null;
    cur = src;
  }
  return null;
}

// "song.mp3 [input]" -> /view?filename=song.mp3&type=input
function audioUrl(name) {
  let type = "input";
  let file = name;
  const m = /^(.*) \[(input|output|temp)\]$/.exec(name);
  if (m) { file = m[1]; type = m[2]; }
  let subfolder = "";
  const slash = file.lastIndexOf("/");
  if (slash >= 0) { subfolder = file.slice(0, slash); file = file.slice(slash + 1); }
  return api.apiURL(`/view?filename=${encodeURIComponent(file)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`);
}

async function openTap(node) {
  ensureStyle();
  const file = upstreamAudioFile(node);
  const wrap = el("div", { className: "apnext-tap-wrap" });
  const panel = el("div", { className: "apnext-tap" });
  const body = el("div", { className: "body" });
  const audio = new Audio();
  const close = () => { audio.pause(); audio.src = ""; wrap.remove(); document.removeEventListener("keydown", onKey); };
  wrap.addEventListener("click", (e) => { if (e.target === wrap) close(); });
  panel.append(
    el("header", {},
      el("h2", { textContent: "Tap the cuts" }),
      el("span", { className: "sub", textContent: file || "no audio file", title: file || "" }),
      el("button", { className: "close", textContent: "×", onclick: close })),
    body,
  );
  wrap.append(panel);
  document.body.append(wrap);

  if (!file) {
    body.append(el("p", { className: "error", textContent: "Connect a Load Audio node to this node's audio input first." }));
    document.addEventListener("keydown", onKey);
    return;
  }

  // taps already on the node
  const taps = String(widget(node, "manual_cuts")?.value || "").split(/[\s,]+/).filter(Boolean)
    .map((t) => (t.includes(":") ? Number(t.split(":")[0]) * 60 + Number(t.split(":")[1]) : Number(t))).filter((n) => !Number.isNaN(n));
  let snapped = [];      // from the preview endpoint: onsets to show as guides (optional)
  let duration = 0;

  const clock = el("div", { className: "clock", textContent: "0:00.00" });
  const clockSmall = el("small", { textContent: `${taps.length} cut(s)` });
  clock.append(clockSmall);
  const strip = el("div", { className: "strip" });
  const head = el("div", { className: "head" });
  strip.append(head);
  const big = el("button", { className: "big", textContent: "CUT  —  tap on the beat  (space)" });
  const play = el("button", { className: "pill", textContent: "▶ Play from start" });
  const pause = el("button", { className: "pill", textContent: "⏯ Play / pause" });
  const back = el("button", { className: "pill", textContent: "↶ 5 s" });
  const undo = el("button", { className: "pill", textContent: "Undo last tap" });
  const clear = el("button", { className: "pill", textContent: "Clear all" });
  body.append(
    el("p", { textContent: "Listen and press SPACE (or the big button) where the video should cut. A tap lands a little after the sound - the node snaps each one to the nearest onset. Stretches between your taps longer than max_seconds are still cut on the music." }),
    clock, strip, big,
    el("div", { className: "row" }, play, pause, back, undo, clear),
  );
  const note = el("span", { className: "note" });
  const apply = el("button", { className: "pill primary", textContent: "Apply to node" });
  panel.append(el("div", { className: "foot" }, note, el("button", { className: "pill", textContent: "Close", onclick: close }), apply));

  function render() {
    for (const n of strip.querySelectorAll("i, .beat, .section")) n.remove();
    if (duration > 0) {
      for (const t of taps) strip.append(el("i", { style: `left:${(t / duration) * 100}%`, title: fmt(t) }));
      for (const t of snapped) strip.append(el("div", { className: "section", style: `left:${(t / duration) * 100}%`, title: `section ${fmt(t)}` }));
      head.style.left = `${(audio.currentTime / duration) * 100}%`;
    }
    clockSmall.textContent = `${taps.length} cut(s)`;
    note.textContent = taps.length ? `Apply writes ${taps.length} cut(s) into manual_cuts.` : "No taps yet.";
  }
  function tap() {
    if (!duration) return;
    const t = Number(audio.currentTime.toFixed(2));
    if (!taps.some((x) => Math.abs(x - t) < 0.25)) taps.push(t);
    taps.sort((a, b) => a - b);
    big.style.background = "#e8b4b8";
    setTimeout(() => (big.style.background = ""), 90);
    render();
  }
  function onKey(e) {
    if (e.key === "Escape") { close(); return; }
    if (e.code === "Space") { e.preventDefault(); tap(); }
    if (e.key === "Backspace") { e.preventDefault(); taps.pop(); render(); }
  }
  document.addEventListener("keydown", onKey);
  big.addEventListener("click", tap);
  play.addEventListener("click", () => { audio.currentTime = 0; audio.play(); });
  pause.addEventListener("click", () => (audio.paused ? audio.play() : audio.pause()));
  back.addEventListener("click", () => { audio.currentTime = Math.max(0, audio.currentTime - 5); });
  undo.addEventListener("click", () => { taps.pop(); render(); });
  clear.addEventListener("click", () => { taps.length = 0; render(); });
  strip.addEventListener("click", (e) => {
    if (!duration) return;
    const r = strip.getBoundingClientRect();
    audio.currentTime = ((e.clientX - r.left) / r.width) * duration;
    render();
  });
  apply.addEventListener("click", () => {
    const w = widget(node, "manual_cuts");
    if (w) { w.value = taps.map((t) => t.toFixed(2)).join(" "); w.callback?.(w.value, app.canvas, node); }
    node.setDirtyCanvas?.(true, true);
    close();
  });

  audio.src = audioUrl(file);
  audio.addEventListener("loadedmetadata", () => { duration = audio.duration || 0; render(); });
  audio.addEventListener("timeupdate", () => { clock.firstChild.textContent = fmt(audio.currentTime); render(); });
  audio.addEventListener("error", () => { body.prepend(el("p", { className: "error", textContent: `Could not load ${file} for playback.` })); });
  // section starts as guides, from the structure the node computes - best effort
  try {
    const res = await api.fetchApi("/apnext/h3/sound_events_preview", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ audio: file, sensitivity: 1.0, min_gap_seconds: 0.18 }),
    });
    const data = await res.json();
    if (res.ok && data.events) snapped = data.events.filter((e) => e.type === "DROP" || e.type === "SECTION").map((e) => e.t);
    render();
  } catch (err) { /* guides are optional */ }
}

app.registerExtension({
  name: "apnext.h3.cut_plan_tap",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_TYPE) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const __btn = this.addWidget("button", "🎮 Tap the cuts", null, () => {
        openTap(this).catch((err) => console.error("[APNext tap]", err));
      });
      __btn.serialize = false;   // a button has no value to save; a saved null would shift later widgets
      return r;
    };
  },
});
