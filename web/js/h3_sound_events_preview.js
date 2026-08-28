// APNext H3 Sound Events - "🎚 Preview & edit events" button
//
// Strength is normalised to each track's loudest hit, so `min_strength` has to
// be tuned per song - and the only honest way to tune it is to SEE the hits
// on the waveform. This button finds the Load Audio node upstream, asks the
// server to run the detectors once with every kind on (see
// nodes/h3/sound_events_preview.py), decodes the audio in the browser, and
// opens a full-screen editor:
//
//   * the whole waveform, zoomable (ctrl+wheel / buttons), with a playhead -
//     click to seek, space to play
//   * every event as a block on the lane under it: its wind-up (cue), the
//     instant it lands, and its settle - the same window the writer's brief
//     lists as `[+cue ->+land ->+settle s]`
//   * drag a block to move it, drag its left / right edge to make the wind-up
//     or the settle longer or shorter, click to select and edit its type,
//     strength and times in the inspector, Delete to strike it out (and again
//     to keep it), double-click the lane to add one
//   * the kind toggles, the strength band, sensitivity / gap and the density
//     readout from before, in the side panel
//   * a Signal section - gain, a dynamics curve and a five-band EQ - that
//     shapes what the detectors listen to (nodes/h3/sound_events.shape_signal).
//     The shaped signal is drawn, translucent, behind the events on the lane
//     (the same chain, run in Web Audio) and can be monitored through the
//     headphones toggle, so what you see and hear is what gets scored.
//   * a beat grid at the track's measured BPM, phase-fitted to the kept hits,
//     so a hit that is off the pulse is visible at a glance
//
// "Apply" writes the detector settings, the signal shaping, the struck-out
// times and the hand edits (as JSON in the node's `edits` widget) back into
// the node; the node applies them after detection so the writer sees exactly
// this list.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_TYPE = "H3SoundEvents";
const STYLE_ID = "apnext-h3-sev-style";

// Which node toggle owns which event kind (mirrors detect_events()).
const KIND_TOGGLE = {
  "BASS HIT": "bass_hits", IMPACT: "impacts", DROP: "drops_and_stops", STOP: "drops_and_stops",
  BUILD: "builds", SECTION: "sections", ACCENT: "accents",
};
const TOGGLES = [
  ["bass_hits", "bass hits"], ["impacts", "impacts"], ["drops_and_stops", "drops & stops"],
  ["builds", "builds"], ["sections", "sections"], ["accents", "accents"],
];
const EVENT_TYPES = ["DROP", "STOP", "SECTION", "BUILD", "IMPACT", "BASS HIT", "ACCENT"];
const HIT_KINDS = new Set(["BASS HIT", "IMPACT", "ACCENT"]);
const STRUCTURAL = new Set(["DROP", "STOP", "SECTION", "BUILD"]);
const KIND_COLOR = {
  "BASS HIT": "#d4a574", IMPACT: "#e8b4b8", DROP: "#84b3a6", STOP: "#88a9c0",
  BUILD: "#a7bd84", SECTION: "#b58fc2", ACCENT: "#ccb777",
};
// The writer's staging window per kind at (light, solid, heavy) - mirrors
// sound_events._LEAD / _TAIL so the blocks here are the windows the brief lists.
const LEAD = { DROP: [0.55, 0.85, 1.20], STOP: [0.40, 0.60, 0.85], SECTION: [0.30, 0.40, 0.55], BUILD: [0.90, 1.40, 2.00],
  IMPACT: [0.32, 0.45, 0.60], "BASS HIT": [0.20, 0.28, 0.36], ACCENT: [0.08, 0.12, 0.16] };
const TAIL = { DROP: [0.55, 0.80, 1.10], STOP: [0.50, 0.75, 1.10], SECTION: [0.35, 0.45, 0.60], BUILD: [0, 0, 0],
  IMPACT: [0.35, 0.50, 0.70], "BASS HIT": [0.22, 0.30, 0.40], ACCENT: [0.10, 0.14, 0.18] };
const tier = (s) => (s >= 0.66 ? 2 : s >= 0.33 ? 1 : 0);
const strengthLabel = (s) => (s >= 0.66 ? "heavy" : s >= 0.33 ? "solid" : "light");

// Density bands, hits per minute. A 9 s piece gets 6 slots (the writer scales
// this to each scene's real length) and drops/stops always take theirs first, so ~24/min
// (one every 2.5 s) fills a piece with 2-4 hits and still leaves room.
const OK_PER_MIN = 24;
const TOO_MANY_PER_MIN = 40;
const PIECE_SECONDS = 9;
const SLOTS_PER_PIECE = 6;
const MATCH = 0.08;          // s - "the same event" when matching times (under min_gap, over the snap shift)
const MAX_CANVAS = 30000;    // px - Chrome's canvas width limit is 32767
const WAVE_H = 250;          // px - the waveform canvas (CSS below carries the same numbers)
const LANE_H = 220;          // px - the event lane under it
const GRID_TOL = 0.045;      // s - a hit within this of a grid line "sits on the beat"

// Signal shaping - mirrors sound_events.EQ_BANDS / SHAPING_KEYS. The five
// biquads are RBJ cookbook filters on both sides, so the Web Audio chain here
// and the torch chain that scores the track have the same magnitude response.
const EQ_BANDS = [
  ["eq_sub_db", "lowshelf", 60, 0.707, "sub", "60 Hz shelf"],
  ["eq_bass_db", "peaking", 150, 0.9, "bass", "150 Hz"],
  ["eq_low_db", "peaking", 400, 0.9, "low", "400 Hz"],
  ["eq_mid_db", "peaking", 1500, 0.9, "mid", "1.5 kHz"],
  ["eq_high_db", "highshelf", 5000, 0.707, "high", "5 kHz shelf"],
];
const SHAPING_DEFAULT = { gain_db: 0, dynamics_curve: 1, eq_sub_db: 0, eq_bass_db: 0, eq_low_db: 0, eq_mid_db: 0, eq_high_db: 0 };
const isNeutral = (sh) => Object.keys(SHAPING_DEFAULT).every((k) => Math.abs((sh[k] ?? SHAPING_DEFAULT[k]) - SHAPING_DEFAULT[k]) < 0.01);

const CSS = `
.apnext-sev-wrap { position: fixed; inset: 0; z-index: 10000; background: #0d0c0a; color-scheme: dark; display: flex; }
.apnext-sev { flex: 1; display: flex; flex-direction: column; min-width: 0; background: #14120e;
  font: 13px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: #e8e4df; }
.apnext-sev * { box-sizing: border-box; }
.apnext-sev :is(input, button, select):focus, .apnext-sev :is(input, button, select):focus-visible { outline: none; }
.apnext-sev header { display: flex; align-items: center; gap: 12px; padding: 10px 16px; border-bottom: 1px solid #2c2820; flex: 0 0 auto; }
.apnext-sev header h2 { margin: 0; font-size: 15px; font-weight: 650; }
.apnext-sev header .sub { color: #9a9590; font-size: 12px; max-width: 40%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.apnext-sev header .spacer { margin-left: auto; }
.apnext-sev header button.close { background: none; border: 0; color: #9a9590; font-size: 22px; line-height: 1; cursor: pointer; padding: 0 4px; }
.apnext-sev header button.close:hover { color: #e8e4df; }
.apnext-sev main { flex: 1; display: flex; min-height: 0; }
.apnext-sev .stage { flex: 1; display: flex; flex-direction: column; min-width: 0; border-right: 1px solid #2c2820; }
.apnext-sev .toolbar { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; padding: 8px 14px; border-bottom: 1px solid #2c2820; flex: 0 0 auto; }
.apnext-sev .toolbar .clock { font: 12px "JetBrains Mono", Consolas, monospace; color: #d4a574; min-width: 150px; }
.apnext-sev .toolbar .hint { color: #9a9590; font-size: 11.5px; margin-left: auto; }
.apnext-sev .scroll { flex: 1; overflow: auto; position: relative; background: #0f0e0b; overscroll-behavior: contain; }
.apnext-sev .track { position: relative; height: 100%; min-height: 516px; }
.apnext-sev canvas.wave { position: absolute; left: 0; top: 22px; height: 250px; display: block; }
.apnext-sev .ruler { position: absolute; left: 0; top: 0; height: 22px; border-bottom: 1px solid #2c2820; }
.apnext-sev .ruler span { position: absolute; top: 3px; font-size: 10px; color: #9a9590; transform: translateX(-50%); white-space: nowrap; }
.apnext-sev .ruler span::after { content: ""; position: absolute; left: 50%; top: 14px; width: 1px; height: 6px; background: #463f33; }
.apnext-sev .lane { position: absolute; left: 0; top: 286px; height: 220px; border-top: 1px solid #2c2820; border-bottom: 1px solid #2c2820; background: #131110; }
.apnext-sev .lane canvas { position: absolute; left: 0; top: 0; height: 100%; pointer-events: none; display: block; }
.apnext-sev .lane .ev { position: absolute; border-radius: 4px; opacity: .88; cursor: grab; border: 1px solid rgba(0,0,0,.4); }
.apnext-sev .lane .ev.struct { opacity: .6; }
.apnext-sev .lane .ev .land { position: absolute; top: -6px; bottom: -6px; width: 2px; background: #fff; opacity: .9; pointer-events: none; }
.apnext-sev .lane .ev .lbl { position: absolute; left: 4px; top: 2px; font-size: 10px; color: #1a1510; white-space: nowrap; pointer-events: none; font-weight: 600; text-shadow: 0 0 2px rgba(255,255,255,.4); }
.apnext-sev .lane .ev .h { position: absolute; top: 0; bottom: 0; width: 7px; cursor: ew-resize; }
.apnext-sev .lane .ev .h.l { left: -3px; } .apnext-sev .lane .ev .h.r { right: -3px; }
.apnext-sev .lane .ev:hover { opacity: 1; }
.apnext-sev .lane .ev.struct:hover { opacity: .8; }
.apnext-sev .lane .ev.sel { outline: 2px solid #fff; opacity: 1; z-index: 3; }
.apnext-sev .lane .ev.cut { opacity: .22; }
.apnext-sev .lane .ev.struck { opacity: .55; background-image: repeating-linear-gradient(135deg, rgba(208,112,112,.9) 0 4px, transparent 4px 9px) !important; }
.apnext-sev .lane .ev.edited::before { content: "✎"; position: absolute; right: 3px; top: 1px; font-size: 10px; color: #1a1510; }
.apnext-sev .lane .ev.added::before { content: "+"; position: absolute; right: 3px; top: 0; font-size: 12px; color: #1a1510; font-weight: 700; }
.apnext-sev .playhead { position: absolute; top: 0; bottom: 0; width: 1px; background: #e8b4b8; pointer-events: none; z-index: 4; }
.apnext-sev .inspector { flex: 0 0 auto; display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; padding: 8px 14px; border-top: 1px solid #2c2820; min-height: 44px; }
.apnext-sev .inspector .none { color: #9a9590; }
.apnext-sev aside { flex: 0 0 340px; overflow: auto; padding: 12px 14px 18px; overscroll-behavior: contain; }
.apnext-sev h3 { margin: 16px 0 8px; font-size: 12px; font-weight: 650; text-transform: uppercase; letter-spacing: .1em; color: #d4a574; }
.apnext-sev aside h3:first-child { margin-top: 0; }
.apnext-sev p { margin: 6px 0; color: #c9c4bd; }
.apnext-sev .row { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; }
.apnext-sev label.ctl { display: inline-flex; align-items: center; gap: 6px; color: #c9c4bd; white-space: nowrap; }
.apnext-sev input[type="number"] { width: 76px; background: #1e1c17; color: #e8e4df; border: 1px solid #2c2820; border-radius: 3px; padding: 3px 6px; font: inherit; }
.apnext-sev input[type="number"]:focus { border-color: #d4a574; }
.apnext-sev select { background: #1e1c17; color: #e8e4df; border: 1px solid #2c2820; border-radius: 3px; padding: 3px 6px; font: inherit; }
.apnext-sev input[type="range"] { accent-color: #d4a574; width: 150px; }
.apnext-sev input[type="checkbox"] { accent-color: #d4a574; width: 14px; height: 14px; margin: 0; }
.apnext-sev button.pill { background: #1e1c17; color: #e8e4df; border: 1px solid #463f33; border-radius: 999px; padding: 4px 12px; cursor: pointer; font: inherit; }
.apnext-sev button.pill:hover { border-color: #d4a574; }
.apnext-sev button.pill:disabled { opacity: .5; cursor: default; }
.apnext-sev button.pill.primary { background: #d4a574; color: #1a1510; border-color: #d4a574; font-weight: 600; }
.apnext-sev button.pill.primary:hover { background: #e8b4b8; border-color: #e8b4b8; }
.apnext-sev button.pill.danger { border-color: #d07070; color: #e8b4b8; }
.apnext-sev .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.apnext-sev .stat { background: #1e1c17; border: 1px solid #2c2820; border-radius: 6px; padding: 6px 9px; }
.apnext-sev .stat .k { color: #9a9590; font-size: 10.5px; text-transform: uppercase; letter-spacing: .06em; }
.apnext-sev .stat .v { font-size: 16px; font-weight: 650; margin-top: 1px; }
.apnext-sev .stat .v small { font-size: 10.5px; color: #9a9590; font-weight: 400; margin-left: 4px; }
.apnext-sev .banner { border-radius: 6px; padding: 8px 10px; margin: 8px 0 0; border: 1px solid; line-height: 1.45; font-size: 12px; }
.apnext-sev .banner.ok { border-color: #84b3a6; background: rgba(132,179,166,.12); }
.apnext-sev .banner.warn { border-color: #d4a574; background: rgba(212,165,116,.12); }
.apnext-sev .banner.bad { border-color: #d07070; background: rgba(208,112,112,.14); }
.apnext-sev .banner b { font-weight: 650; }
.apnext-sev .banner button.pill { margin-left: 8px; padding: 2px 10px; }
.apnext-sev .hist { display: grid; grid-template-columns: repeat(10, 1fr); gap: 3px; align-items: end; height: 70px; margin-top: 6px; }
.apnext-sev .hist .bin { position: relative; height: 100%; display: flex; flex-direction: column; justify-content: flex-end; cursor: pointer; }
.apnext-sev .hist .bin .bar { background: #463f33; border-radius: 3px 3px 0 0; min-height: 2px; transition: height .12s; }
.apnext-sev .hist .bin.kept .bar { background: #d4a574; }
.apnext-sev .hist .bin .n { position: absolute; top: -2px; left: 0; right: 0; text-align: center; font-size: 10px; color: #9a9590; }
.apnext-sev .hist-axis { display: grid; grid-template-columns: repeat(10, 1fr); font-size: 10px; color: #9a9590; text-align: center; margin-top: 2px; }
.apnext-sev .legend { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 11px; color: #9a9590; margin-top: 6px; }
.apnext-sev .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: -1px; margin-right: 4px; }
.apnext-sev .foot { display: flex; gap: 10px; justify-content: flex-end; align-items: center; padding: 10px 16px; border-top: 1px solid #2c2820; flex: 0 0 auto; }
.apnext-sev .foot .note { margin-right: auto; color: #9a9590; font-size: 12px; }
.apnext-sev .error { color: #d07070; white-space: pre-wrap; padding: 16px; }
.apnext-sev .toolbar .bpm { font: 11.5px "JetBrains Mono", Consolas, monospace; color: #84b3a6; }
.apnext-sev button.pill.needs { border-color: #d4a574; color: #d4a574; box-shadow: 0 0 0 1px rgba(212,165,116,.35); }
.apnext-sev .sig { display: grid; grid-template-columns: auto 1fr auto; gap: 6px 10px; align-items: center; margin-top: 6px; }
.apnext-sev .sig .k { color: #c9c4bd; font-size: 12px; white-space: nowrap; }
.apnext-sev .sig input[type="range"] { width: 100%; }
.apnext-sev .sig input[type="number"] { width: 64px; }
.apnext-sev .eq { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin-top: 10px; padding: 8px 6px 6px; background: #1e1c17; border: 1px solid #2c2820; border-radius: 6px; }
.apnext-sev .eq .band { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.apnext-sev .eq .band input[type="range"] { writing-mode: vertical-lr; direction: rtl; width: 22px; height: 96px; margin: 0; }
.apnext-sev .eq .band .db { font: 11px "JetBrains Mono", Consolas, monospace; color: #e8e4df; min-width: 44px; text-align: center; }
.apnext-sev .eq .band .db.hot { color: #d4a574; }
.apnext-sev .eq .band .nm { font-size: 11px; color: #c9c4bd; }
.apnext-sev .eq .band .hz { font-size: 9.5px; color: #9a9590; }
.apnext-sev .sigfoot { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; margin-top: 8px; }
.apnext-sev .sigfoot .mon { color: #c9c4bd; }
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
const widgetValue = (node, name, fallback) => {
  const w = widget(node, name);
  return w == null || w.value == null ? fallback : w.value;
};

function fmtTime(sec) {
  const m = Math.floor(sec / 60), s = sec - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
}
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const r2 = (v) => Math.round(v * 100) / 100;

// Walk the `audio` input back to a Load Audio node and read its file name.
function upstreamAudio(node) {
  const graph = node.graph ?? app.graph;
  const via = [];
  let cur = node;
  for (let hops = 0; cur && hops < 8; hops++) {
    const input = cur.inputs?.find((i) => i.name === "audio") ?? cur.inputs?.find((i) => i.type === "AUDIO");
    if (input?.link == null) return { file: null, via, reason: hops === 0 ? "the `audio` input is not connected" : `${cur.type} has no audio source` };
    const link = graph.links?.[input.link];
    const src = link && graph.getNodeById(link.origin_id);
    if (!src) return { file: null, via, reason: "broken link" };
    if (src.type === "LoadAudio" || src.widgets?.some((w) => w.name === "audio" && typeof w.value === "string")) {
      const w = src.widgets?.find((w) => w.name === "audio");
      return { file: w?.value || null, via, reason: w?.value ? "" : "Load Audio has no file selected" };
    }
    via.push(src.type);
    cur = src;
  }
  return { file: null, via, reason: "no Load Audio node found upstream" };
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

// ---- the event model --------------------------------------------------------
// A detected event is keyed by the time it was DETECTED at (`at`), so a hand
// edit survives a re-detection with the same settings; an added event has no
// `at`. The block on the lane is the writer's window: cue -> land -> settle.

function windowOf(it) {
  const k = tier(it.strength);
  const lead = it.lead != null ? it.lead : (LEAD[it.type] || [0.2, 0.3, 0.4])[k];
  let settle;
  if (it.type === "BUILD") settle = it.until != null ? it.until : it.t;
  else settle = it.until != null ? it.until : it.t + (TAIL[it.type] || [0.2, 0.3, 0.4])[k];
  return { cue: Math.max(0, it.t - lead), land: it.t, settle: Math.max(it.t, settle) };
}

function buildItems(payload, state) {
  const list = state.toggles.impacts ? payload.events : payload.events_no_impacts;
  const items = [];
  for (const e of list) {
    const key = e.t.toFixed(2);
    const ed = state.edits.get(key) || {};
    items.push({
      id: "d:" + key, at: e.t, orig: e,
      t: ed.t != null ? ed.t : e.t, type: ed.type || e.type, strength: ed.strength != null ? ed.strength : e.strength,
      lead: ed.lead != null ? ed.lead : null, until: ed.until != null ? ed.until : (e.until != null ? e.until : null),
      edited: Object.keys(ed).length > 0, added: false,
    });
  }
  state.added.forEach((a, i) => items.push({ id: "a:" + a.id, at: null, orig: null, t: a.t, type: a.type, strength: a.strength,
    lead: a.lead != null ? a.lead : null, until: a.until != null ? a.until : null, edited: false, added: true, ref: a }));
  items.sort((x, y) => x.t - y.t);
  return items;
}

const isStruck = (state, it) => !it.added && [...state.rejected].some((t) => Math.abs(t - it.at) <= MATCH);
const isAdded = (it) => it.added;

function keptByFilters(state, it) {
  const toggle = KIND_TOGGLE[it.type];
  if (toggle && !state.toggles[toggle]) return false;
  if (STRUCTURAL.has(it.type)) return true;
  return it.strength >= state.minStrength - 1e-9 && it.strength <= state.maxStrength + 1e-9;
}

function density(items, duration) {
  const minutes = Math.max(duration, 1) / 60;
  const hits = items.filter((e) => HIT_KINDS.has(e.type));
  const structural = items.length - hits.length;
  const pieces = Math.max(1, Math.ceil(duration / PIECE_SECONDS));
  const perPiece = new Array(pieces).fill(0);
  for (const e of items) perPiece[Math.min(pieces - 1, Math.floor(e.t / PIECE_SECONDS))]++;
  const over = perPiece.filter((n) => n > SLOTS_PER_PIECE).length;
  return {
    total: items.length, hits: hits.length, structural,
    hitsPerMin: hits.length / minutes, hitsPerPiece: hits.length / pieces,
    piecesOver: over, pieces, everySeconds: hits.length ? duration / hits.length : Infinity,
  };
}

// ---- the editor -------------------------------------------------------------

async function openModal(node) {
  ensureStyle();
  const wrap = el("div", { className: "apnext-sev-wrap" });
  const panel = el("div", { className: "apnext-sev" });
  const source = upstreamAudio(node);

  const audio = new Audio();
  let monitor = null;       // Web Audio chain for the 🎧 toggle, built on first use
  const close = () => {
    audio.pause(); audio.src = "";
    try { monitor?.ctx?.close?.(); } catch (_) { /* already closed */ }
    wrap.remove(); document.removeEventListener("keydown", onKey);
  };
  const head = el("header");
  head.append(
    el("h2", { textContent: "Sound events" }),
    el("span", { className: "sub", textContent: source.file || "no audio file", title: source.file || "" }),
    el("span", { className: "spacer" }),
    el("button", { className: "close", textContent: "×", onclick: close }),
  );
  panel.append(head);
  wrap.append(panel);
  document.body.append(wrap);

  if (!source.file) {
    panel.append(el("p", { className: "error", textContent: `Cannot preview: ${source.reason}. Connect Load Audio → this node's audio input and pick a file.` }));
    const onKey = (e) => { if (e.key === "Escape") close(); };
    document.addEventListener("keydown", onKey);
    return;
  }

  // ---- state seeded from the node's widgets
  let savedEdits = {};
  try { savedEdits = JSON.parse(String(widgetValue(node, "edits", "") || "") || "{}"); } catch (_) { savedEdits = {}; }
  const state = {
    sensitivity: Number(widgetValue(node, "sensitivity", 1.0)),
    minGap: Number(widgetValue(node, "min_gap_seconds", 0.18)),
    maxEvents: Number(widgetValue(node, "max_events", 200)),
    minStrength: Number(widgetValue(node, "min_strength", 0.0)),
    maxStrength: Number(widgetValue(node, "max_strength", 1.0)),
    toggles: Object.fromEntries(TOGGLES.map(([k]) => [k, Boolean(widgetValue(node, k, k !== "accents"))])),
    rejected: new Set(String(widgetValue(node, "rejected", "") || "").split(/[\s,]+/).filter(Boolean).map(Number).filter((n) => !Number.isNaN(n))),
    edits: new Map(),                 // "12.40" -> {t, lead, until, strength, type}
    added: [],                        // [{id, t, type, strength, lead, until}]
    selected: null,                   // item id
    pps: 0,                           // px per second (0 = fit)
    addType: "BASS HIT",
    shaping: Object.fromEntries(Object.keys(SHAPING_DEFAULT).map((k) => [k, Number(widgetValue(node, k, SHAPING_DEFAULT[k]))])),
    onBeat: Number(widgetValue(node, "on_beat_weight", 0)) || 0,   // lean hit strengths toward the grid (needs Recompute)
    grid: true,                       // draw the BPM grid on the lane
    monitor: false,                   // play the shaped signal instead of the file
  };
  for (const k of Object.keys(SHAPING_DEFAULT)) if (!Number.isFinite(state.shaping[k])) state.shaping[k] = SHAPING_DEFAULT[k];
  for (const e of savedEdits.events || []) if (e && e.at != null) state.edits.set(Number(e.at).toFixed(2), { ...e, at: undefined });
  let addSeq = 0;
  for (const a of savedEdits.added || []) if (a && a.t != null) state.added.push({ id: ++addSeq, t: Number(a.t), type: a.type || "BASS HIT", strength: Number(a.strength ?? 0.6), lead: a.lead ?? null, until: a.until ?? null });

  let payload = null;
  let decoded = null;       // AudioBuffer of the file, kept for re-shaping
  let peaks = null;         // Float32Array of mono samples (decimated) - the file as it is
  let shapedPeaks = null;   // the same after gain -> EQ -> curve - what the detectors hear
  let peaksRate = 0;        // samples per second of `peaks`
  let duration = 0;
  let items = [];
  let gridFit = { bpm: 0, phase: 0, on: 0, of: 0 };   // the beat grid, phase-fitted to the kept hits

  // ---- layout
  const main = el("main");
  const stage = el("div", { className: "stage" });
  const aside = el("aside");
  main.append(stage, aside);
  panel.append(main);

  // toolbar
  const toolbar = el("div", { className: "toolbar" });
  const playBtn = el("button", { className: "pill", textContent: "▶ play" });
  const clock = el("span", { className: "clock", textContent: "0:00.00 / 0:00.00" });
  const zoomIn = el("button", { className: "pill", textContent: "zoom +" });
  const zoomOut = el("button", { className: "pill", textContent: "zoom −" });
  const zoomFit = el("button", { className: "pill", textContent: "fit" });
  const addSel = el("select");
  for (const t of EVENT_TYPES) addSel.append(el("option", { value: t, textContent: t.toLowerCase() }));
  addSel.value = state.addType;
  addSel.addEventListener("change", () => { state.addType = addSel.value; });
  const gridBox = el("input", { type: "checkbox", checked: state.grid });
  const bpmOut = el("span", { className: "bpm", textContent: "" });
  gridBox.addEventListener("change", () => { state.grid = gridBox.checked; drawGrid(true); });
  toolbar.append(
    playBtn, clock, zoomOut, zoomIn, zoomFit,
    el("label", { className: "ctl" }, "double-click adds a ", addSel),
    el("label", { className: "ctl", title: "Beat grid at the track's measured tempo, phase-fitted to the kept hits. A hit off the grid is off the pulse." }, gridBox, "beat grid ", bpmOut),
    el("span", { className: "hint", textContent: "drag = move · edges = wind-up / settle · click = select · Delete = strike out / keep · space = play · ctrl+wheel = zoom" }),
  );
  stage.append(toolbar);

  // scrolling track
  const scroll = el("div", { className: "scroll" });
  const track = el("div", { className: "track" });
  const ruler = el("div", { className: "ruler" });
  const wave = el("canvas", { className: "wave" });
  const lane = el("div", { className: "lane" });
  const laneWave = el("canvas", { className: "lanewave" });   // the shaped signal, translucent, behind the events
  const gridCanvas = el("canvas", { className: "grid" });     // the BPM grid, over the waveform, under the events
  lane.append(laneWave, gridCanvas);
  const playhead = el("div", { className: "playhead" });
  track.append(ruler, wave, lane, playhead);
  scroll.append(track);
  stage.append(scroll);

  // inspector
  const inspector = el("div", { className: "inspector" });
  stage.append(inspector);

  // side panel
  aside.append(el("h3", { textContent: "Detection" }));
  if (source.via.length) aside.append(el("p", { textContent: `Preview runs on the Load Audio file; on the canvas the audio passes through ${source.via.join(" → ")} first, so counts can differ slightly.` }));
  const sens = el("input", { type: "number", step: "0.05", min: "0.25", max: "4", value: state.sensitivity });
  const gap = el("input", { type: "number", step: "0.01", min: "0.05", max: "2", value: state.minGap });
  const recompute = el("button", { className: "pill", textContent: "Recompute" });
  aside.append(el("div", { className: "row" }, el("label", { className: "ctl" }, "sensitivity ", sens), el("label", { className: "ctl" }, "min_gap ", gap), recompute));
  const onBeat = el("input", { type: "range", min: "0", max: "1", step: "0.05", value: String(state.onBeat), style: "width:120px" });
  const onBeatNum = el("input", { type: "number", min: "0", max: "1", step: "0.05", value: state.onBeat.toFixed(2) });
  const setOnBeat = (v) => { state.onBeat = clamp(Math.round(Number(v) * 20) / 20 || 0, 0, 1); onBeat.value = String(state.onBeat); onBeatNum.value = state.onBeat.toFixed(2); markNeedsRecompute(); };
  onBeat.addEventListener("input", () => setOnBeat(onBeat.value));
  onBeatNum.addEventListener("change", () => setOnBeat(onBeatNum.value));
  aside.append(el("div", { className: "row", style: "margin-top:6px" },
    el("label", { className: "ctl", title: "Prefer hits on the beat grid: a hit within 45 ms of a grid line keeps its strength, one a quarter-beat or more away is scaled by (1 - this). 0 = off, 1 = off-beat hits go to zero. Timing and structural events are never touched." }, "on-beat weight ", onBeat, onBeatNum)));
  aside.append(el("p", { textContent: "These two and the Signal section change what is detected and need Recompute (hand edits are kept where their event still exists); everything under \"What reaches the writer\" filters instantly.", style: "font-size:11.5px;color:#9a9590" }));

  // ---- signal shaping: gain -> EQ -> curve, in front of every detector
  aside.append(el("h3", { textContent: "Signal" }));
  aside.append(el("p", { textContent: "What the detectors listen to. Tilt the mix toward the kick, pull the hats down, bend the dynamics so the big hits stand proud - the teal waveform under the events is the result, and 🎧 plays it. Then Recompute.", style: "font-size:11.5px;color:#9a9590" }));
  const sigInputs = {};   // key -> [range, number | label]
  const fmtDb = (v) => `${v > 0 ? "+" : ""}${Number(v).toFixed(1)} dB`;
  const setShaping = (key, value, { fromRange = false } = {}) => {
    const lo = key === "dynamics_curve" ? 0.25 : -24, hi = key === "dynamics_curve" ? 4 : 24;
    let v = clamp(Number(value), lo, hi);
    if (!Number.isFinite(v)) v = SHAPING_DEFAULT[key];
    v = key === "dynamics_curve" ? Math.round(v * 100) / 100 : Math.round(v * 2) / 2;
    state.shaping[key] = v;
    const [range, out] = sigInputs[key];
    if (!fromRange) range.value = String(v);
    if (out.tagName === "INPUT") out.value = key === "dynamics_curve" ? v.toFixed(2) : v.toFixed(1);
    else { out.textContent = fmtDb(v); out.classList.toggle("hot", Math.abs(v) >= 0.05); }
    markNeedsRecompute();
    updateMonitor();
    scheduleReshape();
  };
  const sigRow = (key, label, title, min, max, step) => {
    const range = el("input", { type: "range", min: String(min), max: String(max), step: String(step), value: String(state.shaping[key]), title });
    const num = el("input", { type: "number", min: String(min), max: String(max), step: String(step), value: key === "dynamics_curve" ? state.shaping[key].toFixed(2) : state.shaping[key].toFixed(1), title });
    range.addEventListener("input", () => setShaping(key, range.value, { fromRange: true }));
    range.addEventListener("dblclick", () => setShaping(key, SHAPING_DEFAULT[key]));
    num.addEventListener("change", () => setShaping(key, num.value));
    sigInputs[key] = [range, num];
    return [el("span", { className: "k", textContent: label, title }), range, num];
  };
  const sig = el("div", { className: "sig" });
  sig.append(
    ...sigRow("gain_db", "gain dB", "Input gain, first in the chain. Matters mostly with a curve above 1: the curve bends relative to full scale, so a quiet master needs lifting first. Double-click the slider to reset.", -24, 24, 0.5),
    ...sigRow("dynamics_curve", "curve", "|x| to this power. 1 = untouched. Above 1 expands: soft material sinks, the big hits stand proud (fewer, cleaner beats). Below 1 compresses: the soft hits come up. Double-click to reset.", 0.25, 4, 0.05),
  );
  aside.append(sig);
  const eq = el("div", { className: "eq" });
  for (const [key, , hz, , name, hzLabel] of EQ_BANDS) {
    const range = el("input", { type: "range", min: "-24", max: "24", step: "0.5", value: String(state.shaping[key]), title: `${name} - ${hzLabel}. Double-click to reset.` });
    const db = el("span", { className: "db" + (Math.abs(state.shaping[key]) >= 0.05 ? " hot" : ""), textContent: fmtDb(state.shaping[key]) });
    range.addEventListener("input", () => setShaping(key, range.value, { fromRange: true }));
    range.addEventListener("dblclick", () => setShaping(key, SHAPING_DEFAULT[key]));
    sigInputs[key] = [range, db];
    eq.append(el("div", { className: "band" }, db, range, el("span", { className: "nm", textContent: name }), el("span", { className: "hz", textContent: hzLabel })));
    void hz;
  }
  aside.append(eq);
  const monitorBox = el("input", { type: "checkbox", checked: state.monitor });
  monitorBox.addEventListener("change", () => { state.monitor = monitorBox.checked; updateMonitor(); });
  const sigReset = el("button", { className: "pill", textContent: "flat" });
  sigReset.addEventListener("click", () => { for (const k of Object.keys(SHAPING_DEFAULT)) setShaping(k, SHAPING_DEFAULT[k]); });
  aside.append(el("div", { className: "sigfoot" },
    el("label", { className: "ctl mon", title: "Play the shaped signal through the same chain (Web Audio) instead of the file, so you can hear what the detectors are scoring." }, monitorBox, "🎧 monitor shaped signal"),
    sigReset,
  ));

  aside.append(el("h3", { textContent: "What reaches the writer" }));
  const toggleRow = el("div", { className: "row" });
  for (const [key, label] of TOGGLES) {
    const box = el("input", { type: "checkbox", checked: state.toggles[key] });
    box.addEventListener("change", () => { state.toggles[key] = box.checked; render(); });
    toggleRow.append(el("label", { className: "ctl" }, box, label));
  }
  aside.append(toggleRow);

  const clamp01 = (v) => Math.max(0, Math.min(1, Math.round(Number(v) * 100) / 100));
  const slider = el("input", { type: "range", min: "0", max: "1", step: "0.01", value: state.minStrength });
  const sliderNum = el("input", { type: "number", min: "0", max: "1", step: "0.01", value: state.minStrength.toFixed(2) });
  const sliderMax = el("input", { type: "range", min: "0", max: "1", step: "0.01", value: state.maxStrength });
  const sliderMaxNum = el("input", { type: "number", min: "0", max: "1", step: "0.01", value: state.maxStrength.toFixed(2) });
  const syncStrengthInputs = () => {
    slider.value = String(state.minStrength); sliderNum.value = state.minStrength.toFixed(2);
    sliderMax.value = String(state.maxStrength); sliderMaxNum.value = state.maxStrength.toFixed(2);
  };
  const setStrength = (v) => { state.minStrength = clamp01(v); if (state.maxStrength < state.minStrength) state.maxStrength = state.minStrength; syncStrengthInputs(); render(); };
  const setMaxStrength = (v) => { state.maxStrength = clamp01(v); if (state.minStrength > state.maxStrength) state.minStrength = state.maxStrength; syncStrengthInputs(); render(); };
  slider.addEventListener("input", () => setStrength(slider.value));
  sliderNum.addEventListener("change", () => setStrength(sliderNum.value));
  sliderMax.addEventListener("input", () => setMaxStrength(sliderMax.value));
  sliderMaxNum.addEventListener("change", () => setMaxStrength(sliderMaxNum.value));
  aside.append(
    el("div", { className: "row", style: "margin-top:8px" }, el("label", { className: "ctl" }, "min ", slider, sliderNum)),
    el("div", { className: "row", style: "margin-top:4px" }, el("label", { className: "ctl" }, "max ", sliderMax, sliderMaxNum)),
    el("p", { textContent: "strength band - only hits between the two are kept; drops / stops / sections / builds are never dropped by it", style: "font-size:11.5px;color:#9a9590" }),
  );

  const stats = el("div", { className: "stats", style: "margin-top:10px" });
  const banner = el("div", { className: "banner ok" });
  aside.append(stats, banner);

  aside.append(el("h3", { textContent: "Hit strength" }));
  aside.append(el("p", { textContent: "Click a bar: min_strength there; shift-click: max_strength at its top.", style: "font-size:11.5px;color:#9a9590" }));
  const hist = el("div", { className: "hist" });
  const axis = el("div", { className: "hist-axis" });
  for (let b = 0; b < 10; b++) axis.append(el("span", { textContent: (b / 10).toFixed(1) }));
  aside.append(hist, axis);

  const legend = el("div", { className: "legend" });
  for (const [kind, color] of Object.entries(KIND_COLOR)) legend.append(el("span", {}, el("i", { style: `background:${color}` }), kind.toLowerCase()));
  legend.append(el("span", { textContent: "faded = removed by the settings · striped = struck out · ✎ edited · + added" }));
  aside.append(el("h3", { textContent: "Legend" }), legend);

  // footer
  const note = el("span", { className: "note" });
  const resetBtn = el("button", { className: "pill danger", textContent: "Discard hand edits" });
  const apply = el("button", { className: "pill primary", textContent: "Apply to node" });
  panel.append(el("div", { className: "foot" }, note, resetBtn, el("button", { className: "pill", textContent: "Close", onclick: close }), apply));

  resetBtn.addEventListener("click", () => { state.edits.clear(); state.added = []; state.rejected.clear(); state.selected = null; render(); });

  apply.addEventListener("click", () => {
    const set = (name, value) => { const w = widget(node, name); if (w) { w.value = value; w.callback?.(value, app.canvas, node); } };
    set("sensitivity", state.sensitivity);
    set("min_gap_seconds", state.minGap);
    set("min_strength", state.minStrength);
    set("max_strength", state.maxStrength);
    for (const [key] of TOGGLES) set(key, state.toggles[key]);
    for (const key of Object.keys(SHAPING_DEFAULT)) set(key, state.shaping[key]);
    set("on_beat_weight", state.onBeat);
    set("rejected", [...state.rejected].sort((a, b) => a - b).map((t) => t.toFixed(2)).join(" "));
    set("edits", editsJson());
    node.setDirtyCanvas?.(true, true);
    close();
  });

  function editsJson() {
    const events = [];
    for (const [key, ed] of state.edits) {
      const clean = {};
      for (const k of ["t", "lead", "until", "strength", "type"]) if (ed[k] != null) clean[k] = ed[k];
      if (Object.keys(clean).length) events.push({ at: Number(key), ...clean });
    }
    const added = state.added.map((a) => ({ t: r2(a.t), type: a.type, strength: r2(a.strength), ...(a.lead != null ? { lead: r2(a.lead) } : {}), ...(a.until != null ? { until: r2(a.until) } : {}) }));
    if (!events.length && !added.length) return "";
    return JSON.stringify({ v: 1, events, added });
  }

  // ---- geometry
  const fitPps = () => Math.max(1, (scroll.clientWidth - 2) / Math.max(duration, 1));
  const pps = () => (state.pps || fitPps());
  const maxPps = () => Math.max(fitPps(), MAX_CANVAS / Math.max(duration, 1));
  const setZoom = (factor, anchorSec) => {
    const before = pps();
    const next = clamp(before * factor, fitPps(), maxPps());
    state.pps = next <= fitPps() + 1e-6 ? 0 : next;
    const anchor = anchorSec ?? (scroll.scrollLeft + scroll.clientWidth / 2) / before;
    layout();
    scroll.scrollLeft = anchor * pps() - scroll.clientWidth / 2;
  };
  zoomIn.addEventListener("click", () => setZoom(1.6));
  zoomOut.addEventListener("click", () => setZoom(1 / 1.6));
  zoomFit.addEventListener("click", () => { state.pps = 0; layout(); });
  scroll.addEventListener("wheel", (e) => {
    if (!e.ctrlKey) return;
    e.preventDefault();
    const sec = (scroll.scrollLeft + e.clientX - scroll.getBoundingClientRect().left) / pps();
    const before = pps();
    setZoom(e.deltaY < 0 ? 1.25 : 1 / 1.25);
    scroll.scrollLeft = sec * pps() - (e.clientX - scroll.getBoundingClientRect().left);
    void before;
  }, { passive: false });
  new ResizeObserver(() => { if (duration) layout(); }).observe(scroll);

  // ---- waveform
  // min / max per pixel column, centred, clipped to the canvas (a boosted
  // signal can exceed full scale - it is drawn touching the edges, not lost)
  function paintPeaks(ctx, data, width, height, color) {
    const spc = peaksRate / pps();          // peak samples per pixel column
    ctx.fillStyle = color;
    for (let x = 0; x < width; x++) {
      const a = Math.floor(x * spc), b = Math.max(a + 1, Math.floor((x + 1) * spc));
      let lo = 1, hi = -1;
      for (let i = a; i < b && i < data.length; i++) { const v = data[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
      if (hi < lo) continue;
      const y1 = clamp((1 - hi) * height / 2, 0, height), y2 = clamp((1 - lo) * height / 2, 0, height);
      ctx.fillRect(x, y1, 1, Math.max(1, y2 - y1));
    }
  }

  function drawWave() {
    const width = Math.min(MAX_CANVAS, Math.ceil(duration * pps()));
    const height = WAVE_H;
    wave.width = width; wave.height = height;
    wave.style.width = `${width}px`;
    const ctx = wave.getContext("2d");
    ctx.fillStyle = "#0f0e0b"; ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = "#463f33"; ctx.fillRect(0, height / 2, width, 1);
    if (!peaks) { ctx.fillStyle = "#9a9590"; ctx.fillText("decoding audio…", 12, 20); return; }
    paintPeaks(ctx, peaks, width, height, "#c9a26e");
  }

  // The shaped signal - what the detectors actually score - translucent
  // behind the events. With everything flat it is the file itself.
  function drawLaneWave() {
    const width = Math.min(MAX_CANVAS, Math.ceil(duration * pps()));
    laneWave.width = width; laneWave.height = LANE_H;
    laneWave.style.width = `${width}px`;
    const ctx = laneWave.getContext("2d");
    ctx.clearRect(0, 0, width, LANE_H);
    ctx.fillStyle = "rgba(132,179,166,.22)"; ctx.fillRect(0, LANE_H / 2, width, 1);
    const data = shapedPeaks || peaks;
    if (!data) return;
    paintPeaks(ctx, data, width, LANE_H, isNeutral(state.shaping) ? "rgba(132,179,166,.30)" : "rgba(132,179,166,.42)");
  }

  // One line per beat at the measured BPM, phase-fitted to the kept hits (see
  // fitGrid). Skipped when the zoom puts beats closer than 5 px.
  let gridKey = "";
  function drawGrid(force = false) {
    const width = Math.min(MAX_CANVAS, Math.ceil(duration * pps()));
    const key = `${width}|${state.grid}|${gridFit.bpm}|${gridFit.phase.toFixed(3)}`;
    if (!force && key === gridKey) return;
    gridKey = key;
    gridCanvas.width = width; gridCanvas.height = LANE_H;
    gridCanvas.style.width = `${width}px`;
    const ctx = gridCanvas.getContext("2d");
    ctx.clearRect(0, 0, width, LANE_H);
    if (!state.grid || !gridFit.bpm) return;
    const period = 60 / gridFit.bpm;
    if (period * pps() < 5) return;
    ctx.fillStyle = "rgba(232,180,184,.16)";
    for (let t = gridFit.phase; t <= duration; t += period) ctx.fillRect(Math.round(t * pps()), 0, 1, LANE_H);
  }

  // Which phase of the beat period lines up with the most kept hits? Brute
  // force over 64 phases (scheb/sound-to-light-osc fits its BPM grid the same
  // way, by quantisation error). The result is only a picture: nothing is
  // moved onto the grid.
  function fitGrid(kept) {
    const bpm = Number(payload?.bpm) || 0;
    const hits = kept.filter((e) => HIT_KINDS.has(e.type)).map((e) => e.t);
    if (!bpm || !hits.length) { gridFit = { bpm, phase: 0, on: 0, of: hits.length }; return; }
    const period = 60 / bpm;
    const onGrid = (phase) => hits.reduce((n, t) => { const r = (t - phase) / period; return n + (Math.abs(r - Math.round(r)) * period <= GRID_TOL ? 1 : 0); }, 0);
    // The server fitted the grid on the full hit list (sound_events.beat_grid)
    // and the on-beat weighting used that phase, so draw that one; fit here
    // only when an older server did not send it.
    if (payload?.grid?.bpm && payload.grid.phase != null) {
      const phase = Number(payload.grid.phase) || 0;
      gridFit = { bpm, phase, on: onGrid(phase), of: hits.length };
      return;
    }
    let best = { phase: 0, on: -1 };
    for (let k = 0; k < 64; k++) {
      const phase = (k / 64) * period;
      const on = onGrid(phase);
      if (on > best.on) best = { phase, on };
    }
    gridFit = { bpm, phase: best.phase, on: best.on, of: hits.length };
  }

  function drawRuler() {
    const width = Math.ceil(duration * pps());
    ruler.style.width = `${width}px`;
    ruler.replaceChildren();
    const steps = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60];
    const step = steps.find((s) => s * pps() >= 70) || 60;
    for (let t = 0; t <= duration; t += step) ruler.append(el("span", { textContent: fmtTime(t), style: `left:${t * pps()}px` }));
  }

  function layout() {
    const width = Math.ceil(duration * pps());
    track.style.width = `${width}px`;
    lane.style.width = `${width}px`;
    drawRuler();
    drawWave();
    drawLaneWave();
    render();
  }

  // ---- the lane
  function render() {
    if (!payload) return;
    items = buildItems(payload, state);
    const p = pps();
    lane.replaceChildren(laneWave, gridCanvas);
    const kept = [];
    for (const it of items) {
      const w = windowOf(it);
      const struck = isStruck(state, it);
      const keep = !struck && keptByFilters(state, it);
      if (keep) kept.push(it);
      // Structural events span the lane; a hit's height is its strength, so
      // the heavy ones read as heavy from across the room.
      const structural = STRUCTURAL.has(it.type);
      const h = structural ? LANE_H - 12 : Math.round(34 + clamp(it.strength, 0, 1) * (LANE_H - 64));
      const top = Math.round((LANE_H - h) / 2);
      const box = el("div", {
        className: "ev" + (structural ? " struct" : "") + (keep ? "" : " cut") + (struck ? " struck" : "")
          + (it.edited ? " edited" : "") + (it.added ? " added" : "") + (state.selected === it.id ? " sel" : ""),
        style: `left:${w.cue * p}px;width:${Math.max(6, (w.settle - w.cue) * p)}px;top:${top}px;height:${h}px;background:${KIND_COLOR[it.type] || "#9a9590"}`,
        title: `${fmtTime(it.t)} ${it.type} ${strengthLabel(it.strength)} (${it.strength.toFixed(2)})  window +${(it.t - w.cue).toFixed(2)} / -${(w.settle - it.t).toFixed(2)} s`
          + (struck ? " - STRUCK OUT" : "") + (it.added ? " - added by hand" : it.edited ? " - edited" : ""),
      });
      box.append(el("div", { className: "land", style: `left:${(it.t - w.cue) * p - 1}px` }));
      if ((w.settle - w.cue) * p > 46) box.append(el("span", { className: "lbl", textContent: it.type.toLowerCase() }));
      box.append(el("div", { className: "h l" }), el("div", { className: "h r" }));
      box.dataset.id = it.id;
      lane.append(box);
    }
    renderInspector();
    renderStats(kept);
    fitGrid(kept);
    drawGrid();
    bpmOut.textContent = gridFit.bpm ? `${gridFit.bpm.toFixed(1)} BPM · ${gridFit.of ? Math.round((100 * gridFit.on) / gridFit.of) : 0}% of hits on it` : "no pulse found";
    const changes = state.edits.size + state.added.length;
    note.textContent = `Apply writes band ${state.minStrength.toFixed(2)}-${state.maxStrength.toFixed(2)}, the kind toggles, sensitivity, min_gap`
      + (isNeutral(state.shaping) ? "" : ", the signal shaping")
      + (state.rejected.size ? `, ${state.rejected.size} struck-out` : "") + (changes ? `, ${changes} hand edit(s)` : "") + " into the node.";
  }

  // ---- drag / select / add on the lane
  let drag = null;
  lane.addEventListener("pointerdown", (e) => {
    const box = e.target.closest(".ev");
    if (!box) return;
    const it = items.find((x) => x.id === box.dataset.id);
    if (!it) return;
    e.preventDefault();
    state.selected = it.id;
    const mode = e.target.classList.contains("h") ? (e.target.classList.contains("l") ? "lead" : "until") : "move";
    const w = windowOf(it);
    drag = { it, mode, x0: e.clientX, t0: it.t, cue0: w.cue, settle0: w.settle, moved: false };
    lane.setPointerCapture(e.pointerId);
    render();
  });
  lane.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dt = (e.clientX - drag.x0) / pps();
    if (Math.abs(e.clientX - drag.x0) > 2) drag.moved = true;
    const it = drag.it;
    const ed = editFor(it);
    if (drag.mode === "move") {
      const t = r2(clamp(drag.t0 + dt, 0, duration));
      ed.t = t;
      // the hand-set window travels with the event; a default window recomputes
    } else if (drag.mode === "lead") {
      const cue = clamp(drag.cue0 + dt, 0, it.t - 0.02);
      ed.lead = r2(it.t - cue);
    } else {
      const settle = clamp(drag.settle0 + dt, it.t + 0.02, duration);
      ed.until = r2(settle);
    }
    commitEdit(it, ed);
    render();
  });
  const endDrag = (e) => {
    if (!drag) return;
    try { lane.releasePointerCapture(e.pointerId); } catch (_) { /* already released */ }
    drag = null;
    render();
  };
  lane.addEventListener("pointerup", endDrag);
  lane.addEventListener("pointercancel", endDrag);
  lane.addEventListener("dblclick", (e) => {
    if (e.target.closest(".ev")) return;
    const t = r2(clamp((e.clientX - lane.getBoundingClientRect().left) / pps(), 0, duration));
    const a = { id: ++addSeq, t, type: state.addType, strength: 0.7, lead: null, until: null };
    state.added.push(a);
    state.selected = "a:" + a.id;
    render();
  });
  lane.addEventListener("click", (e) => {
    if (e.target.closest(".ev")) return;
    if (!drag) { state.selected = null; render(); }
  });

  // an item's editable record (detected -> edits map entry; added -> the record itself)
  function editFor(it) {
    if (it.added) return it.ref;
    const key = it.at.toFixed(2);
    return { ...(state.edits.get(key) || {}) };
  }
  function commitEdit(it, ed) {
    if (it.added) { Object.assign(it.ref, ed); return; }
    const key = it.at.toFixed(2);
    const clean = {};
    for (const k of ["t", "lead", "until", "strength", "type"]) {
      if (ed[k] == null) continue;
      // an edit equal to the detected value is no edit
      if (k === "t" && Math.abs(ed.t - it.orig.t) < 0.005) continue;
      if (k === "strength" && Math.abs(ed.strength - it.orig.strength) < 0.005) continue;
      if (k === "type" && ed.type === it.orig.type) continue;
      if (k === "until" && it.orig.until != null && Math.abs(ed.until - it.orig.until) < 0.005) continue;
      clean[k] = ed[k];
    }
    if (Object.keys(clean).length) state.edits.set(key, clean); else state.edits.delete(key);
  }

  function toggleStruck(it) {
    if (it.added) { state.added = state.added.filter((a) => a !== it.ref); state.selected = null; return; }
    const hit = [...state.rejected].find((t) => Math.abs(t - it.at) <= MATCH);
    if (hit !== undefined) state.rejected.delete(hit); else state.rejected.add(Number(it.at.toFixed(2)));
  }

  function renderInspector() {
    inspector.replaceChildren();
    const it = items.find((x) => x.id === state.selected);
    if (!it) { inspector.append(el("span", { className: "none", textContent: "Select an event on the lane to edit it - or double-click the lane to add one." })); return; }
    const w = windowOf(it);
    const struck = isStruck(state, it);
    const typeSel = el("select");
    for (const t of EVENT_TYPES) typeSel.append(el("option", { value: t, textContent: t.toLowerCase() }));
    typeSel.value = it.type;
    typeSel.addEventListener("change", () => { const ed = editFor(it); ed.type = typeSel.value; commitEdit(it, ed); render(); });
    const tNum = el("input", { type: "number", step: "0.01", min: "0", max: duration.toFixed(2), value: it.t.toFixed(2) });
    tNum.addEventListener("change", () => { const ed = editFor(it); ed.t = r2(clamp(Number(tNum.value), 0, duration)); commitEdit(it, ed); render(); });
    const sNum = el("input", { type: "range", min: "0", max: "1", step: "0.01", value: it.strength, style: "width:110px" });
    const sLbl = el("span", { textContent: `${it.strength.toFixed(2)} ${strengthLabel(it.strength)}`, style: "min-width:78px;color:#c9c4bd" });
    sNum.addEventListener("input", () => { const ed = editFor(it); ed.strength = clamp01(sNum.value); commitEdit(it, ed); sLbl.textContent = `${ed.strength.toFixed(2)} ${strengthLabel(ed.strength)}`; render(); });
    const leadNum = el("input", { type: "number", step: "0.01", min: "0", max: "5", value: (it.t - w.cue).toFixed(2) });
    leadNum.addEventListener("change", () => { const ed = editFor(it); ed.lead = r2(clamp(Number(leadNum.value), 0, 5)); commitEdit(it, ed); render(); });
    const settleNum = el("input", { type: "number", step: "0.01", min: "0", max: "10", value: (w.settle - it.t).toFixed(2) });
    settleNum.addEventListener("change", () => { const ed = editFor(it); ed.until = r2(clamp(it.t + Number(settleNum.value), it.t, duration)); commitEdit(it, ed); render(); });
    const strikeBtn = el("button", { className: "pill" + (struck ? "" : " danger"), textContent: it.added ? "remove" : (struck ? "keep" : "strike out") });
    strikeBtn.addEventListener("click", () => { toggleStruck(it); render(); });
    const resetBtnI = el("button", { className: "pill", textContent: "reset to detected", disabled: it.added || !it.edited });
    resetBtnI.addEventListener("click", () => { state.edits.delete(it.at.toFixed(2)); render(); });
    const seek = el("button", { className: "pill", textContent: "▶ from here" });
    seek.addEventListener("click", () => { audio.currentTime = Math.max(0, it.t - 1.0); audio.play(); });
    inspector.append(
      el("label", { className: "ctl" }, "type ", typeSel),
      el("label", { className: "ctl" }, "lands at ", tNum, "s"),
      el("label", { className: "ctl" }, "strength ", sNum, sLbl),
      el("label", { className: "ctl" }, "wind-up ", leadNum, "s before"),
      el("label", { className: "ctl" }, "settle ", settleNum, "s after"),
      seek, strikeBtn, resetBtnI,
      el("span", { style: "color:#9a9590;font-size:11.5px", textContent: it.added ? "added by hand" : (it.edited ? `detected at ${fmtTime(it.at)}, edited` : `detected at ${fmtTime(it.at)}`) }),
    );
  }

  function renderStats(kept) {
    const d = density(kept, duration);
    stats.replaceChildren(
      stat("track", fmtTime(duration), `${d.pieces} pieces of ~${PIECE_SECONDS}s`),
      stat("events kept", String(d.total), `${d.hits} hits · ${d.structural} structural`),
      stat("hits / minute", d.hitsPerMin.toFixed(0), Number.isFinite(d.everySeconds) ? `one every ${d.everySeconds.toFixed(1)}s` : "none"),
      stat("hits / piece", d.hitsPerPiece.toFixed(1), `${SLOTS_PER_PIECE} slots per piece`),
      stat("pieces overflowing", String(d.piecesOver), d.piecesOver ? "extra hits get cut, strongest kept" : "everything fits"),
      stat("node max_events", String(state.maxEvents), d.total > state.maxEvents ? `will thin ${d.total} → ${state.maxEvents}` : "not reached"),
    );
    const suggested = suggestThreshold();
    const jump = suggested > state.minStrength + 1e-9
      ? el("button", { className: "pill", textContent: `set min ${suggested.toFixed(2)}`, onclick: () => setStrength(suggested) }) : null;
    if (d.hitsPerMin > TOO_MANY_PER_MIN) {
      banner.className = "banner bad";
      banner.replaceChildren(el("b", { textContent: `Too many hits: ${d.hitsPerMin.toFixed(0)} per minute. ` }),
        `A hit every ${d.everySeconds.toFixed(1)}s is a metronome, not a beat to stage. ${suggested.toFixed(2)} keeps the heavy ones (~${OK_PER_MIN}/min).`, jump);
    } else if (d.hitsPerMin > OK_PER_MIN) {
      banner.className = "banner warn";
      banner.replaceChildren(el("b", { textContent: `Dense: ${d.hitsPerMin.toFixed(0)} hits per minute. ` }),
        `Workable, but pieces will fill their ${SLOTS_PER_PIECE} slots and lighter hits crowd out the drops.`, jump);
    } else if (d.hits === 0 && d.structural === 0) {
      banner.className = "banner warn";
      banner.replaceChildren(el("b", { textContent: "Nothing left. " }), "Every kind is off or the strength band holds no hit - the writer will get no sound moments at all.");
    } else {
      banner.className = "banner ok";
      banner.replaceChildren(el("b", { textContent: `Good density: ${d.hitsPerMin.toFixed(0)} hits per minute. ` }),
        `About ${d.hitsPerPiece.toFixed(1)} hits per piece plus the drops and stops - each one can land as a real, visible event.`);
    }
    // histogram of hit strengths (toggled-on kinds, before the band)
    const pool = items.filter((e) => HIT_KINDS.has(e.type) && state.toggles[KIND_TOGGLE[e.type]] && !isStruck(state, e));
    const bins = new Array(10).fill(0);
    for (const e of pool) bins[Math.min(9, Math.floor(e.strength * 10))]++;
    const top = Math.max(1, ...bins);
    hist.replaceChildren(...bins.map((n, b) => {
      const inBand = (b + 1) / 10 > state.minStrength + 1e-9 && b / 10 < state.maxStrength - 1e-9;
      const bin = el("div", { className: "bin" + (inBand ? " kept" : ""), title: `strength ${(b / 10).toFixed(1)}-${((b + 1) / 10).toFixed(1)}: ${n} hits` });
      bin.append(el("span", { className: "n", textContent: n ? String(n) : "" }), el("div", { className: "bar", style: `height:${Math.max(2, (n / top) * 56)}px` }));
      bin.addEventListener("click", (ev) => (ev.shiftKey ? setMaxStrength((b + 1) / 10) : setStrength(b / 10)));
      return bin;
    }));
  }

  function suggestThreshold() {
    for (let t = 0; t <= 1.0001; t += 0.05) {
      const trial = { ...state, minStrength: t };
      const kept = items.filter((it) => !isStruck(state, it) && keptByFilters(trial, it));
      if (density(kept, duration).hitsPerMin <= OK_PER_MIN) return Math.min(state.maxStrength, Math.round(t * 20) / 20);
    }
    return state.maxStrength;
  }

  function stat(k, v, small) {
    return el("div", { className: "stat" }, el("div", { className: "k", textContent: k }), el("div", { className: "v" }, v, small ? el("small", { textContent: small }) : null));
  }

  // ---- audio: playback + decoded waveform
  audio.src = audioUrl(source.file);
  const updateClock = () => { clock.textContent = `${fmtTime(audio.currentTime || 0)} / ${fmtTime(duration)}`; playhead.style.left = `${(audio.currentTime || 0) * pps()}px`; };
  audio.addEventListener("loadedmetadata", () => { if (!duration) { duration = audio.duration || 0; layout(); } updateClock(); });
  audio.addEventListener("play", () => { playBtn.textContent = "❚❚ pause"; tick(); });
  audio.addEventListener("pause", () => { playBtn.textContent = "▶ play"; });
  audio.addEventListener("ended", () => { playBtn.textContent = "▶ play"; });
  let raf = 0;
  function tick() { updateClock(); if (!audio.paused) raf = requestAnimationFrame(tick); else cancelAnimationFrame(raf); }
  playBtn.addEventListener("click", () => { if (audio.paused) audio.play(); else audio.pause(); });
  wave.addEventListener("click", (e) => { audio.currentTime = clamp((e.clientX - wave.getBoundingClientRect().left) / pps(), 0, duration); updateClock(); });
  ruler.addEventListener("click", (e) => { audio.currentTime = clamp((e.clientX - ruler.getBoundingClientRect().left) / pps(), 0, duration); updateClock(); });

  const onKey = (e) => {
    if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
    if (e.key === "Escape") { close(); return; }
    if (e.key === " ") { e.preventDefault(); if (audio.paused) audio.play(); else audio.pause(); return; }
    if ((e.key === "Delete" || e.key === "Backspace") && state.selected) {
      const it = items.find((x) => x.id === state.selected);
      if (it) { toggleStruck(it); render(); }
    }
  };
  document.addEventListener("keydown", onKey);

  // Mono mix of `chans`, decimated to ~4000 peak samples per second (enough for
  // 400 px/s), keeping the largest-magnitude sample of each step. `curve`
  // bends the dynamics like sound_events.shape_signal: sign(x) * |x| ^ curve.
  function decimate(chans, n, sampleRate, curve = 1) {
    const target = 4000;
    const step = Math.max(1, Math.floor(sampleRate / target));
    const out = new Float32Array(Math.ceil(n / step));
    const ch = chans.length;
    const bend = Math.abs(curve - 1) >= 1e-3;
    for (let i = 0, k = 0; i < n; i += step, k++) {
      let m = 0;
      for (let j = i; j < Math.min(n, i + step); j++) { let v = 0; for (let c = 0; c < ch; c++) v += chans[c][j]; v /= ch; if (Math.abs(v) > Math.abs(m)) m = v; }
      out[k] = bend ? Math.sign(m) * Math.pow(Math.abs(m), curve) : m;
    }
    return { out, rate: sampleRate / step };
  }

  // gain -> five biquads on a Web Audio graph. Shared by the offline render
  // that draws the lane waveform and the live 🎧 monitor.
  function buildChain(ctx) {
    const gain = ctx.createGain();
    const filters = EQ_BANDS.map(([, kind, hz, q]) => { const f = ctx.createBiquadFilter(); f.type = kind; f.frequency.value = hz; f.Q.value = q; return f; });
    let last = gain;
    for (const f of filters) { last.connect(f); last = f; }
    const tune = (sh) => {
      gain.gain.value = Math.pow(10, (sh.gain_db || 0) / 20);
      EQ_BANDS.forEach(([key], i) => { filters[i].gain.value = sh[key] || 0; });
    };
    return { input: gain, output: last, tune };
  }

  // Re-render the shaped signal for the lane. Debounced: an EQ slider fires
  // dozens of times a second and an offline render of a 3-minute track takes
  // ~100 ms.
  let reshapeTimer = 0, reshaping = false, reshapeAgain = false;
  function scheduleReshape() { clearTimeout(reshapeTimer); reshapeTimer = setTimeout(reshape, 180); }
  async function reshape() {
    if (!decoded) return;
    if (reshaping) { reshapeAgain = true; return; }
    reshaping = true;
    try {
      const sh = { ...state.shaping };
      if (isNeutral(sh)) { shapedPeaks = null; }
      else {
        const off = new OfflineAudioContext(1, decoded.length, decoded.sampleRate);
        const src = off.createBufferSource(); src.buffer = decoded;
        const chain = buildChain(off); chain.tune(sh);
        src.connect(chain.input); chain.output.connect(off.destination);
        src.start(0);
        const rendered = await off.startRendering();
        shapedPeaks = decimate([rendered.getChannelData(0)], rendered.length, rendered.sampleRate, sh.dynamics_curve).out;
      }
      drawLaneWave();
    } catch (err) {
      console.warn("[APNext sound events] reshape failed:", err);
    } finally {
      reshaping = false;
      if (reshapeAgain) { reshapeAgain = false; scheduleReshape(); }
    }
  }

  // 🎧: route the <audio> element through the same chain. Once an element is
  // captured by createMediaElementSource it only sounds through the graph, so
  // "off" is a straight wire to the destination, not a teardown.
  function updateMonitor() {
    if (!monitor) {
      if (!state.monitor) return;
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const src = ctx.createMediaElementSource(audio);
        const chain = buildChain(ctx);
        const shaper = ctx.createWaveShaper(); shaper.oversample = "2x";
        chain.output.connect(shaper);
        monitor = { ctx, src, chain, shaper, wired: null };
      } catch (err) {
        console.warn("[APNext sound events] monitor unavailable:", err);
        state.monitor = false; monitorBox.checked = false;
        return;
      }
    }
    const m = monitor;
    m.chain.tune(state.shaping);
    const curve = state.shaping.dynamics_curve || 1;
    const table = new Float32Array(2049);
    for (let i = 0; i < table.length; i++) { const x = (i / (table.length - 1)) * 2 - 1; table[i] = Math.sign(x) * Math.pow(Math.abs(x), curve); }
    m.shaper.curve = table;
    const want = state.monitor ? "shaped" : "dry";
    if (m.wired !== want) {
      try { m.src.disconnect(); } catch (_) { /* nothing wired yet */ }
      try { m.shaper.disconnect(); } catch (_) { /* nothing wired yet */ }
      if (want === "shaped") { m.src.connect(m.chain.input); m.shaper.connect(m.ctx.destination); }
      else m.src.connect(m.ctx.destination);
      m.wired = want;
    }
    if (m.ctx.state === "suspended") m.ctx.resume().catch(() => {});
  }

  function markNeedsRecompute() {
    const stale = payload && (
      Object.keys(SHAPING_DEFAULT).some((k) => Math.abs((payload.shaping?.[k] ?? SHAPING_DEFAULT[k]) - state.shaping[k]) >= 0.01)
      || Math.abs((Number(payload.on_beat_weight) || 0) - state.onBeat) >= 0.01);
    recompute.classList.toggle("needs", Boolean(stale));
    recompute.textContent = stale ? "Recompute ●" : "Recompute";
    recompute.title = stale ? "The signal shaping or on-beat weight changed since the events were detected - run the detectors again." : "";
  }

  (async () => {
    try {
      const buf = await (await fetch(audioUrl(source.file))).arrayBuffer();
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      decoded = await ctx.decodeAudioData(buf);
      ctx.close?.();
      const chans = Array.from({ length: decoded.numberOfChannels }, (_, c) => decoded.getChannelData(c));
      const d = decimate(chans, decoded.length, decoded.sampleRate);
      peaks = d.out; peaksRate = d.rate;
      if (!duration) duration = decoded.duration;
      layout();
      if (!isNeutral(state.shaping)) reshape();
    } catch (err) {
      console.warn("[APNext sound events] waveform decode failed:", err);
      const ctx = wave.getContext("2d"); ctx.fillStyle = "#d07070"; ctx.fillText(`waveform unavailable: ${err.message}`, 12, 20);
    }
  })();

  // ---- detection
  async function compute() {
    state.sensitivity = Number(sens.value) || 1.0;
    state.minGap = Number(gap.value) || 0.18;
    banner.className = "banner ok";
    banner.textContent = "Detecting… (a 3-minute track takes a few seconds)";
    recompute.disabled = true;
    try {
      const res = await api.fetchApi("/apnext/h3/sound_events_preview", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio: source.file, sensitivity: state.sensitivity, min_gap_seconds: state.minGap, on_beat_weight: state.onBeat, ...state.shaping }),
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
      payload = data;
      if (!duration) duration = Number(data.duration) || 0;
      markNeedsRecompute();
      layout();
    } catch (err) {
      banner.className = "banner bad";
      banner.textContent = `Preview failed: ${err.message}`;
    } finally {
      recompute.disabled = false;
    }
  }
  recompute.addEventListener("click", compute);
  await compute();
}

app.registerExtension({
  name: "apnext.h3.sound_events_preview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_TYPE) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const __btn = this.addWidget("button", "🎚 Preview & edit events", null, () => {
        openModal(this).catch((err) => console.error("[APNext sound events]", err));
      });
      __btn.serialize = false;   // a button has no value to save; a saved null would shift later widgets
      return r;
    };
  },
});
