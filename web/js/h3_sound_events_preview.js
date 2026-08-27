// APNext H3 Sound Events - "🎚 Preview events" button
//
// Strength is normalised to each track's loudest hit, so `min_strength` has to
// be tuned per song - and until now the only way to see the result was to run
// the whole workflow. This button finds the Load Audio node upstream, asks the
// server to run the detectors once with every kind on (see
// nodes/h3/sound_events_preview.py), and opens a modal where the kind toggles
// and the min_strength slider filter the list instantly. It shows what the
// writer will actually see - hits per minute, hits per 9 s piece, how many
// pieces overflow `events_per_scene` - warns when the track is too dense, and
// suggests the threshold that lands in the comfortable band. "Apply" writes
// the values back into the node's widgets.

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
const HIT_KINDS = new Set(["BASS HIT", "IMPACT", "ACCENT"]);
const STRUCTURAL = new Set(["DROP", "STOP", "SECTION", "BUILD"]);
const KIND_COLOR = {
  "BASS HIT": "#d4a574", IMPACT: "#e8b4b8", DROP: "#84b3a6", STOP: "#88a9c0",
  BUILD: "#a7bd84", SECTION: "#b58fc2", ACCENT: "#ccb777",
};

// Density bands, hits per minute. A 9 s piece gets `events_per_scene` slots
// (6 by default) and drops/stops always take theirs first, so ~24/min
// (one every 2.5 s) fills a piece with 2-4 hits and still leaves room.
const OK_PER_MIN = 24;
const TOO_MANY_PER_MIN = 40;
const PIECE_SECONDS = 9;
const SLOTS_PER_PIECE = 6;

const CSS = `
.apnext-sev-wrap { position: fixed; inset: 0; z-index: 10000; background: rgba(0,0,0,.72);
  display: flex; align-items: center; justify-content: center; padding: 24px; color-scheme: dark; }
.apnext-sev { background: #14120e; border: 1px solid #463f33; border-radius: 10px;
  width: min(1040px, 96vw); max-height: 92vh; display: flex; flex-direction: column;
  box-shadow: 0 18px 60px rgba(0,0,0,.7); font: 13px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: #e8e4df; }
.apnext-sev * { box-sizing: border-box; }
.apnext-sev :is(input, button, select):focus, .apnext-sev :is(input, button, select):focus-visible { outline: none; }
.apnext-sev header { display: flex; align-items: center; gap: 12px; padding: 12px 16px; border-bottom: 1px solid #2c2820; flex: 0 0 auto; }
.apnext-sev header h2 { margin: 0; font-size: 15px; font-weight: 650; }
.apnext-sev header .sub { color: #9a9590; font-size: 12px; margin-left: auto; max-width: 50%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.apnext-sev header button.close { background: none; border: 0; color: #9a9590; font-size: 20px; line-height: 1; cursor: pointer; padding: 0 4px; }
.apnext-sev header button.close:hover { color: #e8e4df; }
.apnext-sev .body { overflow: auto; padding: 14px 16px 18px; overscroll-behavior: contain; }
.apnext-sev h3 { margin: 18px 0 8px; font-size: 12px; font-weight: 650; text-transform: uppercase; letter-spacing: .1em; color: #d4a574; }
.apnext-sev h3:first-child { margin-top: 0; }
.apnext-sev p { margin: 6px 0; color: #c9c4bd; }
.apnext-sev .row { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; }
.apnext-sev label.ctl { display: inline-flex; align-items: center; gap: 6px; color: #c9c4bd; white-space: nowrap; }
.apnext-sev input[type="number"] { width: 72px; background: #1e1c17; color: #e8e4df; border: 1px solid #2c2820; border-radius: 3px; padding: 3px 6px; font: inherit; }
.apnext-sev input[type="number"]:focus { border-color: #d4a574; }
.apnext-sev input[type="range"] { accent-color: #d4a574; width: 260px; }
.apnext-sev input[type="checkbox"] { accent-color: #d4a574; width: 14px; height: 14px; margin: 0; }
.apnext-sev button.pill { background: #1e1c17; color: #e8e4df; border: 1px solid #463f33; border-radius: 999px; padding: 4px 12px; cursor: pointer; font: inherit; }
.apnext-sev button.pill:hover { border-color: #d4a574; }
.apnext-sev button.pill.primary { background: #d4a574; color: #1a1510; border-color: #d4a574; font-weight: 600; }
.apnext-sev button.pill.primary:hover { background: #e8b4b8; border-color: #e8b4b8; }
.apnext-sev .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; }
.apnext-sev .stat { background: #1e1c17; border: 1px solid #2c2820; border-radius: 6px; padding: 8px 10px; }
.apnext-sev .stat .k { color: #9a9590; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
.apnext-sev .stat .v { font-size: 18px; font-weight: 650; margin-top: 2px; }
.apnext-sev .stat .v small { font-size: 11px; color: #9a9590; font-weight: 400; margin-left: 4px; }
.apnext-sev .banner { border-radius: 6px; padding: 10px 12px; margin: 10px 0 0; border: 1px solid; line-height: 1.45; }
.apnext-sev .banner.ok { border-color: #84b3a6; background: rgba(132,179,166,.12); }
.apnext-sev .banner.warn { border-color: #d4a574; background: rgba(212,165,116,.12); }
.apnext-sev .banner.bad { border-color: #d07070; background: rgba(208,112,112,.14); }
.apnext-sev .banner b { font-weight: 650; }
.apnext-sev .banner button.pill { margin-left: 10px; padding: 2px 10px; }
.apnext-sev .hist { display: grid; grid-template-columns: repeat(10, 1fr); gap: 4px; align-items: end; height: 84px; margin-top: 6px; }
.apnext-sev .hist .bin { position: relative; height: 100%; display: flex; flex-direction: column; justify-content: flex-end; cursor: pointer; }
.apnext-sev .hist .bin .bar { background: #463f33; border-radius: 3px 3px 0 0; min-height: 2px; transition: height .12s; }
.apnext-sev .hist .bin.kept .bar { background: #d4a574; }
.apnext-sev .hist .bin .n { position: absolute; top: -2px; left: 0; right: 0; text-align: center; font-size: 10px; color: #9a9590; }
.apnext-sev .hist-axis { display: grid; grid-template-columns: repeat(10, 1fr); font-size: 10px; color: #9a9590; text-align: center; margin-top: 2px; }
.apnext-sev .timeline { position: relative; height: 34px; background: #1e1c17; border: 1px solid #2c2820; border-radius: 4px; margin-top: 8px; overflow: hidden; }
.apnext-sev .timeline i { position: absolute; top: 4px; bottom: 4px; width: 2px; border-radius: 1px; opacity: .9; }
.apnext-sev .timeline i.struct { top: 0; bottom: 0; width: 3px; }
.apnext-sev .timeline i.cut { opacity: .18; }
.apnext-sev .timeline i { cursor: pointer; width: 3px; }
.apnext-sev .timeline i:hover { outline: 1px solid #fff; }
.apnext-sev .timeline i.struck { background: #d07070 !important; opacity: .9; height: 8px; top: auto; bottom: 2px; border-radius: 2px; width: 6px; margin-left: -1.5px; }
.apnext-sev .legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 11px; color: #9a9590; margin-top: 6px; }
.apnext-sev .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: -1px; margin-right: 4px; }
.apnext-sev pre.table { background: #0f0f0f; border: 1px solid #2c2820; border-radius: 6px; padding: 10px 12px; margin: 6px 0 0;
  font: 11.5px/1.45 "JetBrains Mono", Consolas, monospace; color: #ddd6c8; max-height: 220px; overflow: auto; white-space: pre; }
.apnext-sev .foot { display: flex; gap: 10px; justify-content: flex-end; align-items: center; padding: 12px 16px; border-top: 1px solid #2c2820; }
.apnext-sev .foot .note { margin-right: auto; color: #9a9590; font-size: 12px; }
.apnext-sev .error { color: #d07070; white-space: pre-wrap; }
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

// ---- the density maths --------------------------------------------------

function filterEvents(payload, state) {
  const list = state.toggles.impacts ? payload.events : payload.events_no_impacts;
  return list.filter((e) => {
    if (state.rejected && [...state.rejected].some((t) => Math.abs(t - e.t) <= 0.05)) return false;
    const toggle = KIND_TOGGLE[e.type];
    if (toggle && !state.toggles[toggle]) return false;
    if (STRUCTURAL.has(e.type)) return true;           // never dropped by the strength band
    return e.strength >= state.minStrength - 1e-9 && e.strength <= state.maxStrength + 1e-9;
  });
}

function density(events, duration) {
  const minutes = Math.max(duration, 1) / 60;
  const hits = events.filter((e) => HIT_KINDS.has(e.type));
  const structural = events.length - hits.length;
  const pieces = Math.max(1, Math.ceil(duration / PIECE_SECONDS));
  const perPiece = new Array(pieces).fill(0);
  for (const e of events) perPiece[Math.min(pieces - 1, Math.floor(e.t / PIECE_SECONDS))]++;
  const over = perPiece.filter((n) => n > SLOTS_PER_PIECE).length;
  return {
    total: events.length, hits: hits.length, structural,
    hitsPerMin: hits.length / minutes,
    hitsPerPiece: hits.length / pieces,
    piecesOver: over, pieces,
    everySeconds: hits.length ? duration / hits.length : Infinity,
  };
}

function suggestThreshold(payload, state) {
  for (let t = 0; t <= 1.0001; t += 0.05) {
    const d = density(filterEvents(payload, { ...state, minStrength: t }), payload.duration);
    if (d.hitsPerMin <= OK_PER_MIN) return Math.min(state.maxStrength, Math.round(t * 20) / 20);
  }
  return state.maxStrength;
}

// ---- the modal -------------------------------------------------------------

async function openModal(node) {
  ensureStyle();

  const wrap = el("div", { className: "apnext-sev-wrap" });
  const panel = el("div", { className: "apnext-sev" });
  const body = el("div", { className: "body" });
  const close = () => { wrap.remove(); document.removeEventListener("keydown", onKey); };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  wrap.addEventListener("click", (e) => { if (e.target === wrap) close(); });
  document.addEventListener("keydown", onKey);

  const source = upstreamAudio(node);
  const head = el("header");
  head.append(
    el("h2", { textContent: "Sound events preview" }),
    el("span", { className: "sub", textContent: source.file || "no audio file", title: source.file || "" }),
    el("button", { className: "close", textContent: "×", onclick: close }),
  );
  panel.append(head, body);
  wrap.append(panel);
  document.body.append(wrap);

  if (!source.file) {
    body.append(el("p", { className: "error", textContent: `Cannot preview: ${source.reason}. Connect Load Audio → this node's audio input and pick a file.` }));
    return;
  }

  // state seeded from the node's widgets
  const state = {
    sensitivity: Number(widgetValue(node, "sensitivity", 1.0)),
    minGap: Number(widgetValue(node, "min_gap_seconds", 0.18)),
    maxEvents: Number(widgetValue(node, "max_events", 200)),
    minStrength: Number(widgetValue(node, "min_strength", 0.0)),
    maxStrength: Number(widgetValue(node, "max_strength", 1.0)),
    toggles: Object.fromEntries(TOGGLES.map(([k]) => [k, Boolean(widgetValue(node, k, k !== "accents"))])),
    rejected: new Set(String(widgetValue(node, "rejected", "") || "").split(/[\s,]+/).filter(Boolean).map(Number).filter((n) => !Number.isNaN(n))),
  };
  const isRejected = (e) => [...state.rejected].some((t) => Math.abs(t - e.t) <= 0.05);
  let payload = null;

  // ---- controls -----------------------------------------------------------
  body.append(el("h3", { textContent: "1. Detection" }));
  if (source.via.length) {
    body.append(el("p", { textContent: `Preview runs on the Load Audio file; on the canvas the audio passes through ${source.via.join(" → ")} first, so counts can differ slightly.` }));
  }
  const sens = el("input", { type: "number", step: "0.05", min: "0.25", max: "4", value: state.sensitivity });
  const gap = el("input", { type: "number", step: "0.01", min: "0.05", max: "2", value: state.minGap });
  const recompute = el("button", { className: "pill", textContent: "Recompute" });
  body.append(el("div", { className: "row" },
    el("label", { className: "ctl" }, "sensitivity ", sens),
    el("label", { className: "ctl" }, "min_gap_seconds ", gap),
    recompute,
    el("span", { textContent: "(these two change what is detected and need a re-run; everything below filters instantly)", style: "color:#9a9590;font-size:12px" }),
  ));

  body.append(el("h3", { textContent: "2. What reaches the writer" }));
  const toggleRow = el("div", { className: "row" });
  for (const [key, label] of TOGGLES) {
    const box = el("input", { type: "checkbox", checked: state.toggles[key] });
    box.addEventListener("change", () => { state.toggles[key] = box.checked; render(); });
    toggleRow.append(el("label", { className: "ctl" }, box, label));
  }
  body.append(toggleRow);

  // The hit stream is kept inside a strength BAND [min, max]: a floor alone
  // thins a busy track, a ceiling as well targets one layer of the mix
  // (0.35-0.45 = "the hits around 0.4"). Either bound pushes the other along
  // so the band never inverts.
  const clamp01 = (v) => Math.max(0, Math.min(1, Math.round(Number(v) * 100) / 100));
  const slider = el("input", { type: "range", min: "0", max: "1", step: "0.01", value: state.minStrength });
  const sliderNum = el("input", { type: "number", min: "0", max: "1", step: "0.01", value: state.minStrength.toFixed(2) });
  const sliderMax = el("input", { type: "range", min: "0", max: "1", step: "0.01", value: state.maxStrength });
  const sliderMaxNum = el("input", { type: "number", min: "0", max: "1", step: "0.01", value: state.maxStrength.toFixed(2) });
  const syncStrengthInputs = () => {
    slider.value = String(state.minStrength); sliderNum.value = state.minStrength.toFixed(2);
    sliderMax.value = String(state.maxStrength); sliderMaxNum.value = state.maxStrength.toFixed(2);
  };
  const setStrength = (v) => {
    state.minStrength = clamp01(v);
    if (state.maxStrength < state.minStrength) state.maxStrength = state.minStrength;
    syncStrengthInputs();
    render();
  };
  const setMaxStrength = (v) => {
    state.maxStrength = clamp01(v);
    if (state.minStrength > state.maxStrength) state.minStrength = state.maxStrength;
    syncStrengthInputs();
    render();
  };
  slider.addEventListener("input", () => setStrength(slider.value));
  sliderNum.addEventListener("change", () => setStrength(sliderNum.value));
  sliderMax.addEventListener("input", () => setMaxStrength(sliderMax.value));
  sliderMaxNum.addEventListener("change", () => setMaxStrength(sliderMaxNum.value));
  body.append(el("div", { className: "row", style: "margin-top:8px" },
    el("label", { className: "ctl" }, "min_strength ", slider, sliderNum),
    el("label", { className: "ctl" }, "max_strength ", sliderMax, sliderMaxNum),
    el("span", { textContent: "a band: only hits between the two are kept; drops / stops / sections / builds are never dropped by it", style: "color:#9a9590;font-size:12px" }),
  ));

  const stats = el("div", { className: "stats", style: "margin-top:12px" });
  const banner = el("div", { className: "banner ok" });
  body.append(stats, banner);

  body.append(el("h3", { textContent: "3. Hit strength - where the threshold cuts" }));
  body.append(el("p", { textContent: "Bars are the hits (bass hits / impacts / accents) by strength, 0 = quietest kept peak, 1 = the loudest hit in this track. Click a bar to set min_strength there; shift-click to set max_strength to the top of that bar (click 0.3, shift-click 0.4 = the band 0.30-0.50)." }));
  const hist = el("div", { className: "hist" });
  const axis = el("div", { className: "hist-axis" });
  for (let b = 0; b < 10; b++) axis.append(el("span", { textContent: (b / 10).toFixed(1) }));
  body.append(hist, axis);

  body.append(el("h3", { textContent: "4. Timeline - click a tick to strike that hit out (click again to keep it)" }));
  const timeline = el("div", { className: "timeline" });
  const legend = el("div", { className: "legend" });
  for (const [kind, color] of Object.entries(KIND_COLOR)) legend.append(el("span", {}, el("i", { style: `background:${color}` }), kind.toLowerCase()));
  legend.append(el("span", { textContent: "faded = removed by the current settings" }));
  body.append(timeline, legend);

  body.append(el("h3", { textContent: "5. The first lines of the table" }));
  const table = el("pre", { className: "table", textContent: "…" });
  body.append(table);

  const note = el("span", { className: "note" });
  const apply = el("button", { className: "pill primary", textContent: "Apply to node" });
  const foot = el("div", { className: "foot" }, note, el("button", { className: "pill", textContent: "Close", onclick: close }), apply);
  panel.append(foot);

  apply.addEventListener("click", () => {
    const set = (name, value) => { const w = widget(node, name); if (w) { w.value = value; w.callback?.(value, app.canvas, node); } };
    set("sensitivity", state.sensitivity);
    set("min_gap_seconds", state.minGap);
    set("min_strength", state.minStrength);
    set("max_strength", state.maxStrength);
    for (const [key] of TOGGLES) set(key, state.toggles[key]);
    set("rejected", [...state.rejected].sort((a, b) => a - b).map((t) => t.toFixed(2)).join(" "));
    node.setDirtyCanvas?.(true, true);
    close();
  });

  // ---- rendering ----------------------------------------------------------
  function render() {
    if (!payload) return;
    const kept = filterEvents(payload, state);
    const d = density(kept, payload.duration);
    const suggested = suggestThreshold(payload, state);

    stats.replaceChildren(
      stat("track", `${fmtTime(payload.duration)}`, `${d.pieces} pieces of ~${PIECE_SECONDS}s`),
      stat("events kept", String(d.total), `${d.hits} hits · ${d.structural} structural`),
      stat("hits / minute", d.hitsPerMin.toFixed(0), Number.isFinite(d.everySeconds) ? `one every ${d.everySeconds.toFixed(1)}s` : "none"),
      stat("hits / piece", d.hitsPerPiece.toFixed(1), `${SLOTS_PER_PIECE} slots per piece`),
      stat("pieces overflowing", String(d.piecesOver), d.piecesOver ? "extra hits get cut, strongest kept" : "everything fits"),
      stat("node max_events", String(state.maxEvents), d.total > state.maxEvents ? `will thin ${d.total} → ${state.maxEvents}` : "not reached"),
    );

    const jump = suggested > state.minStrength + 1e-9
      ? el("button", { className: "pill", textContent: `set min_strength ${suggested.toFixed(2)}`, onclick: () => setStrength(suggested) })
      : null;
    if (d.hitsPerMin > TOO_MANY_PER_MIN) {
      banner.className = "banner bad";
      banner.replaceChildren(
        el("b", { textContent: `Too many hits: ${d.hitsPerMin.toFixed(0)} per minute. ` }),
        `Every scene will be a wall of "bass hit" cues and the writer cannot make them mean anything - a hit every ${d.everySeconds.toFixed(1)}s is a metronome, not a beat to stage. Raise min_strength: ${suggested.toFixed(2)} keeps the heavy ones on this track (~${OK_PER_MIN}/min).`,
        jump,
      );
    } else if (d.hitsPerMin > OK_PER_MIN) {
      banner.className = "banner warn";
      banner.replaceChildren(
        el("b", { textContent: `Dense: ${d.hitsPerMin.toFixed(0)} hits per minute. ` }),
        `Workable, but pieces will fill their ${SLOTS_PER_PIECE} slots and lighter hits crowd out the drops. ${suggested.toFixed(2)} would land at ~${OK_PER_MIN}/min.`,
        jump,
      );
    } else if (d.hits === 0 && d.structural === 0) {
      banner.className = "banner warn";
      banner.replaceChildren(el("b", { textContent: "Nothing left. " }), "Every kind is off or the strength band holds no hit - the writer will get no sound moments at all.");
    } else {
      banner.className = "banner ok";
      banner.replaceChildren(
        el("b", { textContent: `Good density: ${d.hitsPerMin.toFixed(0)} hits per minute. ` }),
        `About ${d.hitsPerPiece.toFixed(1)} hits per piece plus the drops and stops - each one can land as a real, visible event.`,
      );
    }

    // histogram of hit strengths (all kinds currently toggled on, before the threshold)
    const hitPool = (state.toggles.impacts ? payload.events : payload.events_no_impacts)
      .filter((e) => HIT_KINDS.has(e.type) && state.toggles[KIND_TOGGLE[e.type]]);
    const bins = new Array(10).fill(0);
    for (const e of hitPool) bins[Math.min(9, Math.floor(e.strength * 10))]++;
    const top = Math.max(1, ...bins);
    hist.replaceChildren(...bins.map((n, b) => {
      const inBand = (b + 1) / 10 > state.minStrength + 1e-9 && b / 10 < state.maxStrength - 1e-9;
      const bin = el("div", { className: "bin" + (inBand ? " kept" : ""), title: `strength ${(b / 10).toFixed(1)}-${((b + 1) / 10).toFixed(1)}: ${n} hits - click: min_strength ${(b / 10).toFixed(1)}, shift-click: max_strength ${((b + 1) / 10).toFixed(1)}` });
      bin.append(el("span", { className: "n", textContent: n ? String(n) : "" }), el("div", { className: "bar", style: `height:${Math.max(2, (n / top) * 68)}px` }));
      bin.addEventListener("click", (ev) => (ev.shiftKey ? setMaxStrength((b + 1) / 10) : setStrength(b / 10)));
      return bin;
    }));

    // timeline
    const all = state.toggles.impacts ? payload.events : payload.events_no_impacts;
    const keptSet = new Set(kept);
    timeline.replaceChildren(...all.map((e) => {
      const struck = isRejected(e);
      const tick = el("i", {
        className: (STRUCTURAL.has(e.type) ? "struct" : "") + (keptSet.has(e) ? "" : " cut") + (struck ? " struck" : ""),
        style: `left:${(e.t / payload.duration) * 100}%;background:${KIND_COLOR[e.type] || "#9a9590"}`,
        title: `${fmtTime(e.t)} ${e.type} ${e.label}${struck ? " - STRUCK OUT (click to keep)" : " - click to strike out"}`,
      });
      tick.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const hit = [...state.rejected].find((t) => Math.abs(t - e.t) <= 0.05);
        if (hit !== undefined) state.rejected.delete(hit); else state.rejected.add(Number(e.t.toFixed(2)));
        render();
      });
      return tick;
    }));

    // table head
    table.textContent = kept.slice(0, 60).map((e) => `${fmtTime(e.t).padStart(8)}  ${e.type.padEnd(9)} ${e.strength.toFixed(2)}  ${e.label}`).join("\n")
      + (kept.length > 60 ? `\n… ${kept.length - 60} more` : "");
    note.textContent = `Apply writes the strength band ${state.minStrength.toFixed(2)}-${state.maxStrength.toFixed(2)}, the kind toggles, sensitivity, min_gap`
      + (state.rejected.size ? ` and ${state.rejected.size} struck-out hit(s)` : "") + " into the node.";
  }

  function stat(k, v, small) {
    return el("div", { className: "stat" }, el("div", { className: "k", textContent: k }), el("div", { className: "v" }, v, small ? el("small", { textContent: small }) : null));
  }

  async function compute() {
    state.sensitivity = Number(sens.value) || 1.0;
    state.minGap = Number(gap.value) || 0.18;
    banner.className = "banner ok";
    banner.textContent = "Detecting… (a 3-minute track takes a few seconds)";
    recompute.disabled = true;
    try {
      const res = await api.fetchApi("/apnext/h3/sound_events_preview", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio: source.file, sensitivity: state.sensitivity, min_gap_seconds: state.minGap }),
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
      payload = data;
      render();
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
      const __btn = this.addWidget("button", "🎚 Preview events", null, () => {
        openModal(this).catch((err) => console.error("[APNext sound events]", err));
      });
      __btn.serialize = false;   // a button has no value to save; a saved null would shift later widgets
      return r;
    };
  },
});
