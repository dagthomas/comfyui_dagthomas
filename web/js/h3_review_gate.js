// APNext H3 Dailies Gate - the run pauses here like a screening room.
//
// The writer's scenes land on the "dailies desk" while the run stays open in
// the background. Ways out, film-set style:
//
//   ▶ Print it   - render exactly what is on the desk (hand edits included)
//   ✍ Punch-up   - send the director's notes back into the writer's own
//                  model session; the rewritten takes come back to the desk
//   🎲 New take  - the same rewrite with no notes: ask the model for a
//                  noticeably different version of the selected takes
//   ↩ Undo       - roll back the last rewrite (server-side history, so it
//                  survives a browser reload)
//   ✋ Cut        - end the run, nothing renders
//
// The editor shows all takes or one at a time (selector / ◀ ▶); edits made in
// a single-take view are merged back into the full text, and with the takes
// field empty a rewrite targets the take being viewed. Buttons POST to
// /apnext/h3/review_gate, which resolves the future the backend node is
// awaiting; GET /apnext/h3/review_gate/pending re-attaches after a reload.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ensureStyle, highlight } from "./h3_prompt_preview.js";

const NODE_CLASS = "H3ScenesReviewGate";
const STYLE_ID = "apnext-h3-dailies-style";
const ENV_RE = /===\s*SCENE\s+(\d+)\s*===\s*([\s\S]*?)\s*===\s*END\s+SCENE\s*\1\s*===/gi;

const CSS = `
.apnext-h3-dailies {
  display: flex; flex-direction: column; gap: 7px;
  background: #10161a; border: 1px solid #24343c; border-radius: 8px;
  border-left: 4px solid #2b4a52; padding: 8px; box-sizing: border-box;
}
.apnext-h3-dailies .h3d-head {
  display: flex; align-items: center; gap: 8px;
  font: 600 11px/1.2 ui-sans-serif, "Segoe UI", system-ui, sans-serif;
  letter-spacing: 0.14em; text-transform: uppercase; color: #7aa8a0;
}
.apnext-h3-dailies .h3d-dot {
  width: 8px; height: 8px; border-radius: 50%; background: #3a4a4a; flex: none;
}
.apnext-h3-dailies.h3d-live { border-left-color: #56c8ad; }
.apnext-h3-dailies.h3d-live .h3d-dot {
  background: #56c8ad; animation: h3d-blink 1.6s ease-in-out infinite;
}
.apnext-h3-dailies.h3d-live .h3d-head { color: #8fe3cd; }
@keyframes h3d-blink { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }
.apnext-h3-dailies .h3d-count { margin-left: auto; color: #567; letter-spacing: 0; text-transform: none; font-weight: 400; }
.apnext-h3-dailies .h3d-scope-row { display: flex; align-items: center; gap: 5px; }
.apnext-h3-dailies select.h3d-scope {
  background: #0b1013; color: #8fe3cd; border: 1px solid #1d2b32; border-radius: 5px;
  font: 11px/1.3 ui-monospace, Consolas, monospace; padding: 2px 5px; max-width: 140px;
}
.apnext-h3-dailies .h3d-scope-row button {
  background: #0b1013; color: #7aa8a0; border: 1px solid #1d2b32; border-radius: 5px;
  font-size: 11px; padding: 2px 7px; cursor: pointer;
}
.apnext-h3-dailies .h3d-scope-row button:hover { color: #8fe3cd; border-color: #2b4a52; }
.apnext-h3-dailies .h3d-editwrap {
  position: relative; flex: 1 1 0; min-height: 150px;
  border: 1px solid #1d2b32; border-radius: 6px; overflow: hidden; background: #0b1013;
}
.apnext-h3-dailies .h3d-hl,
.apnext-h3-dailies .h3d-input {
  position: absolute; inset: 0; margin: 0; box-sizing: border-box;
  padding: 8px 10px; overflow: auto; overscroll-behavior: contain;
  font-family: ui-monospace, "Cascadia Code", "JetBrains Mono", Menlo, Consolas, monospace;
  font-size: 12px; line-height: 1.55;
  white-space: pre-wrap; word-break: break-word; tab-size: 4;
}
.apnext-h3-dailies .h3d-hl { pointer-events: none; color: #dfe6e3; }
.apnext-h3-dailies .h3d-input {
  background: transparent; color: transparent; caret-color: #8fe3cd;
  border: 0; outline: none; resize: none; width: 100%; height: 100%;
}
.apnext-h3-dailies .h3d-hl .h3-tag,
.apnext-h3-dailies .h3d-hl .h3-d,
.apnext-h3-dailies .h3d-hl .h3-header {
  display: inline; padding: 0; border: 0; margin: 0; border-radius: 2px;
}
.apnext-h3-dailies .h3d-hl .h3-header { background: rgba(86,200,173,0.10); }
.apnext-h3-dailies .h3d-notes-row { display: flex; gap: 6px; align-items: stretch; }
.apnext-h3-dailies textarea.h3d-notes {
  flex: 1; min-height: 42px; max-height: 96px; resize: vertical;
  background: #0b1013; color: #dfe6e3; border: 1px solid #1d2b32; border-radius: 6px;
  font: 12px/1.4 ui-sans-serif, "Segoe UI", system-ui, sans-serif; padding: 6px 8px;
}
.apnext-h3-dailies input.h3d-takes {
  width: 84px; background: #0b1013; color: #8fe3cd; text-align: center;
  border: 1px solid #1d2b32; border-radius: 6px;
  font: 12px/1.4 ui-monospace, Consolas, monospace; padding: 4px 6px;
}
.apnext-h3-dailies .h3d-actions { display: flex; gap: 6px; }
.apnext-h3-dailies .h3d-actions button {
  flex: 1; background: #16222a; color: #cfe0da; border: 1px solid #24343c;
  border-radius: 6px; font: 600 12px/1.2 ui-sans-serif, "Segoe UI", system-ui, sans-serif;
  padding: 7px 6px; cursor: pointer; white-space: nowrap;
}
.apnext-h3-dailies .h3d-actions button:hover:not(:disabled) { filter: brightness(1.25); }
.apnext-h3-dailies .h3d-actions button:disabled { opacity: 0.35; cursor: default; }
.apnext-h3-dailies button.h3d-print { background: #14322a; border-color: #2c6b57; color: #9fe8cf; }
.apnext-h3-dailies button.h3d-punch { background: #1d2a3a; border-color: #3c5a80; color: #a9c8ef; }
.apnext-h3-dailies button.h3d-take  { background: #2a2436; border-color: #55486e; color: #cbb8ef; }
.apnext-h3-dailies button.h3d-undo  { flex: 0 0 auto; background: #1a2126; border-color: #2b3a42; color: #9ab4ad; }
.apnext-h3-dailies button.h3d-cut   { flex: 0 0 auto; background: #33191d; border-color: #6b3038; color: #e8a9b1; }
.apnext-h3-dailies .h3d-slate {
  font: 11px/1.45 ui-monospace, Consolas, monospace; color: #6d8580; min-height: 15px;
  white-space: pre-wrap; border-top: 1px dashed #1d2b32; padding-top: 5px;
}
`;

function ensureDailiesStyle() {
  ensureStyle();
  if (document.getElementById(STYLE_ID)) return;
  const st = document.createElement("style");
  st.id = STYLE_ID;
  st.textContent = CSS;
  document.head.appendChild(st);
}

async function postDecision(body) {
  try {
    const r = await api.fetchApi("/apnext/h3/review_gate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return r.ok;
  } catch (e) {
    console.error("H3 Dailies Gate decision failed", e);
    return false;
  }
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

// A soft two-note chime, synthesised on the spot (no asset, no autoplay issues
// beyond the usual: the first chime needs one prior interaction with the page).
function playChime() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const note = (freq, t0) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, ctx.currentTime + t0);
      gain.gain.exponentialRampToValueAtTime(0.12, ctx.currentTime + t0 + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + t0 + 0.5);
      osc.connect(gain).connect(ctx.destination);
      osc.start(ctx.currentTime + t0);
      osc.stop(ctx.currentTime + t0 + 0.55);
    };
    note(784, 0);      // G5
    note(1175, 0.12);  // D6
    setTimeout(() => ctx.close(), 1200);
  } catch {}
}

function buildDesk(node) {
  ensureDailiesStyle();

  const root = document.createElement("div");
  root.className = "apnext-h3-dailies";

  const head = document.createElement("div");
  head.className = "h3d-head";
  const dot = document.createElement("span");
  dot.className = "h3d-dot";
  const label = document.createElement("span");
  label.textContent = "Dailies desk";
  const count = document.createElement("span");
  count.className = "h3d-count";
  count.textContent = "nothing screening";
  head.appendChild(dot);
  head.appendChild(label);
  head.appendChild(count);

  const scopeRow = document.createElement("div");
  scopeRow.className = "h3d-scope-row";
  const prev = document.createElement("button");
  prev.textContent = "◀";
  prev.title = "Previous take";
  const scope = document.createElement("select");
  scope.className = "h3d-scope";
  scope.title = "Edit every take at once, or one take at a time";
  const next = document.createElement("button");
  next.textContent = "▶";
  next.title = "Next take";
  scopeRow.appendChild(prev);
  scopeRow.appendChild(scope);
  scopeRow.appendChild(next);

  const wrap = document.createElement("div");
  wrap.className = "h3d-editwrap";
  const hl = document.createElement("pre");
  hl.className = "h3d-hl";
  hl.setAttribute("aria-hidden", "true");
  const ta = document.createElement("textarea");
  ta.className = "h3d-input";
  ta.spellcheck = false;
  ta.placeholder = "Queue the graph - the takes land here while the run holds for your call.";
  wrap.appendChild(hl);
  wrap.appendChild(ta);

  const notesRow = document.createElement("div");
  notesRow.className = "h3d-notes-row";
  const notes = document.createElement("textarea");
  notes.className = "h3d-notes";
  notes.spellcheck = false;
  notes.placeholder = "Director's notes for a punch-up — what should change before this prints?";
  const takes = document.createElement("input");
  takes.className = "h3d-takes";
  takes.placeholder = "takes: all";
  takes.title =
    "Which takes a rewrite targets, e.g. `2, 4-5`. Empty = the take being viewed, or all of them in the all-takes view.";
  notesRow.appendChild(notes);
  notesRow.appendChild(takes);

  const actions = document.createElement("div");
  actions.className = "h3d-actions";
  const btnPrint = document.createElement("button");
  btnPrint.className = "h3d-print";
  btnPrint.textContent = "▶ Print it";
  btnPrint.title = "Continue the run and render exactly what is on the desk, hand edits included";
  const btnPunch = document.createElement("button");
  btnPunch.className = "h3d-punch";
  btnPunch.textContent = "✍ Punch-up";
  const btnTake = document.createElement("button");
  btnTake.className = "h3d-take";
  btnTake.textContent = "🎲 New take";
  const btnUndo = document.createElement("button");
  btnUndo.className = "h3d-undo";
  btnUndo.textContent = "↩";
  btnUndo.title = "Undo the last rewrite (server-side history, survives a reload)";
  const btnCut = document.createElement("button");
  btnCut.className = "h3d-cut";
  btnCut.textContent = "✋";
  btnCut.title = "Cut - end the run, render nothing";
  actions.appendChild(btnPrint);
  actions.appendChild(btnPunch);
  actions.appendChild(btnTake);
  actions.appendChild(btnUndo);
  actions.appendChild(btnCut);

  const slate = document.createElement("div");
  slate.className = "h3d-slate";

  root.appendChild(head);
  root.appendChild(scopeRow);
  root.appendChild(wrap);
  root.appendChild(notesRow);
  root.appendChild(actions);
  root.appendChild(slate);

  for (const ev of ["pointerdown", "mousedown", "wheel", "dblclick", "contextmenu"]) {
    root.addEventListener(ev, (e) => e.stopPropagation());
  }
  root.addEventListener("keydown", (e) => e.stopPropagation());

  const state = {
    token: null, canPunch: false, canUndo: false, busy: false, raf: 0,
    fullText: "", view: "all", deadline: null, timer: 0, chimed: null,
  };

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
    scope.innerHTML = "";
    const all = document.createElement("option");
    all.value = "all";
    all.textContent = `All takes (${envs.length || "-"})`;
    scope.appendChild(all);
    for (const { no } of envs) {
      const o = document.createElement("option");
      o.value = String(no);
      o.textContent = `Take ${String(no).padStart(2, "0")}`;
      scope.appendChild(o);
    }
    scope.value = state.view === "all" || !envs.some((e) => e.no === state.view) ? "all" : String(state.view);
    prev.style.display = next.style.display = envs.length > 1 ? "" : "none";
    return envs;
  };

  const showView = () => {
    const envs = rebuildScope();
    if (state.view !== "all") {
      const env = envs.find((e) => e.no === state.view);
      if (env) {
        ta.value = env.body;
        schedulePaint();
        return;
      }
      state.view = "all";
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
  scope.addEventListener("change", () => switchView(scope.value === "all" ? "all" : Number(scope.value)));
  const step = (d) => {
    const envs = parseEnvelopes(state.fullText);
    if (!envs.length) return;
    if (state.view === "all") return switchView(envs[d > 0 ? 0 : envs.length - 1].no);
    const i = envs.findIndex((e) => e.no === state.view);
    const j = i < 0 ? 0 : (i + d + envs.length) % envs.length;
    switchView(envs[j].no);
  };
  prev.addEventListener("click", (e) => { e.preventDefault(); step(-1); });
  next.addEventListener("click", (e) => { e.preventDefault(); step(1); });

  const refresh = () => {
    const live = !!state.token && !state.busy;
    btnPrint.disabled = !live;
    btnCut.disabled = !live;
    btnUndo.disabled = !live || !state.canUndo;
    btnPunch.disabled = btnTake.disabled = !live || !state.canPunch;
    const punchTitle = state.canPunch
      ? "Send the notes into the writer's own model session and rewrite the selected takes around your hand edits"
      : "Wire the writer's session_id into this node to enable rewrites";
    btnPunch.title = punchTitle;
    btnTake.title = state.canPunch
      ? "Ask the model for a noticeably different version of the selected takes (notes are ignored)"
      : punchTitle;
  };

  const tickCountdown = () => {
    if (!state.token || state.deadline == null) return;
    const left = Math.max(0, state.deadline - Date.now() / 1000);
    const m = Math.floor(left / 60);
    const s = Math.floor(left % 60);
    count.textContent = `${parseEnvelopes(state.fullText).length} take(s) · auto-prints in ${m}:${String(s).padStart(2, "0")}`;
  };

  const stopTimer = () => {
    if (state.timer) clearInterval(state.timer);
    state.timer = 0;
  };

  const show = (payload) => {
    state.token = payload.token;
    state.canPunch = !!payload.can_reroll;
    state.canUndo = !!payload.can_undo;
    state.busy = false;
    state.fullText = payload.text || "";
    // server clocks differ from browser clocks; rebase the deadline
    state.deadline =
      payload.deadline != null && payload.server_now != null
        ? Date.now() / 1000 + (payload.deadline - payload.server_now)
        : null;
    showView();
    count.textContent = `${payload.count} take(s) on the desk`;
    slate.textContent = payload.status || "the run is holding for your call";
    root.classList.add("h3d-live");
    stopTimer();
    if (state.deadline != null) {
      tickCountdown();
      state.timer = setInterval(tickCountdown, 1000);
    }
    if (payload.chime && state.chimed !== payload.token + ":" + payload.revision) {
      state.chimed = payload.token + ":" + payload.revision;
      playChime();
    }
    refresh();
    app.canvas?.setDirty?.(true, true);
  };

  const resolve = (payload) => {
    state.token = null;
    state.busy = false;
    stopTimer();
    root.classList.remove("h3d-live");
    count.textContent = "nothing screening";
    if (payload?.status) slate.textContent = payload.status;
    refresh();
    app.canvas?.setDirty?.(true, true);
  };

  const send = async (action, extra = {}) => {
    if (!state.token || state.busy) return;
    state.busy = true;
    refresh();
    if (action === "reroll") {
      count.textContent = extra.variant ? "rolling a new take…" : "punch-up in progress…";
      slate.textContent = "sent to the writer's session - the new takes land here";
    }
    // in a single-take view an empty takes field means "this take"
    const picked =
      takes.value.trim() || (state.view !== "all" ? String(state.view) : "");
    const ok = await postDecision({
      token: state.token,
      action,
      text: state.fullText,
      feedback: notes.value,
      scenes: picked,
      ...extra,
    });
    if (!ok) {
      state.busy = false;
      slate.textContent = "the desk did not answer (gate already resolved?)";
      refresh();
    }
  };

  btnPrint.addEventListener("click", () => send("approve"));
  btnPunch.addEventListener("click", () => send("reroll"));
  btnTake.addEventListener("click", () => send("reroll", { variant: true }));
  btnUndo.addEventListener("click", () => send("undo"));
  btnCut.addEventListener("click", () => send("stop"));

  refresh();
  return { root, show, resolve };
}

app.registerExtension({
  name: "apnext.h3.dailiesGate",

  setup() {
    const deskOf = (id) => {
      const node = app.graph?.getNodeById?.(Number(id)) ?? app.graph?.getNodeById?.(id);
      if (!node || (node.comfyClass !== NODE_CLASS && node.type !== NODE_CLASS)) return null;
      return node._h3Dailies || null;
    };
    api.addEventListener("apnext.h3.review_gate", (ev) => {
      const d = ev.detail || {};
      deskOf(d.node)?.show(d);
    });
    api.addEventListener("apnext.h3.review_gate_resolved", (ev) => {
      const d = ev.detail || {};
      deskOf(d.node)?.resolve(d);
    });
    // a reload mid-screening: re-attach to whatever is still pending server-side
    setTimeout(async () => {
      try {
        const r = await api.fetchApi("/apnext/h3/review_gate/pending");
        if (!r.ok) return;
        const { reviews } = await r.json();
        for (const payload of reviews || []) deskOf(payload.node)?.show(payload);
      } catch {}
    }, 1200);
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_CLASS) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const desk = buildDesk(this);
      this._h3Dailies = desk;
      this.addDOMWidget("gate", "H3_DAILIES_GATE", desk.root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 380,
      });
      if (!this.size || this.size[0] < 460) this.setSize?.([470, 560]);
      return r;
    };
  },
});
