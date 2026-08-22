// APNext layout button - the floating graph button bottom-left.
//
// One button pinned bottom-left on the ComfyUI canvas (offset past the
// sidebar rail):
//
//   click        - UN-CRASH: keep the workflow's existing form and just push
//                  overlapping nodes apart (iterative least-penetration
//                  separation) until nothing sits on top of anything.
//   Shift+click  - AUTO-DAG: full re-arrange into left-to-right topological
//                  layers (longest-path depth, barycenter ordering), with a
//                  shelf below for unconnected nodes that would collide.
//   click again  - restore the layout from before either operation.
//
// The H3 Studio dashboard itself stays reachable at
// /extensions/comfyui_dagthomas/h3_dashboard.html.
//
// Pure DOM + LiteGraph node positions - no frontend-version APIs to break.

import { app } from "../../../scripts/app.js";

const STYLE_ID = "apnext-h3-studio-btn-style";

const CSS = `
.apnext-h3-fab {
  position: fixed; bottom: 14px; z-index: 999;
  width: 46px; height: 46px; border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  background: #10161a; border: 1px solid #2b4a52; cursor: pointer;
  transition: transform .15s, box-shadow .15s, border-color .15s;
  box-shadow: 0 2px 10px rgba(0,0,0,.45);
}
.apnext-h3-fab:hover {
  transform: translateY(-2px) scale(1.05);
  border-color: #56c8ad;
  box-shadow: 0 4px 18px rgba(86,200,173,.35);
}
.apnext-h3-fab svg { width: 26px; height: 26px; }
.apnext-h3-fab .h3sb-tip {
  position: absolute; left: 54px; bottom: 8px; white-space: nowrap;
  background: #10161a; border: 1px solid #2b4a52; border-radius: 6px;
  color: #8fe3cd; font: 600 11px/1 ui-sans-serif, "Segoe UI", sans-serif;
  letter-spacing: .12em; text-transform: uppercase; padding: 7px 10px;
  opacity: 0; pointer-events: none; transition: opacity .15s;
}
.apnext-h3-fab:hover .h3sb-tip { opacity: 1; }
/* offset past ComfyUI's left sidebar rail so it never overlaps the menu */
#apnext-h3-dag-btn { left: 82px; }
`;

const DAG_LOGO = `
<svg viewBox="0 0 32 32" fill="none" aria-hidden="true">
  <circle cx="6" cy="16" r="3.4" stroke="#56c8ad" stroke-width="2"/>
  <circle cx="20" cy="7" r="3.4" stroke="#8fe3cd" stroke-width="2"/>
  <circle cx="20" cy="25" r="3.4" stroke="#8fe3cd" stroke-width="2"/>
  <circle cx="28.2" cy="16" r="2.4" fill="#56c8ad"/>
  <path d="M9 14.2 16.8 9M9 17.8 16.8 23M23.2 8.8 26.6 14M23.2 23.2 26.6 18"
        stroke="#3c7a6a" stroke-width="2" stroke-linecap="round"/>
</svg>`;

// -------------------------------------------------------------------------
// Auto-DAG: layered left-to-right arrangement of the connected graph
// -------------------------------------------------------------------------

const COL_GAP = 110;
const ROW_GAP = 46;
const dagState = { saved: null };

function edgesOf(graph) {
  // the links store moved/renamed across frontend versions - try them all
  for (const store of [graph?.links, graph?._links]) {
    if (!store) continue;
    let list = [];
    if (store instanceof Map) list = [...store.values()];
    else if (typeof store.values === "function") list = [...store.values()];
    else list = Object.values(store);
    list = list.filter((l) => l && l.origin_id != null && l.target_id != null);
    if (list.length) return list;
  }
  return [];
}

function nodeHeight(n) {
  return (n.flags?.collapsed ? 30 : (n.size?.[1] || 60)) + 34; // + title bar
}

// Newer frontends back node.pos with a layout store and may hand out COPIES:
// mutating pos[0] silently does nothing. Assigning through the setter works
// everywhere, with an index-write fallback for old builds.
function setPos(n, x, y) {
  try { n.pos = [x, y]; } catch (e) { /* fall through */ }
  if (n.pos[0] !== x || n.pos[1] !== y) {
    n.pos[0] = x;
    n.pos[1] = y;
  }
}

function repaint(graph) {
  graph.setDirtyCanvas?.(true, true);
  app.canvas?.setDirty?.(true, true);
  try { app.canvas?.draw?.(true, true); } catch (e) {}
}

function restoreLayout(graph, nodes) {
  for (const n of nodes) {
    const pos = dagState.saved.get(n.id);
    if (pos) setPos(n, pos[0], pos[1]);
  }
  dagState.saved = null;
  repaint(graph);
  return "restored";
}

// UN-CRASH: keep the workflow's form, just separate whatever overlaps.
// A placed-set sweep with a ZERO-overlap guarantee: nodes are settled in
// reading order (top-left first) - each one is nudged along the axis of
// least penetration until it collides with nothing already placed, and a
// settled node is never disturbed again. A node stuck ping-ponging in a
// tight corridor escapes straight down below everything placed.
function deOverlap() {
  const graph = app.graph;
  const nodes = (graph?._nodes || []).filter(Boolean);
  if (!nodes.length) return false;
  if (dagState.saved) return restoreLayout(graph, nodes);

  const GAP = 18;
  const rect = (n) => ({
    x: n.pos[0] - GAP / 2,
    y: n.pos[1] - 34 - GAP / 2,
    w: (n.size?.[0] || 200) + GAP,
    h: nodeHeight(n) + GAP,
  });
  const hits = (a, b) =>
    Math.min(a.x + a.w, b.x + b.w) > Math.max(a.x, b.x) &&
    Math.min(a.y + a.h, b.y + b.h) > Math.max(a.y, b.y);

  // compute on a SNAPSHOT (node.pos may be store-backed copies), apply at the
  // end through the pos setter
  const entries = nodes.map((n) => ({
    n, x: n.pos[0], y: n.pos[1],
    w: n.size?.[0] || 200, h: nodeHeight(n),
  }));
  const erect = (e) => ({ x: e.x - GAP / 2, y: e.y - 34 - GAP / 2, w: e.w + GAP, h: e.h + GAP });
  const saved = new Map(entries.map((e) => [e.n.id, [e.x, e.y]]));

  const order = [...entries].sort((a, b) => (a.y - b.y) || (a.x - b.x));
  const placed = [];
  let movedCount = 0;
  for (const e of order) {
    const before = [e.x, e.y];
    let guard = 0;
    for (;;) {
      const a = erect(e);
      const other = placed.find((p) => hits(a, erect(p)));
      if (!other) break;
      if (++guard > 40) {
        // corridor escape: drop straight below everything placed so far
        e.y = Math.max(...placed.map((p) => erect(p).y + erect(p).h)) + 34 + GAP;
        break;
      }
      const b = erect(other);
      const penX = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const penY = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (penX <= penY) e.x += ((a.x + a.w / 2) >= (b.x + b.w / 2) ? 1 : -1) * (penX + 1);
      else e.y += ((a.y + a.h / 2) >= (b.y + b.h / 2) ? 1 : -1) * (penY + 1);
    }
    if (e.x !== before[0] || e.y !== before[1]) movedCount++;
    placed.push(e);
  }

  // apply through the setter, then verify the writes actually LANDED
  for (const e of entries) setPos(e.n, e.x, e.y);
  let applied = 0;
  for (const e of entries) {
    if (Math.abs(e.n.pos[0] - e.x) < 0.5 && Math.abs(e.n.pos[1] - e.y) < 0.5) applied++;
  }
  let remaining = 0;
  for (let i = 0; i < entries.length; i++) {
    for (let j = i + 1; j < entries.length; j++) {
      if (hits(erect(entries[i]), erect(entries[j]))) remaining++;
    }
  }
  console.log(
    `[APNext layout] un-crash: moved ${movedCount} node(s), positions applied `
    + `${applied}/${entries.length}, ${remaining} overlap(s) remaining`,
  );
  if (!movedCount) return "clean";
  dagState.saved = saved;
  repaint(graph);
  return applied < entries.length ? "error" : "uncrashed";
}

function autoDag() {
  const graph = app.graph;
  const nodes = (graph?._nodes || []).filter(Boolean);
  if (!nodes.length) return false;

  // second click restores the layout from before the arrange
  if (dagState.saved) return restoreLayout(graph, nodes);

  const edges = edgesOf(graph);
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const outs = new Map(nodes.map((n) => [n.id, []]));
  const indeg = new Map(nodes.map((n) => [n.id, 0]));
  const linked = new Set();
  for (const e of edges) {
    if (!byId.has(e.origin_id) || !byId.has(e.target_id)) continue;
    outs.get(e.origin_id).push(e.target_id);
    indeg.set(e.target_id, (indeg.get(e.target_id) || 0) + 1);
    linked.add(e.origin_id);
    linked.add(e.target_id);
  }
  if (!linked.size) return false;

  // longest-path depth via Kahn - unconnected nodes (notes) keep their spot
  const depth = new Map();
  const queue = [];
  const remaining = new Map();
  for (const id of linked) {
    remaining.set(id, indeg.get(id) || 0);
    if (!indeg.get(id)) { depth.set(id, 0); queue.push(id); }
  }
  while (queue.length) {
    const id = queue.shift();
    for (const to of outs.get(id) || []) {
      if (!linked.has(to)) continue;
      depth.set(to, Math.max(depth.get(to) || 0, (depth.get(id) || 0) + 1));
      remaining.set(to, remaining.get(to) - 1);
      if (remaining.get(to) === 0) queue.push(to);
    }
  }
  for (const id of linked) if (!depth.has(id)) depth.set(id, 0); // cycles: flatten

  const columns = [];
  for (const id of linked) {
    const d = depth.get(id);
    (columns[d] ??= []).push(byId.get(id));
  }

  dagState.saved = new Map(nodes.map((n) => [n.id, [n.pos[0], n.pos[1]]]));

  // anchor the layout at the current top-left of the connected nodes
  const anchorX = Math.min(...[...linked].map((id) => byId.get(id).pos[0]));
  const anchorY = Math.min(...[...linked].map((id) => byId.get(id).pos[1]));

  // crossing reduction: order each column by the average row of its inputs
  const rowOf = new Map();
  const inputsOf = new Map();
  for (const e of edges) {
    if (!linked.has(e.target_id)) continue;
    (inputsOf.get(e.target_id) ?? inputsOf.set(e.target_id, []).get(e.target_id)).push(e.origin_id);
  }
  let x = anchorX;
  let layoutBottom = anchorY;
  for (const col of columns) {
    if (!col) continue;
    col.sort((a, b) => {
      const bary = (n) => {
        const ins = (inputsOf.get(n.id) || []).map((id) => rowOf.get(id)).filter((v) => v != null);
        return ins.length ? ins.reduce((s, v) => s + v, 0) / ins.length : n.pos[1];
      };
      return bary(a) - bary(b);
    });
    let y = anchorY;
    col.forEach((n, i) => {
      rowOf.set(n.id, i);
      setPos(n, x, y);
      y += nodeHeight(n) + ROW_GAP;
    });
    layoutBottom = Math.max(layoutBottom, y);
    x += Math.max(...col.map((n) => n.size?.[0] || 200)) + COL_GAP;
  }

  // collision resolution: any node the layout did not place (notes, parked
  // nodes) that now intersects an arranged node is moved to a shelf below the
  // layout, so nothing can end up on top of anything
  const rect = (n) => ({
    x: n.pos[0] - 8, y: n.pos[1] - 34,
    w: (n.size?.[0] || 200) + 16, h: nodeHeight(n) + 8,
  });
  const hits = (a, b) =>
    a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
  const placed = [...linked].map((id) => rect(byId.get(id)));
  let shelfX = anchorX;
  const shelfY = layoutBottom + 90;
  for (const n of nodes) {
    if (linked.has(n.id)) continue;
    if (!placed.some((r) => hits(r, rect(n)))) { placed.push(rect(n)); continue; }
    setPos(n, shelfX, shelfY);
    placed.push(rect(n));
    shelfX += (n.size?.[0] || 200) + 60;
  }

  repaint(graph);
  return "arranged";
}

// -------------------------------------------------------------------------

function makeFab(id, logo, tip, onClick) {
  const btn = document.createElement("div");
  btn.id = id;
  btn.className = "apnext-h3-fab";
  btn.innerHTML = logo + `<span class="h3sb-tip">${tip}</span>`;
  btn.addEventListener("click", onClick);
  document.body.appendChild(btn);
  return btn;
}

const BUTTON_VERSION = "layout-button v3 (un-crash + shift-DAG)";

app.registerExtension({
  name: "apnext.h3.studioButton",
  setup() {
    console.log(`[APNext layout] ${BUTTON_VERSION} loaded`);
    if (!document.getElementById(STYLE_ID)) {
      const st = document.createElement("style");
      st.id = STYLE_ID;
      st.textContent = CSS;
      document.head.appendChild(st);
    }

    const TIP_DEFAULT = "Un-crash nodes (Shift: full DAG)";
    const dag = makeFab("apnext-h3-dag-btn", DAG_LOGO, TIP_DEFAULT, (ev) => {
      let result = false;
      try {
        result = ev.shiftKey ? autoDag() : deOverlap();
        console.log("[APNext layout]", result || "nothing to do");
      } catch (err) {
        console.error("[APNext layout] failed:", err);
        result = "error";
      }
      const tip = dag.querySelector(".h3sb-tip");
      if (tip) {
        tip.textContent = result === "restored" ? "Layout restored"
          : result === "arranged" ? "DAG'd — click to restore"
          : result === "uncrashed" ? "Un-crashed — click to restore"
          : result === "clean" ? "No overlaps found"
          : result === "error" ? "Failed — see console (F12)"
          : "Nothing to arrange";
        setTimeout(() => { tip.textContent = TIP_DEFAULT; }, 2400);
      }
    });
    // a hand-moved graph invalidates the stored restore layout
    document.addEventListener("pointerup", () => {
      if (dagState.saved && app.canvas?.node_dragged) dagState.saved = null;
    }, true);
  },
});
