// APNext graph layout tools
//
// Canvas right-click → "APNext layout" (and the command palette):
//   Auto-layout    - dagre (vendored, MIT) lays the graph out left-to-right
//                    along the data flow; falls back to a built-in layered
//                    layout if dagre cannot load.
//   Space out      - resolves overlaps with minimal movement: nodes are pushed
//                    apart just enough to stop touching (margin configurable),
//                    the overall arrangement is kept. Pinned nodes stay put.
//   Align          - left / right / top / bottom / row (centre Y) / column
//                    (centre X) for the selected nodes.
//   Distribute     - even gaps horizontally or vertically (3+ nodes).
//
// A floating "DAG sort" button on the canvas triggers Auto-layout directly.
// Scope: 2+ selected nodes = the selection; otherwise the whole graph.
// Groups are re-fitted around the nodes they contained before a whole-graph
// operation. Spacing lives in Settings → APNext → Layout.

import { app } from "../../../scripts/app.js";

const S_NODESEP = "APNext.Layout.NodeSep";
const S_RANKSEP = "APNext.Layout.RankSep";
const S_MARGIN = "APNext.Layout.SpreadMargin";

function setting(id, fallback) {
  const v = Number(app.ui?.settings?.getSettingValue?.(id));
  return Number.isFinite(v) && v >= 0 ? v : fallback;
}

// ---------------------------------------------------------------------------
// Geometry helpers - all in "full rect" space (title bar included)
// ---------------------------------------------------------------------------

function titleH() {
  return (window.LiteGraph && window.LiteGraph.NODE_TITLE_HEIGHT) || 30;
}

function rectOf(node) {
  try {
    const b = node.getBounding?.();
    if (b && b.length >= 4 && Number.isFinite(b[0])) return { x: b[0], y: b[1], w: b[2], h: b[3] };
  } catch (e) { /* fall through */ }
  const th = titleH();
  return { x: node.pos[0], y: node.pos[1] - th, w: node.size[0], h: node.size[1] + th };
}

function moveTo(node, x, y) {
  const r = rectOf(node);
  node.pos[0] += x - r.x;
  node.pos[1] += y - r.y;
}

function isPinned(node) {
  return !!(node.flags?.pinned || node.pinned);
}

function scopeNodes() {
  const sel = Object.values(app.canvas?.selected_nodes || {});
  if (sel.length >= 2) return { nodes: sel, whole: false };
  return { nodes: (app.graph?._nodes || []).slice(), whole: true };
}

function edgesAmong(nodes) {
  const ids = new Set(nodes.map((n) => String(n.id)));
  const out = [];
  const table = app.graph?.links ?? app.graph?._links;
  if (!table) return out;
  const each = (cb) => {
    if (typeof table.forEach === "function" && !(table instanceof Array)) table.forEach(cb);
    else for (const k in table) cb(table[k]);
  };
  each((l) => {
    if (!l) return;
    const a = String(l.origin_id), b = String(l.target_id);
    if (ids.has(a) && ids.has(b) && a !== b) out.push([a, b]);
  });
  return out;
}

// group membership is derived from node positions, so capture it BEFORE
// moving anything, and refit the group frames afterwards
function captureGroups(whole) {
  if (!whole) return null;
  const map = new Map();
  for (const g of app.graph?._groups || []) {
    try { g.recomputeInsideNodes?.(); } catch (e) { /* older frontend */ }
    const members = (g._nodes || []).slice();
    if (members.length) map.set(g, members);
  }
  return map;
}

function refitGroups(map) {
  if (!map) return;
  for (const [g, members] of map) {
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (const n of members) {
      const r = rectOf(n);
      minx = Math.min(minx, r.x); miny = Math.min(miny, r.y);
      maxx = Math.max(maxx, r.x + r.w); maxy = Math.max(maxy, r.y + r.h);
    }
    if (!Number.isFinite(minx)) continue;
    g.pos = [minx - 16, miny - 44];
    g.size = [maxx - minx + 32, maxy - miny + 60];
  }
}

function commit() {
  app.graph?.afterChange?.();
  app.canvas?.setDirty?.(true, true);
}

// ---------------------------------------------------------------------------
// Auto-layout (dagre, with a layered fallback)
// ---------------------------------------------------------------------------

let dagreLoading = null;
function ensureDagre() {
  if (window.dagre) return Promise.resolve(window.dagre);
  if (!dagreLoading) {
    dagreLoading = new Promise((resolve) => {
      const s = document.createElement("script");
      s.src = new URL("./vendor/dagre.min.js", import.meta.url).href;
      s.onload = () => resolve(window.dagre || null);
      s.onerror = () => { console.warn("[APNext layout] dagre failed to load; using fallback"); resolve(null); };
      document.head.appendChild(s);
    });
  }
  return dagreLoading;
}

// fallback: longest-path layering, rows ordered by current y
function layeredLayout(nodes, edges, nodesep, ranksep) {
  const byId = new Map(nodes.map((n) => [String(n.id), n]));
  const rank = new Map(nodes.map((n) => [String(n.id), 0]));
  for (let i = 0; i < nodes.length; i++) {
    let changed = false;
    for (const [a, b] of edges) {
      if (rank.get(b) < rank.get(a) + 1) { rank.set(b, rank.get(a) + 1); changed = true; }
    }
    if (!changed) break;
  }
  const cols = new Map();
  for (const [id, r] of rank) {
    if (!cols.has(r)) cols.set(r, []);
    cols.get(r).push(byId.get(id));
  }
  const out = new Map();
  let x = 0;
  for (const r of [...cols.keys()].sort((a, b) => a - b)) {
    const col = cols.get(r).sort((a, b) => rectOf(a).y - rectOf(b).y);
    let y = 0, maxw = 0;
    for (const n of col) {
      const rc = rectOf(n);
      out.set(n, { x, y });
      y += rc.h + nodesep;
      maxw = Math.max(maxw, rc.w);
    }
    x += maxw + ranksep;
  }
  return out;
}

async function autoLayout() {
  const { nodes, whole } = scopeNodes();
  if (nodes.length < 2) return;
  const groups = captureGroups(whole);
  const edges = edgesAmong(nodes);
  const nodesep = setting(S_NODESEP, 40);
  const ranksep = setting(S_RANKSEP, 90);
  // anchor: keep the arrangement where the graph already is
  let ax = Infinity, ay = Infinity;
  for (const n of nodes) { const r = rectOf(n); ax = Math.min(ax, r.x); ay = Math.min(ay, r.y); }

  const dagre = await ensureDagre();
  app.graph?.beforeChange?.();
  let placed;
  if (dagre) {
    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: "LR", nodesep, ranksep, marginx: 0, marginy: 0 });
    g.setDefaultEdgeLabel(() => ({}));
    for (const n of nodes) {
      const r = rectOf(n);
      g.setNode(String(n.id), { width: r.w, height: r.h });
    }
    for (const [a, b] of edges) g.setEdge(a, b);
    dagre.layout(g);
    placed = new Map();
    for (const n of nodes) {
      const p = g.node(String(n.id));
      if (p) placed.set(n, { x: p.x - p.width / 2, y: p.y - p.height / 2 });
    }
  } else {
    placed = layeredLayout(nodes, edges, nodesep, ranksep);
  }
  let minx = Infinity, miny = Infinity;
  for (const p of placed.values()) { minx = Math.min(minx, p.x); miny = Math.min(miny, p.y); }
  const dx = (Number.isFinite(ax) ? ax : 0) - (Number.isFinite(minx) ? minx : 0);
  const dy = (Number.isFinite(ay) ? ay : 0) - (Number.isFinite(miny) ? miny : 0);
  for (const [n, p] of placed) moveTo(n, p.x + dx, p.y + dy);
  refitGroups(groups);
  commit();
}

// ---------------------------------------------------------------------------
// Space out - minimal-movement overlap resolution
// ---------------------------------------------------------------------------

function spaceOut() {
  const { nodes, whole } = scopeNodes();
  if (nodes.length < 2) return;
  const groups = captureGroups(whole);
  const margin = setting(S_MARGIN, 24);
  app.graph?.beforeChange?.();
  const items = nodes.map((n) => {
    const r = rectOf(n);
    return { n, x: r.x, y: r.y, w: r.w, h: r.h, pinned: isPinned(n) };
  });
  for (let iter = 0; iter < 80; iter++) {
    let any = false;
    for (let i = 0; i < items.length; i++) {
      for (let j = i + 1; j < items.length; j++) {
        const a = items[i], b = items[j];
        if (a.pinned && b.pinned) continue;
        const acx = a.x + a.w / 2, acy = a.y + a.h / 2;
        const bcx = b.x + b.w / 2, bcy = b.y + b.h / 2;
        const px = (a.w + b.w) / 2 + margin - Math.abs(acx - bcx);
        const py = (a.h + b.h) / 2 + margin - Math.abs(acy - bcy);
        if (px <= 0 || py <= 0) continue;
        any = true;
        // push along the axis that needs the smaller correction
        if (px <= py) {
          const dir = acx <= bcx ? 1 : -1;
          const jitter = acx === bcx ? 0.5 : 0;
          if (a.pinned) { b.x += dir * px; }
          else if (b.pinned) { a.x -= dir * px; }
          else { a.x -= dir * (px / 2 + jitter); b.x += dir * (px / 2 + jitter); }
        } else {
          const dir = acy <= bcy ? 1 : -1;
          const jitter = acy === bcy ? 0.5 : 0;
          if (a.pinned) { b.y += dir * py; }
          else if (b.pinned) { a.y -= dir * py; }
          else { a.y -= dir * (py / 2 + jitter); b.y += dir * (py / 2 + jitter); }
        }
      }
    }
    if (!any) break;
  }
  for (const it of items) if (!it.pinned) moveTo(it.n, it.x, it.y);
  refitGroups(groups);
  commit();
}

// ---------------------------------------------------------------------------
// Align / distribute (selection of 2+, distribute 3+)
// ---------------------------------------------------------------------------

function selected(min) {
  const sel = Object.values(app.canvas?.selected_nodes || {});
  if (sel.length < min) {
    app.extensionManager?.toast?.add?.({
      severity: "info", summary: "APNext layout",
      detail: `Select at least ${min} nodes first.`, life: 2500,
    });
    return null;
  }
  return sel;
}

function align(edge) {
  const sel = selected(2);
  if (!sel) return;
  app.graph?.beforeChange?.();
  const rects = sel.map((n) => ({ n, ...rectOf(n) }));
  const minx = Math.min(...rects.map((r) => r.x));
  const maxx = Math.max(...rects.map((r) => r.x + r.w));
  const miny = Math.min(...rects.map((r) => r.y));
  const maxy = Math.max(...rects.map((r) => r.y + r.h));
  for (const r of rects) {
    if (edge === "left") moveTo(r.n, minx, r.y);
    else if (edge === "right") moveTo(r.n, maxx - r.w, r.y);
    else if (edge === "top") moveTo(r.n, r.x, miny);
    else if (edge === "bottom") moveTo(r.n, r.x, maxy - r.h);
    else if (edge === "row") moveTo(r.n, r.x, (miny + maxy) / 2 - r.h / 2);
    else if (edge === "column") moveTo(r.n, (minx + maxx) / 2 - r.w / 2, r.y);
  }
  commit();
}

function distribute(axis) {
  const sel = selected(3);
  if (!sel) return;
  app.graph?.beforeChange?.();
  const rects = sel.map((n) => ({ n, ...rectOf(n) }));
  const horizontal = axis === "h";
  rects.sort((a, b) => (horizontal ? a.x - b.x : a.y - b.y));
  const first = rects[0], last = rects[rects.length - 1];
  const span = horizontal
    ? (last.x + last.w) - first.x
    : (last.y + last.h) - first.y;
  const total = rects.reduce((s, r) => s + (horizontal ? r.w : r.h), 0);
  const gap = (span - total) / (rects.length - 1);
  let cursor = horizontal ? first.x : first.y;
  for (const r of rects) {
    if (horizontal) { moveTo(r.n, cursor, r.y); cursor += r.w + gap; }
    else { moveTo(r.n, r.x, cursor); cursor += r.h + gap; }
  }
  commit();
}

// ---------------------------------------------------------------------------
// Floating "DAG sort" button on the canvas: one click = auto-layout
// ---------------------------------------------------------------------------

function injectDagSortButton() {
  if (document.getElementById("apnext-dag-sort-btn")) return;
  const btn = document.createElement("button");
  btn.id = "apnext-dag-sort-btn";
  btn.type = "button";
  btn.title = "DAG sort: auto-layout the graph along the data flow (selection only when 2+ nodes are selected)";
  btn.innerHTML = '<i class="pi pi-sitemap" style="font-size:0.85em"></i><span>DAG sort</span>';
  Object.assign(btn.style, {
    position: "fixed",
    right: "16px",
    bottom: "72px",
    zIndex: "999",
    display: "flex",
    alignItems: "center",
    gap: "6px",
    padding: "6px 12px",
    font: "500 12px/1 inherit",
    color: "var(--fg-color, #ddd)",
    background: "var(--comfy-menu-bg, #202020)",
    border: "1px solid var(--border-color, #4e4e4e)",
    borderRadius: "8px",
    cursor: "pointer",
  });
  btn.addEventListener("mouseenter", () => { btn.style.borderColor = "var(--fg-color, #ddd)"; });
  btn.addEventListener("mouseleave", () => { btn.style.borderColor = "var(--border-color, #4e4e4e)"; });
  btn.addEventListener("click", () => autoLayout());
  document.body.appendChild(btn);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

app.registerExtension({
  name: "apnext.layout",

  settings: [
    {
      id: S_NODESEP,
      category: ["APNext", "Layout", "Node spacing"],
      name: "Auto-layout: spacing between nodes in a column",
      type: "slider", attrs: { min: 10, max: 150, step: 5 }, defaultValue: 40,
    },
    {
      id: S_RANKSEP,
      category: ["APNext", "Layout", "Column spacing"],
      name: "Auto-layout: spacing between columns (along the flow)",
      type: "slider", attrs: { min: 20, max: 300, step: 5 }, defaultValue: 90,
    },
    {
      id: S_MARGIN,
      category: ["APNext", "Layout", "Space-out margin"],
      name: "Space out: minimum gap between nodes",
      type: "slider", attrs: { min: 4, max: 100, step: 2 }, defaultValue: 24,
    },
  ],

  commands: [
    { id: "APNext.Layout.Auto", label: "APNext: auto-layout graph (dagre)", icon: "pi pi-sitemap", function: () => autoLayout() },
    { id: "APNext.Layout.Spread", label: "APNext: space out nodes (de-overlap)", icon: "pi pi-arrows-alt", function: () => spaceOut() },
    { id: "APNext.Layout.AlignLeft", label: "APNext: align left", function: () => align("left") },
    { id: "APNext.Layout.AlignRight", label: "APNext: align right", function: () => align("right") },
    { id: "APNext.Layout.AlignTop", label: "APNext: align top", function: () => align("top") },
    { id: "APNext.Layout.AlignBottom", label: "APNext: align bottom", function: () => align("bottom") },
    { id: "APNext.Layout.AlignRow", label: "APNext: align as row (centre Y)", function: () => align("row") },
    { id: "APNext.Layout.AlignColumn", label: "APNext: align as column (centre X)", function: () => align("column") },
    { id: "APNext.Layout.DistributeH", label: "APNext: distribute horizontally", function: () => distribute("h") },
    { id: "APNext.Layout.DistributeV", label: "APNext: distribute vertically", function: () => distribute("v") },
  ],
  menuCommands: [
    { path: ["APNext", "Layout"], commands: ["APNext.Layout.Auto", "APNext.Layout.Spread"] },
  ],

  getCanvasMenuItems() {
    return [
      {
        content: "APNext: layout",
        has_submenu: true,
        submenu: {
          options: [
            { content: "Auto-layout (dagre, flow left→right)", callback: () => autoLayout() },
            { content: "Space out (de-overlap, keep arrangement)", callback: () => spaceOut() },
            null,
            { content: "Align left", callback: () => align("left") },
            { content: "Align right", callback: () => align("right") },
            { content: "Align top", callback: () => align("top") },
            { content: "Align bottom", callback: () => align("bottom") },
            { content: "Align as row (centre Y)", callback: () => align("row") },
            { content: "Align as column (centre X)", callback: () => align("column") },
            null,
            { content: "Distribute horizontally", callback: () => distribute("h") },
            { content: "Distribute vertically", callback: () => distribute("v") },
          ],
        },
      },
    ];
  },

  setup() {
    injectDagSortButton();
    // warm the dagre cache in the background so the first layout is instant
    setTimeout(() => { ensureDagre(); }, 2000);
  },
});
