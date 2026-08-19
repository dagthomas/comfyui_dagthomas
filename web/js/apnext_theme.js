// APNext "Graphgen" theme for ComfyUI
//
// Optional look-and-feel borrowed from graphgen (X:/KODE/graphgen): the "Dark
// Botanical" palette (warm near-black, tan accent, dusty-pink highlight, muted
// botanical port colours), IBM Plex Sans on the canvas and the UI chrome,
// rounder nodes, and "gravity" wires - every link is a little verlet rope that
// hangs between its two ports and swings when you drag a node (direct port of
// graphgen's src/lib/flow/rope.svelte.ts).
//
// Everything is opt-in and reversible:
//   Settings → APNext → Graphgen theme (On/Off) and Wire style (ComfyUI default /
//              Bezier / Smooth step / Step / Straight / Cable / Gravity)
//   top menu  → APNext → toggle theme / next wire style / Wire style submenu
//   canvas right-click → APNext theme + wire style entries
// Turning it off restores the palette that was active before, the default
// font and radius, and stock bezier links. The palette is installed as a normal
// custom ComfyUI colour palette ("APNext Graphgen"), so it also shows up in the
// regular palette picker.

import { app } from "../../../scripts/app.js";

const PALETTE_ID = "apnext_graphgen";
const S_MODE = "APNext.Theme.Mode";
const S_PREV = "APNext.Theme.PreviousPalette";
const S_SLACK = "APNext.Theme.WireSlack";
const S_GRAVITY = "APNext.Theme.WireGravity";
const S_SEGMENTS = "APNext.Theme.WireSegments";

const MODE_OFF = "off";
const MODE_THEME = "theme";
const MODE_BOTH = "theme+gravity";
const MODE_GRAVITY = "gravity";

const FONT_SANS = "'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif";
const FONT_MONO = "'JetBrains Mono', 'Cascadia Code', Consolas, monospace";
const FONT_DISPLAY = "'Cormorant', Georgia, 'Times New Roman', serif";
const FONTS_HREF =
  "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Cormorant:wght@500;600&family=JetBrains+Mono:wght@400;500&display=swap";

// ---------------------------------------------------------------------------
// Palette - graphgen's tokens (src/routes/layout.css) mapped onto ComfyUI's
// colour-palette schema. Port colours follow the botanical category hues.
// ---------------------------------------------------------------------------

const C = {
  bg: "#0f0f0f",
  panel: "#161512",
  panel2: "#1e1c17",
  border: "#2c2820",
  borderLight: "#463f33",
  text: "#e8e4df",
  textDim: "#9a9590",
  accent: "#d4a574", // warm tan
  accent2: "#e8b4b8", // dusty pink
  gold: "#c9b896",
  sage: "#a7bd84",
  slate: "#88a9c0",
  rose: "#d49aa6",
  mauve: "#b58fc2",
  neutral: "#ddd6c8",
  goldPort: "#ccb777",
  terracotta: "#d08a5e",
  honey: "#cda06a",
  teal: "#84b3a6",
};

function dotGrid() {
  // A quiet dot grid in the border colour, like graphgen's canvas, as a data URL
  // for LiteGraph's BACKGROUND_IMAGE (drawn tiled; 64px matches the default).
  try {
    // graphgen: <Background variant=Dots gap=22 /> in #2b2b33 over a transparent
    // pane, so the fixed corner glows (CSS on the canvas element) show through
    const c = document.createElement("canvas");
    c.width = 22;
    c.height = 22;
    const g = c.getContext("2d");
    g.clearRect(0, 0, 22, 22);
    g.fillStyle = "#2b2b33";
    g.beginPath();
    g.arc(11, 11, 1.0, 0, Math.PI * 2);
    g.fill();
    return c.toDataURL("image/png");
  } catch (e) {
    return "";
  }
}

function buildPalette() {
  return {
    id: PALETTE_ID,
    name: "APNext Graphgen (Dark Botanical)",
    colors: {
      node_slot: {
        CLIP: C.honey,
        CLIP_VISION: C.teal,
        CLIP_VISION_OUTPUT: C.terracotta,
        CONDITIONING: C.accent,
        CONTROL_NET: C.sage,
        IMAGE: C.slate,
        LATENT: C.mauve,
        MASK: C.neutral,
        MODEL: C.rose,
        STYLE_MODEL: C.sage,
        VAE: C.terracotta,
        NOISE: C.textDim,
        GUIDER: C.teal,
        SAMPLER: C.accent2,
        SIGMAS: C.goldPort,
        TAESD: C.gold,
        AUDIO: C.goldPort,
        VIDEO: C.honey,
        STRING: C.neutral,
        INT: C.slate,
        FLOAT: C.slate,
        BOOLEAN: C.textDim,
        APNEXT_LLM: C.accent2,
      },
      litegraph_base: {
        BACKGROUND_IMAGE: dotGrid(),
        CLEAR_BACKGROUND_COLOR: "rgba(15, 15, 15, 0)", // transparent: the glows live on the canvas element
        NODE_TITLE_COLOR: "#f4f1ec",
        NODE_SELECTED_TITLE_COLOR: C.accent,
        NODE_TEXT_COLOR: C.textDim,
        NODE_TEXT_HIGHLIGHT_COLOR: C.text,
        NODE_DEFAULT_COLOR: C.panel,
        NODE_DEFAULT_BGCOLOR: C.panel,
        NODE_DEFAULT_BOXCOLOR: C.borderLight,
        NODE_DEFAULT_SHAPE: 2,
        NODE_BOX_OUTLINE_COLOR: C.accent,
        NODE_BYPASS_BGCOLOR: C.mauve,
        NODE_ERROR_COLOUR: "#d07070",
        DEFAULT_SHADOW_COLOR: "rgba(0, 0, 0, 0.55)",
        WIDGET_BGCOLOR: C.panel2,
        WIDGET_OUTLINE_COLOR: C.border,
        WIDGET_TEXT_COLOR: C.text,
        WIDGET_SECONDARY_TEXT_COLOR: C.textDim,
        WIDGET_DISABLED_TEXT_COLOR: "#5e5850",
        LINK_COLOR: C.accent,
        EVENT_LINK_COLOR: C.accent2,
        CONNECTING_LINK_COLOR: C.accent2,
        BADGE_FG_COLOR: C.text,
        BADGE_BG_COLOR: C.panel2,
      },
      comfy_base: {
        "fg-color": C.text,
        "bg-color": C.bg,
        "comfy-menu-bg": C.panel,
        "comfy-menu-secondary-bg": C.panel2,
        "comfy-input-bg": C.panel,
        "input-text": C.text,
        "descrip-text": C.textDim,
        "drag-text": C.textDim,
        "error-text": "#d07070",
        "border-color": C.border,
        "tr-even-bg-color": C.panel,
        "tr-odd-bg-color": C.panel2,
        "content-bg": C.panel2,
        "content-fg": C.text,
        "content-hover-bg": C.border,
        "content-hover-fg": C.text,
        "bar-shadow": "rgba(0, 0, 0, 0.6) 0 0 0.5rem",
      },
    },
  };
}

// ---------------------------------------------------------------------------
// Chrome (fonts + a few CSS tokens for the Vue UI)
// ---------------------------------------------------------------------------

const STYLE_ID = "apnext-graphgen-style";
const FONT_LINK_ID = "apnext-graphgen-fonts";
const CSS = `
html.apnext-graphgen {
  --p-font-family: ${FONT_SANS};
  --apnext-font-display: ${FONT_DISPLAY};
  --apnext-font-mono: ${FONT_MONO};
  --p-primary-color: ${C.accent};
  --p-primary-hover-color: ${C.accent2};
  --p-primary-active-color: ${C.accent};
  --p-highlight-background: rgba(212, 165, 116, 0.16);
  --p-highlight-color: ${C.accent};
  --p-focus-ring-color: ${C.accent};
  --p-content-border-color: ${C.border};
  --p-surface-ground: ${C.bg};
  --p-surface-section: ${C.panel};
  --p-surface-card: ${C.panel};
  --p-surface-overlay: ${C.panel};
  --p-surface-border: ${C.border};
  --p-text-color: ${C.text};
  --p-text-muted-color: ${C.textDim};
}
html.apnext-graphgen #graph-canvas,
html.apnext-graphgen .graph-canvas-container,
html.apnext-graphgen canvas.litegraph {
  background-color: ${C.bg} !important;
  background-image:
    radial-gradient(52% 48% at 8% 2%, rgba(212,165,116,0.09), transparent 70%),
    radial-gradient(48% 44% at 100% 100%, rgba(232,180,184,0.08), transparent 72%),
    radial-gradient(40% 38% at 78% 6%, rgba(201,184,150,0.06), transparent 70%) !important;
}
html.apnext-graphgen body,
html.apnext-graphgen .p-component,
html.apnext-graphgen .comfy-menu,
html.apnext-graphgen .litegraph,
html.apnext-graphgen .litegraph * {
  font-family: ${FONT_SANS};
  -webkit-font-smoothing: antialiased;
}
html.apnext-graphgen code, html.apnext-graphgen pre, html.apnext-graphgen textarea.comfy-multiline-input,
html.apnext-graphgen .apnext-h3-vars { font-family: ${FONT_MONO}; }
html.apnext-graphgen .comfy-multiline-input,
html.apnext-graphgen .comfy-multiline-input textarea {
  background: ${C.panel} !important; color: ${C.text} !important;
  border: 1px solid ${C.border} !important; border-radius: 8px !important;
}
html.apnext-graphgen .comfy-multiline-input:focus-within { border-color: ${C.accent} !important; }
html.apnext-graphgen .p-panel, html.apnext-graphgen .side-tool-bar-container,
html.apnext-graphgen .comfyui-menu, html.apnext-graphgen .actionbar,
html.apnext-graphgen .p-dialog, html.apnext-graphgen .p-popover, html.apnext-graphgen .p-menu,
html.apnext-graphgen .p-tieredmenu, html.apnext-graphgen .p-contextmenu, html.apnext-graphgen .litecontextmenu {
  background: ${C.panel} !important; color: ${C.text} !important; border-color: ${C.border} !important;
}
html.apnext-graphgen .litecontextmenu .litemenu-entry { color: ${C.text} !important; }
html.apnext-graphgen .litecontextmenu .litemenu-entry:hover,
html.apnext-graphgen .litemenu-entry.submenu:hover { background: ${C.border} !important; color: ${C.accent} !important; }
html.apnext-graphgen .p-button, html.apnext-graphgen .comfyui-button { border-radius: 8px; }
html.apnext-graphgen .p-button.p-button-primary, html.apnext-graphgen .comfyui-queue-button .p-button {
  background: ${C.accent} !important; border-color: ${C.accent} !important; color: #1a1510 !important;
}
html.apnext-graphgen .p-button.p-button-primary:hover { background: ${C.accent2} !important; border-color: ${C.accent2} !important; }
html.apnext-graphgen .p-toast-message, html.apnext-graphgen .p-inputtext, html.apnext-graphgen .p-select,
html.apnext-graphgen .p-inputnumber-input, html.apnext-graphgen .p-textarea {
  background: ${C.panel2} !important; color: ${C.text} !important; border-color: ${C.border} !important;
}
html.apnext-graphgen h1, html.apnext-graphgen h2, html.apnext-graphgen .p-dialog-title { font-family: ${FONT_DISPLAY}; letter-spacing: 0.01em; }
`;

function ensureFonts() {
  if (!document.getElementById(FONT_LINK_ID)) {
    const link = document.createElement("link");
    link.id = FONT_LINK_ID;
    link.rel = "stylesheet";
    link.href = FONTS_HREF;
    document.head.appendChild(link);
  }
  try {
    document.fonts?.load?.("14px 'IBM Plex Sans'")?.then?.(() => app.canvas?.setDirty?.(true, true));
  } catch (e) { /* offline: fall back to system fonts */ }
}

function applyChrome(on) {
  if (on) {
    if (!document.getElementById(STYLE_ID)) {
      const st = document.createElement("style");
      st.id = STYLE_ID;
      st.textContent = CSS;
      document.head.appendChild(st);
    }
    ensureFonts();
    document.documentElement.classList.add("apnext-graphgen");
  } else {
    document.documentElement.classList.remove("apnext-graphgen");
  }
}

// ---------------------------------------------------------------------------
// LiteGraph look: font + radius (the colours come through the palette)
// ---------------------------------------------------------------------------

const saved = { font: null, radius: null, applied: false };

function applyLiteGraphLook(on) {
  const LG = window.LiteGraph;
  if (!LG) return;
  if (on && !saved.applied) {
    saved.font = LG.NODE_FONT;
    saved.radius = LG.ROUND_RADIUS;
    LG.NODE_FONT = FONT_SANS;
    LG.ROUND_RADIUS = 8; // graphgen rounded-lg
    saved.applied = true;
  } else if (!on && saved.applied) {
    LG.NODE_FONT = saved.font ?? "Inter";
    LG.ROUND_RADIUS = saved.radius ?? 8;
    saved.applied = false;
  }
  app.canvas?.setDirty?.(true, true);
}

// ---------------------------------------------------------------------------
// Palette install / switch through ComfyUI's own colour-palette store
// ---------------------------------------------------------------------------

function settingGet(id) {
  return app.ui?.settings?.getSettingValue?.(id);
}
async function settingSet(id, value) {
  if (app.extensionManager?.setting?.set) return app.extensionManager.setting.set(id, value);
  return app.ui.settings.setSettingValueAsync(id, value);
}

function paletteStore() {
  try {
    const root = document.getElementById("vue-app") || document.querySelector("[data-v-app]");
    const vue = root?.__vue_app__;
    const pinia = vue?.config?.globalProperties?.$pinia;
    return pinia?._s?.get?.("colorPalette") || null;
  } catch (e) {
    return null;
  }
}

async function registerPalette() {
  const palette = buildPalette();
  // 1. persist as a custom palette so it is there on the next page load too
  const customs = { ...(settingGet("Comfy.CustomColorPalettes") || {}) };
  const stored = customs[PALETTE_ID];
  if (!stored || JSON.stringify(stored) !== JSON.stringify(palette)) {
    customs[PALETTE_ID] = palette;
    await settingSet("Comfy.CustomColorPalettes", customs);
  }
  // 2. make the running store know it right now
  const store = paletteStore();
  if (store) {
    if (!store.palettesLookup?.[PALETTE_ID]) {
      const active = store.activePaletteId;
      store.addCustomPalette(palette);
      store.activePaletteId = active; // addCustomPalette activates; we switch via the setting
    } else if (store.customPalettes) {
      store.customPalettes[PALETTE_ID] = palette;
    }
    return true;
  }
  return false;
}

// Fallback when the Pinia store is not reachable: push colours straight into
// LiteGraph. Restoring then means re-selecting a palette in Settings.
function applyPaletteDirect(palette) {
  const LG = window.LiteGraph;
  const canvas = app.canvas;
  if (!LG || !canvas) return;
  const base = palette.colors.litegraph_base;
  for (const [k, v] of Object.entries(base)) {
    if (k === "BACKGROUND_IMAGE") canvas.background_image = v;
    else if (k === "CLEAR_BACKGROUND_COLOR") canvas.clear_background_color = v;
    else LG[k] = v;
  }
  canvas.default_link_color = base.LINK_COLOR;
  canvas.default_connection_color_byType = { ...canvas.default_connection_color_byType, ...palette.colors.node_slot };
  if (window.LGraphCanvas) window.LGraphCanvas.link_type_colors = { ...window.LGraphCanvas.link_type_colors, ...palette.colors.node_slot };
  for (const [k, v] of Object.entries(palette.colors.comfy_base)) document.documentElement.style.setProperty(`--${k}`, v);
  canvas.setDirty(true, true);
}

async function applyPalette(on) {
  const current = settingGet("Comfy.ColorPalette");
  if (on) {
    const ok = await registerPalette();
    if (current !== PALETTE_ID) {
      await settingSet(S_PREV, current || "dark");
      if (ok) await settingSet("Comfy.ColorPalette", PALETTE_ID);
      else applyPaletteDirect(buildPalette());
    }
  } else if (current === PALETTE_ID) {
    const prev = settingGet(S_PREV) || "dark";
    await settingSet("Comfy.ColorPalette", prev === PALETTE_ID ? "dark" : prev);
  }
}

// ---------------------------------------------------------------------------
// Gravity wires - verlet rope per link (port of graphgen rope.svelte.ts)
// ---------------------------------------------------------------------------

const rope = {
  ropes: new Map(),
  raf: 0,
  frame: 0,
  cfg: { segments: 22, gravity: 0.55, friction: 0.94, iterations: 12, slack: 1.16, settle: 0.05, maxExtra: 420 },

  build(ax, ay, bx, by) {
    const n = Math.max(4, this.cfg.segments | 0);
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const t = i / n;
      const x = ax + (bx - ax) * t;
      const y = ay + (by - ay) * t;
      pts.push({ x, y, ox: x, oy: y });
    }
    return { pts, ax, ay, bx, by, seen: this.frame };
  },

  points(id, ax, ay, bx, by) {
    let r = this.ropes.get(id);
    if (!r || r.pts.length !== (Math.max(4, this.cfg.segments | 0) + 1)) {
      r = this.build(ax, ay, bx, by);
      this.ropes.set(id, r);
      this.wake();
    }
    const moved = Math.abs(r.ax - ax) + Math.abs(r.ay - ay) + Math.abs(r.bx - bx) + Math.abs(r.by - by) > 0.01;
    r.ax = ax; r.ay = ay; r.bx = bx; r.by = by;
    r.seen = this.frame;
    if (moved) {
      // keep the ends glued even between ticks, so a drag never shows a gap
      const a = r.pts[0], b = r.pts[r.pts.length - 1];
      a.x = a.ox = ax; a.y = a.oy = ay; b.x = b.ox = bx; b.y = b.oy = by;
      this.wake();
    }
    return r.pts;
  },

  prune() {
    for (const [id, r] of this.ropes) if (r.seen < this.frame - 2) this.ropes.delete(id);
  },

  clear() { this.ropes.clear(); this.stop(); },
  stop() { if (this.raf) cancelAnimationFrame(this.raf); this.raf = 0; },

  wake() {
    if (this.raf || typeof requestAnimationFrame === "undefined") return;
    this.raf = requestAnimationFrame(() => this.step());
  },

  step() {
    if (wireState.style !== WIRE_GRAVITY) { this.raf = 0; return; }
    const { gravity, friction, iterations, slack, settle, maxExtra } = this.cfg;
    let awake = false;
    for (const r of this.ropes.values()) {
      const n = r.pts.length;
      for (let i = 1; i < n - 1; i++) {
        const p = r.pts[i];
        const vx = (p.x - p.ox) * friction;
        const vy = (p.y - p.oy) * friction;
        p.ox = p.x; p.oy = p.y;
        p.x += vx; p.y += vy + gravity;
      }
      const a = r.pts[0], b = r.pts[n - 1];
      a.x = a.ox = r.ax; a.y = a.oy = r.ay;
      b.x = b.ox = r.bx; b.y = b.oy = r.by;
      const span = Math.hypot(r.bx - r.ax, r.by - r.ay);
      const rest = span + Math.min(span * (slack - 1), maxExtra);
      const seg = rest / (n - 1);
      for (let it = 0; it < iterations; it++) {
        for (let i = 0; i < n - 1; i++) {
          const p1 = r.pts[i], p2 = r.pts[i + 1];
          const dx = p2.x - p1.x, dy = p2.y - p1.y;
          const d = Math.hypot(dx, dy) || 1e-6;
          const diff = ((seg - d) / d) * 0.5;
          const ox = dx * diff, oy = dy * diff;
          if (i !== 0) { p1.x -= ox; p1.y -= oy; }
          if (i + 1 !== n - 1) { p2.x += ox; p2.y += oy; }
        }
      }
      let motion = 0;
      for (let i = 1; i < n - 1; i++) {
        const p = r.pts[i];
        motion += Math.abs(p.x - p.ox) + Math.abs(p.y - p.oy);
        if (motion > settle) break;
      }
      if (motion > settle) awake = true;
    }
    // redraw the link layer while anything still swings
    app.canvas?.setDirty?.(false, true);
    this.raf = awake ? requestAnimationFrame(() => this.step()) : 0;
  },
};

// ---------------------------------------------------------------------------
// Cable wires - a damped spring per link chases "midpoint + sag", so the wire
// droops and wobbles after a drag (port of graphgen cable.svelte.ts)
// ---------------------------------------------------------------------------

const cable = {
  springs: new Map(),
  raf: 0,
  last: 0,
  frame: 0,
  cfg: { stiffness: 90, damping: 9, settle: 0.25, sagMax: 70, sagRatio: 0.18 },

  point(id, tx, ty) {
    let s = this.springs.get(id);
    if (!s) {
      s = { px: tx, py: ty, vx: 0, vy: 0, tx, ty, seen: this.frame };
      this.springs.set(id, s);
    } else {
      s.tx = tx; s.ty = ty; s.seen = this.frame;
    }
    if (!this.raf && (Math.hypot(s.tx - s.px, s.ty - s.py) > this.cfg.settle || Math.hypot(s.vx, s.vy) > this.cfg.settle)) {
      this.wake();
    }
    return { x: s.px, y: s.py };
  },

  prune() { for (const [id, s] of this.springs) if (s.seen < this.frame - 2) this.springs.delete(id); },
  clear() { this.springs.clear(); this.stop(); },
  stop() { if (this.raf) cancelAnimationFrame(this.raf); this.raf = 0; },

  wake() {
    if (this.raf || typeof requestAnimationFrame === "undefined") return;
    this.last = performance.now();
    this.raf = requestAnimationFrame((t) => this.step(t));
  },

  step(now) {
    if (wireState.style !== WIRE_CABLE) { this.raf = 0; return; }
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    const { stiffness, damping, settle } = this.cfg;
    let awake = false;
    for (const s of this.springs.values()) {
      s.vx += (s.tx - s.px) * stiffness * dt - s.vx * damping * dt;
      s.vy += (s.ty - s.py) * stiffness * dt - s.vy * damping * dt;
      s.px += s.vx * dt;
      s.py += s.vy * dt;
      if (Math.hypot(s.tx - s.px, s.ty - s.py) > settle || Math.hypot(s.vx, s.vy) > settle) awake = true;
      else { s.px = s.tx; s.py = s.ty; s.vx = 0; s.vy = 0; }
    }
    app.canvas?.setDirty?.(false, true);
    this.raf = awake ? requestAnimationFrame((t) => this.step(t)) : 0;
  },
};

// ---------------------------------------------------------------------------
// Wire geometry
// ---------------------------------------------------------------------------

const WIRE_DEFAULT = "default";
const WIRE_BEZIER = "bezier";
const WIRE_SMOOTHSTEP = "smoothstep";
const WIRE_STEP = "step";
const WIRE_STRAIGHT = "straight";
const WIRE_CABLE = "cable";
const WIRE_GRAVITY = "gravity";
const WIRE_STYLES = [WIRE_DEFAULT, WIRE_BEZIER, WIRE_SMOOTHSTEP, WIRE_STEP, WIRE_STRAIGHT, WIRE_CABLE, WIRE_GRAVITY];
const WIRE_COMBO = [WIRE_DEFAULT, WIRE_BEZIER, WIRE_SMOOTHSTEP, WIRE_STEP, WIRE_STRAIGHT, WIRE_CABLE]; // gravity is its own toggle
const S_GRAVITY_ON = "APNext.Theme.GravityWires";

const wireState = { style: WIRE_DEFAULT, patched: false };

function isCustomWire(style) {
  return style === WIRE_SMOOTHSTEP || style === WIRE_STEP || style === WIRE_STRAIGHT || style === WIRE_CABLE || style === WIRE_GRAVITY;
}

// Points along a rounded corner from p0 -> corner -> p1 (radius r); the corner
// itself is replaced by a small arc, sampled so the polyline stays a polyline.
function roundedCorner(p0, c, p1, r) {
  const d0 = Math.hypot(c.x - p0.x, c.y - p0.y);
  const d1 = Math.hypot(p1.x - c.x, p1.y - c.y);
  const rr = Math.min(r, d0 / 2, d1 / 2);
  if (rr <= 0.5) return [c];
  const ux0 = (c.x - p0.x) / d0, uy0 = (c.y - p0.y) / d0;
  const ux1 = (p1.x - c.x) / d1, uy1 = (p1.y - c.y) / d1;
  const a = { x: c.x - ux0 * rr, y: c.y - uy0 * rr };
  const b = { x: c.x + ux1 * rr, y: c.y + uy1 * rr };
  const out = [];
  for (let i = 0; i <= 5; i++) {
    const t = i / 5;
    // quadratic bezier a -> c -> b approximates the arc well enough at this size
    out.push({
      x: (1 - t) * (1 - t) * a.x + 2 * (1 - t) * t * c.x + t * t * b.x,
      y: (1 - t) * (1 - t) * a.y + 2 * (1 - t) * t * c.y + t * t * b.y,
    });
  }
  return out;
}

// xyflow-style smooth step between a right-facing output and a left-facing
// input: horizontal, vertical, horizontal - with a detour when the target sits
// behind the source. radius 0 => hard steps.
function stepPoints(a, b, radius) {
  const off = 20;
  let corners;
  if (b[0] >= a[0] + 2 * off) {
    const mx = (a[0] + b[0]) / 2;
    corners = [{ x: mx, y: a[1] }, { x: mx, y: b[1] }];
  } else {
    const my = (a[1] + b[1]) / 2;
    corners = [
      { x: a[0] + off, y: a[1] }, { x: a[0] + off, y: my },
      { x: b[0] - off, y: my }, { x: b[0] - off, y: b[1] },
    ];
  }
  const all = [{ x: a[0], y: a[1] }, ...corners, { x: b[0], y: b[1] }];
  if (!radius) return all;
  const pts = [all[0]];
  for (let i = 1; i < all.length - 1; i++) pts.push(...roundedCorner(all[i - 1], all[i], all[i + 1], radius));
  pts.push(all[all.length - 1]);
  return pts;
}

function cablePoints(key, a, b) {
  const span = Math.hypot(b[0] - a[0], b[1] - a[1]);
  const sag = Math.min(cable.cfg.sagMax, span * cable.cfg.sagRatio);
  const p = cable.point(key, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + sag);
  const pts = [];
  for (let i = 0; i <= 24; i++) {
    const t = i / 24;
    pts.push({
      x: (1 - t) * (1 - t) * a[0] + 2 * (1 - t) * t * p.x + t * t * b[0],
      y: (1 - t) * (1 - t) * a[1] + 2 * (1 - t) * t * p.y + t * t * b[1],
    });
  }
  return pts;
}

function wirePoints(style, key, a, b) {
  switch (style) {
    case WIRE_GRAVITY: return rope.points(key, a[0], a[1], b[0], b[1]);
    case WIRE_CABLE: return cablePoints(key, a, b);
    case WIRE_SMOOTHSTEP: return stepPoints(a, b, 8);
    case WIRE_STEP: return stepPoints(a, b, 0);
    case WIRE_STRAIGHT: return [{ x: a[0], y: a[1] }, { x: b[0], y: b[1] }];
    default: return null;
  }
}

function tracePoly(ctx, pts, smooth) {
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  if (smooth) {
    // smooth through the chain with quadratic midpoints (round joins do the rest)
    for (let i = 1; i < pts.length - 1; i++) {
      const mx = (pts[i].x + pts[i + 1].x) / 2;
      const my = (pts[i].y + pts[i + 1].y) / 2;
      ctx.quadraticCurveTo(pts[i].x, pts[i].y, mx, my);
    }
    ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
  } else {
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  }
}

function pointAt(pts, t) {
  // position at normalised arc-length t along the polyline
  let total = 0;
  const seg = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const d = Math.hypot(pts[i + 1].x - pts[i].x, pts[i + 1].y - pts[i].y);
    seg.push(d); total += d;
  }
  let target = t * total;
  for (let i = 0; i < seg.length; i++) {
    if (target <= seg[i]) {
      const u = seg[i] ? target / seg[i] : 0;
      return [pts[i].x + (pts[i + 1].x - pts[i].x) * u, pts[i].y + (pts[i + 1].y - pts[i].y) * u];
    }
    target -= seg[i];
  }
  const p = pts[pts.length - 1];
  return [p.x, p.y];
}

function patchLinkRenderer() {
  if (wireState.patched) return;
  const LGC = window.LGraphCanvas || app.canvas?.constructor;
  if (!LGC?.prototype?.renderLink) {
    console.warn("[APNext theme] LGraphCanvas.renderLink not found - custom wire styles unavailable on this frontend");
    return;
  }
  const origRender = LGC.prototype.renderLink;
  LGC.prototype.renderLink = function (ctx, a, b, link, skip_border, flow, color, start_dir, end_dir, opts) {
    const LG = window.LiteGraph;
    const style = wireState.style;
    if (LG && this.links_render_mode === LG.HIDDEN_LINK) return origRender.apply(this, arguments);
    if (style === WIRE_BEZIER && LG && LG.SPLINE_LINK !== undefined) {
      // force LiteGraph's own spline regardless of the user's render-mode setting
      const prev = this.links_render_mode;
      this.links_render_mode = LG.SPLINE_LINK;
      try { return origRender.apply(this, arguments); } finally { this.links_render_mode = prev; }
    }
    if (!isCustomWire(style) || !link || !a || !b || opts?.disabled) {
      return origRender.apply(this, arguments);
    }
    const key = `${link.id ?? "tmp"}|${opts?.reroute?.id ?? ""}|${opts?.startControl ? "s" : ""}`;
    const pts = wirePoints(style, key, a, b);
    if (!pts) return origRender.apply(this, arguments);
    const col =
      color || link.color || (LGC.link_type_colors && LGC.link_type_colors[link.type]) || this.default_link_color;
    const smooth = style === WIRE_GRAVITY;

    ctx.save();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    const width = this.connections_width || 3;
    if (!skip_border && this.render_connections_border && (this.ds?.scale ?? 1) > 0.6) {
      ctx.lineWidth = width + 4;
      ctx.strokeStyle = "rgba(0,0,0,0.5)";
      tracePoly(ctx, pts, smooth);
      ctx.stroke();
    }
    ctx.lineWidth = width;
    ctx.strokeStyle = col;
    tracePoly(ctx, pts, smooth);
    ctx.stroke();

    // link centre (used by the link menu / tooltip)
    const [mx, my] = pointAt(pts, 0.5);
    link._pos = [mx, my];
    const markerOn = this.linkMarkerShape === undefined || this.linkMarkerShape !== 0;
    if (markerOn && (this.ds?.scale ?? 1) > 0.5 && !flow) {
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(mx, my, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = C.bg;
      ctx.beginPath();
      ctx.arc(mx, my, 2, 0, Math.PI * 2);
      ctx.fill();
    }
    // execution "flow" sparks travelling along the wire
    if (flow) {
      const now = (typeof performance !== "undefined" ? performance.now() : Date.now()) * 0.0005;
      ctx.fillStyle = color || "#ffffff";
      for (let i = 0; i < 5; i++) {
        const t = (now + i * 0.2) % 1;
        const [x, y] = pointAt(pts, t);
        ctx.beginPath();
        ctx.arc(x, y, width * 0.75, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  };

  // prune springs / ropes whose links are gone, once per background-layer draw
  const origBack = LGC.prototype.drawBackCanvas;
  if (origBack) {
    LGC.prototype.drawBackCanvas = function () {
      rope.frame++;
      cable.frame++;
      const r = origBack.apply(this, arguments);
      if (rope.frame % 30 === 0) { rope.prune(); cable.prune(); }
      return r;
    };
  }
  wireState.patched = true;
}

function applyWires(style) {
  if (!WIRE_STYLES.includes(style)) style = WIRE_DEFAULT;
  if (style !== WIRE_DEFAULT) patchLinkRenderer();
  wireState.style = wireState.patched ? style : WIRE_DEFAULT;
  if (style !== WIRE_GRAVITY) rope.clear();
  if (style !== WIRE_CABLE) cable.clear();
  app.canvas?.setDirty?.(false, true);
}

function readWireConfig() {
  const slack = Number(settingGet(S_SLACK));
  const gravity = Number(settingGet(S_GRAVITY));
  const segments = Number(settingGet(S_SEGMENTS));
  if (Number.isFinite(slack) && slack >= 1) rope.cfg.slack = slack;
  if (Number.isFinite(gravity) && gravity >= 0) rope.cfg.gravity = gravity;
  if (Number.isFinite(segments) && segments >= 4) rope.cfg.segments = segments;
  rope.clear();
  if (wireState.style === WIRE_GRAVITY) rope.wake();
  app.canvas?.setDirty?.(false, true);
}

// ---------------------------------------------------------------------------
// Recolour nodes & groups that carry their own colour (right-click → Colors,
// or packs that pre-colour their nodes) to the nearest botanical hue while the
// theme is on. Done at draw time, so the stored colours are never modified and
// turning the theme off shows them again. The node-colour menu swatches
// (LGraphCanvas.node_colors) are remapped too, and restored on exit.
// ---------------------------------------------------------------------------

const S_RECOLOR = "APNext.Theme.RecolorNodes";

// hue (deg) -> swatch; picked by nearest hue for saturated colours
const SWATCHES = [
  [0, C.rose], [30, C.terracotta], [45, C.honey], [55, C.goldPort], [90, C.sage],
  [170, C.teal], [210, C.slate], [275, C.mauve], [330, C.accent2], [360, C.rose],
];
const NODE_COLOR_MAP = {
  red: C.rose, brown: C.terracotta, green: C.sage, blue: C.slate, pale_blue: C.teal,
  cyan: C.teal, purple: C.mauve, yellow: C.goldPort, black: C.textDim,
};

function parseColor(str) {
  if (typeof str !== "string") return null;
  const s = str.trim();
  let m = s.match(/^#([0-9a-f]{3,4})$/i);
  if (m) {
    const h = m[1];
    return [parseInt(h[0] + h[0], 16), parseInt(h[1] + h[1], 16), parseInt(h[2] + h[2], 16)];
  }
  m = s.match(/^#([0-9a-f]{6})([0-9a-f]{2})?$/i);
  if (m) return [parseInt(m[1].slice(0, 2), 16), parseInt(m[1].slice(2, 4), 16), parseInt(m[1].slice(4, 6), 16)];
  m = s.match(/^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/i);
  if (m) return [Number(m[1]), Number(m[2]), Number(m[3])];
  return null;
}
function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h;
  if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) * 60;
  else if (max === g) h = ((b - r) / d + 2) * 60;
  else h = ((r - g) / d + 4) * 60;
  return [h, s, l];
}
function mix(hexA, hexB, t) {
  const a = parseColor(hexA), b = parseColor(hexB);
  if (!a || !b) return hexA;
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
  return "#" + c.map((v) => v.toString(16).padStart(2, "0")).join("");
}
function swatchFor(hue) {
  let best = SWATCHES[0][1], dist = 1e9;
  for (const [h, col] of SWATCHES) {
    const d = Math.min(Math.abs(hue - h), 360 - Math.abs(hue - h));
    if (d < dist) { dist = d; best = col; }
  }
  return best;
}

const recolorCache = new Map();
// kind: "title" (node.color), "body" (node.bgcolor), "group" (group.color)
function botanical(color, kind) {
  if (!color) return color;
  const key = kind + "|" + color;
  if (recolorCache.has(key)) return recolorCache.get(key);
  const rgb = parseColor(color);
  let out = color;
  if (rgb) {
    const [h, s, l] = rgbToHsl(...rgb);
    if (s < 0.12) {
      // greys: sit on the panel tones by lightness
      out = kind === "group" ? (l > 0.5 ? C.borderLight : C.border)
        : kind === "title" ? (l > 0.35 ? C.border : C.panel)
        : (l > 0.35 ? C.border : C.panel2);
    } else {
      const sw = swatchFor(h);
      out = kind === "group" ? mix(sw, C.panel, 0.15)          // groups draw at 50% alpha already
        : kind === "title" ? mix(sw, C.panel, 0.72)             // header: tinted panel
        : C.panel;                                              // body: neutral - the colour lives in the header
    }
  }
  recolorCache.set(key, out);
  return out;
}

const recolorState = { on: false, patched: false, savedNodeColors: null };

function patchNodeRecolor() {
  if (recolorState.patched) return;
  const LGC = window.LGraphCanvas || app.canvas?.constructor;
  if (!LGC?.prototype?.drawNode) return;
  const origDrawNode = LGC.prototype.drawNode;
  LGC.prototype.drawNode = function (node, ctx) {
    if (!recolorState.on || !node || (!node.color && !node.bgcolor) || node.properties?.apnext_autocolor) {
      return origDrawNode.apply(this, arguments);
    }
    const c = node.color, bg = node.bgcolor;
    try {
      if (c) node.color = botanical(c, "title");
      if (bg) node.bgcolor = botanical(bg, "body");
      return origDrawNode.apply(this, arguments);
    } finally {
      node.color = c;
      node.bgcolor = bg;
    }
  };
  const origDrawGroups = LGC.prototype.drawGroups;
  if (origDrawGroups) {
    LGC.prototype.drawGroups = function (canvas, ctx) {
      const groups = recolorState.on ? (this.graph?._groups || []) : [];
      const saved = groups.map((g) => g.color);
      try {
        if (recolorState.on) for (const g of groups) if (g.color) g.color = botanical(g.color, "group");
        return origDrawGroups.apply(this, arguments);
      } finally {
        groups.forEach((g, i) => { g.color = saved[i]; });
      }
    };
  }
  recolorState.patched = true;
}

function applyNodeRecolor(on) {
  const LGC = window.LGraphCanvas || app.canvas?.constructor;
  if (on) patchNodeRecolor();
  recolorState.on = on && recolorState.patched;
  if (LGC?.node_colors) {
    if (on && !recolorState.savedNodeColors) {
      recolorState.savedNodeColors = JSON.parse(JSON.stringify(LGC.node_colors));
      for (const [name, sw] of Object.entries(NODE_COLOR_MAP)) {
        if (!LGC.node_colors[name]) continue;
        LGC.node_colors[name] = {
          ...LGC.node_colors[name],
          color: mix(sw, C.panel, 0.72),
          bgcolor: C.panel,
          groupcolor: mix(sw, C.panel, 0.15),
        };
      }
    } else if (!on && recolorState.savedNodeColors) {
      for (const [name, v] of Object.entries(recolorState.savedNodeColors)) LGC.node_colors[name] = v;
      recolorState.savedNodeColors = null;
    }
  }
  app.canvas?.setDirty?.(true, true);
}

// ---------------------------------------------------------------------------
// Drag highlight: while a link is being dragged, every slot that could take it
// (matching type, other node) gets a pulsing ring, so the valid targets are
// obvious. Works for output→input and input→output drags. Independent of the
// theme; the ring uses the accent colour.
// ---------------------------------------------------------------------------

const S_HIGHLIGHT = "APNext.Theme.HighlightTargets";
const S_SPARKS = "APNext.Theme.Sparks";
const hlState = { on: true, patched: false };

function dragSources(canvas) {
  // [{type, node, toInputs}] for every link currently being dragged
  const lc = canvas.linkConnector;
  const out = [];
  if (!lc || !lc.isConnecting) return out;
  const toInputs = lc.state?.connectingTo === "input";
  for (const rl of lc.renderLinks || []) {
    const slot = rl.fromSlot;
    if (!slot) continue;
    out.push({ type: slot.type, node: rl.node, toInputs });
  }
  return out;
}

function patchHighlight() {
  if (hlState.patched) return;
  const LGC = window.LGraphCanvas || app.canvas?.constructor;
  if (!LGC?.prototype?.drawNode) return;
  const orig = LGC.prototype.drawNode;
  LGC.prototype.drawNode = function (node, ctx) {
    const r = orig.apply(this, arguments);
    if (!hlState.on || !node) return r;
    const LG = window.LiteGraph;
    const sources = dragSources(this);
    if (!sources.length || !LG) return r;
    const slots = [];
    for (const src of sources) {
      if (src.node === node) continue;
      if (src.toInputs) {
        (node.inputs || []).forEach((inp, i) => {
          if (inp && LG.isValidConnection(src.type, inp.type)) slots.push([true, i, inp.link != null]);
        });
      } else {
        (node.outputs || []).forEach((outp, i) => {
          if (outp && LG.isValidConnection(outp.type, src.type)) slots.push([false, i, false]);
        });
      }
    }
    if (!slots.length) return r;
    const t = (performance.now() % 1000) / 1000;
    const pulse = 0.5 + 0.5 * Math.sin(t * Math.PI * 2);
    // sizes in SCREEN pixels, so the rings stay readable when zoomed out
    const k = 1 / Math.max(0.15, this.ds?.scale ?? 1);
    const rCore = (10 + pulse * 3) * k;
    const rHalo = (18 + pulse * 10) * k;
    ctx.save();
    for (const [isInput, i, taken] of slots) {
      const p = isInput ? node.getInputPos(i) : node.getOutputPos(i);
      if (!p) continue;
      const x = p[0] - node.pos[0], y = p[1] - node.pos[1];
      const rgb = taken ? "232,180,184" : "212,165,116";
      // soft halo that breathes outward
      const grad = ctx.createRadialGradient(x, y, rCore * 0.6, x, y, rHalo);
      grad.addColorStop(0, `rgba(${rgb},${0.45 - 0.2 * pulse})`);
      grad.addColorStop(1, `rgba(${rgb},0)`);
      ctx.beginPath();
      ctx.arc(x, y, rHalo, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
      // solid core ring
      ctx.beginPath();
      ctx.arc(x, y, rCore, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${rgb},${0.22 + 0.18 * pulse})`;
      ctx.fill();
      ctx.lineWidth = 3 * k;
      ctx.strokeStyle = `rgba(${rgb},${0.85 + 0.15 * pulse})`;
      ctx.stroke();
      // thin white inner ring for contrast on any node colour
      ctx.beginPath();
      ctx.arc(x, y, rCore - 2.5 * k, 0, Math.PI * 2);
      ctx.lineWidth = 1 * k;
      ctx.strokeStyle = `rgba(255,255,255,${0.5 + 0.3 * pulse})`;
      ctx.stroke();
    }
    ctx.restore();
    // keep pulsing while the drag lasts
    this.setDirty(true, false);
    return r;
  };
  hlState.patched = true;
}

function applyHighlight(on) {
  if (on) patchHighlight();
  hlState.on = on && hlState.patched;
}

// ---------------------------------------------------------------------------
// Connect sparks: a little burst of particles at the input slot when a link is
// made (graphgen's Sparks.svelte, drawn on the front canvas).
// ---------------------------------------------------------------------------

const sparks = { list: [], raf: 0, patched: false, on: true, listening: new WeakSet() };

function spawnSparks(x, y, color) {
  const now = performance.now();
  const k = 1 / Math.max(0.15, app.canvas?.ds?.scale ?? 1); // canvas px per screen px
  for (let i = 0; i < 26; i++) {
    const ang = Math.random() * Math.PI * 2;
    const speed = (70 + Math.random() * 190) * k; // ~ screen px / s
    sparks.list.push({
      x, y, vx: Math.cos(ang) * speed, vy: Math.sin(ang) * speed - 40 * k,
      born: now, life: 500 + Math.random() * 400, size: (1.8 + Math.random() * 2.4) * k,
      color: color || C.accent, k,
    });
  }
  // one expanding ring so the connect reads even from far out
  sparks.list.push({ x, y, vx: 0, vy: 0, born: now, life: 520, size: 0, ring: true, color: color || C.accent, k });
  if (!sparks.raf) sparks.raf = requestAnimationFrame(tickSparks);
}

function tickSparks() {
  sparks.raf = 0;
  const now = performance.now();
  sparks.list = sparks.list.filter((p) => now - p.born < p.life);
  app.canvas?.setDirty?.(true, false);
  if (sparks.list.length) sparks.raf = requestAnimationFrame(tickSparks);
}

function drawSparks(canvas, ctx) {
  if (!sparks.list.length) return;
  const now = performance.now();
  const ds = canvas.ds;
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  for (const p of sparks.list) {
    const age = (now - p.born) / 1000;
    const k = 1 - (now - p.born) / p.life;
    if (p.ring) {
      const sx = (p.x + ds.offset[0]) * ds.scale, sy = (p.y + ds.offset[1]) * ds.scale;
      ctx.globalAlpha = Math.max(0, k) * 0.9;
      ctx.strokeStyle = p.color;
      ctx.lineWidth = 3 * (0.4 + 0.6 * k);
      ctx.beginPath();
      ctx.arc(sx, sy, 6 + (1 - k) * 38, 0, Math.PI * 2); // screen px
      ctx.stroke();
      continue;
    }
    const x = p.x + p.vx * age, y = p.y + p.vy * age + 120 * (p.k || 1) * age * age; // a little gravity
    const sx = (x + ds.offset[0]) * ds.scale, sy = (y + ds.offset[1]) * ds.scale;
    ctx.globalAlpha = Math.max(0, k);
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.arc(sx, sy, Math.max(1.2, p.size * ds.scale * (0.6 + 0.4 * k)), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function linkEndpoint(link) {
  const graph = app.canvas?.graph;
  const node = graph?.getNodeById?.(link.target_id);
  if (!node) return null;
  const p = node.getInputPos?.(link.target_slot);
  return p ? [p[0], p[1]] : null;
}

function patchSparks() {
  if (sparks.patched) return;
  const LGC = window.LGraphCanvas || app.canvas?.constructor;
  if (!LGC?.prototype?.drawFrontCanvas) return;
  const orig = LGC.prototype.drawFrontCanvas;
  LGC.prototype.drawFrontCanvas = function () {
    const r = orig.apply(this, arguments);
    if (sparks.on) {
      try { drawSparks(this, this.ctx); } catch (e) { /* ignore */ }
      listenForLinks(this);
    }
    return r;
  };
  sparks.patched = true;
}

function listenForLinks(canvas) {
  const lc = canvas.linkConnector;
  if (!lc?.events || sparks.listening.has(lc)) return;
  sparks.listening.add(lc);
  lc.events.addEventListener("link-created", (e) => {
    if (!sparks.on) return;
    const link = e?.detail;
    if (!link) return;
    const p = linkEndpoint(link);
    if (!p) return;
    const LGC = window.LGraphCanvas || app.canvas?.constructor;
    const color = link.color || (LGC?.link_type_colors && LGC.link_type_colors[link.type]) || C.accent;
    spawnSparks(p[0], p[1], color);
  });
}

function applySparks(on) {
  if (on) patchSparks();
  sparks.on = on && sparks.patched;
  if (!on) sparks.list = [];
}

// ---------------------------------------------------------------------------
// Colour-coded APNext nodes: every node of this pack gets a family colour on
// creation (unless the user already coloured it), in the same hues as the
// links/ports, so it is obvious what is what and what plugs where:
//   sage     - H3 writers (they produce prompts / scenes)       → STRING-ish outputs
//   rose     - H3 Characters (cast)                              → cast_N sockets
//   pink     - H3 LLM Backend                                    → llm socket (pink link)
//   slate    - H3 Prompt Preview / viewers
//   gold     - scene utilities (Scene Pick, Scenes Join, Chain Plan, resolution)
//   mauve    - APNext context / prompt-generator nodes (Time, Scene, Poses ...)
//   teal     - vision / captioning nodes
//   terracotta - everything else from the pack
// ---------------------------------------------------------------------------

const S_COLORCODE = "APNext.Theme.ColorCodeNodes";
const colorCodeState = { on: true };

function familySwatch(name) {
  if (/^H3.*(Writer|Refiner)$/.test(name)) return C.sage;
  if (name === "H3Characters") return C.rose;
  if (name === "H3LLMBackend") return C.accent2;
  if (/Preview|Viewer|Show/.test(name)) return C.slate;
  if (/^H3(ScenePick|ScenesJoin|ScenesToChainPlan|Resolution)/.test(name)) return C.goldPort;
  if (/PromptNode$|Prompt(Generator|Builder|Composer)|Random|Wildcard|Category/.test(name)) return C.mauve;
  if (/Vision|QwenVL|MiniCPM|Phi|Caption|Describe|Gemini|Gpt|Claude|Grok|Groq|Ollama/.test(name)) return C.teal;
  return C.terracotta;
}

function familyColors(name) {
  const sw = familySwatch(name);
  return { color: mix(sw, C.panel, 0.72), bgcolor: undefined }; // header only; body stays neutral
}

function isOurs(nodeData) {
  if (nodeData?.python_module) return String(nodeData.python_module).includes("comfyui_dagthomas");
  return /^H3|^APNext|^ClaudeCodeNode$/.test(nodeData?.name || "");
}

function colorCodeNode(node) {
  if (!colorCodeState.on || !node) return;
  if (node.color || node.bgcolor) return; // user (or a saved workflow) already chose
  const { color } = familyColors(node.type || node.comfyClass || "");
  node.color = color;
  node.properties = node.properties || {};
  node.properties.apnext_autocolor = true;
}

function uncolorCodeAll() {
  for (const n of app.graph?._nodes || []) {
    if (n.properties?.apnext_autocolor && n.color === familyColors(n.type || "").color) {
      delete n.color;
      delete n.properties.apnext_autocolor;
    }
  }
  app.canvas?.setDirty?.(true, true);
}

function colorCodeAll() {
  for (const n of app.graph?._nodes || []) if (n._apnextOurs) colorCodeNode(n);
  app.canvas?.setDirty?.(true, true);
}

function applyColorCode(on) {
  colorCodeState.on = on;
  if (on) colorCodeAll(); else uncolorCodeAll();
}

// ---------------------------------------------------------------------------
// Graphgen node look (theme on): 1px panel border on a rounded-lg box, a
// header that carries the node's hue as a faint tint + a stronger hued bottom
// border (BaseNode.svelte), white semibold title text, and port "tabs" -
// small vertical bars pinned to the node edge that extend outward when
// connected or hovered (layout.css .svelte-flow__handle) instead of circles.
// ---------------------------------------------------------------------------

const S_GGNODES = "APNext.Theme.GraphgenNodes";
const ggState = { on: false, shapePatched: false, slotPatched: false, fontPatched: false, savedTitleSize: null };

function hexToRgba(hex, a) {
  const c = parseColor(hex);
  return c ? `rgba(${c[0]},${c[1]},${c[2]},${a})` : hex;
}

// the hue a node's header carries: family swatch for APNext nodes, the node's
// own (botanical-mapped) title colour when coloured, else its first output's
// link colour, else the border colour
function nodeHue(node) {
  if (node.properties?.apnext_autocolor) return familySwatch(node.type || "");
  const LGC = window.LGraphCanvas;
  if (node.color) {
    const rgb = parseColor(node.color);
    if (rgb) {
      const [h, s] = rgbToHsl(...rgb);
      if (s >= 0.12) return swatchFor(h);
    }
  }
  const out = (node.outputs || []).find((o) => o && o.type && o.type !== "*");
  const col = out && LGC?.link_type_colors?.[out.type];
  return col || C.borderLight;
}

function patchNodeShape() {
  if (ggState.shapePatched) return;
  const LGC = window.LGraphCanvas || app.canvas?.constructor;
  if (!LGC?.prototype?.drawNodeShape) return;
  const orig = LGC.prototype.drawNodeShape;
  LGC.prototype.drawNodeShape = function (node, ctx, size, fgcolor, bgcolor, selected) {
    if (ggState.on && !ggState.slotPatched && node) patchSlots(node.inputs?.[0] || node.outputs?.[0]);
    const r = orig.apply(this, arguments);
    if (!ggState.on || !node || this.low_quality) return r;
    const LG = window.LiteGraph;
    const b = node.boundingRect;
    if (!b) return r;
    const x = b[0] - node.pos[0], y = b[1] - node.pos[1], w = b[2], h = b[3];
    const radius = LG?.ROUND_RADIUS ?? 8;
    const collapsed = !!node.flags?.collapsed;
    const titleH = LG?.NODE_TITLE_HEIGHT ?? 30;
    const hue = nodeHue(node);
    ctx.save();
    // header: hue tint + hued bottom border (skip when the title is hidden)
    const hideTitle = node.title_mode === LG?.TRANSPARENT_TITLE || node.title_mode === LG?.NO_TITLE;
    if (!collapsed && !hideTitle) {
      ctx.beginPath();
      ctx.roundRect(x, y, w, titleH, [radius, radius, 0, 0]);
      ctx.fillStyle = hexToRgba(hue, 0.12);
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(x, y + titleH - 0.5);
      ctx.lineTo(x + w, y + titleH - 0.5);
      ctx.lineWidth = 1;
      ctx.strokeStyle = mix(hue, C.border, 0.35);
      ctx.stroke();
    }
    // 1px panel border around the whole box (selection outline is drawn by LiteGraph)
    if (!selected) {
      ctx.beginPath();
      ctx.roundRect(x + 0.5, y + 0.5, w - 1, h - 1, collapsed ? radius : radius);
      ctx.lineWidth = 1;
      ctx.strokeStyle = C.border;
      ctx.stroke();
    }
    ctx.restore();
    return r;
  };
  ggState.shapePatched = true;
}

// bold white title: LiteGraph builds the title font from NODE_TEXT_SIZE + NODE_FONT,
// so the weight has to be injected through the canvas getter
function patchTitleFont() {
  if (ggState.fontPatched) return;
  const LGC = window.LGraphCanvas || app.canvas?.constructor;
  if (!LGC) return;
  let proto = LGC.prototype, desc = null;
  while (proto && !desc) { desc = Object.getOwnPropertyDescriptor(proto, "title_text_font"); if (!desc) proto = Object.getPrototypeOf(proto); }
  if (!desc?.get) return;
  const origGet = desc.get;
  Object.defineProperty(proto, "title_text_font", {
    configurable: true,
    get() {
      const f = origGet.call(this);
      return ggState.on ? `600 ${f}` : f;
    },
  });
  ggState.fontPatched = true;
}

// slot tabs: wrap NodeSlot.prototype.draw (class is not exported; reach it via
// any live slot instance)
function patchSlots(sampleSlot) {
  if (ggState.slotPatched || !sampleSlot) return;
  let proto = Object.getPrototypeOf(sampleSlot), owner = null;
  while (proto && !owner) { if (Object.prototype.hasOwnProperty.call(proto, "draw")) owner = proto; else proto = Object.getPrototypeOf(proto); }
  // the draw we want is NodeSlot's (parent of NodeInputSlot / NodeOutputSlot)
  if (!owner) return;
  const orig = owner.draw;
  owner.draw = function (ctx, opts = {}) {
    const r = orig.call(this, ctx, opts);
    if (!ggState.on || opts.lowQuality || !this.node) return r;
    if (this.isWidgetInputSlot && !this.isConnected) return r;
    const node = this.node;
    const u = this._centreOffset;
    if (!u) return r;
    const LG = window.LiteGraph;
    const isInput = (node.inputs || []).includes(this);
    const isOutput = !isInput && (node.outputs || []).includes(this);
    if (!isInput && !isOutput) return r;
    const body = node.bgcolor || LG?.NODE_DEFAULT_BGCOLOR || C.panel2;
    const color = (typeof this.renderingColor === "function" && opts.colorContext) ? this.renderingColor(opts.colorContext) : (this.color_on || C.accent);
    const w = opts.highlight ? 14 : this.isConnected ? 10 : 6;
    const W = node.size?.[0] ?? 0;
    ctx.save();
    // hide the stock circle
    ctx.fillStyle = body;
    ctx.fillRect(u[0] - 6.5, u[1] - 6.5, 13, 13);
    // the tab, pinned to the edge, extending outward
    const x0 = isInput ? 3 - w : W - 3;
    ctx.beginPath();
    ctx.roundRect(x0, u[1] - 7, w, 14, 2);
    ctx.fillStyle = color;
    ctx.fill();
    if (opts.highlight) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = hexToRgba(C.accent, 0.9);
      ctx.stroke();
    }
    ctx.restore();
    return r;
  };
  ggState.slotPatched = true;
}

function findSampleSlot() {
  for (const n of app.graph?._nodes || []) {
    const s = n.inputs?.[0] || n.outputs?.[0];
    if (s && typeof s.draw === "function") return s;
  }
  return null;
}

function applyGraphgenNodes(on) {
  const LG = window.LiteGraph;
  if (on) {
    patchNodeShape();
    patchTitleFont();
    patchSlots(findSampleSlot());
    if (LG && ggState.savedTitleSize === null) { ggState.savedTitleSize = LG.NODE_TEXT_SIZE; LG.NODE_TEXT_SIZE = 13; }
  } else if (LG && ggState.savedTitleSize !== null) {
    LG.NODE_TEXT_SIZE = ggState.savedTitleSize; ggState.savedTitleSize = null;
  }
  ggState.on = on;
  app.canvas?.setDirty?.(true, true);
}

// ---------------------------------------------------------------------------
// Mode plumbing
// ---------------------------------------------------------------------------

const S_WIRES = "APNext.Theme.Wires";

function themeOn(mode) {
  return mode === MODE_THEME || mode === MODE_BOTH;
}

function currentMode() {
  return settingGet(S_MODE) || MODE_OFF;
}

function currentWires() {
  // the gravity toggle wins; otherwise the wire-style combo (gravity there is legacy)
  if (settingGet(S_GRAVITY_ON) === true) return WIRE_GRAVITY;
  const w = settingGet(S_WIRES);
  if (w === WIRE_GRAVITY) return WIRE_DEFAULT;
  return w && WIRE_STYLES.includes(w) ? w : WIRE_DEFAULT;
}

async function migrateLegacyGravity() {
  // old combined mode values / gravity in the combo -> the explicit toggle
  const m = currentMode();
  const w = settingGet(S_WIRES);
  if (m === MODE_BOTH || m === MODE_GRAVITY || w === WIRE_GRAVITY) {
    await settingSet(S_GRAVITY_ON, true);
    if (w === WIRE_GRAVITY) await settingSet(S_WIRES, WIRE_DEFAULT);
    await settingSet(S_MODE, m === MODE_BOTH ? MODE_THEME : m === MODE_GRAVITY ? MODE_OFF : m);
  }
}

let applying = false;
async function applyAll() {
  if (applying) return;
  applying = true;
  try {
    const theme = themeOn(currentMode());
    applyChrome(theme);
    applyLiteGraphLook(theme);
    await applyPalette(theme);
    applyNodeRecolor(theme && settingGet(S_RECOLOR) !== false);
    applyGraphgenNodes(theme && settingGet(S_GGNODES) !== false);
    applyWires(currentWires());
    applyHighlight(settingGet(S_HIGHLIGHT) !== false);
    applySparks(settingGet(S_SPARKS) !== false);
    applyColorCode(settingGet(S_COLORCODE) !== false);
  } catch (e) {
    console.warn("[APNext theme] apply failed:", e);
  } finally {
    applying = false;
  }
}

async function setTheme(on) {
  await migrateLegacyGravity();
  await settingSet(S_MODE, on ? MODE_THEME : MODE_OFF);
  await applyAll();
}
async function setWires(style) {
  await migrateLegacyGravity();
  if (style === WIRE_GRAVITY) return setGravity(true);
  await settingSet(S_WIRES, style);
  await applyAll();
}
async function setGravity(on) {
  await settingSet(S_GRAVITY_ON, !!on);
  await applyAll();
}
function toggleTheme() {
  return setTheme(!themeOn(currentMode()));
}
function toggleGravity() {
  return setGravity(settingGet(S_GRAVITY_ON) !== true);
}
function cycleWires() {
  const cur = settingGet(S_WIRES) || WIRE_DEFAULT;
  const i = WIRE_COMBO.indexOf(cur);
  return setWires(WIRE_COMBO[(i + 1) % WIRE_COMBO.length]);
}
const WIRE_LABELS = {
  [WIRE_DEFAULT]: "ComfyUI default",
  [WIRE_BEZIER]: "Bezier (spline)",
  [WIRE_SMOOTHSTEP]: "Smooth step",
  [WIRE_STEP]: "Step",
  [WIRE_STRAIGHT]: "Straight",
  [WIRE_CABLE]: "Cable (sag + wobble)",
  [WIRE_GRAVITY]: "Gravity (hanging rope)",
};

app.registerExtension({
  name: "apnext.theme.graphgen",

  settings: [
    {
      id: S_MODE,
      category: ["APNext", "Graphgen theme", "Theme"],
      name: "Graphgen theme",
      tooltip:
        "Restyle ComfyUI like graphgen: Dark Botanical palette, IBM Plex Sans, rounder nodes. " +
        "Off restores the previous palette, font and radius. Wire style is a separate setting below.",
      type: "combo",
      options: [
        { text: "Off", value: MODE_OFF },
        { text: "On", value: MODE_THEME },
      ],
      defaultValue: MODE_OFF,
      onChange: (value, old) => {
        if (old === undefined) return; // initial load handled in setup()
        applyAll();
      },
    },
    {
      id: S_WIRES,
      category: ["APNext", "Graphgen theme", "Wire style"],
      name: "Wire style",
      tooltip:
        "How links are drawn - graphgen's edge styles: Bezier, Smooth step, Step, Straight, " +
        "Cable (a springy wire that sags and wobbles after a drag) or Gravity (a hanging verlet rope). " +
        "ComfyUI default leaves the stock renderer alone.",
      type: "combo",
      options: WIRE_COMBO.map((v) => ({ text: WIRE_LABELS[v], value: v })),
      defaultValue: WIRE_DEFAULT,
      onChange: (value, old) => {
        if (old === undefined) return;
        applyAll();
      },
    },
    {
      id: S_GRAVITY_ON,
      category: ["APNext", "Graphgen theme", "Gravity wires"],
      name: "Gravity wires (hanging rope physics)",
      tooltip:
        "ON: every link is a verlet rope that hangs and swings (overrides the wire style above). " +
        "OFF: the physics is fully stopped - no simulation, no extra redraws. Off by default; " +
        "turn it off if the canvas feels slow on a big graph.",
      type: "boolean",
      defaultValue: false,
      onChange: (value, old) => {
        if (old === undefined) return;
        applyAll();
      },
    },
    {
      id: S_RECOLOR,
      category: ["APNext", "Graphgen theme", "Recolour nodes"],
      name: "Recolour coloured nodes & groups",
      tooltip:
        "While the theme is on, nodes and groups that carry their own colour (right-click → Colors, " +
        "or packs that pre-colour their nodes) are drawn in the nearest botanical hue, and the node-colour " +
        "menu swatches become botanical. Stored colours are untouched; theme off shows them again.",
      type: "boolean",
      defaultValue: true,
      onChange: (value, old) => {
        if (old === undefined) return;
        applyAll();
      },
    },
    {
      id: S_GGNODES,
      category: ["APNext", "Graphgen theme", "Graphgen nodes"],
      name: "Graphgen node look (header, border, slot tabs)",
      tooltip: "While the theme is on: rounded-lg box with a 1px panel border, a header carrying the node's hue as a faint tint plus a hued bottom border, white semibold title, and port tabs pinned to the node edge that extend outward when connected or hovered (instead of circles).",
      type: "boolean",
      defaultValue: true,
      onChange: (value, old) => { if (old !== undefined) applyAll(); },
    },
    {
      id: S_HIGHLIGHT,
      category: ["APNext", "Canvas helpers", "Highlight drop targets"],
      name: "Highlight compatible slots while dragging a link",
      tooltip: "While you drag a link, every slot on every node that can take it pulses with a ring (pink ring = already connected, would be replaced). Works from outputs and from inputs.",
      type: "boolean",
      defaultValue: true,
      onChange: (value, old) => { if (old !== undefined) applyAll(); },
    },
    {
      id: S_SPARKS,
      category: ["APNext", "Canvas helpers", "Connect sparks"],
      name: "Spark burst when a link connects",
      tooltip: "A small particle burst in the link's colour at the input when a connection is made (graphgen's Sparks).",
      type: "boolean",
      defaultValue: true,
      onChange: (value, old) => { if (old !== undefined) applyAll(); },
    },
    {
      id: S_COLORCODE,
      category: ["APNext", "Canvas helpers", "Colour-code APNext nodes"],
      name: "Colour-code APNext nodes by family",
      tooltip: "Give every APNext node a family colour in the palette hues - sage writers, rose Characters, pink LLM Backend, slate previews, gold scene utilities, mauve context generators, teal vision - so it is obvious what plugs where. Only nodes you have not coloured yourself; turning it off removes the automatic colours again.",
      type: "boolean",
      defaultValue: true,
      onChange: (value, old) => { if (old !== undefined) applyAll(); },
    },
    {
      id: S_SLACK,
      category: ["APNext", "Graphgen theme", "Wire slack"],
      name: "Gravity wire slack",
      tooltip: "Rope length as a multiple of the straight distance. 1.0 = taut, 1.16 = graphgen default, 1.5 = very droopy.",
      type: "slider",
      attrs: { min: 1.0, max: 1.6, step: 0.01 },
      defaultValue: 1.16,
      onChange: () => readWireConfig(),
    },
    {
      id: S_GRAVITY,
      category: ["APNext", "Graphgen theme", "Wire gravity"],
      name: "Gravity wire weight",
      tooltip: "How hard the wires fall (px/frame²). 0 = weightless, 0.55 = graphgen default.",
      type: "slider",
      attrs: { min: 0, max: 1.5, step: 0.05 },
      defaultValue: 0.55,
      onChange: () => readWireConfig(),
    },
    {
      id: S_SEGMENTS,
      category: ["APNext", "Graphgen theme", "Wire segments"],
      name: "Gravity wire segments",
      tooltip: "Chain resolution per wire. More = smoother and heavier to simulate. 22 = graphgen default.",
      type: "slider",
      attrs: { min: 6, max: 40, step: 1 },
      defaultValue: 22,
      onChange: () => readWireConfig(),
    },
    {
      id: S_PREV,
      category: ["APNext", "Graphgen theme", "Previous palette"],
      name: "Palette to restore when the theme is turned off",
      type: "hidden",
      defaultValue: "dark",
    },
  ],

  commands: [
    { id: "APNext.Theme.Toggle", label: "APNext: toggle Graphgen theme", icon: "pi pi-palette", function: () => toggleTheme() },
    { id: "APNext.Theme.CycleWires", label: "APNext: next wire style", icon: "pi pi-link", function: () => cycleWires() },
    { id: "APNext.Theme.ToggleGravity", label: "APNext: toggle gravity wires", icon: "pi pi-sort-amount-down", function: () => toggleGravity() },
    { id: "APNext.Theme.WiresDefault", label: "APNext: wires - ComfyUI default", function: () => setWires(WIRE_DEFAULT) },
    { id: "APNext.Theme.WiresBezier", label: "APNext: wires - Bezier", function: () => setWires(WIRE_BEZIER) },
    { id: "APNext.Theme.WiresSmoothStep", label: "APNext: wires - Smooth step", function: () => setWires(WIRE_SMOOTHSTEP) },
    { id: "APNext.Theme.WiresStep", label: "APNext: wires - Step", function: () => setWires(WIRE_STEP) },
    { id: "APNext.Theme.WiresStraight", label: "APNext: wires - Straight", function: () => setWires(WIRE_STRAIGHT) },
    { id: "APNext.Theme.WiresCable", label: "APNext: wires - Cable (sag + wobble)", function: () => setWires(WIRE_CABLE) },
    { id: "APNext.Theme.WiresGravity", label: "APNext: wires - Gravity (rope) on/off", function: () => toggleGravity() },
    { id: "APNext.Theme.Off", label: "APNext: theme off (stock ComfyUI)", icon: "pi pi-times", function: async () => { await setGravity(false); await setWires(WIRE_DEFAULT); await setTheme(false); } },
  ],
  menuCommands: [
    { path: ["APNext"], commands: ["APNext.Theme.Toggle", "APNext.Theme.CycleWires", "APNext.Theme.ToggleGravity", "APNext.Theme.Off"] },
    {
      path: ["APNext", "Wire style"],
      commands: [
        "APNext.Theme.WiresDefault", "APNext.Theme.WiresBezier", "APNext.Theme.WiresSmoothStep",
        "APNext.Theme.WiresStep", "APNext.Theme.WiresStraight", "APNext.Theme.WiresCable", "APNext.Theme.WiresGravity",
      ],
    },
  ],

  getCanvasMenuItems() {
    const theme = themeOn(currentMode());
    const wires = currentWires();
    return [
      null,
      { content: `APNext: Graphgen theme ${theme ? "✓" : ""}`, callback: () => toggleTheme() },
      { content: `APNext: gravity wires ${settingGet(S_GRAVITY_ON) === true ? "✓" : ""}`, callback: () => toggleGravity() },
      {
        content: `APNext: wire style (${WIRE_LABELS[wires] || wires})`,
        has_submenu: true,
        submenu: {
          options: WIRE_COMBO.map((v) => ({
            content: `${WIRE_LABELS[v]} ${v === (settingGet(S_WIRES) || WIRE_DEFAULT) ? "✓" : ""}`,
            callback: () => setWires(v),
          })),
        },
      },
    ];
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!isOurs(nodeData)) return;
    nodeType.prototype._apnextOurs = true;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      colorCodeNode(this);
      return r;
    };
  },

  async setup() {
    await migrateLegacyGravity();
    readWireConfig();
    // the palette store and canvas exist by now; apply once the first frame is in
    setTimeout(() => applyAll(), 50);
  },
});
