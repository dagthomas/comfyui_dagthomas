// APNext "Graphgen" theme for ComfyUI
//
// Optional look-and-feel borrowed from graphgen (X:/KODE/graphgen): the "Dark
// Botanical" palette (warm near-black, tan accent, dusty-pink highlight, muted
// botanical port colours), IBM Plex Sans on the canvas and the UI chrome,
// rounder nodes, and alternative wire styles. All wire styles are static -
// the old physics wires (verlet rope / spring cable) were removed because the
// constant redraws hurt UI responsiveness.
//
// Everything is opt-in and reversible:
//   Settings → APNext → Graphgen theme (On/Off) and Wire style (ComfyUI default /
//              Bezier / Smooth step / Step / Straight / Cable)
//   top menu  → APNext → toggle theme / next wire style / Wire style submenu
//   canvas right-click → APNext theme + wire style entries
// Turning it off restores the palette that was active before, the default
// font and radius, and stock bezier links. The palette is installed as a normal
// custom ComfyUI colour palette ("APNext Graphgen"), so it also shows up in the
// regular palette picker.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const PALETTE_ID = "apnext_graphgen";
const S_MODE = "APNext.Theme.Mode";
const S_PREV = "APNext.Theme.PreviousPalette";

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
    g.fillStyle = "#33323a";
    g.beginPath();
    g.arc(11, 11, 1.1, 0, Math.PI * 2);
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
        CLEAR_BACKGROUND_COLOR: C.bg,
        NODE_TITLE_COLOR: "#ffffff",
        NODE_SELECTED_TITLE_COLOR: "#ffffff",
        NODE_TEXT_COLOR: C.textDim,
        NODE_TEXT_HIGHLIGHT_COLOR: C.text,
        NODE_DEFAULT_COLOR: C.panel,
        NODE_DEFAULT_BGCOLOR: C.panel,
        NODE_DEFAULT_BOXCOLOR: C.borderLight,
        NODE_DEFAULT_SHAPE: 4, // CARD: rounded header corners, square body
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
:is(html, body).apnext-graphgen {
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
/* ---- ComfyUI "Nodes 2.0" (Vue-rendered nodes): dark body, coloured header, white bold title,
        square corners except the header, square inputs ---- */
:is(html, body).apnext-graphgen [data-testid="node-inner-wrapper"] {
  --component-node-background: ${C.panel} !important;
  --node-component-header-surface: ${C.panel};
  border-radius: 8px 8px 0 0 !important;
  box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}
:is(html, body).apnext-graphgen [data-testid="node-inner-wrapper"] > *:not([data-testid^="node-header"]) { border-radius: 0 !important; }
:is(html, body).apnext-graphgen [data-testid^="node-header-"],
:is(html, body).apnext-graphgen [data-testid^="node-header-"] span,
:is(html, body).apnext-graphgen [data-testid^="node-header-"] input,
:is(html, body).apnext-graphgen [data-testid^="node-header-"] [contenteditable] {
  color: #ffffff !important; font-weight: 700 !important; font-family: ${FONT_SANS};
}
:is(html, body).apnext-graphgen [data-testid^="node-header-"] { position: relative; }
:is(html, body).apnext-graphgen [data-testid^="node-header-"]::after {
  content: ""; position: absolute; left: 0; right: 0; bottom: -1px; height: 2px; pointer-events: none;
  background: rgba(212,165,116,0.85);
}
@supports (backdrop-filter: brightness(1)) {
  :is(html, body).apnext-graphgen [data-testid^="node-header-"]::after {
    background: rgba(255,255,255,0.18);
    backdrop-filter: brightness(2.2) saturate(1.8);
  }
}
:is(html, body).apnext-graphgen [data-testid="node-inner-wrapper"] input:not([type="checkbox"]):not([type="range"]),
:is(html, body).apnext-graphgen [data-testid="node-inner-wrapper"] textarea,
:is(html, body).apnext-graphgen [data-testid="node-inner-wrapper"] select,
:is(html, body).apnext-graphgen [data-testid="node-inner-wrapper"] .p-inputtext,
:is(html, body).apnext-graphgen [data-testid="node-inner-wrapper"] .p-select,
:is(html, body).apnext-graphgen [data-testid="node-inner-wrapper"] .p-inputnumber,
:is(html, body).apnext-graphgen [data-testid="node-inner-wrapper"] .p-textarea,
:is(html, body).apnext-graphgen [data-testid="node-inner-wrapper"] [class*="rounded-lg"]:not([class*="rounded-full"]),
:is(html, body).apnext-graphgen [data-testid="node-inner-wrapper"] [class*="rounded-md"]:not([class*="rounded-full"]) {
  border-radius: 2px !important;
}
:is(html, body).apnext-graphgen body,
:is(html, body).apnext-graphgen .p-component,
:is(html, body).apnext-graphgen .comfy-menu,
:is(html, body).apnext-graphgen .litegraph,
:is(html, body).apnext-graphgen .litegraph * {
  font-family: ${FONT_SANS};
  -webkit-font-smoothing: antialiased;
}
:is(html, body).apnext-graphgen code, :is(html, body).apnext-graphgen pre, :is(html, body).apnext-graphgen textarea.comfy-multiline-input,
:is(html, body).apnext-graphgen .apnext-h3-vars { font-family: ${FONT_MONO}; }
:is(html, body).apnext-graphgen .comfy-multiline-input,
:is(html, body).apnext-graphgen .comfy-multiline-input textarea {
  background: ${C.panel} !important; color: ${C.text} !important;
  border: 1px solid ${C.border} !important; border-radius: 2px !important;
}
:is(html, body).apnext-graphgen .comfy-multiline-input:focus-within { border-color: ${C.accent} !important; }
:is(html, body).apnext-graphgen .p-panel, :is(html, body).apnext-graphgen .side-tool-bar-container,
:is(html, body).apnext-graphgen .comfyui-menu, :is(html, body).apnext-graphgen .actionbar,
:is(html, body).apnext-graphgen .p-dialog, :is(html, body).apnext-graphgen .p-popover, :is(html, body).apnext-graphgen .p-menu,
:is(html, body).apnext-graphgen .p-tieredmenu, :is(html, body).apnext-graphgen .p-contextmenu, :is(html, body).apnext-graphgen .litecontextmenu {
  background: ${C.panel} !important; color: ${C.text} !important; border-color: ${C.border} !important;
}
:is(html, body).apnext-graphgen .litecontextmenu .litemenu-entry { color: ${C.text} !important; }
:is(html, body).apnext-graphgen .litecontextmenu .litemenu-entry:hover,
:is(html, body).apnext-graphgen .litemenu-entry.submenu:hover { background: ${C.border} !important; color: ${C.accent} !important; }
:is(html, body).apnext-graphgen .p-button, :is(html, body).apnext-graphgen .comfyui-button, :is(html, body).apnext-graphgen .p-inputtext,
:is(html, body).apnext-graphgen .p-select, :is(html, body).apnext-graphgen .p-textarea, :is(html, body).apnext-graphgen .p-dialog,
:is(html, body).apnext-graphgen .p-panel, :is(html, body).apnext-graphgen .p-popover, :is(html, body).apnext-graphgen .apnext-h3,
:is(html, body).apnext-graphgen .apnext-h3-vars, :is(html, body).apnext-graphgen .apnext-h3-ac { border-radius: 2px !important; }
:is(html, body).apnext-graphgen .p-button.p-button-primary, :is(html, body).apnext-graphgen .comfyui-queue-button .p-button {
  background: ${C.accent} !important; border-color: ${C.accent} !important; color: #1a1510 !important;
}
:is(html, body).apnext-graphgen .p-button.p-button-primary:hover { background: ${C.accent2} !important; border-color: ${C.accent2} !important; }
:is(html, body).apnext-graphgen .p-toast-message, :is(html, body).apnext-graphgen .p-inputtext, :is(html, body).apnext-graphgen .p-select,
:is(html, body).apnext-graphgen .p-inputnumber-input, :is(html, body).apnext-graphgen .p-textarea {
  background: ${C.panel2} !important; color: ${C.text} !important; border-color: ${C.border} !important;
}
:is(html, body).apnext-graphgen h1, :is(html, body).apnext-graphgen h2, :is(html, body).apnext-graphgen .p-dialog-title { font-family: ${FONT_DISPLAY}; letter-spacing: 0.01em; }
:is(html, body).apnext-graphgen h1, :is(html, body).apnext-graphgen h2, :is(html, body).apnext-graphgen h3, :is(html, body).apnext-graphgen h4,
:is(html, body).apnext-graphgen .p-dialog-title, :is(html, body).apnext-graphgen .p-panel-header, :is(html, body).apnext-graphgen .p-accordion-header,
:is(html, body).apnext-graphgen .p-tabview-nav-link, :is(html, body).apnext-graphgen .p-tab, :is(html, body).apnext-graphgen .side-bar-panel .title,
:is(html, body).apnext-graphgen .comfyui-menu .p-menubar-item-label, :is(html, body).apnext-graphgen .apnext-h3-title,
:is(html, body).apnext-graphgen .apnext-h3 .h3-header, :is(html, body).apnext-graphgen .node-title, :is(html, body).apnext-graphgen .lg-node-title,
:is(html, body).apnext-graphgen [class*="node-header"] [class*="title"], :is(html, body).apnext-graphgen [class*="NodeHeader"] {
  color: #ffffff !important; font-weight: 700 !important;
}
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

const chrome = { on: false, observer: null };

function assertChromeClass() {
  if (!chrome.on) return;
  const html = document.documentElement;
  if (!html.classList.contains("apnext-graphgen")) html.classList.add("apnext-graphgen");
  if (document.body && !document.body.classList.contains("apnext-graphgen")) document.body.classList.add("apnext-graphgen");
  if (!document.getElementById(STYLE_ID)) {
    const st = document.createElement("style");
    st.id = STYLE_ID;
    st.textContent = CSS;
    document.head.appendChild(st);
  }
}

function applyChrome(on) {
  chrome.on = on;
  if (on) {
    assertChromeClass();
    ensureFonts();
    if (!chrome.observer && typeof MutationObserver !== "undefined") {
      // ComfyUI's theme code may rewrite <html class="..."> wholesale; put ours back
      chrome.observer = new MutationObserver(() => assertChromeClass());
      chrome.observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
      if (document.body) chrome.observer.observe(document.body, { attributes: true, attributeFilter: ["class"] });
    }
  } else {
    document.documentElement.classList.remove("apnext-graphgen");
    document.body?.classList.remove("apnext-graphgen");
    if (chrome.observer) { chrome.observer.disconnect(); chrome.observer = null; }
  }
}

// ---------------------------------------------------------------------------
// LiteGraph look: font + radius (the colours come through the palette)
// ---------------------------------------------------------------------------

const saved = { font: null, radius: null, applied: false };

// The node defaults every node without its own colour uses. Re-asserted each
// background frame while the theme is on (cheap equality checks), so nothing a
// later palette load or another extension sets can drift the look.
const THEME_LG = () => ({
  NODE_TITLE_COLOR: "#ffffff",
  NODE_SELECTED_TITLE_COLOR: "#ffffff",
  NODE_TEXT_COLOR: C.textDim,
  NODE_TEXT_HIGHLIGHT_COLOR: C.text,
  NODE_DEFAULT_COLOR: C.panel,
  NODE_DEFAULT_BGCOLOR: C.panel,
  NODE_DEFAULT_BOXCOLOR: C.borderLight,
  NODE_DEFAULT_SHAPE: 4, // CARD: only the header corners are rounded
  NODE_BOX_OUTLINE_COLOR: C.accent,
  WIDGET_BGCOLOR: C.panel2,
  WIDGET_OUTLINE_COLOR: C.border,
  WIDGET_TEXT_COLOR: C.text,
  WIDGET_SECONDARY_TEXT_COLOR: C.textDim,
  LINK_COLOR: C.accent,
  CONNECTING_LINK_COLOR: C.accent2,
});
const VUE_NODE_VARS = {
  "--component-node-background": C.panel,
  "--node-component-header-surface": C.panel,
  "--node-component-header": "#ffffff",
  "--node-component-header-icon": C.borderLight,
  "--component-node-widget-background": C.panel2,
  "--component-node-foreground": C.text,
  "--component-node-border": C.border,
};

function assertThemeColors() {
  const LG = window.LiteGraph;
  if (!LG) return;
  for (const [k, v] of Object.entries(THEME_LG())) if (LG[k] !== v) LG[k] = v;
  const cv = app.canvas;
  if (cv) {
    if (cv.node_title_color !== "#ffffff") cv.node_title_color = "#ffffff";
    if (cv.default_link_color !== C.accent) cv.default_link_color = C.accent;
  }
}

function applyVueNodeVars(on) {
  const st = document.documentElement?.style;
  if (!st) return;
  for (const [k, v] of Object.entries(VUE_NODE_VARS)) {
    if (on) st.setProperty(k, v); else st.removeProperty(k);
  }
}

function applyLiteGraphLook(on) {
  applyVueNodeVars(on);
  const LG = window.LiteGraph;
  if (!LG) return;
  if (on && !saved.applied) {
    saved.font = LG.NODE_FONT;
    saved.radius = LG.ROUND_RADIUS;
    saved.titleColor = LG.NODE_TITLE_COLOR;
    saved.selectedTitleColor = LG.NODE_SELECTED_TITLE_COLOR;
    LG.NODE_FONT = FONT_SANS;
    LG.ROUND_RADIUS = 8; // graphgen rounded-lg
    saved.applied = true;
  } else if (!on && saved.applied) {
    LG.NODE_FONT = saved.font ?? "Inter";
    LG.ROUND_RADIUS = saved.radius ?? 8;
    saved.applied = false;
  }
  // headers: bold + white on every node while the theme is on (the palette sets the
  // colour too, but a later palette load must not dim it)
  patchTitleFont();
  ggState.boldTitles = on;
  if (on) assertThemeColors();
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
// Cable wires - a static drooping quadratic per link (midpoint + sag). No
// physics, no per-frame simulation: the old verlet rope / spring wobble cost
// UI responsiveness and is gone.
// ---------------------------------------------------------------------------

const cable = {
  cfg: { sagMax: 70, sagRatio: 0.18 },
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
const WIRE_GRAVITY = "gravity"; // legacy value only - migrated to default on load
const WIRE_STYLES = [WIRE_DEFAULT, WIRE_BEZIER, WIRE_SMOOTHSTEP, WIRE_STEP, WIRE_STRAIGHT, WIRE_CABLE];
const WIRE_COMBO = WIRE_STYLES;
const themeState = { on: false };

const wireState = { style: WIRE_DEFAULT, patched: false };

function isCustomWire(style) {
  return style === WIRE_SMOOTHSTEP || style === WIRE_STEP || style === WIRE_STRAIGHT || style === WIRE_CABLE;
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
// behind the source. radius 0 => hard steps. `lane` shifts this wire's vertical
// run left/right so parallel wires ride separate lanes instead of stacking on
// the shared mid-corridor and crossing each other.
function stepPoints(a, b, radius, lane = 0) {
  const off = 20;
  let corners;
  if (b[0] >= a[0] + 2 * off) {
    const mx = Math.min(Math.max((a[0] + b[0]) / 2 + lane, a[0] + off), b[0] - off);
    corners = [{ x: mx, y: a[1] }, { x: mx, y: b[1] }];
  } else {
    const bump = Math.abs(lane);
    const my = (a[1] + b[1]) / 2 + lane;
    corners = [
      { x: a[0] + off + bump, y: a[1] }, { x: a[0] + off + bump, y: my },
      { x: b[0] - off - bump, y: my }, { x: b[0] - off - bump, y: b[1] },
    ];
  }
  const all = [{ x: a[0], y: a[1] }, ...corners, { x: b[0], y: b[1] }];
  if (!radius) return all;
  const pts = [all[0]];
  for (let i = 1; i < all.length - 1; i++) pts.push(...roundedCorner(all[i - 1], all[i], all[i + 1], radius));
  pts.push(all[all.length - 1]);
  return pts;
}

// Which lane a step wire rides in - PCB-style trace bundling. Every step wire
// registers the corridor its vertical run wants (x position + y span); wires
// whose corridors coincide are spread onto evenly spaced parallel lanes,
// ordered by height, like traces etched next to each other instead of on top
// of each other. Grouping uses the previous frame's corridors (the canvas
// redraws constantly, so the one-frame lag is invisible).
const LANE_SPACING = 14;
const laneBuf = { tick: -1, prev: new Map(), cur: new Map() };

function stepLane(canvas, link, a, b) {
  if (!link || !a || !b) return 0;
  const tick =
    canvas?.frame ??
    Math.floor((typeof performance !== "undefined" ? performance.now() : Date.now()) / 50);
  if (tick !== laneBuf.tick) {
    laneBuf.prev = laneBuf.cur;
    laneBuf.cur = new Map();
    laneBuf.tick = tick;
  }
  const off = 20;
  const mx = b[0] >= a[0] + 2 * off ? (a[0] + b[0]) / 2 : a[0] + off;
  const y0 = Math.min(a[1], b[1]);
  const y1 = Math.max(a[1], b[1]);
  laneBuf.cur.set(link.id, { mx, y0, y1 });

  const mates = [];
  for (const [id, r] of laneBuf.prev) {
    if (Math.abs(r.mx - mx) < LANE_SPACING * 3 && r.y0 < y1 && r.y1 > y0) mates.push([id, r]);
  }
  if (mates.length <= 1) return 0;
  mates.sort(
    (p, q) => p[1].y0 + p[1].y1 - (q[1].y0 + q[1].y1) || p[0] - q[0],
  );
  const idx = mates.findIndex((p) => p[0] === link.id);
  if (idx < 0) return 0;
  return (idx - (mates.length - 1) / 2) * LANE_SPACING;
}

function cablePoints(a, b) {
  const span = Math.hypot(b[0] - a[0], b[1] - a[1]);
  const sag = Math.min(cable.cfg.sagMax, span * cable.cfg.sagRatio);
  const p = { x: (a[0] + b[0]) / 2, y: (a[1] + b[1]) / 2 + sag };
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

function wirePoints(style, a, b, lane = 0) {
  switch (style) {
    case WIRE_CABLE: return cablePoints(a, b);
    case WIRE_SMOOTHSTEP: return stepPoints(a, b, 8, lane);
    case WIRE_STEP: return stepPoints(a, b, 0, lane);
    case WIRE_STRAIGHT: return [{ x: a[0], y: a[1] }, { x: b[0], y: b[1] }];
    default: return null;
  }
}

function tracePoly(ctx, pts) {
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
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
  const styledRender = function (ctx, a, b, link, skip_border, flow, color, start_dir, end_dir, opts) {
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
    const lane =
      style === WIRE_STEP || style === WIRE_SMOOTHSTEP ? stepLane(this, link, a, b) : 0;
    const pts = wirePoints(style, a, b, lane);
    if (!pts) return origRender.apply(this, arguments);
    const col =
      color || link.color || (LGC.link_type_colors && LGC.link_type_colors[link.type]) || this.default_link_color;

    ctx.save();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    const width = this.connections_width || 3;
    if (!skip_border && this.render_connections_border && (this.ds?.scale ?? 1) > 0.6) {
      ctx.lineWidth = width + 4;
      ctx.strokeStyle = "rgba(0,0,0,0.5)";
      tracePoly(ctx, pts);
      ctx.stroke();
    }
    ctx.lineWidth = width;
    ctx.strokeStyle = col;
    tracePoly(ctx, pts);
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

  // outermost pass: links that touch a selected node draw brighter (saturated
  // version of their own colour) at their normal width, wrapped in a crisp
  // border of the same hue but lighter (no blur/glow), and with flow dots
  // travelling in the data direction (output → input)
  LGC.prototype.renderLink = function (ctx, a, b, link, skip_border, flow, color, start_dir, end_dir, opts) {
    const boost = selGlowColor(this, link, color);
    if (!boost) return styledRender.apply(this, arguments);
    const savedW = this.connections_width;
    const baseW = savedW || 3;
    ctx.save();
    try {
      // under-stroke first: a wider line in a lighter version of the boosted
      // colour — reads as a bright border around the wire instead of a glow
      this.connections_width = baseW + 3;
      styledRender.call(this, ctx, a, b, link, true, false, haloColorOf(boost), start_dir, end_dir, opts);
      // then the wire itself on top at its normal width, brighter + saturated
      this.connections_width = baseW;
      // suppress the uniform built-in flow dots; the varied ovals below
      // replace them (different speeds and sizes per link, shell + hot core)
      const r = styledRender.call(this, ctx, a, b, link, true,
        selGlow.flow ? false : flow, boost, start_dir, end_dir, opts);
      if (selGlow.flow) {
        try { drawFlowSpheres(this, ctx, a, b, link, boost, start_dir, end_dir, opts); }
        catch (e) { /* never break the link render */ }
      }
      return r;
    } finally {
      this.connections_width = savedW;
      ctx.restore();
    }
  };

  // per-frame upkeep on the background-layer draw
  const origBack = LGC.prototype.drawBackCanvas;
  if (origBack) {
    LGC.prototype.drawBackCanvas = function () {
      if (themeState.on) { assertThemeColors(); assertChromeClass(); }
      if (sparks.on) { try { watchLinks(this); } catch (err) { console.warn("[APNext sparks] watch failed:", err); } }
      if (selGlow.on || runGlow.on) { try { selGlowFrame(this); } catch (err) { /* never block the draw */ } }
      return origBack.apply(this, arguments);
    };
  }
  wireState.patched = true;
}

function applyWires(style) {
  if (!WIRE_STYLES.includes(style)) style = WIRE_DEFAULT;
  if (style !== WIRE_DEFAULT) patchLinkRenderer();
  wireState.style = wireState.patched ? style : WIRE_DEFAULT;
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

// While the graphgen node look is on, groups are repainted node-style ON TOP
// of the stock rendering: an opaque dark panel (no see-through fill), a header
// band carrying the group's hue as a solid tint with a hued bottom border, a
// 1px panel border and a bold white title - the same recipe as the nodes.
function overdrawGroups(canvas, ctx) {
  const LG = window.LiteGraph;
  const groups = canvas.graph?._groups || [];
  if (!groups.length) return;
  const radius = LG?.ROUND_RADIUS ?? 8;
  ctx.save();
  ctx.globalAlpha = 1;
  for (const g of groups) {
    const x = g.pos?.[0] ?? g._pos?.[0], y = g.pos?.[1] ?? g._pos?.[1];
    const w = g.size?.[0] ?? g._size?.[0], h = g.size?.[1] ?? g._size?.[1];
    if (![x, y, w, h].every(Number.isFinite) || w <= 0 || h <= 0) continue;
    const hue = (recolorState.on ? botanical(g.color, "group") : g.color) || C.borderLight;
    const fs = g.font_size || LG?.DEFAULT_GROUP_FONT || 24;
    const th = Math.max(24, Math.round(fs * 1.5));
    // opaque dark body, slightly darker than the nodes so they stay readable
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, radius);
    ctx.fillStyle = mix(C.panel, C.bg, 0.45);
    ctx.fill();
    // header: solid hue tint + hued bottom border, like the node headers
    ctx.beginPath();
    ctx.roundRect(x, y, w, Math.min(th, h), [radius, radius, 0, 0]);
    ctx.fillStyle = mix(hue, C.panel, 0.75);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(x, y + Math.min(th, h) - 1);
    ctx.lineTo(x + w, y + Math.min(th, h) - 1);
    ctx.lineWidth = 2;
    ctx.strokeStyle = mix(hue, "#ffffff", 0.22);
    ctx.stroke();
    // 1px border around the whole box (accent outline when selected)
    const selected = canvas.selectedItems instanceof Set && canvas.selectedItems.has(g);
    ctx.beginPath();
    ctx.roundRect(x + 0.5, y + 0.5, w - 1, h - 1, radius);
    ctx.lineWidth = selected ? 2 : 1;
    ctx.strokeStyle = selected ? C.accent : C.border;
    ctx.stroke();
    // resize handle (the stock one is painted over by the body)
    ctx.beginPath();
    ctx.moveTo(x + w - 12, y + h);
    ctx.lineTo(x + w, y + h - 12);
    ctx.lineTo(x + w, y + h);
    ctx.closePath();
    ctx.fillStyle = C.borderLight;
    ctx.fill();
    // bold white title, node-style
    ctx.font = `700 ${fs}px ${FONT_SANS}`;
    ctx.fillStyle = C.text;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(g.title || "Group", x + 10, y + Math.min(th, h) / 2 + 1);
  }
  ctx.restore();
}

function patchNodeRecolor() {
  if (recolorState.patched) return;
  const LGC = window.LGraphCanvas || app.canvas?.constructor;
  if (!LGC?.prototype?.drawNode) return;
  const origDrawNode = LGC.prototype.drawNode;
  LGC.prototype.drawNode = function (node, ctx) {
    if (!recolorState.on || !node || (!node.color && !node.bgcolor)) {
      return origDrawNode.apply(this, arguments);
    }
    const c = node.color, bg = node.bgcolor;
    try {
      // header keeps (a botanical version of) its hue; the body is always the dark panel
      if (c && !node.properties?.apnext_autocolor) node.color = botanical(c, "title");
      if (bg) node.bgcolor = C.panel;
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
        const r = origDrawGroups.apply(this, arguments);
        if (ggState.on && !this.low_quality) {
          try { overdrawGroups(this, ctx); } catch (err) { /* never block the draw */ }
        }
        return r;
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
// (matching type, other node) is drawn scaled up a little with its NAME in a
// readable label pill - no pulsing, just a clear picture of where you can
// drop. The label size is zoom-aware: it never falls below a readable
// screen-space size, so zoomed-out targets stay legible. Works for
// output→input and input→output drags.
// ---------------------------------------------------------------------------

const S_HIGHLIGHT = "APNext.Theme.HighlightTargets";
const S_SPARKS = "APNext.Theme.Sparks";
const hlState = { on: true, patched: false };

function dragSources(canvas) {
  // [{type, node, toInputs}] for every link currently being dragged
  const out = [];
  const lc = canvas?.linkConnector;
  if (lc?.isConnecting) {
    const toInputs = lc.state?.connectingTo === "input";
    for (const rl of lc.renderLinks || []) {
      const slot = rl.fromSlot;
      if (!slot) continue;
      out.push({ type: slot.type, node: rl.node, toInputs });
    }
  } else if (Array.isArray(canvas?.connecting_links) && canvas.connecting_links.length) {
    for (const cl of canvas.connecting_links) {
      const slot = cl.output || cl.input;
      if (!slot) continue;
      out.push({ type: slot.type, node: cl.node, toInputs: !!cl.output });
    }
  }
  return out;
}

// slots that could take the dragged link(s), with what to print next to them:
// [{x, y, taken, name, type, side}] in graph space
function highlightTargets(canvas) {
  const LG = window.LiteGraph;
  const sources = dragSources(canvas);
  if (!sources.length || !LG) return [];
  const out = [];
  for (const node of canvas.graph?._nodes || []) {
    if (!node || node.flags?.collapsed) continue;
    for (const src of sources) {
      if (src.node === node) continue;
      if (src.toInputs) {
        (node.inputs || []).forEach((inp, i) => {
          if (!inp || !LG.isValidConnection(src.type, inp.type)) return;
          if (inp.widget && inp.link == null && !inp.pos) return; // hidden widget sockets
          let p = null;
          try { p = node.getInputPos(i); } catch (e) { p = null; }
          if (p) out.push({
            x: p[0], y: p[1], taken: inp.link != null, nodeId: node.id,
            name: inp.label || inp.name || "", type: inp.type, side: "input",
          });
        });
      } else {
        (node.outputs || []).forEach((outp, i) => {
          if (!outp || !LG.isValidConnection(outp.type, src.type)) return;
          let p = null;
          try { p = node.getOutputPos(i); } catch (e) { p = null; }
          if (p) out.push({
            x: p[0], y: p[1], taken: false, nodeId: node.id,
            name: outp.label || outp.name || "", type: outp.type, side: "output",
          });
        });
      }
    }
  }
  return out;
}

function slotTypeColor(canvas, type) {
  if (!type || type === "*" || type === -1) return null;
  const LGC = window.LGraphCanvas || canvas?.constructor;
  return canvas?.default_connection_color_byType?.[type] || LGC?.link_type_colors?.[type] || null;
}

// the "you will hit THIS one" colour - deliberately used nowhere else in the
// theme, so the live snap target can never be confused with a candidate
const ACTIVE_TARGET_DARK = "#6dff8e";
const ACTIVE_TARGET_LIGHT = "#0e8a34";
// how close (screen px) the cursor must be for a target to count as the one
// the release would connect to
const ACTIVE_SNAP_PX = 48;
// cursor position in canvas-pixel space, maintained by armDragWatch
const overlayMouse = { x: NaN, y: NaN, k: 1 };

function drawHighlightOverlay(ctx) {
  if (!hlState.on) return false;
  const canvas = app.canvas;
  const targets = highlightTargets(canvas);
  if (!targets.length) {
    hlState.snapLogged = false; // next drag logs its snap measurement again
    return false;
  }
  const scale = canvas?.ds?.scale ?? 1;
  const lightBg = canvasIsLight();
  // the slot, scaled up a touch beyond its normal size (~4px graph radius),
  // with a screen-space floor so far-out targets stay hittable-looking
  const r = Math.max(5.5, 6.5 * scale);
  // zoom-aware label size: grows a little when zoomed in, but NEVER drops
  // below a readable screen size when zoomed out
  const px = Math.max(12, Math.min(22, 14 * scale));
  ctx.globalAlpha = 1;
  ctx.textBaseline = "middle";
  const fallback = lightBg ? "#92621e" : "#d4a574";
  const activeCol = lightBg ? ACTIVE_TARGET_LIGHT : ACTIVE_TARGET_DARK;

  // which target would the release actually land on? the nearest one to the
  // cursor within snapping distance. Coordinate spaces differ across frontend
  // versions and display scaling, so try every plausible mouse position (CSS
  // pixels, backing-store pixels, graph_mouse) and use whichever agrees best.
  let activeIdx = -1;
  {
    const mice = [];
    if (Number.isFinite(overlayMouse.x)) {
      mice.push([overlayMouse.x, overlayMouse.y]);
      const k = overlayMouse.k || 1;
      if (Math.abs(k - 1) > 0.01) mice.push([overlayMouse.x * k, overlayMouse.y * k]);
    }
    const gm = canvas?.graph_mouse;
    if (gm && Number.isFinite(gm[0])) mice.push(graphToScreen(gm[0], gm[1]));
    let best = Infinity;
    targets.forEach((tg, i) => {
      const [sx, sy] = graphToScreen(tg.x, tg.y);
      for (const m of mice) {
        const d = Math.hypot(sx - m[0], sy - m[1]);
        if (d < best) { best = d; activeIdx = i; }
      }
    });
    const limit = ACTIVE_SNAP_PX * Math.max(1, overlayMouse.k || 1);
    if (best > limit) activeIdx = -1;
    // coordinate-free fallback: LiteGraph itself tracks which node the cursor
    // is over - hovering a node with compatible slots activates the nearest
    // (or only) one even if every mouse-space guess above was wrong
    const over = canvas?.node_over;
    if (activeIdx < 0 && over) {
      const own = [];
      targets.forEach((tg, i) => { if (tg.nodeId === over.id) own.push(i); });
      if (own.length === 1) activeIdx = own[0];
      else if (own.length > 1 && mice.length) {
        let b2 = Infinity;
        for (const i of own) {
          const [sx, sy] = graphToScreen(targets[i].x, targets[i].y);
          for (const m of mice) {
            const d = Math.hypot(sx - m[0], sy - m[1]);
            if (d < b2) { b2 = d; activeIdx = i; }
          }
        }
      } else if (own.length > 1) {
        activeIdx = own[0];
      }
    }
    // once per drag: say what we measured, so a missing green is debuggable
    if (!hlState.snapLogged) {
      hlState.snapLogged = true;
      console.log(
        `[APNext theme] snap-check: ${mice.length} mouse candidate(s), nearest target `
        + `${Math.round(best)}px (limit ${Math.round(limit)}px) -> `
        + (activeIdx >= 0 ? "ACTIVE" : "none"),
      );
    }
  }

  const drawTarget = (tg, isActive) => {
    const [x, y] = graphToScreen(tg.x, tg.y);
    const col = isActive ? activeCol : (slotTypeColor(canvas, tg.type) || fallback);
    const rr = isActive ? r * 1.45 : r;
    const fpx = isActive ? px * 1.15 : px;
    ctx.font = `${isActive ? 700 : 600} ${fpx}px 'Segoe UI', ui-sans-serif, system-ui, sans-serif`;
    if (isActive) {
      // a wide soft halo + outer ring: unmistakably "this is the one"
      const grad = ctx.createRadialGradient(x, y, rr * 0.5, x, y, rr * 2.6);
      grad.addColorStop(0, lightBg ? "rgba(14,138,52,0.35)" : "rgba(109,255,142,0.35)");
      grad.addColorStop(1, "rgba(0,0,0,0)");
      ctx.beginPath();
      ctx.arc(x, y, rr * 2.6, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, y, rr + 4, 0, Math.PI * 2);
      ctx.lineWidth = 2;
      ctx.strokeStyle = col;
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.arc(x, y, rr, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = lightBg ? "rgba(40,30,20,0.9)" : "rgba(10,9,7,0.9)";
    ctx.stroke();
    if (tg.taken) {
      // occupied input: hollow centre as a quiet "this will replace" hint
      ctx.beginPath();
      ctx.arc(x, y, Math.max(2, rr * 0.42), 0, Math.PI * 2);
      ctx.fillStyle = lightBg ? "#ffffff" : "#1a1712";
      ctx.fill();
    }
    // the slot's name in a label pill, inputs to the right, outputs to the left
    const name = tg.name;
    if (!name) return;
    const wText = ctx.measureText(name).width;
    const padX = fpx * 0.45;
    const h = fpx * 1.5;
    const left = tg.side === "input" ? x + rr + 6 : x - rr - 6 - wText - padX * 2;
    const top = y - h / 2;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(left, top, wText + padX * 2, h, h / 2);
    else ctx.rect(left, top, wText + padX * 2, h);
    if (isActive) {
      // inverted pill in the active colour: dark text on bright green
      ctx.fillStyle = col;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = lightBg ? "rgba(255,255,255,0.9)" : "rgba(10,9,7,0.9)";
      ctx.stroke();
      ctx.fillStyle = lightBg ? "#ffffff" : "#0c1a10";
    } else {
      ctx.fillStyle = lightBg ? "rgba(255,255,255,0.93)" : "rgba(16,14,11,0.93)";
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = col;
      ctx.stroke();
      ctx.fillStyle = lightBg ? "#2a2118" : "#efe9df";
    }
    ctx.fillText(name, left + padX, y + fpx * 0.05);
  };

  targets.forEach((tg, i) => { if (i !== activeIdx) drawTarget(tg, false); });
  if (activeIdx >= 0) drawTarget(targets[activeIdx], true); // on top
  return true;
}

// The dragged link takes its source slot's type colour instead of LiteGraph's
// single CONNECTING_LINK_COLOR, so what you drag looks like what you will get.
const dragColor = { patched: false, on: false };
function patchDragLinkColor() {
  if (dragColor.patched) return;
  const LGC = window.LGraphCanvas || app.canvas?.constructor;
  const LG = window.LiteGraph;
  if (!LGC?.prototype?._drawConnectingLinks || !LG) return;
  const orig = LGC.prototype._drawConnectingLinks;
  LGC.prototype._drawConnectingLinks = function (ctx) {
    if (!dragColor.on) return orig.apply(this, arguments);
    const rl = this.linkConnector?.renderLinks?.[0];
    const type = rl?.fromSlot?.type;
    const col = type && type !== "*" && type !== -1
      ? (this.default_connection_color_byType?.[type] || LGC.link_type_colors?.[type])
      : null;
    if (!col) return orig.apply(this, arguments);
    const saved = LG.CONNECTING_LINK_COLOR;
    LG.CONNECTING_LINK_COLOR = col;
    try { return orig.apply(this, arguments); } finally { LG.CONNECTING_LINK_COLOR = saved; }
  };
  dragColor.patched = true;
}

function applyHighlight(on) {
  hlState.on = !!on;
  hlState.patched = true;
  if (on) {
    console.log("[APNext theme] drop-target overlay v2 (snap colour) active");
    ensureOverlayLoop();
    armDragWatch();
    patchDragLinkColor();
  }
  dragColor.on = !!on && dragColor.patched;
}

// one rAF loop drives the overlay while there is something to show: a drag in
// progress or live sparks. It is armed by pointerdown / a spark and stops by itself.
const overlayLoop = { raf: 0, armed: false };
function ensureOverlayLoop() {
  if (overlayLoop.raf) return;
  overlayLoop.raf = requestAnimationFrame(overlayTick);
}
function overlayTick() {
  overlayLoop.raf = 0;
  let busy = false;
  try { busy = drawOverlayFrame(); } catch (err) { console.warn("[APNext overlay] draw failed:", err); }
  if (busy) overlayLoop.raf = requestAnimationFrame(overlayTick);
}
function armDragWatch() {
  if (overlayLoop.armed) return;
  overlayLoop.armed = true;
  document.addEventListener("pointerdown", () => ensureOverlayLoop(), true);
  document.addEventListener("pointermove", (e) => {
    // track the cursor relative to the canvas element in CSS pixels; the
    // backing-store factor is kept separately so the snap check can try both
    // spaces (display scaling changes which one graphToScreen matches)
    const el = app.canvas?.canvas;
    if (el) {
      const r = el.getBoundingClientRect();
      if (r.width > 0) {
        overlayMouse.x = e.clientX - r.left;
        overlayMouse.y = e.clientY - r.top;
        overlayMouse.k = el.width / r.width;
      }
    }
    if (dragSources(app.canvas).length) ensureOverlayLoop();
  }, true);
}

// ---------------------------------------------------------------------------
// Selection link highlight: while nodes are selected, every link that touches
// one of them draws in a much brighter, more saturated version of its own
// colour, wrapped in a crisp lighter border of the same hue (no blur/glow) —
// so you can see at a glance where a
// node's data comes from and goes to. On top of that, flow dots travel along
// the highlighted wires in the data direction (output → input). Independent
// of the theme; works with the stock renderer and every custom wire style.
// ---------------------------------------------------------------------------

const S_SELGLOW = "APNext.Theme.SelectionGlow";
const S_SELFLOW = "APNext.Theme.SelectionFlow";
const S_RUNGLOW = "APNext.Theme.RunGlow";
const selGlow = { on: true, flow: true, ids: null, raf: 0, cache: new Map() };
// while a prompt runs, the executing node's links glow too, so you can see
// which step the run is on and where its data goes
const runGlow = { on: true, id: null, armed: false };

function executingNodeId(detail) {
  if (detail == null) return null;
  if (typeof detail === "string" || typeof detail === "number") return detail;
  return detail.display_node ?? detail.node ?? null;
}

function armRunGlow() {
  if (runGlow.armed) return;
  runGlow.armed = true;
  try {
    api.addEventListener("executing", (ev) => {
      runGlow.id = executingNodeId(ev?.detail);
      app.canvas?.setDirty?.(false, true);
    });
    for (const evName of ["execution_interrupted", "execution_error"]) {
      api.addEventListener(evName, () => {
        runGlow.id = null;
        app.canvas?.setDirty?.(false, true);
      });
    }
  } catch (e) {
    console.warn("[APNext theme] run-glow events unavailable:", e);
  }
}

function hslToHex(h, s, l) {
  h = ((h % 360) + 360) % 360;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r, g, b;
  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return "#" + [r, g, b].map((v) => Math.round((v + m) * 255).toString(16).padStart(2, "0")).join("");
}

// true when the canvas background is light (stock light palette): highlights
// must then get DARKER and more saturated, not lighter, to stay contrasty
function canvasIsLight() {
  const rgb = parseColor(app.canvas?.clear_background_color || "#202020");
  return !!rgb && rgbToHsl(...rgb)[2] > 0.5;
}

// a link colour → its "hot" version: saturation pushed up hard, lightness
// pushed AWAY from the canvas background (up on dark canvases, down on light
// ones). Greys stay grey (pushed toward white or black) instead of picking up
// a false hue.
function boostLinkColor(col) {
  const light = canvasIsLight();
  const key = (light ? "L|" : "D|") + col;
  if (selGlow.cache.has(key)) return selGlow.cache.get(key);
  const rgb = parseColor(col);
  let out = col;
  if (rgb) {
    const [h, s, l] = rgbToHsl(...rgb);
    if (light) {
      out = s < 0.1
        ? mix(col, "#000000", 0.55)
        : hslToHex(h, Math.min(1, s * 1.5 + 0.25), Math.max(0.26, Math.min(0.42, l * 0.72)));
    } else {
      out = s < 0.1
        ? mix(col, "#ffffff", 0.5)
        : hslToHex(h, Math.min(1, s * 1.6 + 0.25), Math.min(0.75, l * 1.25 + 0.12));
    }
  }
  selGlow.cache.set(key, out);
  return out;
}

// deterministic 0..1 from any id - keeps each sphere's speed/size/phase stable
// across frames so the traffic looks alive instead of flickering
function hashSeed(x) {
  const s = String(x);
  let n = 0;
  for (let i = 0; i < s.length; i++) n = (Math.imul(n, 31) + s.charCodeAt(i)) | 0;
  n = (n ^ 61) ^ (n >>> 16);
  n = (n + (n << 3)) | 0;
  n ^= n >>> 4;
  n = Math.imul(n, 0x27d4eb2d);
  n ^= n >>> 15;
  return (n >>> 0) / 4294967296;
}

// point at t along the STOCK link path: cubic bezier like LiteGraph's spline
// (control points 25% of the distance along the slot directions); straight
// render modes fall back to a plain lerp
function stockFlowPoint(canvas, a, b, t, start_dir, end_dir) {
  const LG = window.LiteGraph;
  if (LG && (canvas.links_render_mode === LG.STRAIGHT_LINK || canvas.links_render_mode === LG.LINEAR_LINK)) {
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  }
  const f = Math.hypot(b[0] - a[0], b[1] - a[1]) * 0.25;
  const off = (dir, m) => {
    if (LG && dir === LG.UP) return [0, -m];
    if (LG && dir === LG.DOWN) return [0, m];
    if (LG && dir === LG.LEFT) return [-m, 0];
    return [m, 0];
  };
  const o1 = off(start_dir, f), o2 = off(end_dir, f);
  const p1x = a[0] + o1[0], p1y = a[1] + o1[1];
  const p2x = b[0] + o2[0], p2y = b[1] + o2[1];
  const u = 1 - t;
  return [
    u * u * u * a[0] + 3 * u * u * t * p1x + 3 * u * t * t * p2x + t * t * t * b[0],
    u * u * u * a[1] + 3 * u * u * t * p1y + 3 * u * t * t * p2y + t * t * t * b[1],
  ];
}

const flowCore = new Map(); // boosted colour -> hot-core colour
function coreColorOf(col) {
  let c = flowCore.get(col);
  if (!c) { c = mix(col, "#ffffff", 0.6); flowCore.set(col, c); }
  return c;
}

const haloCache = new Map(); // boosted colour -> lighter border colour
function haloColorOf(col) {
  let c = haloCache.get(col);
  if (!c) { c = mix(col, "#ffffff", 0.45); haloCache.set(col, c); }
  return c;
}

// the "data traffic" on a highlighted link: ovals of different sizes stretched
// along the wire, moving at different (per-link deterministic) speeds, each a
// coloured shell + hot core (no blur)
function drawFlowSpheres(canvas, ctx, a, b, link, col, start_dir, end_dir, opts) {
  const style = wireState.style;
  let pts = null;
  if (isCustomWire(style) && link && a && b && !opts?.disabled) {
    const lane =
      style === WIRE_STEP || style === WIRE_SMOOTHSTEP ? stepLane(canvas, link, a, b) : 0;
    pts = wirePoints(style, a, b, lane);
  }
  const LG = window.LiteGraph;
  const sd = start_dir ?? LG?.RIGHT ?? 4;
  const ed = end_dir ?? LG?.LEFT ?? 3;
  const id = link?.id ?? 0;
  const time = (typeof performance !== "undefined" ? performance.now() : Date.now()) * 0.001;
  const width = canvas.connections_width || 3;
  const core = coreColorOf(col);
  ctx.save();
  const at = (t) => (pts ? pointAt(pts, t) : stockFlowPoint(canvas, a, b, t, sd, ed));
  for (let i = 0; i < 6; i++) {
    const r1 = hashSeed(id + ":" + i);
    const r2 = hashSeed(i + "#" + id);
    const speed = 0.1 + 0.35 * r1;            // laps/second: some crawl, some race
    const t = (r2 + i / 6 + time * speed) % 1;
    const size = width * (0.45 + 1.0 * r2);   // some small, some chunky
    const [x, y] = at(t);
    // ovals stretched along the wire: angle from a nearby sample point
    const [x2, y2] = at(t < 0.98 ? t + 0.02 : t - 0.02);
    const ang = Math.atan2(y2 - y, x2 - x);
    const rx = size * 1.7, ry = size * 0.55;  // long axis rides the line
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, ang, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 0.95;
    ctx.fillStyle = core;
    ctx.beginPath();
    ctx.ellipse(x, y, Math.max(1.2, rx * 0.45), Math.max(0.6, ry * 0.45), ang, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function selectedNodeIds(canvas) {
  const ids = new Set();
  const sn = canvas?.selected_nodes;
  if (sn) for (const k in sn) { const n = sn[k]; if (n?.id !== undefined) { ids.add(n.id); ids.add(String(n.id)); } }
  const items = canvas?.selectedItems;
  if (items instanceof Set) {
    for (const it of items) {
      if (it?.id !== undefined && (it.inputs || it.outputs)) { ids.add(it.id); ids.add(String(it.id)); }
    }
  }
  return ids;
}

// the boosted colour when this link touches a selected (or executing) node
function selGlowColor(canvas, link, color) {
  if ((!selGlow.on && !runGlow.on) || !link || link.origin_id === undefined) return null;
  const ids = selGlow.ids;
  if (!ids || !ids.size) return null;
  if (!ids.has(link.origin_id) && !ids.has(link.target_id)) return null;
  const LGC = window.LGraphCanvas || canvas.constructor;
  const base = color || link.color || LGC?.link_type_colors?.[link.type] || canvas.default_link_color || C.accent;
  return boostLinkColor(base);
}

// once per background draw: refresh the selected-id set, and while flow dots
// should animate keep the link layer redrawing (self-sustaining rAF that
// stops by itself when the selection is cleared)
function selGlowFrame(canvas) {
  const ids = selGlow.on ? selectedNodeIds(canvas) : new Set();
  if (runGlow.on && runGlow.id != null) {
    ids.add(runGlow.id);
    ids.add(String(runGlow.id));
    const num = Number(runGlow.id);
    if (!Number.isNaN(num)) ids.add(num);
  }
  selGlow.ids = ids;
  if (selGlow.flow && selGlow.ids.size && !selGlow.raf) {
    selGlow.raf = requestAnimationFrame(() => {
      selGlow.raf = 0;
      if (selGlow.on && selGlow.flow && selGlow.ids?.size) app.canvas?.setDirty?.(false, true);
    });
  }
}

function applySelGlow(on, flowOn) {
  if (on) patchLinkRenderer();
  selGlow.on = !!on && wireState.patched;
  selGlow.flow = !!flowOn;
  if (!selGlow.on && !runGlow.on) selGlow.ids = null;
  app.canvas?.setDirty?.(false, true);
}

function applyRunGlow(on) {
  if (on) { patchLinkRenderer(); armRunGlow(); }
  runGlow.on = !!on && wireState.patched;
  if (!selGlow.on && !runGlow.on) selGlow.ids = null;
  app.canvas?.setDirty?.(false, true);
}

// ---------------------------------------------------------------------------
// Connect sparks: a little burst of particles at the input slot when a link is
// made (graphgen's Sparks.svelte, drawn on the front canvas).
// ---------------------------------------------------------------------------

const sparks = { list: [], raf: 0, patched: false, on: true, listening: new WeakSet(), overlay: null, octx: null };

function sparksOverlay() {
  // a fixed, pointer-transparent canvas on <body> that tracks the graph canvas
  // rect - above every layer (classic canvas or the Vue node layer)
  const base = app.canvas?.canvas;
  if (!base) return null;
  let ov = sparks.overlay;
  if (!ov || !ov.isConnected) {
    ov = document.createElement("canvas");
    ov.id = "apnext-sparks-overlay";
    Object.assign(ov.style, { position: "fixed", pointerEvents: "none", zIndex: "2147483000", left: "0", top: "0" });
    document.body.appendChild(ov);
    sparks.overlay = ov;
    sparks.octx = ov.getContext("2d");
  }
  const r = base.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
  if (ov.width !== Math.round(w * dpr) || ov.height !== Math.round(h * dpr)) {
    ov.width = Math.round(w * dpr);
    ov.height = Math.round(h * dpr);
    ov.style.width = w + "px";
    ov.style.height = h + "px";
  }
  if (ov._l !== r.left || ov._t !== r.top) {
    ov.style.transform = `translate(${r.left}px, ${r.top}px)`;
    ov._l = r.left; ov._t = r.top;
  }
  return ov;
}

function graphToScreen(x, y) {
  const ds = app.canvas?.ds;
  if (!ds) return [x, y];
  if (typeof ds.convertOffsetToCanvas === "function") return ds.convertOffsetToCanvas([x, y]);
  return [(x + ds.offset[0]) * ds.scale, (y + ds.offset[1]) * ds.scale];
}

function spawnSparks(x, y, color) {
  // x, y in GRAPH space; particles move in screen pixels so they read the same at any zoom
  const now = performance.now();
  if (sparks.last && now - sparks.last.t < 150 && Math.abs(sparks.last.x - x) < 2 && Math.abs(sparks.last.y - y) < 2) return;
  sparks.last = { t: now, x, y };
  console.debug("[APNext sparks] burst", x.toFixed(0), y.toFixed(0), "overlay:", !!sparksOverlay());
  for (let i = 0; i < 26; i++) {
    const ang = Math.random() * Math.PI * 2;
    const speed = 70 + Math.random() * 190; // screen px / s
    sparks.list.push({
      x, y, vx: Math.cos(ang) * speed, vy: Math.sin(ang) * speed - 40,
      born: now, life: 500 + Math.random() * 400, size: 1.8 + Math.random() * 2.4, color: color || C.accent,
    });
  }
  sparks.list.push({ x, y, vx: 0, vy: 0, born: now, life: 520, size: 0, ring: true, color: color || C.accent });
  ensureOverlayLoop();
}

// one frame of the overlay: clear, then highlight rings + sparks. Returns true
// while there is still something animating.
function drawOverlayFrame() {
  const ov = sparksOverlay();
  const ctx = sparks.octx;
  if (!ov || !ctx) return false;
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, ov.width, ov.height);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const now = performance.now();
  sparks.list = sparks.list.filter((p) => now - p.born < p.life);
  let busy = false;
  if (hlState.on && drawHighlightOverlay(ctx)) busy = true;
  if (sparks.on && sparks.list.length) { drawSparksInto(ctx, now); busy = true; }
  return busy;
}

function drawSparksInto(ctx, now) {
  for (const p of sparks.list) {
    const age = (now - p.born) / 1000;
    const life = 1 - (now - p.born) / p.life;
    if (life <= 0) continue;
    const [sx, sy] = graphToScreen(p.x, p.y);
    if (p.ring) {
      ctx.globalAlpha = life * 0.9;
      ctx.strokeStyle = p.color;
      ctx.lineWidth = 3 * (0.4 + 0.6 * life);
      ctx.beginPath();
      ctx.arc(sx, sy, 6 + (1 - life) * 38, 0, Math.PI * 2);
      ctx.stroke();
      continue;
    }
    const x = sx + p.vx * age, y = sy + p.vy * age + 120 * age * age;
    ctx.globalAlpha = life;
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.arc(x, y, Math.max(1.2, p.size * (0.6 + 0.4 * life)), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function linkEndpoint(link) {
  // graph-space position of the target input
  const graph = app.canvas?.graph;
  const node = graph?.getNodeById?.(link.target_id);
  if (!node) return null;
  let p = null;
  try { p = node.getInputPos?.(link.target_slot); } catch (e) { p = null; }
  const nx = node.pos?.[0] ?? 0, ny = node.pos?.[1] ?? 0, nw = node.size?.[0] ?? 200;
  if (!p) return [nx, ny];
  let [x, y] = p;
  // some layouts report slot positions relative to the node; detect and fix
  if (x < nx - 40 || x > nx + nw + 40) { x += nx; y += ny; }
  // with graphgen slot tabs the visible port sits on the node edge, not 10px in
  if (ggState.on && !node.flags?.collapsed) x = nx;
  return [x, y];
}

function patchSparks() {
  if (sparks.patched) return;
  const LGC = window.LGraphCanvas || app.canvas?.constructor;
  if (!LGC?.prototype?.drawFrontCanvas) return;
  const orig = LGC.prototype.drawFrontCanvas;
  LGC.prototype.drawFrontCanvas = function () {
    const r = orig.apply(this, arguments);
    if (sparks.on) listenForLinks(this);
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
    try {
      const link = e?.detail?.link || e?.detail;
      if (!link || link.target_id === undefined) { console.debug("[APNext sparks] link-created without a link payload", e?.detail); return; }
      const p = linkEndpoint(link);
      if (!p) return;
      const LGC = window.LGraphCanvas || app.canvas?.constructor;
      const color = link.color || (LGC?.link_type_colors && LGC.link_type_colors[link.type]) || C.accent;
      spawnSparks(p[0], p[1], color);
    } catch (err) {
      console.warn("[APNext sparks] failed:", err);
    }
  });
}

function listenForGraphLinks() {
  // second source: LGraph-level "link-created"-ish signals are not uniform across
  // versions, so also watch onConnectionChange of the root graph
  const g = app.canvas?.graph;
  if (!g || sparks.listening.has(g)) return;
  sparks.listening.add(g);
  const prev = g.onConnectionChange;
  g.onConnectionChange = function (node, link) {
    try {
      if (sparks.on && link && link.target_id !== undefined && !sparks.seen?.has(link.id)) {
        sparks.seen = sparks.seen || new Set();
        sparks.seen.add(link.id);
        setTimeout(() => sparks.seen.delete(link.id), 500);
        const p = linkEndpoint(link);
        if (p) {
          const LGC = window.LGraphCanvas || app.canvas?.constructor;
          spawnSparks(p[0], p[1], link.color || (LGC?.link_type_colors && LGC.link_type_colors[link.type]) || C.accent);
        }
      }
    } catch (err) { /* ignore */ }
    return prev?.apply(this, arguments);
  };
}

// Renderer-independent trigger: diff the graph's link table every background
// frame (it is redrawn whenever links change). New ids -> burst. Bulk arrivals
// (workflow load, paste, undo) are ignored.
const linkWatch = { known: null, graph: null };

function watchLinks(canvas) {
  const graph = canvas?.graph;
  if (!graph) return;
  const table = graph._links ?? graph.links;
  if (!table) return;
  const ids = [];
  if (table instanceof Map) for (const k of table.keys()) ids.push(k);
  else for (const k in table) ids.push(k);
  if (linkWatch.graph !== graph || !linkWatch.known) {
    linkWatch.graph = graph;
    linkWatch.known = new Set(ids.map(String));
    return;
  }
  const current = new Set(ids.map(String));
  const fresh = ids.filter((id) => !linkWatch.known.has(String(id)));
  linkWatch.known = current;
  if (!fresh.length || fresh.length > 3 || !sparks.on) return;
  for (const id of fresh) {
    const link = table instanceof Map ? table.get(id) ?? table.get(Number(id)) : table[id];
    if (!link || link.target_id === undefined) continue;
    const p = linkEndpoint(link);
    if (!p) continue;
    const LGC = window.LGraphCanvas || canvas.constructor;
    const color = link.color || (LGC?.link_type_colors && LGC.link_type_colors[link.type]) || C.accent;
    console.debug("[APNext sparks] new link", id, "→ burst at", p[0].toFixed(0), p[1].toFixed(0));
    spawnSparks(p[0], p[1], color);
  }
}

function applySparks(on) {
  if (on) { patchSparks(); patchLinkRenderer(); listenForLinks(app.canvas || {}); listenForGraphLinks(); }
  sparks.on = on && sparks.patched;
  if (!on) { sparks.list = []; if (sparks.overlay) sparks.overlay.remove(); sparks.overlay = null; }
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
  const { color } = familyColors(node.type || node.comfyClass || "");
  if (node.properties?.apnext_autocolor) {
    // ours from an earlier session: refresh to the header-only scheme
    node.color = color;
    delete node.bgcolor;
    return;
  }
  if (node.color || node.bgcolor) return; // user (or a saved workflow) already chose
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
const ggState = { on: false, shapePatched: false, slotPatched: false, fontPatched: false, savedTitleSize: null, boldTitles: false };

function hexToRgba(hex, a) {
  const c = parseColor(hex);
  return c ? `rgba(${c[0]},${c[1]},${c[2]},${a})` : hex;
}

// the hue a node's header carries: family swatch for APNext nodes, the node's
// own (botanical-mapped) title colour when coloured, else its first output's
// link colour, else the border colour
function nodeHue(node) {
  const key = (node.properties?.apnext_autocolor ? "A" : "") + (node.color || "") + "|" + (node.outputs?.[0]?.type || "");
  if (node._apnextHueKey === key && node._apnextHue) return node._apnextHue;
  const hue = nodeHueCompute(node);
  node._apnextHueKey = key;
  node._apnextHue = hue;
  return hue;
}

function nodeHueCompute(node) {
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
      // solid tint (hue folded into the panel colour) - no alpha in the layout
      ctx.fillStyle = mix(hue, C.panel, 0.88);
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(x, y + titleH - 1);
      ctx.lineTo(x + w, y + titleH - 1);
      ctx.lineWidth = 2;
      ctx.strokeStyle = mix(hue, "#ffffff", 0.22);
      ctx.stroke();
    }
    // 1px panel border around the whole box (selection outline is drawn by LiteGraph)
    if (!selected) {
      ctx.beginPath();
      ctx.roundRect(x + 0.5, y + 0.5, w - 1, h - 1, collapsed ? radius : [radius, radius, 0, 0]);
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
      return (ggState.on || ggState.boldTitles) ? `700 ${f}` : f;
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
    patchNodeRecolor(); // hosts the group overdraw (node-style group panels)
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
  const w = settingGet(S_WIRES);
  if (w === WIRE_GRAVITY) return WIRE_DEFAULT; // legacy value
  return w && WIRE_STYLES.includes(w) ? w : WIRE_DEFAULT;
}

async function migrateLegacyGravity() {
  // gravity wires are gone: map any stored gravity mode/style back to defaults
  const m = currentMode();
  const w = settingGet(S_WIRES);
  if (m === MODE_BOTH) await settingSet(S_MODE, MODE_THEME);
  else if (m === MODE_GRAVITY) await settingSet(S_MODE, MODE_OFF);
  if (w === WIRE_GRAVITY) await settingSet(S_WIRES, WIRE_DEFAULT);
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
    if (theme) patchLinkRenderer(); // hosts the background-pass hook
    themeState.on = theme;
    applyNodeRecolor(theme && settingGet(S_RECOLOR) !== false);
    applyGraphgenNodes(theme && settingGet(S_GGNODES) !== false);
    applyWires(currentWires());
    applyHighlight(settingGet(S_HIGHLIGHT) !== false);
    applySparks(settingGet(S_SPARKS) !== false);
    applySelGlow(settingGet(S_SELGLOW) !== false, settingGet(S_SELFLOW) !== false);
    applyRunGlow(settingGet(S_RUNGLOW) !== false);
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
  if (style === WIRE_GRAVITY) style = WIRE_DEFAULT;
  await settingSet(S_WIRES, style);
  await applyAll();
}
function toggleTheme() {
  return setTheme(!themeOn(currentMode()));
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
  [WIRE_CABLE]: "Cable (sag)",
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
        "How links are drawn - graphgen's edge styles: Bezier, Smooth step, Step, Straight or " +
        "Cable (a drooping wire with a static sag). ComfyUI default leaves the stock renderer alone.",
      type: "combo",
      options: WIRE_COMBO.map((v) => ({ text: WIRE_LABELS[v], value: v })),
      defaultValue: WIRE_DEFAULT,
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
      id: S_SELGLOW,
      category: ["APNext", "Canvas helpers", "Selection link glow"],
      name: "Highlight the links of selected nodes",
      tooltip: "While one or more nodes are selected, every link connected to them is drawn brighter (a more saturated version of its own colour) with a crisp lighter border in the same hue, so you can see at a glance where the data comes from and goes to.",
      type: "boolean",
      defaultValue: true,
      onChange: (value, old) => { if (old !== undefined) applyAll(); },
    },
    {
      id: S_SELFLOW,
      category: ["APNext", "Canvas helpers", "Selection flow animation"],
      name: "Animate data flow on highlighted links",
      tooltip: "Adds travelling dots to the highlighted links of the selected nodes, moving in the data direction (output → input). Turn off to keep the static highlight without the animation (saves redraws on big graphs).",
      type: "boolean",
      defaultValue: true,
      onChange: (value, old) => { if (old !== undefined) applyAll(); },
    },
    {
      id: S_RUNGLOW,
      category: ["APNext", "Canvas helpers", "Running-node link glow"],
      name: "Highlight the links of the node being executed",
      tooltip: "While a prompt runs, the links of the currently executing node are highlighted (and carry the flow dots) exactly like a selected node's, so you can see which step the run is on and where its data goes. Clears when the run finishes.",
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
    { id: "APNext.Theme.WiresDefault", label: "APNext: wires - ComfyUI default", function: () => setWires(WIRE_DEFAULT) },
    { id: "APNext.Theme.WiresBezier", label: "APNext: wires - Bezier", function: () => setWires(WIRE_BEZIER) },
    { id: "APNext.Theme.WiresSmoothStep", label: "APNext: wires - Smooth step", function: () => setWires(WIRE_SMOOTHSTEP) },
    { id: "APNext.Theme.WiresStep", label: "APNext: wires - Step", function: () => setWires(WIRE_STEP) },
    { id: "APNext.Theme.WiresStraight", label: "APNext: wires - Straight", function: () => setWires(WIRE_STRAIGHT) },
    { id: "APNext.Theme.WiresCable", label: "APNext: wires - Cable (sag)", function: () => setWires(WIRE_CABLE) },
    { id: "APNext.Theme.Off", label: "APNext: theme off (stock ComfyUI)", icon: "pi pi-times", function: async () => { await setWires(WIRE_DEFAULT); await setTheme(false); } },
  ],
  menuCommands: [
    { path: ["APNext"], commands: ["APNext.Theme.Toggle", "APNext.Theme.CycleWires", "APNext.Theme.Off"] },
    {
      path: ["APNext", "Wire style"],
      commands: [
        "APNext.Theme.WiresDefault", "APNext.Theme.WiresBezier", "APNext.Theme.WiresSmoothStep",
        "APNext.Theme.WiresStep", "APNext.Theme.WiresStraight", "APNext.Theme.WiresCable",
      ],
    },
  ],

  getCanvasMenuItems() {
    const theme = themeOn(currentMode());
    const wires = currentWires();
    return [
      null,
      { content: `APNext: Graphgen theme ${theme ? "✓" : ""}`, callback: () => toggleTheme() },
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
    // the palette store and canvas exist by now; apply once the first frame is in
    setTimeout(() => applyAll(), 50);
  },
});
