// APNext H3 template variables - show what {vars} a node can use
//
// The H3 writer nodes expand {character1}, {actor1}, {franchise1}, {cast1},
// {characters}, {cast}, {context_1} ... in their text boxes from whatever is
// wired into them (see nodes/h3/template_vars.py). This extension adds a small
// read-only strip under the inputs that lists the variables currently available
// from the node's connections, refreshed whenever links change. Click a chip to
// copy it.

import { app } from "../../../scripts/app.js";

const NODE_CLASSES = new Set([
  "H3ClaudeCodeBaseWriter",
  "H3ClaudeCodeRefWriter",
  "H3ClaudeCodeCrossoverWriter",
  "H3ClaudeCodeScenesWriter",
  "H3ClaudeCodeContinueWriter",
  "H3ClaudeCodeRefiner",
  "H3ClaudeCodeMusicVideoWriter",
  "H3ClaudeCodePresentationWriter",
  "H3BasePromptWriter",
  "H3RefPromptWriter",
]);
const CHARACTERS_CLASS = "H3Characters";
const RANDOM_PREFIX = "🎲";
const STYLE_ID = "apnext-h3-vars-style";
const WIDGET_NAME = "template_vars";

const CSS = `
.apnext-h3-vars {
  font: 11px/1.5 ui-monospace, "JetBrains Mono", Consolas, monospace;
  color: var(--input-text, #e8e4df);
  background: var(--comfy-input-bg, #161512);
  border: 1px solid var(--border-color, #2c2820);
  border-radius: 6px;
  padding: 4px 6px;
  overflow: auto;
  box-sizing: border-box;
  user-select: text;
}
.apnext-h3-vars .t { opacity: 0.6; margin-right: 4px; }
.apnext-h3-vars .chip {
  display: inline-block; margin: 1px 3px 1px 0; padding: 0 6px;
  border-radius: 10px; background: rgba(212,165,116,0.14); cursor: pointer;
  white-space: nowrap;
}
.apnext-h3-vars .chip:hover { background: rgba(212,165,116,0.32); }
.apnext-h3-vars .chip .v { opacity: 0.6; margin-left: 4px; }
.apnext-h3-vars .none { opacity: 0.55; font-style: italic; }
`;

function ensureStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = CSS + AC_CSS;
  document.head.appendChild(style);
}

function sourceNode(node, input) {
  if (input?.link == null) return null;
  const link = node.graph?.links?.[input.link] ?? app.graph?.links?.[input.link];
  if (!link) return null;
  return { src: node.graph?.getNodeById?.(link.origin_id) ?? app.graph.getNodeById(link.origin_id), slot: link.origin_slot };
}

// Oldest-first chain of H3 Characters nodes that end up in `node`'s cast line.
function charactersChain(node, seen) {
  if (!node || node.type !== CHARACTERS_CLASS || seen.has(node.id)) return [];
  seen.add(node.id);
  const out = [];
  const castIn = node.inputs?.find((i) => i.name === "cast_in");
  const up = castIn ? sourceNode(node, castIn) : null;
  if (up?.src) out.push(...charactersChain(up.src, seen));
  out.push(node);
  return out;
}

function characterLabel(charNode) {
  const custom = charNode.widgets?.find((x) => x.name === "custom_character");
  const customText = typeof custom?.value === "string" ? custom.value.trim() : "";
  const w = charNode.widgets?.find((x) => x.name === "character");
  const label = typeof w?.value === "string" ? w.value : "";
  if (customText && !label.startsWith(RANDOM_PREFIX)) {
    const head = customText.split(/\r?\n/)[0];
    const c = head.match(/^(.*?)(?: \(played by .*?\))? from /) || head.match(/^([^:]+):/);
    return c ? c[1].trim() : head.slice(0, 40);
  }
  // "Character — Actor (Show)" → "Character"
  const m = label.match(/^(.*?)\s+—\s+/);
  return m ? m[1] : label.replace(/^🎲.*$/, "random").replace(/^✏️.*$/, "custom");
}

function rank(name) {
  if (name.startsWith("cast_")) return 0;
  if (name.startsWith("context_")) return 1;
  return 2;
}

function computeVars(node) {
  const inputs = (node.inputs || [])
    .filter((i) => i.link != null)
    .slice()
    .sort((a, b) => rank(a.name) - rank(b.name) || a.name.localeCompare(b.name));

  const vars = [];
  const seen = new Set();
  let k = 0;
  const names = [];
  for (const input of inputs) {
    const s = sourceNode(node, input);
    if (!s?.src) continue;
    for (const charNode of charactersChain(s.src, seen)) {
      k += 1;
      const name = characterLabel(charNode);
      names.push(name);
      vars.push({ key: `character${k}`, hint: name });
      vars.push({ key: `actor${k}` });
      vars.push({ key: `franchise${k}` });
      vars.push({ key: `cast${k}` });
      const wd = charNode.widgets?.find((x) => x.name === "wardrobe");
      if (typeof wd?.value === "string" && wd.value.trim()) vars.push({ key: `wardrobe${k}`, hint: wd.value.trim() });
    }
  }
  if (k > 0) {
    vars.push({ key: "characters", hint: names.join(", ") });
    vars.push({ key: "cast" });
  }
  for (const input of inputs) {
    if (input.name.startsWith("context_") || input.name.startsWith("cast_")) {
      const s = sourceNode(node, input);
      vars.push({ key: input.name, hint: s?.src?.title || s?.src?.type || "" });
    }
  }
  return vars;
}

function render(node, el) {
  const vars = computeVars(node);
  el.innerHTML = "";
  const t = document.createElement("span");
  t.className = "t";
  t.textContent = "{vars}:";
  el.appendChild(t);
  if (!vars.length) {
    const n = document.createElement("span");
    n.className = "none";
    n.textContent = "connect H3 Characters / context nodes → {character1}, {context_1} …";
    el.appendChild(n);
    return;
  }
  for (const v of vars) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = `{${v.key}}`;
    if (v.hint) {
      const h = document.createElement("span");
      h.className = "v";
      h.textContent = v.hint.length > 28 ? v.hint.slice(0, 27) + "…" : v.hint;
      chip.appendChild(h);
    }
    chip.title = `Click to copy {${v.key}}` + (v.hint ? ` — ${v.hint}` : "");
    chip.addEventListener("click", (e) => {
      e.stopPropagation();
      navigator.clipboard?.writeText(`{${v.key}}`);
      chip.style.outline = "1px solid var(--p-primary-color, #d4a574)";
      setTimeout(() => (chip.style.outline = ""), 400);
    });
    el.appendChild(chip);
  }
}

// ---------------------------------------------------------------------------
// Autocomplete: typing "{" in a multiline text box of an H3 node pops up the
// variables currently available from the node's connections; keep typing to
// filter, ↑/↓ to move, Enter/Tab/click to insert, Esc to dismiss.
// ---------------------------------------------------------------------------

const AC_CSS = `
.apnext-h3-ac {
  position: fixed; z-index: 10001; min-width: 180px; max-width: 360px; max-height: 220px; overflow: auto;
  background: var(--comfy-menu-bg, #1e1c17); color: var(--input-text, #e8e4df);
  border: 1px solid var(--border-color, #2c2820); border-radius: 6px; box-shadow: 0 6px 24px rgba(0,0,0,0.5);
  font: 12px/1.5 ui-monospace, "JetBrains Mono", Consolas, monospace; padding: 3px 0;
}
.apnext-h3-ac .row { padding: 2px 10px; cursor: pointer; white-space: nowrap; display: flex; gap: 8px; }
.apnext-h3-ac .row.sel, .apnext-h3-ac .row:hover { background: rgba(212,165,116,0.22); }
.apnext-h3-ac .row .k { color: var(--p-primary-color, #d4a574); }
.apnext-h3-ac .row .h { opacity: 0.55; overflow: hidden; text-overflow: ellipsis; font-family: system-ui, sans-serif; }
.apnext-h3-ac .none { padding: 2px 10px; opacity: 0.6; font-style: italic; font-family: system-ui, sans-serif; }
`;

let acBox = null;
let acState = null; // { ta, node, items, sel, start }

function acClose() {
  if (acBox) acBox.remove();
  acBox = null;
  acState = null;
}

function acRender() {
  if (!acState) return;
  const { ta, items, sel } = acState;
  if (!acBox) {
    acBox = document.createElement("div");
    acBox.className = "apnext-h3-ac";
    acBox.addEventListener("mousedown", (e) => { e.preventDefault(); e.stopPropagation(); });
    acBox.addEventListener("wheel", (e) => e.stopPropagation());
    document.body.appendChild(acBox);
  }
  acBox.innerHTML = "";
  if (!items.length) {
    const n = document.createElement("div");
    n.className = "none";
    n.textContent = "no matching variable — connect H3 Characters / context nodes";
    acBox.appendChild(n);
  }
  items.forEach((v, i) => {
    const row = document.createElement("div");
    row.className = "row" + (i === sel ? " sel" : "");
    const k = document.createElement("span");
    k.className = "k";
    k.textContent = `{${v.key}}`;
    row.appendChild(k);
    if (v.hint) {
      const h = document.createElement("span");
      h.className = "h";
      h.textContent = v.hint;
      row.appendChild(h);
    }
    row.addEventListener("click", (e) => { e.stopPropagation(); acAccept(i); });
    acBox.appendChild(row);
  });
  const r = ta.getBoundingClientRect();
  const h = Math.min(220, acBox.scrollHeight + 4);
  const below = r.bottom + h < window.innerHeight;
  acBox.style.left = `${Math.max(4, Math.min(r.left + 8, window.innerWidth - 380))}px`;
  acBox.style.top = below ? `${r.bottom + 2}px` : `${Math.max(4, r.top - h - 2)}px`;
  const selEl = acBox.children[sel];
  selEl?.scrollIntoView?.({ block: "nearest" });
}

function acAccept(i) {
  if (!acState) return;
  const { ta, items, start } = acState;
  const v = items[i ?? acState.sel];
  if (!v) return acClose();
  const end = ta.selectionStart;
  const before = ta.value.slice(0, start);
  const after = ta.value.slice(end);
  // swallow a "}" the user may already have typed right after the caret
  const tail = after.startsWith("}") ? after.slice(1) : after;
  const ins = `{${v.key}}`;
  ta.value = before + ins + tail;
  const pos = before.length + ins.length;
  ta.setSelectionRange(pos, pos);
  ta.dispatchEvent(new Event("input", { bubbles: true }));
  ta.dispatchEvent(new Event("change", { bubbles: true }));
  acClose();
  ta.focus();
}

function acUpdate(ta, node) {
  const caret = ta.selectionStart;
  const text = ta.value.slice(0, caret);
  const m = text.match(/\{([A-Za-z0-9_]*)$/);
  if (!m) return acClose();
  const prefix = m[1].toLowerCase();
  let vars;
  try { vars = computeVars(node); } catch (e) { vars = []; }
  const items = vars.filter((v) => v.key.toLowerCase().startsWith(prefix));
  const start = caret - m[0].length;
  const keepSel = acState && acState.ta === ta && acState.items[acState.sel] ? acState.items[acState.sel].key : null;
  let sel = Math.max(0, items.findIndex((v) => v.key === keepSel));
  acState = { ta, node, items, sel, start };
  acRender();
}

function attachAutocomplete(node) {
  for (const w of node.widgets || []) {
    const ta = w.element || w.inputEl;
    if (!ta || ta.tagName !== "TEXTAREA" || ta._apnextAc) continue;
    ta._apnextAc = true;
    ta.addEventListener("input", () => acUpdate(ta, node));
    ta.addEventListener("click", () => acState && acUpdate(ta, node));
    ta.addEventListener("blur", () => setTimeout(() => acState?.ta === ta && acClose(), 120));
    ta.addEventListener("keydown", (e) => {
      if (!acState || acState.ta !== ta) {
        if (e.key === "{") setTimeout(() => acUpdate(ta, node), 0);
        return;
      }
      if (e.key === "ArrowDown") { e.preventDefault(); e.stopPropagation(); acState.sel = (acState.sel + 1) % Math.max(1, acState.items.length); acRender(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); e.stopPropagation(); acState.sel = (acState.sel - 1 + Math.max(1, acState.items.length)) % Math.max(1, acState.items.length); acRender(); }
      else if ((e.key === "Enter" || e.key === "Tab") && acState.items.length) { e.preventDefault(); e.stopPropagation(); acAccept(); }
      else if (e.key === "Escape" || e.key === "}") { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); } acClose(); }
    });
  }
}

app.registerExtension({
  name: "apnext.h3.template_vars",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_CLASSES.has(nodeData?.name)) return;
    ensureStyle();

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      if (this.widgets?.some((w) => w.name === WIDGET_NAME)) return r;
      const el = document.createElement("div");
      el.className = "apnext-h3-vars";
      const widget = this.addDOMWidget(WIDGET_NAME, "div", el, {
        serialize: false,
        hideOnZoom: true,
        getMinHeight: () => 28,
        getMaxHeight: () => 60,
      });
      widget.computeSize = () => [this.size?.[0] ?? 300, 44];
      this._apnextVarsEl = el;
      const refresh = () => {
        try { render(this, el); } catch (e) { /* graph not ready yet */ }
      };
      this._apnextVarsRefresh = refresh;
      setTimeout(refresh, 0);
      // text boxes may be created after us; attach autocomplete once they exist
      const attach = () => attachAutocomplete(this);
      setTimeout(attach, 0);
      setTimeout(attach, 500);
      this._apnextAttachAc = attach;
      return r;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const r = onConnectionsChange?.apply(this, arguments);
      // upstream chains may have changed too; defer so links are final
      setTimeout(() => this._apnextVarsRefresh?.(), 0);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      setTimeout(() => { this._apnextVarsRefresh?.(); this._apnextAttachAc?.(); }, 50);
      return r;
    };

    // the textareas of DOM widgets are (re)created when the node is drawn; make
    // sure a freshly attached one gets the autocomplete too
    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function () {
      const r = onDrawForeground?.apply(this, arguments);
      if (!this._apnextAcChecked) {
        this._apnextAcChecked = true;
        setTimeout(() => { this._apnextAttachAc?.(); this._apnextAcChecked = false; }, 1000);
      }
      return r;
    };
  },

  // A character picked on an upstream H3 Characters node changes the hints;
  // refresh every H3 writer in the graph cheaply after any widget change there.
  async nodeCreated(node) {
    if (node?.type !== CHARACTERS_CLASS) return;
    const w = node.widgets?.find((x) => x.name === "character");
    if (!w) return;
    const prev = w.callback;
    w.callback = function () {
      const out = prev?.apply(this, arguments);
      for (const n of app.graph?._nodes || []) n._apnextVarsRefresh?.();
      return out;
    };
  },
});
