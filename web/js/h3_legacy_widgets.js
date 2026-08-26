// APNext H3 - load workflows saved before a widget was removed
//
// LiteGraph restores `widgets_values` by position. When a widget is removed
// from a node, every saved workflow that still carries its value shifts the
// values that follow by one - the old `wildness` number lands in `model`,
// the model name lands in `research`, and so on. Rather than making people
// hand-edit JSON, this extension recognises the old shape while a workflow
// is being configured and drops the stale value before the widgets are
// filled. Saving the workflow again writes the current shape.
//
// A node fixed this way gets an amber "!" badge in its title bar for the rest
// of the session; hovering it explains what happened and suggests adding a
// fresh copy of the node from the node list.
//
// Each entry: node type -> { index, widget, legacy } where `legacy(values)`
// returns true when the value at `index` can only be the removed widget's.

import { app } from "../../../scripts/app.js";

const REMOVED = {
  // `wildness` (INT 0-100) sat right before `model` (always a string).
  H3ClaudeCodeMusicVideoWriter: {
    index: 9, widget: "wildness",
    legacy: (v) => typeof v[9] === "number" && typeof v[10] === "string",
  },
  H3MusicVideoMinimal: {
    index: 4, widget: "wildness",
    legacy: (v) => typeof v[4] === "number" && typeof v[5] === "string",
  },
};

const BADGE_R = 7;                 // badge radius, node-space px
const BADGE_MARGIN = 10;           // from the node's right edge
const TITLE_H = () => LiteGraph.NODE_TITLE_HEIGHT || 30;

function badgeCenter(node) {
  return [node.size[0] - BADGE_MARGIN - BADGE_R, -TITLE_H() / 2];
}

function mouseOverBadge(node) {
  const gm = app.canvas?.graph_mouse;
  if (!gm) return false;
  const [cx, cy] = badgeCenter(node);
  const dx = gm[0] - node.pos[0] - cx, dy = gm[1] - node.pos[1] - cy;
  return dx * dx + dy * dy <= (BADGE_R + 3) * (BADGE_R + 3);
}

function drawBadge(node, ctx) {
  const [cx, cy] = badgeCenter(node);
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, BADGE_R, 0, Math.PI * 2);
  ctx.fillStyle = "#e0a020";
  ctx.fill();
  ctx.fillStyle = "#1a1a1a";
  ctx.font = `bold ${BADGE_R * 1.6}px sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("!", cx, cy + 0.5);
  ctx.restore();
}

function drawTooltip(node, ctx, lines) {
  const [cx, cy] = badgeCenter(node);
  const pad = 8, lineH = 15;
  ctx.save();
  ctx.font = "12px sans-serif";
  const width = Math.max(...lines.map((l) => ctx.measureText(l).width)) + pad * 2;
  const height = lines.length * lineH + pad * 2;
  // above the title bar, right-aligned to the badge, kept inside the node's width
  const x = Math.max(0, Math.min(cx + BADGE_R - width, node.size[0] - width));
  const y = cy - BADGE_R - 6 - height;
  ctx.fillStyle = "rgba(20, 20, 20, 0.96)";
  ctx.strokeStyle = "#e0a020";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, 6);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#f0f0f0";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  lines.forEach((l, i) => ctx.fillText(l, x + pad, y + pad + i * lineH));
  ctx.restore();
}

// Nodes whose JS button used to be saved as a trailing `null`: strip those
// nulls so the real widgets that were added later keep their defaults.
const BUTTON_NODES = new Set(["H3LLMBackend", "H3SoundEvents", "H3CutPlan"]);

app.registerExtension({
  name: "apnext.h3.legacy_widgets",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (BUTTON_NODES.has(nodeData?.name)) {
      const configure = nodeType.prototype.configure;
      nodeType.prototype.configure = function (info) {
        const values = info?.widgets_values;
        if (Array.isArray(values) && values.length && values[values.length - 1] === null) {
          const trimmed = [...values];
          while (trimmed.length && trimmed[trimmed.length - 1] === null) trimmed.pop();
          info = { ...info, widgets_values: trimmed };
        }
        return configure?.call(this, info);
      };
    }
    const rule = REMOVED[nodeData?.name];
    if (!rule) return;
    const displayName = nodeData.display_name || nodeData.name;

    const configure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      const values = info?.widgets_values;
      if (Array.isArray(values) && values.length > rule.index && rule.legacy(values)) {
        const dropped = values[rule.index];
        info = { ...info, widgets_values: values.filter((_, i) => i !== rule.index) };
        this.__h3Legacy = { widget: rule.widget, dropped };
        console.info(`[APNext H3] ${nodeData.name} #${info.id}: dropped legacy widget value ${JSON.stringify(dropped)} (removed '${rule.widget}')`);
      }
      return configure?.call(this, info);
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      const r = onDrawForeground?.apply(this, arguments);
      if (!this.__h3Legacy || this.flags?.collapsed) return r;
      drawBadge(this, ctx);
      if (mouseOverBadge(this)) {
        drawTooltip(this, ctx, [
          `Loaded from an older version of ${displayName}.`,
          `The removed '${this.__h3Legacy.widget}' value (${this.__h3Legacy.dropped}) was dropped, so the`,
          "widgets line up and the node works - but to get the current layout,",
          `add a fresh "${displayName}" from the node list, reconnect it, and delete this one.`,
        ]);
      }
      return r;
    };

    // Redraw while the pointer moves over the node so the tooltip tracks it.
    const onMouseMove = nodeType.prototype.onMouseMove;
    nodeType.prototype.onMouseMove = function () {
      const r = onMouseMove?.apply(this, arguments);
      if (this.__h3Legacy) this.setDirtyCanvas?.(true, false);
      return r;
    };
    const onMouseLeave = nodeType.prototype.onMouseLeave;
    nodeType.prototype.onMouseLeave = function () {
      const r = onMouseLeave?.apply(this, arguments);
      if (this.__h3Legacy) this.setDirtyCanvas?.(true, false);
      return r;
    };
  },
});
