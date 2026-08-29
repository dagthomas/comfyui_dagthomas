// APNext H3 - failed-run banner.
//
// When an H3 run dies (Ollama context overflow, dead backend, CLI missing,
// timeout), the Python side calls report_node_error() with the reason before
// raising. This draws that reason as a red, word-wrapped panel pinned under
// the failed node, so the WHY is readable on the graph itself instead of
// flashing by in a toast or hiding in the server console. The banner clears
// on the next queue.
//
// Generic on purpose: any node that sends "apnext.h3.node_error" with its
// UNIQUE_ID gets the banner - nothing here is writer-specific.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const FONT = "12px ui-monospace, 'Cascadia Code', 'JetBrains Mono', Menlo, Consolas, monospace";
const TITLE_FONT = "bold 12px Inter, Arial, sans-serif";
const LINE_H = 16;
const PAD = 10;
const MAX_LINES = 14;
const MIN_W = 300;

function wrapText(ctx, text, maxWidth) {
  const lines = [];
  for (const raw of String(text).split("\n")) {
    let line = "";
    for (const word of raw.split(/\s+/)) {
      const probe = line ? `${line} ${word}` : word;
      if (ctx.measureText(probe).width <= maxWidth) {
        line = probe;
        continue;
      }
      if (line) lines.push(line);
      // hard-break a single word wider than the panel (paths, model ids)
      line = word;
      while (ctx.measureText(line).width > maxWidth && line.length > 1) {
        let cut = line.length - 1;
        while (cut > 1 && ctx.measureText(line.slice(0, cut)).width > maxWidth) cut--;
        lines.push(line.slice(0, cut));
        line = line.slice(cut);
      }
    }
    lines.push(line);
  }
  if (lines.length > MAX_LINES) {
    lines.length = MAX_LINES;
    lines[MAX_LINES - 1] += " …";
  }
  return lines;
}

function drawErrorPanel(node, ctx) {
  const w = Math.max(node.size?.[0] ?? MIN_W, MIN_W);
  const top = (node.flags?.collapsed ? 0 : node.size?.[1] ?? 0) + 6;

  ctx.save();
  ctx.font = FONT;
  const lines = wrapText(ctx, node.__h3Error, w - PAD * 2);
  const h = PAD * 2 + LINE_H * (lines.length + 1) + 4;

  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(0, top, w, h, 6);
  else ctx.rect(0, top, w, h);
  ctx.fillStyle = "rgba(60, 18, 18, 0.96)";
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "#e5484d";
  ctx.stroke();

  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.font = TITLE_FONT;
  ctx.fillStyle = "#ff8a8e";
  ctx.fillText("⛔ Run failed — fix, then queue again", PAD, top + PAD + 10);
  ctx.font = FONT;
  ctx.fillStyle = "#f4d7d7";
  let y = top + PAD + 10 + LINE_H + 4;
  for (const line of lines) {
    ctx.fillText(line, PAD, y);
    y += LINE_H;
  }
  ctx.restore();
}

function hookDraw(node) {
  if (node.__h3ErrorHooked) return;
  node.__h3ErrorHooked = true;
  const prev = node.onDrawForeground;
  node.onDrawForeground = function (ctx) {
    const r = prev?.apply(this, arguments);
    if (this.__h3Error) drawErrorPanel(this, ctx);
    return r;
  };
}

app.registerExtension({
  name: "apnext.h3.nodeError",

  setup() {
    api.addEventListener("apnext.h3.node_error", (ev) => {
      const { node: id, text } = ev.detail || {};
      if (!id || !text) return;
      const node =
        app.graph?.getNodeById?.(Number(id)) ?? app.graph?.getNodeById?.(id);
      if (!node) return;
      hookDraw(node);
      node.__h3Error = String(text);
      node.setDirtyCanvas?.(true, true);
      app.canvas?.setDirty?.(true, true);
    });
    // a new queue means the user is retrying: clear every banner
    api.addEventListener("execution_start", () => {
      let dirty = false;
      for (const n of app.graph?._nodes || []) {
        if (n?.__h3Error) {
          n.__h3Error = null;
          dirty = true;
        }
      }
      if (dirty) app.canvas?.setDirty?.(true, true);
    });
  },
});
