// APNext context socket autogrow + labelling
//
// The Claude Code / H3 writer nodes declare eight optional STRING sockets
// (context_1..context_8) for steering input from other APNext nodes (Time,
// Scene, Poses, Plots, Feelings, Cinematic, ...). Eight empty sockets are
// noise, so only the connected ones plus one spare are shown. When a socket is
// connected, its label shows what it is fed by ("context: time"), mirroring
// what the back end works out from the graph at run time.
//
// Purely cosmetic: every socket is a declared optional input, so a workflow
// keeps working with this file removed.

import { app } from "../../../scripts/app.js";

const NODE_CLASSES = new Set([
  "ClaudeCodeNode",
  "H3BasePromptWriter",
  "H3RefPromptWriter",
  "H3ClaudeCodeBaseWriter",
  "H3ClaudeCodeRefWriter",
  "H3ClaudeCodeRefiner",
  "H3ClaudeCodeContinueWriter",
  "H3ClaudeCodeCrossoverWriter",
  "H3ClaudeCodeScenesWriter",
  "H3ClaudeCodeMusicVideoWriter",
  "H3ClaudeCodePresentationWriter",
  "H3ClaudeCodeShortFilmWriter",
]);
const MAX_CONTEXT = 8;
const slotName = (i) => `context_${i}`;
const SLOT_RE = /^context_(\d+)$/;

const KNOWN = {
  H3Characters: "cast",
  H3ClaudeCodeBaseWriter: "h3 prompt",
  H3ClaudeCodeRefWriter: "h3 prompt",
  H3ClaudeCodeRefiner: "h3 prompt",
  H3ClaudeCodeContinueWriter: "h3 prompt",
  H3BasePromptWriter: "h3 prompt",
  H3RefPromptWriter: "h3 prompt",
  H3ScenePick: "h3 prompt",
};

function categoryOf(type) {
  if (!type) return null;
  if (KNOWN[type]) return KNOWN[type];
  if (type.endsWith("PromptNode")) {
    return type
      .slice(0, -"PromptNode".length)
      .replace(/(?!^)([A-Z])/g, " $1")
      .toLowerCase();
  }
  return null;
}

function findInput(node, i) {
  return (node.inputs || []).findIndex((input) => input && input.name === slotName(i));
}

function relabel(node) {
  const graph = node.graph || app.graph;
  for (let i = 1; i <= MAX_CONTEXT; i++) {
    const idx = findInput(node, i);
    if (idx < 0) continue;
    const input = node.inputs[idx];
    let label = null;
    if (input.link != null && graph) {
      const link = graph.links?.[input.link];
      const origin = link ? graph.getNodeById(link.origin_id) : null;
      const cat = categoryOf(origin?.type);
      label = cat ? `context: ${cat}` : "context";
    }
    input.label = label || undefined;
  }
}

function syncContextSlots(node) {
  if (!node || node._apnextCtxSyncing) return;
  node._apnextCtxSyncing = true;
  try {
    let highest = 0;
    for (let i = 1; i <= MAX_CONTEXT; i++) {
      const idx = findInput(node, i);
      if (idx >= 0 && node.inputs[idx].link != null) highest = i;
    }
    const wanted = Math.min(MAX_CONTEXT, highest + 1);

    const rowsBefore = (node.inputs || []).length;
    let changed = false;
    for (let i = 1; i <= wanted; i++) {
      if (findInput(node, i) < 0) { node.addInput(slotName(i), "STRING"); changed = true; }
    }
    for (let i = MAX_CONTEXT; i > wanted; i--) {
      const idx = findInput(node, i);
      if (idx >= 0 && node.inputs[idx].link == null) { node.removeInput(idx); changed = true; }
    }
    relabel(node);

    if (changed) {
      const slotHeight = (globalThis.LiteGraph && globalThis.LiteGraph.NODE_SLOT_HEIGHT) || 20;
      const minimum = node.computeSize();
      const height = Math.max(minimum[1], node.size[1] + ((node.inputs || []).length - rowsBefore) * slotHeight);
      node.setSize([Math.max(node.size[0], minimum[0]), height]);
    }
    node.setDirtyCanvas(true, true);
  } finally {
    node._apnextCtxSyncing = false;
  }
}

app.registerExtension({
  name: "apnext.contextInputs",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_CLASSES.has(nodeData?.name)) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      setTimeout(() => syncContextSlots(this), 0);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      setTimeout(() => syncContextSlots(this), 0);
      return result;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo, ioSlot) {
      const result = onConnectionsChange?.apply(this, arguments);
      if (SLOT_RE.test(ioSlot?.name || "")) setTimeout(() => syncContextSlots(this), 0);
      return result;
    };
  },
});
