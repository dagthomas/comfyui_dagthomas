// APNext H3 reference image autogrow
//
// The H3 reference writers declare nine optional image sockets (image_1..image_9)
// on the back end, mirroring ComfyUI's "MiniMax H3 Reference to Video" node. Nine
// empty sockets make the node tall and hard to read, so this keeps only the
// connected ones plus one spare: connect image_1 and image_2 appears, and so on.
// The matching pass-through outputs grow and shrink with them.
//
// Nothing here changes what the back end sees. Every socket is a declared
// optional input, so a workflow keeps working with this file removed.

import { app } from "../../../scripts/app.js";

const NODE_CLASSES = new Set([
  "H3ClaudeCodeRefWriter",
  "H3RefPromptWriter",
  "H3ClaudeCodeCrossoverWriter",
  "H3ClaudeCodeMusicVideoWriter",
  "H3ClaudeCodePresentationWriter",
  "H3ClaudeCodeShortFilmWriter",
  "H3MusicVideoMinimal",
  "H3PromptPreview", // inputs only (thumbnails), no pass-through outputs
]);
const MAX_IMAGES = 9;

const slotName = (i) => `image_${i}`;

function findInput(node, i) {
  return (node.inputs || []).findIndex((input) => input && input.name === slotName(i));
}

function findOutput(node, i) {
  return (node.outputs || []).findIndex((output) => output && output.name === slotName(i));
}

function outputHasLinks(output) {
  return Array.isArray(output.links) && output.links.length > 0;
}

function imageOutputsAreTail(node) {
  // The image outputs may only be grown/trimmed while they are the LAST
  // outputs of the node. The server addresses outputs by slot index, so
  // removing an image output that sits before a later non-image output
  // (e.g. the Music Video Writer's `clip_starts`) would shift that output's
  // index and silently re-point its links at an IMAGE slot.
  const outs = node.outputs || [];
  let sawImage = false;
  for (const o of outs) {
    if (/^image_\d+$/.test(o?.name || "")) sawImage = true;
    else if (sawImage) return false;
  }
  return true;
}

function syncImageSlots(node) {
  if (!node || node._apnextSyncing) return;
  node._apnextSyncing = true;
  try {
    let highest = 0;
    for (let i = 1; i <= MAX_IMAGES; i++) {
      const idx = findInput(node, i);
      if (idx >= 0 && node.inputs[idx].link != null) highest = i;
    }
    // Any output that is wired downstream must keep its socket as well.
    for (let i = 1; i <= MAX_IMAGES; i++) {
      const idx = findOutput(node, i);
      if (idx >= 0 && outputHasLinks(node.outputs[idx])) highest = Math.max(highest, i);
    }
    const wanted = Math.min(MAX_IMAGES, highest + 1);
    const manageOutputs = node._apnextImageOutputs && imageOutputsAreTail(node);

    // Sockets 1..wanted must exist. Adding appends at the tail; the image
    // sockets are declared last among the sockets, so numeric order holds.
    const rows = () => Math.max((node.inputs || []).length, (node.outputs || []).length);
    const rowsBefore = rows();
    let added = false;
    let removed = false;
    for (let i = 1; i <= wanted; i++) {
      if (findInput(node, i) < 0) { node.addInput(slotName(i), "IMAGE"); added = true; }
      if (manageOutputs && findOutput(node, i) < 0) { node.addOutput(slotName(i), "IMAGE"); added = true; }
    }

    // Trailing sockets past the spare go, but never one that carries a link.
    for (let i = MAX_IMAGES; i > wanted; i--) {
      const inIdx = findInput(node, i);
      if (inIdx >= 0 && node.inputs[inIdx].link == null) { node.removeInput(inIdx); removed = true; }
      if (manageOutputs) {
        const outIdx = findOutput(node, i);
        if (outIdx >= 0 && !outputHasLinks(node.outputs[outIdx])) { node.removeOutput(outIdx); removed = true; }
      }
    }

    if (added || removed) {
      // Adjust the height by exactly the rows that came or went, so a size the
      // user chose survives; never drop below what the widgets need.
      const slotHeight = (globalThis.LiteGraph && globalThis.LiteGraph.NODE_SLOT_HEIGHT) || 20;
      const minimum = node.computeSize();
      const height = Math.max(minimum[1], node.size[1] + (rows() - rowsBefore) * slotHeight);
      node.setSize([Math.max(node.size[0], minimum[0]), height]);
      node.setDirtyCanvas(true, true);
    }
  } finally {
    node._apnextSyncing = false;
  }
}

app.registerExtension({
  name: "apnext.h3.referenceImages",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_CLASSES.has(nodeData?.name)) return;
    // Only writers pass the images through; the preview just shows them.
    nodeType.prototype._apnextImageOutputs = (nodeData.output_name || []).includes("image_1");

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      // Let ComfyUI finish building widgets before trimming.
      setTimeout(() => syncImageSlots(this), 0);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      // Links are restored after configure; sync once they are in place.
      setTimeout(() => syncImageSlots(this), 0);
      return result;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo, ioSlot) {
      const result = onConnectionsChange?.apply(this, arguments);
      const name = ioSlot?.name || "";
      if (/^image_\d+$/.test(name)) setTimeout(() => syncImageSlots(this), 0);
      return result;
    };
  },
});
