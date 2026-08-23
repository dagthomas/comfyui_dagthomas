// APNext H3 project names - auto-fill the `project_name` widget
//
// Every writer run belongs to a "project": a memorable cinematography-flavoured
// tag (NeonDollyFoley, GoldenOrbitSlate, ...) that the writer also emits on its
// `project_name` output, so Save Video's filename_prefix can carry it and the
// output folder shows which clips belong together. This extension fills the
// widget with a fresh random name when the node is created (or when a workflow
// saved without one is loaded), so the name is visible and editable before the
// run. Keep the word pools in the same spirit as the Python fallback in
// nodes/h3/claude_code_support.py (used for headless/API runs).

import { app } from "../../../scripts/app.js";

const NODE_TYPES = new Set([
  "H3ClaudeCodeMusicVideoWriter",
  "H3ClaudeCodeShortFilmWriter",
  "H3ClaudeCodePresentationWriter",
  "H3ClaudeCodeScenesWriter",
  "H3ClaudeCodeCrossoverWriter",
  "H3MusicVideoMinimal",
]);

const LOOKS = [
  "Golden", "Silver", "Amber", "Noir", "Neon", "Velvet", "Crimson", "Cobalt",
  "Sepia", "Chrome", "Indigo", "Emerald", "Scarlet", "Midnight", "Pastel",
  "Tungsten", "Halide", "Matte", "Anamorphic", "Technicolor",
];
const MOVES = [
  "Dolly", "Crane", "Zoom", "Orbit", "Boom", "Gimbal", "Rack", "Whip",
  "Glide", "Pan", "Tilt", "Push", "Steadicam", "Tracking", "Vertigo",
];
const GEAR = [
  "Slate", "Reel", "Lens", "Shutter", "Bokeh", "Gaffer", "Grip", "Rig",
  "Flare", "Foley", "Scrim", "Frame", "Take", "Clapper", "Montage",
];

const pick = (pool) => pool[Math.floor(Math.random() * pool.length)];
const generate = () => pick(LOOKS) + pick(MOVES) + pick(GEAR);

function fill(node) {
  const w = node.widgets?.find((x) => x.name === "project_name");
  if (w && !String(w.value ?? "").trim()) {
    w.value = generate();
    node.setDirtyCanvas?.(true, true);
  }
}

app.registerExtension({
  name: "apnext.h3.project_name",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_TYPES.has(nodeData?.name)) return;

    // New node dropped on the canvas.
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      fill(this);
      return r;
    };

    // Workflow loaded from disk: configure() restores the saved (possibly
    // empty) value after onNodeCreated, so top up again here.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      fill(this);
      return r;
    };
  },
});
