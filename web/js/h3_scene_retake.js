// APNext H3 Scene Retake - "🎬 Retake this scene" button
//
// Queues ONLY this node (ComfyUI partial execution: app.queuePrompt with the
// node id), so the writer, the sound nodes and every other scene stay put -
// the retake reads the saved bundle from disk and needs nothing upstream but
// the model loaders, which are cached anyway. Each click also bumps the seed
// unless it is linked, so a click is always a NEW take.

import { app } from "../../../scripts/app.js";

const NODE_TYPE = "H3SceneRetake";

app.registerExtension({
  name: "apnext.h3.scene_retake",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_TYPE) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const node = this;
      const btn = this.addWidget("button", "🎬 Retake this scene (queues only this node)", null, async () => {
        const seed = node.widgets?.find((w) => w.name === "seed");
        const linked = node.inputs?.some((i) => i.name === "seed" && i.link != null);
        if (seed && !linked) {
          seed.value = Math.floor(Math.random() * 0x1fffffffffffff);
          seed.callback?.(seed.value, app.canvas, node);
        }
        const scene = node.widgets?.find((w) => w.name === "scene_number")?.value;
        try {
          // frontend >= 1.3: queuePrompt(number, batchCount, queueNodeIds)
          await app.queuePrompt(0, 1, [String(node.id)]);
          console.info(`[APNext H3] retake of scene ${scene} queued (this node only)`);
        } catch (err) {
          console.error("[APNext H3] partial queue failed, queueing the whole graph", err);
          await app.queuePrompt(0, 1);
        }
      });
      btn.serialize = false;   // a button has no value to save; a saved null would shift later widgets
      return r;
    };
  },
});
