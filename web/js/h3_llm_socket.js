// APNext H3 - make the `llm` socket visibly win over the model dropdown
//
// Every H3 Claude Code node carries an optional `llm` input. While an
// `APNext H3 LLM Backend` is connected, the backend's model overrides the
// node's own `model` (and `draft_model`) widget on the Python side - see
// resolve_backend_model / resolve_draft_model in nodes/h3/claude_code_support.py.
// The widgets kept showing "sonnet" as if it mattered, which read as "still
// using Claude Code". This extension greys those widgets out while the socket
// is linked and relabels them with the model that is actually writing, then
// restores them when the link goes away.

import { app } from "../../../scripts/app.js";

const SOCKET_TYPE = "APNEXT_LLM";
const OVERRIDDEN = ["model", "draft_model"];
const BACKEND_TYPE = "H3LLMBackend";

function backendModel(node) {
  // Mirrors H3LLMBackend.build(): `model` unless it is the custom entry, then
  // `model_name`; a bare Ollama tag is understood as ollama:<tag>.
  const w = (name) => node?.widgets?.find((x) => x.name === name)?.value;
  let chosen = String(w("model") ?? "").trim();
  if (!chosen || chosen.startsWith("custom")) chosen = String(w("model_name") ?? "").trim();
  if (chosen && !chosen.includes(":")) chosen = `ollama:${chosen}`;
  return chosen;
}

function linkedBackend(node) {
  const input = node.inputs?.find((i) => i.name === "llm");
  if (input?.link == null) return null;
  const link = node.graph?.links?.[input.link] ?? app.graph?.links?.[input.link];
  return link ? node.graph?.getNodeById?.(link.origin_id) ?? app.graph.getNodeById(link.origin_id) : null;
}

function syncOverride(node) {
  if (!node?.widgets) return;
  const backend = linkedBackend(node);
  const linked = !!backend;
  const model = linked ? backendModel(backend) : "";
  for (const name of OVERRIDDEN) {
    const widget = node.widgets.find((w) => w.name === name);
    if (!widget) continue;
    if (widget.__h3Label === undefined) widget.__h3Label = widget.label;
    if (linked) {
      const tag = name === "draft_model" ? "llm backend" : model || "llm backend";
      widget.label = `${name} → ${tag}`;
      widget.disabled = true;
    } else {
      widget.label = widget.__h3Label;
      widget.disabled = false;
    }
  }
  node.setDirtyCanvas?.(true, true);
}

function syncAllConsumers(backendNode) {
  const graph = backendNode.graph ?? app.graph;
  for (const out of backendNode.outputs ?? []) {
    for (const linkId of out.links ?? []) {
      const link = graph.links?.[linkId];
      const target = link && graph.getNodeById(link.target_id);
      if (target) syncOverride(target);
    }
  }
}

app.registerExtension({
  name: "apnext.h3.llm_socket",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === BACKEND_TYPE) {
      // A backend's model pick changed: refresh the label on every node it feeds.
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated?.apply(this, arguments);
        for (const widget of this.widgets ?? []) {
          if (widget.name !== "model" && widget.name !== "model_name") continue;
          const cb = widget.callback;
          widget.callback = (...args) => {
            const out = cb?.apply(widget, args);
            setTimeout(() => syncAllConsumers(this), 0);
            return out;
          };
        }
        return r;
      };
      return;
    }

    const optional = nodeData?.input?.optional ?? {};
    if (optional.llm?.[0] !== SOCKET_TYPE) return;

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo, ioSlot) {
      const result = onConnectionsChange?.apply(this, arguments);
      if ((ioSlot?.name || "") === "llm") setTimeout(() => syncOverride(this), 0);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      setTimeout(() => syncOverride(this), 0);
      return result;
    };

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      setTimeout(() => syncOverride(this), 0);
      return result;
    };
  },
});
