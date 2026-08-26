// APNext H3 - make a connected socket visibly win over the widgets it replaces
//
// Some optional inputs on the H3 nodes override widgets on the same node:
//
//   llm       (APNext H3 LLM Backend) -> `model`, `draft_model`
//             The backend's model is the one that writes; the dropdown kept
//             showing "sonnet", which read as "still using Claude Code".
//   cut_plan  (APNext H3 Cut Plan)    -> `segment_mode`, `max_segment_seconds`,
//             `min_segment_seconds`. The plan's scenes are used verbatim.
//
// While such a socket is linked this greys the overridden widgets out and
// relabels them with what is actually in charge (the backend's model name,
// or "cut plan"), and restores them when the link goes away.

import { app } from "../../../scripts/app.js";

const BACKEND_TYPE = "H3LLMBackend";

// input name -> { type: socket type to match, widgets, tag(sourceNode) -> label }
const RULES = {
  llm: {
    type: "APNEXT_LLM",
    widgets: ["model", "draft_model"],
    tag: (source, widgetName) => (widgetName === "draft_model" ? "llm backend" : backendModel(source) || "llm backend"),
  },
  cut_plan: {
    type: "STRING",
    widgets: ["segment_mode", "max_segment_seconds", "min_segment_seconds"],
    tag: () => "cut plan",
  },
};

function backendModel(node) {
  // Mirrors H3LLMBackend.build(): `model` unless it is the custom entry, then
  // `model_name`; a bare Ollama tag is understood as ollama:<tag>.
  const w = (name) => node?.widgets?.find((x) => x.name === name)?.value;
  let chosen = String(w("model") ?? "").trim();
  if (!chosen || chosen.startsWith("custom")) chosen = String(w("model_name") ?? "").trim();
  if (chosen && !chosen.includes(":")) chosen = `ollama:${chosen}`;
  return chosen;
}

function linkedSource(node, inputName) {
  const input = node.inputs?.find((i) => i.name === inputName);
  if (input?.link == null) return null;
  const graph = node.graph ?? app.graph;
  const link = graph.links?.[input.link];
  return link ? graph.getNodeById(link.origin_id) : null;
}

function syncOverride(node) {
  if (!node?.widgets) return;
  for (const [inputName, rule] of Object.entries(RULES)) {
    if (!node.inputs?.some((i) => i.name === inputName)) continue;
    const source = linkedSource(node, inputName);
    for (const name of rule.widgets) {
      const widget = node.widgets.find((w) => w.name === name);
      if (!widget) continue;
      if (widget.__h3Label === undefined) widget.__h3Label = widget.label;
      if (source) {
        widget.label = `${name} → ${rule.tag(source, name)}`;
        widget.disabled = true;
      } else {
        widget.label = widget.__h3Label;
        widget.disabled = false;
      }
    }
  }
  node.setDirtyCanvas?.(true, true);
}

function syncAllConsumers(sourceNode) {
  const graph = sourceNode.graph ?? app.graph;
  for (const out of sourceNode.outputs ?? []) {
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
    const watched = Object.entries(RULES).filter(([name, rule]) => optional[name]?.[0] === rule.type).map(([name]) => name);
    if (!watched.length) return;

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo, ioSlot) {
      const result = onConnectionsChange?.apply(this, arguments);
      if (watched.includes(ioSlot?.name || "")) setTimeout(() => syncOverride(this), 0);
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
