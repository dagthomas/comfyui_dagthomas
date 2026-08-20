// APNext H3 Scene Counter
//
// Renders the scene-render progress ("X / N" with a progress bar and the
// remaining count) inside the H3SceneCounter node. The python side sends
// {index, count, remaining, status} in the execution message; the last value
// is kept in node.properties so a page reload still shows it.

import { app } from "../../../scripts/app.js";

const NODE_CLASS = "H3SceneCounter";
const STYLE_ID = "apnext-h3-counter-style";

const CSS = `
.apnext-h3-counter {
  color-scheme: dark;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 6px;
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  padding: 10px 14px;
  background: #161512;
  border: 1px solid #2c2820;
  border-radius: 6px;
  color: #e8e4df;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  user-select: none;
}
.apnext-h3-counter .h3c-num {
  font-size: 30px;
  font-weight: 700;
  line-height: 1.1;
  letter-spacing: 0.02em;
  font-variant-numeric: tabular-nums;
}
.apnext-h3-counter .h3c-num small { font-size: 15px; font-weight: 500; color: #9a9590; }
.apnext-h3-counter .h3c-bar {
  height: 6px;
  border-radius: 999px;
  background: #2c2820;
  overflow: hidden;
}
.apnext-h3-counter .h3c-fill {
  height: 100%;
  width: 0%;
  border-radius: 999px;
  background: linear-gradient(90deg, #a7bd84, #d4a574);
  transition: width 0.25s ease;
}
.apnext-h3-counter .h3c-sub { font-size: 11.5px; color: #9a9590; }
.apnext-h3-counter.h3c-done .h3c-sub { color: #a7bd84; }
`;

function ensureStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = CSS;
  document.head.appendChild(style);
}

function buildCounter() {
  ensureStyle();
  const root = document.createElement("div");
  root.className = "apnext-h3-counter";

  const num = document.createElement("div");
  num.className = "h3c-num";
  const bar = document.createElement("div");
  bar.className = "h3c-bar";
  const fill = document.createElement("div");
  fill.className = "h3c-fill";
  bar.appendChild(fill);
  const sub = document.createElement("div");
  sub.className = "h3c-sub";

  root.appendChild(num);
  root.appendChild(bar);
  root.appendChild(sub);

  const set = (d) => {
    if (!d || !d.count) {
      num.innerHTML = `&ndash; <small>/ &ndash;</small>`;
      sub.textContent = "run the graph to count scenes";
      fill.style.width = "0%";
      root.classList.remove("h3c-done");
      return;
    }
    const count = Number(d.count) || 0;
    const done = Math.max(0, Math.min(Number(d.index) + 1, count));
    const remaining = Number.isFinite(Number(d.remaining)) ? Number(d.remaining) : count - done;
    num.innerHTML = `${done} <small>/ ${count}</small>`;
    fill.style.width = `${count ? Math.round((done / count) * 100) : 0}%`;
    sub.textContent = remaining > 0
      ? `${remaining} scene${remaining === 1 ? "" : "s"} remaining`
      : "all scenes rendered ✓";
    root.classList.toggle("h3c-done", remaining === 0);
  };

  set(null);
  return { root, set };
}

app.registerExtension({
  name: "apnext.h3.sceneCounter",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_CLASS) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const counter = buildCounter();
      this.addDOMWidget("h3_scene_counter", "H3_SCENE_COUNTER", counter.root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 96,
      });
      this._h3Counter = counter;
      const size = this.computeSize();
      this.setSize([Math.max(size[0], 210), Math.max(size[1], 150)]);
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const d = Array.isArray(message?.counter) ? message.counter[0] : message?.counter;
      if (!d) return;
      this._h3Counter?.set(d);
      this.properties = this.properties || {};
      this.properties.h3_counter = d;
    };

    // restore the last reading when a saved workflow is loaded
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      if (this.properties?.h3_counter) this._h3Counter?.set(this.properties.h3_counter);
      return r;
    };
  },
});
