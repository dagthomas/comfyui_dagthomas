// APNext H3 project names - auto-fill the `project_name` widget
//
// Every writer run belongs to a "project": a memorable cinematography-flavoured
// tag (NeonDollyFoley, GoldenOrbitSlate, ...) that the writer also emits on its
// `project_name` output, so Save Video's filename_prefix can carry it and the
// output folder shows which clips belong together. This extension fills the
// widget when the node is created (or when a workflow saved without one is
// loaded) and swaps in a new name every time the `seed` widget changes - a
// hand edit or `control_after_generate` randomizing after a queue - so each
// seed gets its own project folder. A name the user typed themselves (one not
// built from the pools below) is left alone.
//
// A seed >= 0 maps to its name deterministically (splitmix64, mirrored by
// generate_project_name in nodes/h3/claude_code_support.py), so the UI and
// headless/API runs agree on what a seed is called; seed -1 draws a random one.
// Keep the word pools identical to the Python ones - same words, same order.

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
const POOLS = [LOOKS, MOVES, GEAR];

const MASK64 = (1n << 64n) - 1n;

// splitmix64 - the same stream as the Python side.
function* splitmix64(seed) {
  let state = BigInt.asUintN(64, seed);
  for (;;) {
    state = (state + 0x9E3779B97F4A7C15n) & MASK64;
    let z = state;
    z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & MASK64;
    yield z ^ (z >> 31n);
  }
}

function seedOf(node) {
  const w = node.widgets?.find((x) => x.name === "seed");
  if (!w) return null;
  const n = Number(w.value);
  return Number.isFinite(n) ? Math.floor(n) : null;
}

const pickRandom = (pool) => pool[Math.floor(Math.random() * pool.length)];

function generate(seed) {
  if (typeof seed === "number" && seed >= 0) {
    const stream = splitmix64(BigInt(seed));
    return POOLS.map((pool) => pool[Number(stream.next().value % BigInt(pool.length))]).join("");
  }
  return POOLS.map(pickRandom).join("");
}

// True for names this extension (or the Python fallback) could have produced,
// i.e. LOOK+MOVE+GEAR from the pools. Anything else was typed by the user.
const AUTO_NAME = new RegExp(`^(${LOOKS.join("|")})(${MOVES.join("|")})(${GEAR.join("|")})$`);
const isAutoName = (v) => AUTO_NAME.test(String(v ?? "").trim());

function projectWidget(node) {
  return node.widgets?.find((x) => x.name === "project_name");
}

// Fill an empty widget (new node / workflow saved without a name).
function fill(node) {
  const w = projectWidget(node);
  if (w && !String(w.value ?? "").trim()) {
    w.value = generate(seedOf(node));
    node.setDirtyCanvas?.(true, true);
  }
}

// The seed changed: give the run a new name unless the user named it.
function refresh(node, seed) {
  const w = projectWidget(node);
  if (!w) return;
  const current = String(w.value ?? "").trim();
  if (current && !isAutoName(current)) return;
  const next = generate(seed);
  if (next !== current) {
    w.value = next;
    node.setDirtyCanvas?.(true, true);
  }
}

// Follow the seed widget through every path that changes it: the number
// input, the linked control_after_generate widget after a queue, and
// programmatic assignment. Intercepting `value` catches all of them;
// wrapping `callback` is the fallback for widgets that only fire it.
function watchSeed(node) {
  const w = node.widgets?.find((x) => x.name === "seed");
  if (!w || w.__apnextProjectNameWatched) return;
  w.__apnextProjectNameWatched = true;

  let last = seedOf(node);
  const onChange = () => {
    const seed = seedOf(node);
    if (seed === last) return;
    last = seed;
    refresh(node, seed);
  };

  let desc = null;
  for (let o = w; o && !desc; o = Object.getPrototypeOf(o)) {
    desc = Object.getOwnPropertyDescriptor(o, "value");
  }
  try {
    if (desc && (desc.get || desc.set)) {
      Object.defineProperty(w, "value", {
        configurable: true,
        enumerable: desc.enumerable,
        get() { return desc.get ? desc.get.call(this) : undefined; },
        set(v) { desc.set?.call(this, v); onChange(); },
      });
    } else {
      let stored = w.value;
      Object.defineProperty(w, "value", {
        configurable: true,
        enumerable: true,
        get() { return stored; },
        set(v) { stored = v; onChange(); },
      });
    }
  } catch (e) {
    console.warn("[apnext.h3.project_name] could not watch seed widget", e);
  }

  const callback = w.callback;
  w.callback = function () {
    const r = callback?.apply(this, arguments);
    onChange();
    return r;
  };
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
      watchSeed(this);
      return r;
    };

    // Workflow loaded from disk: configure() restores the saved (possibly
    // empty) value after onNodeCreated, so top up again here.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      fill(this);
      watchSeed(this);
      return r;
    };
  },
});
