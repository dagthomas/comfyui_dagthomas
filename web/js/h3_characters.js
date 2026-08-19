// APNext H3 Characters - franchise follows the picked character
//
// `franchise_filter` only narrows the pool when `character` is "random". When
// a named character is picked, this snaps the filter to that character's show
// so flipping to random afterwards draws a castmate instead of anyone at all.
// Labels look like "Character — Actor (Show)"; the show is matched against the
// filter's own option list, so shows containing parentheses still resolve.

import { app } from "../../../scripts/app.js";

const NODE_CLASS = "H3Characters";
const RANDOM_PREFIX = "🎲";

function franchiseOf(label, options) {
  if (!label || label.startsWith(RANDOM_PREFIX)) return null;
  let best = null;
  for (const opt of options) {
    if (opt === "(all)") continue;
    if (label.endsWith(`(${opt})`) && (!best || opt.length > best.length)) best = opt;
  }
  return best;
}

app.registerExtension({
  name: "apnext.h3.characters",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_CLASS) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const character = this.widgets?.find((w) => w.name === "character");
      const franchise = this.widgets?.find((w) => w.name === "franchise_filter");
      if (!character || !franchise) return r;

      const sync = () => {
        const options = franchise.options?.values || [];
        const show = franchiseOf(character.value, options);
        if (show && franchise.value !== show) {
          franchise.value = show;
          this.setDirtyCanvas(true, true);
        }
      };

      const prev = character.callback;
      character.callback = function () {
        const out = prev?.apply(this, arguments);
        sync();
        return out;
      };
      return r;
    };
  },
});
