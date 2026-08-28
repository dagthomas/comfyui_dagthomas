// APNext H3 - resizable multiline text fields on nodes
//
// The frontend's Vue nodes render every multiline STRING widget as a
// 64 px textarea with `resize: none` and `overflow: hidden` - two lines of a
// direction, a lyric sheet or scene briefs, and the rest only on hover-scroll.
// This lets the field be dragged taller by its corner (the node grows with
// it) and keeps the scrollbar visible once the text is longer than the box.
// The classic canvas nodes get the same corner handle.
//
// Purely cosmetic: the dragged height is not saved with the workflow.

const CSS = `
/* Vue nodes */
.lg-node-widget textarea { resize: vertical !important; overflow-y: auto !important; max-height: 80vh; }
.lg-node-widget textarea:not(:focus):not(:hover) { scrollbar-width: thin; }
/* classic canvas nodes */
textarea.comfy-multiline-input { resize: vertical !important; }
`;

const style = document.createElement("style");
style.id = "apnext-h3-textarea-resize";
style.textContent = CSS;
document.head.appendChild(style);
