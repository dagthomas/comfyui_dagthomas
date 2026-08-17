---
name: h3-ref2va
description: The six-section MiniMax H3 full-reference (Ref2VA) contract the APNext H3 Reference Prompt Writer nodes emit - subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music - with <Subject N>/<Picture N>/<Video N>/<Audio N> roles, the retention vocabulary, summary task types, how the node's image_1..image_9 sockets become <Picture 1>..<Picture 9>, what its reference_role dropdown means, and reference-controls-HOW / user-controls-WHAT style transfer. Load with h3-prompt-director whenever reference media drives identity, style, motion, camera, performance or voice.
---

# H3 Ref2VA (full-reference mode)

This is what the `APNext H3 Reference Prompt Writer` and `APNext H3 Claude Code Reference
Writer` nodes expect back. No alignment line. Six labels, exactly once, in this order,
each followed by one blank line - the node splits on them:

```
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:
```

## How the node's images map

The node has sockets `image_1 ... image_9`, mirrors ComfyUI's *MiniMax H3 Reference to
Video* node, and passes each image back out on the same-numbered output. So:

- **Attached image k IS `<Picture k>`.** Never renumber, skip or merge pictures; the video
  model will receive them in that order.
- Videos and audio cannot be attached to the writer node. When `reference_notes` describe
  them ("Video 1 is the dance clip", "Audio 1 is her voice"), define `<Video N>` /
  `<Audio N>` from the notes in the order given, and treat those notes as fact.
- Only the first frame of a batched input counts as that reference.

## The `reference_role` directive

The node states how to treat the images:

- **Auto** - decide per image: a `<Subject N>` for reusable visible content, a standalone
  `<Picture N>` for a concrete frame or composition anchor, or only a source cited inside
  another item's definition.
- **Subject** - one `<Subject N>` per image, citing `<Picture k>` inside the definition;
  no standalone picture entries.
- **Picture** - one standalone `<Picture N>` per image, stating which shot and position
  (first frame, keyframe, last frame) it anchors.
- **Style reference only** - no standalone `<Picture N>` entries; fold the style provenance
  into the relevant `<Subject N>` definitions and the style sentence before `[Shot 1]`.
- **Storyboard** - standalone `<Picture N>` entries stating which shots they map to and
  what planning information they carry.

## Roles

- `<Subject N>` = reusable visible reference content, or reusable visual / performance /
  style attributes. One line per subject, sourced from the pictures, videos or audio it
  draws on.
- Standalone `<Picture N>` = frame, keyframe or composition anchor only.
- `<Video N>` = editing, continuation, camera, cuts, rhythm or temporal structure.
- `<Audio N>` = copied or referenced audio.

**Reference controls HOW; the user's idea controls WHAT.** Protect the user's identity,
anatomy, clothing, props, setting, composition, action, dialogue, visible text and story
from leaking in from the source. When only a style is wanted, do not import recognizable
IP characters, costumes, logos, products, vehicles or franchise silhouettes.

For a concept-only still with maximum freedom, define a `<Subject N>` sourced from the
picture, mark it `weak_reference`, retain only the stated premise, and explicitly release
composition, palette, design, lighting and animation treatment.

## Fields

- `subject_definitions:` one line per tracked item.
- `summary:` a short paragraph that begins with the applicable task types in square
  brackets, joined by ` + `: `keyframe completion`, `reference generation`,
  `video editing`, `video continuation`, `audio reuse`, `audio reference`. The node's
  `task_type` directive fixes this prefix when it is not Auto.
- `retention_analysis:` one line per item, using only
  visual - `fully_preserved`, `partially_preserved`, `attribute_transfer`, `weak_reference`
  audio - `fully_copy`, `partially_copy`, `reference`, `weak_reference`
  and saying what is kept and what is released.
- `detailed_description:` one or two style sentences, then `[Shot 1]` and one continuous
  audiovisual timeline. Aim for the node's `word_target` (350-500 words is the norm for
  generation tasks); go longer only when the references demand it.
- `overall_soundscape:` and `non_diegetic_music:` as in the core skill.

## Reference library

| Read this | When |
|---|---|
| `13_REF2VA_GOLD_EXAMPLES.md` | Always for a first Ref2VA prompt in a session: full six-section examples with retention markers. |
| `07_REF2VA_FULL_REFERENCE_GUIDE.md` | Roles, retention vocabulary and section contracts in detail. |
| `16_H3_REF2VA_STYLE_TRANSFER_LAB.md` | A video reference supplies style, motion, editing rhythm or technique - keep WHAT separate from HOW. |
