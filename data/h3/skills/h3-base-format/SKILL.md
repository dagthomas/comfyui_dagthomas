---
name: h3-base-format
description: The three-field MiniMax H3 contract the APNext H3 Prompt Writer nodes emit for T2VA, I2VA, FL2VA and L2VA - integrated_multimodal_description / overall_soundscape / non_diegetic_music, the exact first-frame, first-and-last-frame and last-frame alignment lines, how the node's image batch maps to first_frame and last_frame, and how each mode develops from or converges onto its keyframes. Load with h3-prompt-director whenever the task is not Ref2VA.
---

# H3 base format (T2VA / I2VA / FL2VA / L2VA)

This is what the `APNext H3 Prompt Writer` and `APNext H3 Claude Code Writer` nodes
expect back. The node splits the text on these three labels, so they must appear exactly
once, in this order, each followed by one blank line:

```
integrated_multimodal_description: [Shot 1] <style>, <one continuous timeline> ...

overall_soundscape: ...

non_diegetic_music: ...
```

The stated visual style opens `[Shot 1]` (the node names it, or asks you to choose one).

## Alignment lines

T2VA begins directly with `integrated_multimodal_description:`. The other modes put one
alignment line first, then exactly one blank line, then the three fields. Copy the wording
character for character; only `N` (the number of the final shot) and `S.SS` (the duration,
two decimals) change.

I2VA:
`For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.`

FL2VA:
`How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.`

L2VA:
`How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.`

The node's own directive repeats the right line for the chosen task type; if it and this
file ever differ, follow the node.

## How the node's images map

The node sends whatever was connected to its `image` socket as an ordered batch and hands
the same frames back out as `first_frame` and `last_frame` for the video node:

- **I2VA** - the image is `<Picture 1>`, the exact opening frame. Preserve its identity,
  clothing, colours, key objects and geography, then develop forward from it.
- **L2VA** - the image is `<Picture 1>`, the exact final frame, and belongs to the last
  shot. Infer a plausible earlier state, then a transition path, then a gradual convergence,
  then land on the image.
- **FL2VA** - frame 0 of the batch is `<Picture 1>` (opening), the last frame is
  `<Picture 2>` (ending). Describe the change between them and land exactly on the ending.
- **T2VA** with an image attached - the image is visual context to describe from, not a
  keyframe. No alignment line, no `<Picture N>` labels.

Describe what you actually see in the frames: subject, wardrobe, colour, light direction,
objects, spatial layout. Do not paraphrase the user's idea back when the picture already
answers the question.

## Reference library

| Read this | When |
|---|---|
| `11_T2VA_GOLD_EXAMPLES.md` | Task is T2VA - match its density, shot pacing and audio balance. |
| `12_KEYFRAME_GOLD_EXAMPLES.md` | Task is I2VA, FL2VA or L2VA - alignment grammar and how to develop from or converge onto frames. |
| `06_OFFICIAL_GUIDE_CLEAN_COPY.md` | Condensed official guide, if the full guide is not already in your context. |
