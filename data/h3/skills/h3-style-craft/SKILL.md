---
name: h3-style-craft
description: Turning the APNext H3 node's visual_style, camera and wildness settings into observable MiniMax H3 prompt language - one visual-medium pack plus at most one motion, finish and audio pack, translating a named studio, era, genre or artist into traits, describing traditional animation timing (held poses, stepped cadence, smears, line boil, painted texture) without over-claiming frame rates, and scaling authorial risk to the wildness band. Load with h3-prompt-director whenever a style is stated, a reference look is implied, or wildness is above Conservative.
---

# H3 style and motion craft

## What the node gives you

- **`visual_style`** - either a label to open `[Shot 1]` with (`Cinematic`, `live-action`,
  `2D-animated`, `3D CG`, `claymation`, `watercolor`, `vintage film`) or "choose one that
  fits". The label is the opening words, not the whole style: expand it into craft the video
  model can see.
- **Camera** - motion, amplitude (`with small amplitude` / `with large amplitude`) and speed
  (`at slow speed` / `at fast speed`) in the guide's vocabulary. Use the exact phrases the
  directive gives; describe what the move reveals, not just that it happens.
- **Wildness band** - Conservative and Grounded keep to ordinary physics and motivated
  choices; Bold allows one memorable visual idea; Wild and Unhinged invite surreal logic and
  the node lists concrete surreal elements that must appear. Style intensity follows the
  band: a Conservative watercolour is a quiet, faithful watercolour; an Unhinged one can
  bleed, run and re-form.

## Style construction

Translate any style into observable craft rather than a name. In priority order:

1. medium and material construction (what the image is physically made of)
2. shape, edge and shadow grammar
3. palette and value logic
4. motion and deformation grammar
5. animated surface and texture behaviour
6. cadence terminology

Use at most one dominant visual medium, one motion system, one finish system and one audio
treatment unless the user explicitly asks for a hybrid; then say which layer each medium
controls. When a unified style is asked for, apply it to everything - subject, crowd,
vehicles, props, architecture, signage, pavement, reflections, atmosphere, smears and
transitions - not only the hero.

If the user names a studio, era, genre or artist, translate it into traits and do not rely
on the proper name. Never import recognizable IP characters, costumes, logos or franchise
silhouettes when only the look was asked for.

## Animation and motion vocabulary

Use visible motion terms when they help: anticipation, compression, contact, passing,
suspension, extension, impact, rebound, overshoot, follow-through, overlap, secondary
action, smears, replacement drawings, line boil, animated pigment, moving texture,
registration shift.

Smears are brief transition drawings that resolve immediately into readable anatomy; they
are not generic motion blur.

For ones / twos / fours, limited frame rate or flip-book timing, describe the visible
cadence you want rather than promising literal repeated frames: `visible stepped timing`,
`held key poses`, `selective in-betweens`, `abrupt drawing changes`, `minimal
interpolation`. Exact cadence is verified from frames or enforced in the workflow, never
claimed in the prompt.

## Reference library

| Read this | When |
|---|---|
| `10_H3_STYLE_PICKER_RULESET.md` | Any style, medium or look is requested or implied - how to pick one visual pack and at most one motion / finish / audio pack. |
| `08_H3_AESTHETIC_MOTION_AUDIO_LIBRARY.md` | The pack catalogue (V/M/F/A ids) the picker refers to, with concrete injection language. |
| `09_H3_STYLE_REFERENCE_ANCHORS.md` | The user names a studio, era, genre, artist or production - translate it into traits. |
| `15_H3_TEMPORAL_ANIMATION_TECHNIQUES.md` | Traditional animation timing: ones / twos, held poses, smears, line boil, limited in-betweens. |
