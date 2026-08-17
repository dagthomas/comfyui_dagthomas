# H3 Prompt Grammar and Validation Reference

## Canonical structure

T2VA/I2VA/FL2VA/L2VA prompts have three fields in this order:

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

I2VA, FL2VA, and L2VA add their exact alignment instruction before this body, followed by one blank line. T2VA has no alignment instruction. Ref2VA instead uses the six-section full-reference structure documented in `07_REF2VA_FULL_REFERENCE_GUIDE.md`.

## Exact alignment lines

I2VA:

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
```

FL2VA:

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.
```

L2VA:

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.
```

`N` is the actual final shot. `S.SS` is the effective duration with exactly two decimals.

## Timeline writing

Start Shot 1 with the medium/style, composition, subjects, environment, and current object states. Do not timestamp Shot 1.

Later shots use sequential numbers and strictly increasing cut times:

```text
[Shot 2] At 00:03.500, the camera cuts to...
```

A cut should reveal meaningfully new subject, space, state, viewpoint, or time. Small changes in distance or angle usually belong to camera motion within one shot.

## Camera vocabulary

Use camera motion as natural prose:

- Zoom In / Zoom Out: lens changes focal length while the body stays fixed.
- Push In / Pull Out: camera moves forward/backward.
- Pan Left / Pan Right: fixed camera pivots horizontally.
- Truck Left / Truck Right: camera translates horizontally.
- Tilt Up / Tilt Down: fixed camera pivots vertically.
- Pedestal Up / Pedestal Down: entire camera moves vertically.
- Arc Shot: camera moves around the subject.
- Tracking Shot: camera follows a moving subject.
- Static Shot: camera and lens remain still.
- Shake Slightly / Shake Strongly: controlled camera shake.
- POV: subject's point of view.
- Roll Clockwise / Roll Counterclockwise: rotation around the lens axis.

Add `with small amplitude` or `with large amplitude` only when useful. Add `at slow speed` or `at fast speed` only when useful. Normal speed and medium amplitude are implicit.

Good:

`The camera pushes in with small amplitude at slow speed toward the key in her palm.`

Avoid detached command stacks such as `Camera: push in, slow, small amplitude.`

## Keyframe paths

I2VA path:

`first-frame anchor → action onset → continuous development → result or reaction`

FL2VA path:

`first-frame state → observable intermediate changes → narrowing differences → exact last-frame state`

L2VA path:

`plausible preceding state → explicit causal transition → convergence in final shot → exact last-frame landing`

Ref2VA structure:

`subject_definitions → summary → retention_analysis → detailed_description → overall_soundscape → non_diegetic_music`

## Dialogue and singing

Assign stable IDs only to vocal subjects. On first vocal appearance, identify the person and voice outside the dialogue tag:

```text
The older woman with a low, textured voice (S1) says: <d>[English] Leave the light on.</d>
```

Inside `<d>`, include only the language label and exact words. Preserve the user's content verbatim.

For voiceover:

```text
The man (S1) says in an off-screen voiceover: <d>[English] I still remember that road.</d> while his lips remain completely closed.
```

If speech crosses a cut, place `<scenetrans>` at the connecting point in both segments and explicitly state that the audio continues across the cut. Use `<cutoff>` when the clip ends mid-utterance.

## Visible text

Quote every visible banner, sign, label, subtitle, UI string, or neon message with English double quotation marks. Preserve spelling and punctuation:

`A red sign reading "营业中" glows above the doorway.`

## Sound separation

Integrated description:

- Dialogue and singing.
- Diegetic music or media audio audible to characters.
- Sounds synchronized to particular visual events when timing matters.

Overall soundscape:

- Ambient sound.
- Physical action and object sounds.
- Non-verbal human sounds.
- One paragraph, 1–4 English sentences.
- `N/A` only for explicitly requested total silence.

Non-diegetic music:

- Audience-only score.
- Instrumentation, tempo, rhythm, and dynamic change.
- 1–3 English sentences.
- `N/A` when absent.

Avoid abstract score descriptions such as `emotional music`. Prefer `Sparse piano notes at a slow tempo, joined by sustained low strings that rise gradually in volume.`

## Density control

H3 can follow complex instructions, but the clip still has finite time. Use one dominant action beat per roughly 1–3 seconds. For short clips, prefer one location, one causal action chain, and one purposeful camera move. Multi-shot is supported; it is not a reason to cram unrelated scenes into five seconds.

## Silent validator

Before delivering a prompt, verify:

1. Mode, alignment line, duration, and final shot number agree.
2. All reference tags exist and their roles are explicit.
3. Shot numbers and cut times are valid.
4. Keyframes are treated as boundary states, not merely restated.
5. Dialogue is exact and speakers remain stable.
6. Soundscape and non-diegetic music contain the correct kinds of sound.
7. No template placeholders remain.

## Style-pressure grammar — empirical revision 2026-08-13

When the user is intentionally stress-testing style strength, the prompt should not rely on a style name, cadence term, or surface adjective alone. Describe the style through a small set of mutually reinforcing observable constraints.

Use this priority order:

1. **medium geometry** — what forms are physically or graphically made from: cut paper, clay, silhouette, liquid, marble, geometric blocks, inked cel shapes;
2. **shape grammar** — contour simplification, proportion language, shadow construction, edge type, internal mark-making;
3. **palette logic** — restricted color families, flat separations, posterized values, off-register layers, fluorescent relationships;
4. **motion grammar** — pose rhythm, anticipation, compression, extension, smears, replacement shapes, follow-through, recovery;
5. **surface process** — grain, bleed, hatch boil, paint crawl, registration drift, paper fibers, brush texture;
6. **cadence vocabulary** — on twos, on fours, on eights, exposure patterns.

For H3 style forcing, the earlier items generally carry more visual leverage than the later items. Cadence terminology is especially weak when it is the only style cue.

### Whole-frame coverage rule

If the requested look should dominate the video, explicitly apply it to:

- primary subject;
- secondary subjects and crowds;
- props and vehicles;
- architecture and background;
- lighting/reflections;
- transitions and smear drawings;
- frame-to-frame texture behavior.

A style that affects only the hero can read as compositing rather than a coherent world.

### Style reinforcement rule

Prefer 4–6 correlated cues that all imply the same visual system rather than a long list of unrelated adjectives.

Good:

`articulated cut-paper silhouettes, visible paper edges, flat hinged limbs, layered parallax, stepped replacement poses, slight registration jitter`

Weak:

`vintage, artistic, handmade, cinematic, painterly, animated`

### Style-vs-timing separation

Do not assume that a visible style and an exposure cadence are the same control problem. A prompt can strongly transfer a cut-paper or comic-print look while failing to produce literal on-fours timing. Treat cadence as a separate temporal request and avoid weakening the main style description by overloading it with frame-count language.
