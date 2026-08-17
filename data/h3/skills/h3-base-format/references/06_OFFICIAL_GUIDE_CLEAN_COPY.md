# Video Prompt Writing Guide — T2VA / I2VA / FL2VA / L2VA

This is a clean retrieval copy of the official guide supplied for this project. When this guide and a later official MiniMax/ComfyUI source disagree, report the conflict rather than silently blending them.

## Task overview

- T2VA builds a complete audiovisual timeline from text.
- I2VA uses the T2VA body plus a first-frame instruction and a visual path that develops forward from the first frame.
- FL2VA uses the T2VA body plus a first-and-last-frame instruction and a continuous path from the first frame to the last frame.
- L2VA uses the T2VA body plus a last-frame instruction and a path that converges from a plausible preceding state to the last frame.

## Final prompt structure

### Instruction line

T2VA has no image-alignment instruction.

I2VA always uses:

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
```

FL2VA always uses:

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.
```

L2VA always uses:

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.
```

`N` is the actual final shot. `S.SS` is the effective duration with exactly two decimal places. The instruction is the first line, followed by one blank line.

### Three core fields

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

The integrated description contains visuals, actions, shots, speakers, dialogue, singing, and diegetic audio along the timeline. The soundscape summarizes ambient, physical-action, and non-verbal human sounds. Non-diegetic music is audience-only background music.

## Keyframe logic

I2VA: `<Picture 1>` is the actual first frame at 0.00 seconds and belongs to Shot 1. Establish style, subjects, composition, scene anchors, identity, clothing, colors, objects, and spatial relationships before developing action forward.

Path: first-frame anchor → action onset → continuous development → result or reaction.

FL2VA: Picture 1 is the opening and Picture 2 the ending. Describe how subject movement, pose, object state, composition, scene, and lighting change between them. Prefer one continuous shot unless multiple shots are explicit. Reach the final frame in final Shot N.

Path: first-frame state → observable intermediate changes → progressively narrowing differences → last-frame state.

L2VA: `<Picture 1>` is the final frame and belongs to final Shot N, not inherently Shot 1. Infer a plausible earlier state and describe gradual convergence.

Path: plausible preceding state → explicit action and transition → final-shot convergence → last-frame landing.

## Integrated timeline

At the start of Shot 1, state the overall style and initial composition. Possible styles include cinematic, live-action, 2D animation, 3D CG, claymation, watercolor, and vintage film. Derive keyframe-task style from the reference; select T2VA style from user intent.

Do not timestamp the first shot. Later shots are sequential and begin with strictly increasing cut times inside the duration:

```text
[Shot 2] At 00:03.500, the camera cuts to...
```

Ordinary transitions include `the camera cuts to`, `the shot cuts to`, `the shot transitions to`, `the shot changes to`, and `the shot switches to`. Use cross-dissolve, fade, or wipe only when requested. A cut should introduce new subject, space, state, viewpoint, or time information. Prefer camera motion for small distance or angle changes.

## Camera motion

Motion types: Zoom In/Out, Push In/Pull Out, Pan Left/Right, Truck Left/Right, Tilt Up/Down, Pedestal Up/Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly/Strongly, POV, and Roll Clockwise/Counterclockwise.

Amplitude: `with small amplitude` or `with large amplitude`.

Speed: `at slow speed` or `at fast speed`.

Add amplitude and speed only when meaningful. Medium amplitude and normal speed are usually omitted. Write motion as natural action:

```text
The camera pushes in with small amplitude at slow speed toward the folded letter in her hands.
```

## Speakers, dialogue, and singing

Use stable `(S1)`, `(S2)` IDs for speakers, singers, or off-screen human voices. Use compound IDs such as `(S1,S2)` when established speakers vocalize together. Non-vocal characters receive no ID.

On first vocal appearance, establish stable identity using relevant character and voice traits. Put the identifying phrase, ID, action, and delivery outside `<d>`. Inside `<d>`, include only the language tag and verbatim user-provided words and punctuation. Do not translate or rewrite them.

```text
The young woman with a quiet, breathy voice (S1) says: <d>[English] I get off at the next station.</d>
```

For voiceover, use the exact phrase `says in an off-screen voiceover`. Immediately after its `<d>` block, state that the corresponding on-screen character's lips remain closed.

When a line crosses a cut, use `<scenetrans>` at the connecting points in both parts and explicitly state that audio continues across the cut. Use `<cutoff>` when speech is truncated by the video end.

## On-screen text

Place visible banners, signs, labels, subtitles, or neon text in English double quotation marks. Preserve original spelling and punctuation without translation.

## Overall soundscape

Use one continuous paragraph of 1–4 English sentences for ambience, physical-action sounds, and non-verbal human sounds. Do not repeat dialogue, singing, or diegetic music. Use `N/A` only for explicitly requested complete silence.

## Non-diegetic music

Use 1–3 English sentences for audience-only music. Describe instrumentation, speed, rhythm, and dynamics rather than abstract mood or emotional purpose. Singing and music from instruments, radios, televisions, or phones audible to characters are diegetic and belong in the integrated description. Use `N/A` when there is no non-diegetic music.
