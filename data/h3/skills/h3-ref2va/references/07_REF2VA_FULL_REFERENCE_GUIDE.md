# MiniMax H3 Ref2VA Full-Reference Guide

Source: [MiniMaxAI/MiniMax-H3 — Full-Reference Mode Rewrite Output Format Guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)

This is a retrieval-oriented reference for the official MiniMax guide. Use it for Ref2VA/full-reference prompts. T2VA/I2VA/FL2VA/L2VA continue to use the separate three-field format.

## Required six-section structure

Write all six sections in English, preserving original language only for dialogue/lyrics inside `<d>` and visible text:

```text
subject_definitions:
...

summary:
...

retention_analysis:
...

detailed_description:
...

overall_soundscape:
...

non_diegetic_music:
...
```

Make `detailed_description` explicit rather than a plot summary or reference list. Establish composition, appearance, positions, environment, lighting, actions, state changes, camera, current sound, and when references take effect in every shot.

## Reference labels

Each label retains one meaning throughout all sections.

### `<Subject N>`

Reusable visible content abstracted from reference assets: person, animal, object, environment, clothing, prop, interface, visual effect, style, action, expression, or pose. It represents content used in the output, not the source file itself. One subject may use several sources; one source may define several subjects.

```text
<Subject 1> is the woman whose appearance comes from <Picture 1> and whose walking motion comes from <Video 1>.
```

### `<Picture N>`

Use a standalone picture label only when the image itself is a concrete first frame, keyframe, last frame, edited keyframe, composition anchor, or storyboard/planning reference. If it only defines a character, scene, costume, or style, cite it inside the relevant subject definition instead.

```text
<Picture 2> is the first frame of [Shot 1], showing a woman beside a café window.
```

### `<Video N>`

Reserve for whole-video relationships: directly editing a source video, continuing from it, or referencing its camera, cuts, rhythm, or temporal structure. A person/object/action taken from a video remains a `<Subject N>`; the video label does not replace subject labels.

### `<Audio N>`

Represents a standalone audio asset or an enabled synchronized track from a reference video. It may provide copied audio, music style, voice timbre/delivery, dialogue or lyrics, effects, beat, rhythm, or continuity.

If bound to a defined speaking subject, reuse the target speaker ID:

```text
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).
```

Video and audio labels have independent numbering. A source video may be `<Video 1>` while its enabled audio is `<Audio 2>`. A video file containing sound does not automatically create an audio label; the audio must be enabled and used.

## `subject_definitions`

Give each tracked item one line. Define its label, role, attributes to follow, and source provenance when needed. Do not create separate picture/video entries when the source tag only supports a subject and is not used independently later.

## `summary`

Use one short paragraph beginning with one or more exact task types joined by ` + `:

- `keyframe completion`: image is a concrete frame anchor.
- `reference generation`: media guides identity, scene, style, action, camera, storyboard, etc.
- `video editing`: a source video is directly modified.
- `video continuation`: output extends or resumes a source video.
- `audio reuse`: the same audio signal is copied in whole or part.
- `audio reference`: signal is not copied; its style, timbre, content, texture, beat, or continuity guides generation.

Examples:

```text
[reference generation + audio reference] ...
[video editing + reference generation + audio reuse] ...
```

Mere presence of video/audio does not create a task type. Camera/cut/rhythm guidance from a video is normally `reference generation`, not editing. Introduce no new labels in the summary. For editing, begin after the prefix with `The target video is an edited version of <Video 1>.`

## `retention_analysis`

Use one line per label, keeping its defined role unchanged.

Fixed visible-content markers:

- `fully_preserved`: defined role fully retained.
- `partially_preserved`: still used, with some defined traits changed or only partly retained.
- `attribute_transfer`: traits transferred to a different identifiable target.
- `weak_reference`: broad style/category/composition/atmosphere similarity only.

```text
<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - ...
<Picture 2> ([Shot 1] first frame): fully_preserved - ...
<Video 1> (cut and pacing structure): weak_reference - ...
```

Fixed audio markers:

- `fully_copy`: entire source audio is the entire final audio track.
- `partially_copy`: only part/layers are copied or sounds are added, removed, or replaced.
- `reference`: signal is not copied; attributes/content guide generation.
- `weak_reference`: broad category/atmosphere similarity only.

Do not count newly added target actions, backgrounds, or plot events as reference-fidelity losses.

## `detailed_description`

Before `[Shot 1]`, establish overall style in one or two English sentences. Then write the target timeline. Shot 1 has no timestamp; later shots use `[Shot N] At MM:SS.mmm, ...`. Use the shared camera, dialogue, transition, and visible-text grammar.

Generation tasks normally use 350–500 English words. Dialogue-dense content prioritizes the complete spoken timeline. Editing descriptions scale with source complexity. A single shot does not automatically justify a short description.

At first clear appearance, describe each important subject's referenced attributes, position, and visible action. Continue using the label without redefining it. Use natural anchor phrases such as `the shot begins from <Picture 1>`, `the shot's keyframe corresponds to <Picture 2>`, and `the shot ends on <Picture 3>`.

## Speakers and referenced audio

When a referenced subject speaks, combine visual label and global speaker ID:

```text
<Subject 2> (S1) turns and says: <d>[English] Last summer, I went home.</d>
```

Assign `(Sx)` once in order of actual vocal events. Reuse it in the description and any audio definition bound to that target speaker. Do not put speaker IDs in retention analysis.

Verbal content heard only inside directly reused background music or a complete soundtrack uses `<Audio N>` as source without inventing a speaker. A voice physically produced by a person, character, or narrator receives `(Sx)`.

For copied or explicitly reperformed dialogue/lyrics, preserve exact source language and words inside `<d>`. Use `[unclear]` rather than guessing. When only timbre, rhythm, emotion, or delivery is referenced, do not import the original words.

## Audio sections

`overall_soundscape` summarizes ambience and physical sounds. `non_diegetic_music` describes audience-only score. Put synchronized dialogue, lyrics, and shot-specific audio events in `detailed_description`.

When reference audio supplies a layer, cite its relationship in the matching section:

```text
overall_soundscape: The copied ambience layer from <Audio 1> continues throughout the target video.
non_diegetic_music: <Audio 2> is directly reused as the complete audience-only score.
```

Do not repeat complete dialogue or lyrics in either audio-summary field.


## Empirical note — video references used as style/animation priors

The sections above summarize the official full-reference format. The following is **empirical Prompt Director guidance**, not an official MiniMax guarantee.

When a source video supplies style, performance, motion grammar, or editing rhythm rather than literal source-content preservation:

- use `reference generation` as the task type unless the source video is actually being edited or continued;
- preserve official label semantics: define reusable visible style, action, performance, or material behavior as `<Subject N>` sourced from `<Video N>`;
- reserve standalone `<Video N>` for whole-video relationships such as camera, cuts, rhythm, continuation, editing, or temporal structure;
- normally use `attribute_transfer` for abstract style/performance content;
- state which non-identifying attributes should transfer and which user-controlled content must remain independent;
- avoid `fully_preserved` unless literal source content is intentionally retained.

A practical style-transfer role can be written as:

```text
<Subject 1> is the abstract animation style from <Video 1>, including line behavior, palette relationships, deformation, and secondary-motion character.
<Video 1> is the whole-video temporal reference for pose rhythm and cut timing when those relationships are also needed.
```

For multiple video references, explicit role separation is preferable to vague blending when each source contributes a different strength.

For physically manipulated media, distinguish the appearance of a medium from its actual frame-by-frame process. Exact redraw cadence, wet-paint redistribution, charcoal erase/rebuild, and similar physical mechanics can be high variance even when surface appearance transfers well.

See `16_H3_REF2VA_STYLE_TRANSFER_LAB.md` for empirical style-transfer strategy and limitations.

## Empirical addendum — weak still reference + strong prompt-controlled style

The official Ref2VA role grammar remains unchanged. The following is an empirical prompting policy from controlled style stress tests.

When a single still image is connected only to preserve a broad concept and the user wants maximum freedom:

- define a reusable `<Subject N>` sourced from `<Picture N>`;
- use `weak_reference`;
- state exactly which broad concept survives;
- explicitly release composition, palette, character design, proportions, architecture, lighting, and treatment;
- do not separately use the picture as a frame/composition anchor unless the user actually wants that anchor.

Recommended wording pattern:

`<Subject 1>: extremely weak conceptual influence sourced from <Picture 1>; retain only [broad premise]. All character design, proportions, composition, palette, setting details, staging, and animation treatment are newly invented.`

This construction produced more freedom in style experiments than treating the same still as an opening I2VA frame.

## Prompt-controlled style leverage

In the tested weak-reference setup, H3 responded most reliably when style was expressed as a coherent visual system rather than a named influence. Stronger prompts combined several of the following:

- a distinct medium or material construction;
- strong shape grammar;
- restricted or unusual palette relationships;
- specific contour/shadow treatment;
- explicit whole-frame coverage;
- style-specific motion or smear behavior;
- animated surface/process behavior.

Subtle style labels, exact cadence terms, and isolated texture adjectives were less dependable when they were not supported by stronger structural cues.
