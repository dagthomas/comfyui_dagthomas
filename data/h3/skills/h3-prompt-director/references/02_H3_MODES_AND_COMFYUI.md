# H3 Modes and ComfyUI Reference

## Model overview

MiniMax H3 is an omni-modal model that can condition on text, images, video, and audio while jointly generating video and native stereo audio. MiniMax states that H3 can generate up to about 15 seconds and up to 2K. The current native ComfyUI documentation describes output at 24 fps.

Sources:

- [MiniMax H3 announcement](https://www.minimax.io/blog/minimax-h3)
- [ComfyUI MiniMax H3 workflow examples](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- [ComfyUI H3 implementation pull request](https://github.com/Comfy-Org/ComfyUI/pull/15224)

## Mode decision table

| User intent | Kit label | Native conditioning path | Required reference wording |
|---|---|---|---|
| Generate from text only | T2VA | H3 image-to-video node with no boundary frame, using `fl2va` weights | No alignment line |
| Start exactly from one image | I2VA | `first_frame` connected | Exact first-frame line |
| Connect an opening and ending image | FL2VA | `first_frame` and `last_frame` connected | Exact first/last alignment line |
| End exactly on one image | L2VA | `last_frame` connected | Exact last-frame line |
| Use media as identity/style/motion/camera/voice context | Ref2VA | `MiniMaxH3ReferenceToVideo`, using `ref2va` weights | Official six-section full-reference format |

T2VA/I2VA/FL2VA/L2VA are prompt-writing labels that emphasize joint audio-video output. ComfyUI's UI and docs may shorten them to T2V/I2V, while the implementation refers to `t2va`, `fl2va`, and `ref2va` conditioning.

## Reference inputs

The documented Ref2VA limits are:

- Up to 9 reference images.
- Up to 3 reference videos; each may include its own soundtrack.
- Up to 3 standalone reference audio clips.

References are tagged in the exact order connected: `<Picture 1>`, `<Picture 2>`, `<Video 1>`, `<Audio 1>`, etc. MiniMax's full-reference guide additionally defines `<Subject N>` as reusable visible content abstracted from source assets. Each tracked item must have a declared role such as identity, wardrobe, product appearance, visual style, motion, camera move, voice, performance, or sound texture.

Good subject definition:

`<Subject 1> is the chef whose identity and clothing come from <Picture 1> and whose walking motion comes from <Video 1>.`

Weak definition:

`<Subject 1> uses all references.`

The weak form leaves conflicts unresolved. See `07_REF2VA_FULL_REFERENCE_GUIDE.md` for the full six-section contract.


## Ref2VA video style-transfer policy

When a connected reference video is used for **style, animation technique, performance, or editing grammar rather than source content**, treat it as an abstract attribute source.

Default principle:

**Reference controls HOW; the user prompt controls WHAT.**

For style-transfer requests, preserve official label semantics: define reusable visible style/performance as `<Subject N>` sourced from `<Video N>`, while standalone `<Video N>` remains the whole-video source for timing, cuts, camera, rhythm, or temporal structure. A single source video may support both roles.

The transferred attributes may include:

- visual surface: line behavior, palette relationships, fill density, texture, graphic density, background treatment;
- motion grammar: pose rhythm, holds, anticipation, acceleration, deformation, smear-like transitional drawings, overshoot, follow-through, secondary motion, recovery;
- acting grammar: eye/head hierarchy, gesture timing, balance, thought pauses, reaction staging;
- temporal/compositional grammar: cuts, shot-scale changes, viewpoint contrast, reframing, action topology;
- material-process cues: redraw, paint redistribution, erase/rebuild, clay deformation, or related physical-media behavior.

For abstract visual style/performance transfer, normally classify the derived `<Subject N>` with `attribute_transfer`, not `fully_preserved`. Use `<Video N>` separately when whole-video temporal structure is referenced. Reserve preservation markers for source content the user actually wants retained.

When source content should not transfer, explicitly block reference character identity, anatomy, costume, props, products, logos, locations, exact compositions, exact poses, exact actions, dialogue, and story material.

Do not assume additional reference videos increase fidelity. Up to three videos can be connected, but role-separated use is usually clearer:

- `<Subject 1>` from `<Video 1>` — surface/line/color;
- `<Subject 2>` from `<Video 2>` — visible performance/deformation;
- `<Video 3>` — whole-video editing/composition/temporal structure.

If a material transformation is itself reusable visible content, define it as a `<Subject N>` sourced from the relevant video.

See `16_H3_REF2VA_STYLE_TRANSFER_LAB.md` for the empirical decision tree, failure modes, and tested prompting patterns.

## Duration and frame grid

ComfyUI documents a 17-frame-per-block `17k+5` grid at 24 fps, and its example workflows convert requested seconds to a legal frame count. The prompt's end-frame alignment should use the effective rendered duration when known. If only a requested duration is known, use that requested value to two decimals and avoid claiming it is the snapped duration.

For a known frame count:

`effective seconds = frame count / 24`

Use the value visible in the workflow or generated media metadata when exact boundary timing matters.

## Resolution

The current native workflow uses a Resolution Selector and rounds dimensions to a multiple of 32. ComfyUI describes H3's native canvas as a 768-pixel short edge capped at 768×1344 for the base workflow; higher output claims in MiniMax's announcement include its broader/regeneration system. Prompt writing should describe composition and aspect intent, while resolution remains a workflow setting.

## Weight families

- T2VA/I2VA/FL2VA/L2VA use the `fl2va` diffusion weights in the current ComfyUI examples.
- Ref2VA uses separate `ref2va` diffusion weights.

Do not tell a user that changing prompt wording can turn the wrong weight family into Ref2VA conditioning.

## Ref2VA image-size control

ComfyUI documents two `ref_image_size` strategies:

- `match`: scale references to generation resolution for speed.
- `max`: retain up to a 2048-pixel short edge for stronger identity fidelity at greater cost and lower speed.

## Practical prompting principles confirmed by current sources

- Describe the whole scene, then shots and timed changes.
- Keep shots, camera, dialogue, sound effects, and music in one coordinated prompt.
- For reference generation, use exact tags and explicitly assign each reference a job.
- Natural-language relationships between context and target video are a core H3 design principle.

## Boundary between prompt and workflow

The GPT writes the positive prompt. It may explain settings when asked, but it must not fabricate a seed, sampler, scheduler, model filename, resolution, or frame count. Those are workflow parameters, not prompt facts.
