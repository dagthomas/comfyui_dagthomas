# H3 Ref2VA Video-Reference Style Transfer — Empirical Guide

Version: 1.1
Status: Empirical prompting reference, not an official MiniMax specification  
Updated: 2026-08-13
Purpose: Help H3 Prompt Director use Ref2VA video inputs as style, motion, performance, editing-rhythm, or material-technique references while keeping the user's subject, action, setting, composition, props, dialogue, and story independently controllable.

## 1. Target behavior

The practical goal is adapter-like separation:

**Reference video controls HOW. User prompt controls WHAT.**

For style-transfer requests, use `<Video N>` to transfer abstract, non-identifying attributes rather than source content. The desired transfer may include:

- visual surface: line behavior, palette relationships, fill density, texture, brush character, graphic density, background treatment;
- motion grammar: pose rhythm, holds, anticipation, acceleration, deformation, smear-like transitional drawings, overshoot, follow-through, secondary motion, recovery;
- acting grammar: eye/head hierarchy, thought-to-pose progression, gesture timing, weight and balance;
- temporal/compositional grammar: cut rhythm, shot-scale changes, viewpoint changes, graphic reframing, action topology;
- physical-process cues: redraw, paint redistribution, charcoal erase/rebuild, sand movement, clay deformation, or other material behavior.

Do not assume all layers transfer equally well.

## 2. Evidence status and observed transfer hierarchy

Controlled H3 Ref2VA tests across sparse line animation, graphic black/white animation, painterly animation, wet oil-paint-on-glass references, and acting references showed the following tendencies.

### Relatively strong transfer

- broad palette and color-density logic;
- graphic density and shape simplification;
- sparse versus dense rendering;
- high-contrast black/white mass relationships;
- pose rhythm and readable holds;
- anticipation, acceleration, overshoot, follow-through, and recovery;
- expressive transitional deformation and smear-like poses;
- nuanced acting hierarchy when explicitly reinforced;
- cut rhythm and shot-scale contrast when explicitly described;
- graphic object-motion organization.

### Weaker or high-variance transfer

- exact frame-by-frame redraw cadence;
- literal line-boil frequency during held poses;
- exact exposure patterns;
- physical wet-paint redistribution;
- exact charcoal erase/rebuild or similar destructive material process;
- exact source editing rhythm when the prompt merely says "let the reference decide";
- strong material-process fidelity under complex human anatomy, props, dialogue, and environments.

Treat these as model tendencies, not guarantees.

## 3. Default style-transfer policy

When the user says "use this video for the style" and does not ask to copy source content:

1. Use Ref2VA.
2. Preserve official label semantics. Define reusable visual style, visible performance, action language, or material behavior as `<Subject N>` sourced from `<Video N>`.
3. Keep standalone `<Video N>` for whole-video relationships such as camera, cuts, rhythm, temporal structure, editing, or continuation.
4. A single source video may support both a derived `<Subject N>` and a `<Video N>` temporal role.
5. In `retention_analysis`, normally use `attribute_transfer` for abstract style/performance content, not `fully_preserved`.
6. State that the reference controls the animation language while the user's subject, action, setting, props, composition, dialogue, and story remain prompt-controlled.
7. Explicitly reinforce only the important attributes that are visible in the reference and relevant to the user's request.
8. Avoid adding competing named style labels unless the user asks for them.
9. Add a content firewall when reference pressure could leak source character design or semantics.

Reusable role wording:

`<Subject 1> is the abstract animation style sourced from <Video 1>, including [specific non-identifying surface/performance traits]. It excludes source character identity, anatomy, clothing, props, setting, exact composition, exact poses, exact actions, dialogue, and story content.`

If whole-video timing matters, add:

`<Video 1> is the whole-video temporal reference for [pose rhythm / cut timing / camera path / shot-scale relationships].`

Reusable retention wording:

`<Subject 1>: attribute_transfer — transfer the reference-derived visual and performance language while preserving the user-defined subject, action, setting, props, composition, dialogue, and narrative.`

## 4. Analyze before writing

For a video-reference style request, silently classify the reference into the following attribute families.

### A. Visual surface

Look for:
- line thickness, roughness, boil, redraw instability;
- flat versus modeled shapes;
- palette relationships and saturation;
- negative-space usage;
- fill density;
- pigment or material texture;
- shadow and highlight treatment;
- background rendering density;
- edge softness, breakup, or dissolution.

### B. Motion grammar

Look for:
- held poses versus continuous interpolation;
- anticipation;
- acceleration/deceleration;
- pose spacing;
- arcs;
- squash/stretch;
- smear-like transitional drawings;
- overshoot;
- follow-through;
- secondary motion;
- recovery rhythm.

### C. Acting grammar

Look for:
- eyes leading head;
- head leading torso;
- thought pauses;
- expression timing;
- posture and balance changes;
- gesture economy;
- weight transfer;
- reaction hierarchy.

### D. Temporal/compositional grammar

Look for:
- direct cuts;
- shot-length contrast;
- extreme scale changes;
- viewpoint changes;
- graphic reframing;
- repeated compositional motifs;
- action topology such as hold → anticipation → burst → landing → settle.

### E. Material process

Look for whether the source visibly depends on:
- continual redraw;
- wet paint being pushed and repainted;
- charcoal being erased/rebuilt;
- sand being redistributed;
- clay being resculpted;
- collage pieces being physically repositioned.

Material appearance and material process are different targets.

## 5. Reference-role patterns

### Style only

Use when the user wants the look but supplies their own motion.

`<Subject 1>` sourced from `<Video 1>` controls line, palette, fill, texture, graphic density, background treatment, and surface behavior. Do not ask it to control action timing unless requested.

### Motion/performance only

Use when the user wants new rendering but reference-derived movement.

`<Subject 1>` sourced from `<Video 1>` controls visible pose/performance traits; keep `<Video 1>` itself only if whole-video temporal relationships are also being referenced. Explicitly state that source visual rendering is not required.

### Style + motion

Use for the default "animate this in the reference video's style" behavior.

`<Subject 1>` sourced from `<Video 1>` controls visual surface and visible motion/performance grammar; add `<Video 1>` as a temporal source only when whole-video rhythm/cuts/camera matter. User content remains authoritative.

### Editing/temporal structure

Use when style lives partly in cuts or reframing.

Keep `<Video 1>` as the source for cut rhythm, shot-scale relationships, viewpoint contrast, and temporal structure. Explicit cut times in the target prompt are more reliable than merely telling H3 to infer source editing rhythm.

### Material process

Use only when the user specifically cares about how the animation is physically constructed. Explicitly describe the visible process and keep scene complexity conservative when fidelity is the priority.

## 6. Multi-reference strategy

Do not automatically blend multiple reference videos equally.

Prefer **role separation** when references contribute different strengths:

- `<Subject 1>` from `<Video 1>` — surface/color/line treatment;
- `<Subject 2>` from `<Video 2>` — visible performance/deformation;
- `<Video 3>` — whole-video cuts/composition/temporal structure.

A material transformation used as reusable visible content should likewise be a `<Subject N>` sourced from its video.

Role-separated references behaved more predictably than vague "use all three for style" assignments.

Use an equal broad blend only when the user explicitly wants a consensus style and the references are compatible.

Never assume more references increase fidelity. Strong multi-reference pressure can increase source-content leakage.

## 7. Content firewall

When the reference is for style rather than content, protect the user's scene explicitly.

The firewall may exclude:
- character identity and facial construction;
- body proportions and anatomy;
- hairstyle and costume;
- props and product categories;
- logos, branding, packaging, vehicles;
- locations and background layouts;
- exact compositions and poses;
- exact gestures/actions;
- dialogue and narrative events.

For strict independence:

`Transfer HOW the reference is drawn and animated, not WHAT appears in it. The new subject, anatomy, clothing, props, setting, composition, action, dialogue, and story are controlled entirely by this prompt.`

If character-design leakage appears likely, add explicit user-defined identity anchors such as compact stature, round face, small nose, hair silhouette, or other requested traits. Do not over-specify unnecessary rendering details, because excessive subject styling can compete with reference-style transfer.

## 8. Avoid competing style vocabulary

A strong video reference may already provide the style prior.

Do not automatically add phrases such as:
- "traditional cel animation";
- "watercolor background";
- "clean digital cartoon";
- "cinematic painterly illustration";

unless the user explicitly asks for them or the reference clearly needs reinforcement.

Empirically, extra medium labels sometimes pulled H3 toward a generic learned style instead of the supplied video reference.

Prefer:
1. reference role;
2. observable reference attributes;
3. only the minimum corrective language needed.

## 9. Temporal correspondence

When a reference has distinctive motion grammar, a new action with analogous **abstract temporal topology** can improve transfer without copying content.

Example topology:

`held setup → small anticipation → sudden acceleration → extreme transitional state → clean landing → secondary settling`

The target action may be completely different from the source action. Preserve relationships, not literal poses or narrative beats.

## 10. Physical-process transfer and complexity budget

For difficult material animation, separate:

- **surface style transfer**: "looks like wet oil paint";
- **material process transfer**: "each new state is physically rebuilt by redistributing wet paint."

The latter is much harder.

Observed pattern:
- simple abstract forms can show material transformation;
- recognizable objects encourage object permanence;
- articulated humans strongly encourage stable anatomy;
- detailed environments, props, dialogue, and identity constraints further reduce process fidelity.

When the user prioritizes material-process fidelity:
- reduce simultaneous semantic demands;
- favor one subject and one causal action;
- use a locked or simple camera;
- choose an action compatible with the material transformation;
- remove unnecessary dialogue and environment detail;
- explain the physical mechanism explicitly.

Do not promise exact physical-process replication.

## 11. Hand-drawn redraw and line boil

H3 can transfer:
- sparse line construction;
- expressive pose deformation;
- smear-like transitional drawings;
- redraw-like visual character.

It is less reliable at:
- exact redraw frequency;
- literal whole-frame redraw on every playback frame;
- exact line-boil cadence during holds.

When line boil matters, describe the visible behavior, but do not claim a literal cadence unless the output is inspected. Use `15_H3_TEMPORAL_ANIMATION_TECHNIQUES.md` for exposure/cadence diagnostics.

## 12. Recommended prompt skeleton

```text
subject_definitions:
<Subject 1> is the abstract animation style sourced from <Video 1>, including [specific surface/performance traits]. It excludes source character identity, anatomy, clothing, props, branding, settings, exact compositions, exact poses, exact actions, dialogue, and narrative content.
<Video 1> is the whole-video temporal reference for [only if needed: pose rhythm, cuts, camera, or other temporal structure].

summary:
[reference generation] The target video applies <Subject 1>'s abstract animation language and, when needed, <Video 1>'s temporal structure to a completely new user-defined subject, action, and setting.

retention_analysis:
<Subject 1>: attribute_transfer — transfer [selected visual/performance families] while preserving the user-defined content.
<Video 1>: attribute_transfer — transfer [only the whole-video temporal relationships actually used].

detailed_description:
Use <Subject 1> continuously as the dominant source for HOW the animation looks and performs. When present, <Video 1> controls only the declared temporal relationships. The prompt controls WHAT is depicted. [Then describe the user's scene and explicitly reinforce only the reference attributes that matter.]
```

Follow the exact summary syntax required by `07_REF2VA_FULL_REFERENCE_GUIDE.md`; the bracketed prefix above reflects this kit's current official-format examples.

## 13. Decision tree

When user provides a video and asks to use its style:

**Does the user want source content copied?**
- Yes → define reusable `<Subject N>` items for the requested source content and preserve only those.
- No → derive abstract style/performance as `<Subject N>` from the source video and add a content firewall; keep `<Video N>` only for whole-video temporal roles.

**What seems to define the style?**
- Surface → reinforce line/color/fill/texture.
- Performance → reinforce timing/pose/deformation/acting hierarchy.
- Editing → reinforce cuts/scale/viewpoint rhythm.
- Physical process → explicitly describe the material mechanism and warn internally that fidelity is high variance.
- Several layers → combine only the needed families.

**Multiple reference videos?**
- Same role, compatible → broad blend is allowed.
- Different strengths → separate responsibilities.
- Strong semantic leakage risk → reduce references or strengthen the firewall.

## 14. Failure diagnosis

### Looks stylistically generic
- Remove competing named-medium language.
- Make the video-derived `<Subject N>` the dominant abstract style prior.
- Reinforce the most visible surface traits.

### Correct look, wrong motion
- Add motion grammar explicitly.
- Match the reference's abstract action topology.
- If cuts are important, write target cut times.

### Correct motion, wrong subject
- Strengthen content firewall.
- State user-defined identity anchors.
- Change `fully_preserved` on the video to `attribute_transfer` unless literal source content is intended.

### Looks handmade but lacks actual redraw/process
- Do not keep adding synonyms indefinitely.
- Treat exact process as high variance.
- Reduce semantic complexity if process fidelity is essential.

### Multi-reference result becomes source-like
- Reduce the number of references.
- Give each reference one narrow role.
- Exclude source subject/prop/location semantics explicitly.

## 15. Documentation rule

When using empirical R2V findings in user-facing answers or generated prompts:
- describe them as practical tendencies, not official MiniMax guarantees;
- do not claim H3 has an internal "style embedding" or IP-Adapter mechanism;
- use "adapter-like" only as an analogy for the desired separation of style from content;
- never claim exact cadence, redraw frequency, or physical production behavior without output inspection.

## 10. Weak-still style stress-test findings

A separate controlled Ref2VA series used one still image at deliberately weak influence, preserving only the broad astronaut-riding-unicorn premise while the prompt controlled all style, staging, palette, and animation treatment.

Observed pattern:

- strong material/geometry cues transferred well;
- strong shape and palette systems transferred well;
- whole-frame stylization helped prevent reversion to photographic/cinematic baseline;
- explicit smear, anticipation, follow-through, line boil, and animated texture could reinforce a style;
- exact exposure cadence remained high variance;
- subtle painterly or post-process-only distinctions were easier for the model to wash out.

### Practical implication

For style transfer, the most effective "HOW" description is not merely a style label. It is a compact generative system describing:

- what the image is made from;
- how shapes are built;
- how values/colors are separated;
- how motion deforms forms;
- how the surface changes from drawing to drawing.

### Style firewall for weak stills

When the reference still is concept-only, explicitly free the following:

`composition, character design, anatomy stylization, costume specifics, architecture, palette, lighting, camera staging, texture, and animation treatment`

Then separately lock only the premise the user actually wants.

### Strong-prompt template

`Use [medium/construction]. Build forms with [shape/edge/shadow grammar]. Restrict color to [palette/value logic]. Animate with [pose/smear/follow-through grammar]. Let [surface process] change subtly frame to frame. Apply the same system to subject, crowd, vehicles, architecture, signs, pavement, reflections, and transitions.`

This template is preferable to adding multiple style names or relying on `in the style of` language.
