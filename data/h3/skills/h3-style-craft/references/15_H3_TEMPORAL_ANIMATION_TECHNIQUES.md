# H3 Temporal Animation and Traditional-Technique Reference

Version: 1.1
Status: Empirical prompting reference, not an official MiniMax specification  
Updated: 2026-08-13
Purpose: Help H3 Prompt Director push traditional animation timing, reduced in-betweens, exposure cadence, pose-to-pose motion, smear drawings, anticipation, squash/stretch, follow-through, line boil, and moving hand-painted texture without confusing those goals with freeze-frame editing or duplicated-pose imagery.

## 1. Scope and evidence status

This file records practical observations from controlled MiniMax H3 T2VA tests at 24 fps playback. The tests used short static-camera clips, simple backgrounds, and progressively simplified subjects to isolate temporal behavior.

Treat all findings as model tendencies. They are not guarantees about H3's internal frame synthesis, training data, or renderer. Do not claim a literal exposure cadence unless a generated file or workflow output has been inspected frame by frame.

The primary goal is to translate traditional animation vocabulary into visible temporal behavior that H3 has a better chance of following.

## 2. Core distinction: playback frames versus authored states

Traditional animation terminology describes how long an authored drawing is exposed during playback.

At 24 fps:

- on ones: target 24 unique authored states per second; pattern `A, B, C, D...`
- on twos: target 12 unique authored states per second; pattern `A, A, B, B, C, C...`
- on threes: target 8 unique authored states per second; pattern `A, A, A, B, B, B...`
- on fours: target 6 unique authored states per second; pattern `A, A, A, A, B, B, B, B...`
- on eights: target 3 unique authored states per second; eight playback exposures per authored state

When the user cares about cadence, include both the animation term and its observable playback pattern. Do not rely on `on twos` or `on fours` alone.

## 3. Empirical findings from the controlled test series

### Finding A — terminology alone is weak

Requests such as `animate on twelves`, `on fours`, or `on eights` did not reliably produce literal repeated-frame exposures. H3 often interpreted them as a general limited-animation aesthetic or continued to synthesize smooth in-betweens.

### Finding B — spatial locomotion encourages smoothing

A woman walking from left to right repeatedly encouraged continuous interpolation. The model sometimes represented sparse pose descriptions as multiple spatial copies, exposure-chart imagery, or smooth travel between stated poses.

For cadence experiments, prefer a character walking or marching in place before asking the same cycle to translate across the frame.

### Finding C — explicit state patterns are more useful than jargon alone

The strongest paired-cadence character tests included a direct temporal statement such as:

`frame 1 = Drawing A; frame 2 = the exact same Drawing A; frame 3 = Drawing B; frame 4 = the exact same Drawing B`

and the compact pattern:

`A, A, B, B, C, C`

This does not guarantee literal duplicate frames, but it gives H3 a clearer target than the phrase `on twos` by itself.

### Finding D — long holds can turn into editing semantics

Large cadence values and explicit half-second pose intervals can produce:

- long freeze-like holds,
- smooth movement between the requested landmarks,
- hard-cut pose substitution,
- timing drift,
- or a result that reads as editing rather than hand-drawn exposure.

Use these methods for diagnostics, not as the first choice for natural limited animation.

### Finding E — hard pose substitution can create real discontinuity

A two-pose A/B experiment framed as hard substitution produced genuine held states and abrupt changes more successfully than earlier exposure language. However, the exact requested schedule was not perfectly reliable and the result behaved more like editing than classical in-between reduction.

Use this technique when the desired look genuinely includes abrupt temporal popping, or as a diagnostic to test whether H3 can produce discontinuity at all.

### Finding F — paired cadence appeared strongly in some articulated character tests

Several in-place 2D woman and simplified humanoid walk tests displayed a strong apparent paired temporal rhythm even when prompts requested ON 1, ON 3, or ON 4. This suggests that some articulated character contexts can naturally fall toward a two-frame-like cadence.

Do not generalize this into a universal H3 rule.

### Finding G — 24 fps output is capable of per-frame change

A simple moving-dot control progressed smoothly across successive 24 fps playback frames. Simple geometric and mechanical motion could also remain continuously updated. Therefore a paired character cadence is not adequately explained by the video container being 24 fps.

### Finding H — articulation complexity is not a simple threshold

Simplifying the character into a stick figure, mannequin, robot, or isolated-joint mechanism changed temporal behavior, but the progression was not monotonic. Some simple humanoid walks showed paired behavior; some isolated or multi-joint controls remained smooth.

Do not tell users that a specific number of moving joints causes twos. Subject semantics, action type, rendering style, motion prior, and prompt structure can all interact.

## 4. Recommended cadence-prompting ladder

Use the least forceful level that serves the user's goal.

### Level 1 — aesthetic limited animation

Use when the user wants the feel of limited animation rather than measurable frame repetition.

Prompt traits:

`strong held key poses, sparse in-betweens, clear silhouette changes, decisive contact/down/passing/up positions, stepped pose-to-pose timing, restrained secondary motion, and no generic photographic motion blur`

Do not mention exact frame counts unless the user asks.

### Level 2 — explicit exposure target

Use when the user explicitly asks for ones/twos/fours/eights.

State:

1. playback rate when supplied;
2. desired unique-state rate;
3. the repeated-state pattern;
4. that the overall action speed remains normal;
5. that only in-between density changes.

Example for twos:

`The video plays at 24 fps. Target approximately 12 unique authored character states per second. Each authored state persists for two consecutive playback frames: A, A, B, B, C, C. Keep the walk rhythm normal; reduce unique in-between states rather than slowing the action.`

### Level 3 — controlled in-place diagnostic

If Level 2 remains smooth, reduce spatial complexity.

Use:

- one subject,
- static camera,
- plain background,
- fixed screen position,
- simple arm raise, head turn, march in place, or walk in place,
- no independent secondary motion unless that is the variable being tested.

Change one variable per generation.

### Level 4 — explicit discrete-state diagnostic

If the model still interpolates, define a small number of states and say there are no permitted intermediate states.

Example:

`Only Pose A and Pose B exist. Pose A does not travel toward Pose B. The current complete state is replaced directly by the next state. No intermediate arm angle is permitted.`

Use this to diagnose discontinuity, not as the default language for natural character animation.

### Level 5 — hard-cut substitution

Use only when the user is testing the boundary between animation and editing, or wants a visibly abrupt result.

`Hard cut to the same composition with the new pose; no transitional frame exists.`

This can create edit-like behavior and should not be presented as equivalent to classical animation exposure.

## 5. Failure modes and repairs

### Failure: multiple copies of the character appear

Cause: sparse pose descriptions can be interpreted spatially as a pose chart, onion skin, or motion trail.

Repair:

`Exactly one opaque character is visible at every instant. Previous and future poses never remain on screen. Do not show onion skins, ghost poses, sequential figures, contact sheets, or duplicated bodies.`

Then simplify to an in-place action.

### Failure: requested holds become smooth transitions

Cause: H3 treats scheduled pose landmarks as destinations and fills the interval with motion.

Repair: describe the entire interval as one state only if a true hold is desired, or return to Level 2 with a repeated-state pattern if the goal is limited animation rather than freeze frames.

### Failure: ON 4 or ON 8 collapses toward ON 2

Do not keep stacking stronger `hold`, `freeze`, or `unchanged` language. That often pushes the result toward freeze-frame editing.

Instead:

1. keep the same in-place action;
2. verify whether paired cadence is stable;
3. test ON 1 as a control;
4. test a simpler subject or action;
5. treat literal four/eight-frame repetition as unverified unless output inspection confirms it.

### Failure: animation becomes slow motion

State that action duration and rhythm remain unchanged. Only the number of unique authored states is reduced; successive authored poses cover larger motion increments.

### Failure: motion blur replaces a smear drawing

Specify that the smear is an intentionally authored hand-drawn breakdown pose with elongated anatomy or directional contours. Explicitly reject photographic blur when needed.

## 6. Fundamental traditional-animation techniques

These are generally more reliable when written as observable pose and timing behavior rather than only as terminology.

### Anticipation

Describe a preparation pose that moves opposite or compresses before the primary action.

Useful phrasing:

`brief preparatory lean and knee compression, then the decisive action`

Keep anticipation proportional to the action so it does not become a separate event.

### Squash and stretch

Describe shape change while preserving perceived volume.

Useful phrasing:

`compresses into a broad low squash on impact, then elongates through the fast upward acceleration before returning to normal proportions`

Avoid indiscriminate rubber deformation unless the style supports it.

### Overshoot and settle

State the target, a small controlled pass beyond it, then a diminishing correction.

Useful phrasing:

`lands on the target pose, overshoots slightly, then settles back with one smaller corrective motion`

### Follow-through and overlapping action

Separate the primary body's stop from delayed secondary elements.

Useful phrasing:

`the torso stops first; hair tips and jacket hem continue past the stop, reverse, and settle with diminishing amplitude`

### Smear drawings

Treat smears as authored breakdown drawings, not blur effects.

Useful phrasing:

`during the fastest transition, use one or two hand-drawn smear breakdowns with elongated facial/limb shapes, directional contour trails, and simplified anatomy, then resolve immediately into a clean landing pose`

For one visible subject, do not accidentally request multiple fully rendered bodies at once.

### Pose-to-pose limited timing

Use a small number of strong readable states and sparse in-betweens.

Useful phrasing:

`prioritize decisive silhouette changes and key poses; use only enough intermediate drawings to preserve the action's readability`

### In-between density

When the user says `fewer frames`, translate that into fewer unique intermediate states rather than a slower action.

Useful phrasing:

`maintain the same overall action duration while using fewer unique intermediate poses, so successive authored states have larger spatial differences`

## 7. Hand-drawn line boil and moving texture

For rough hand-drawn 2D animation, distinguish living surface variation from transitional repainting.

Preferred baseline:

`frame-by-frame hand-drawn 2D animation, not a still illustration with added motion; rough medium-fine black contours, persistent nervous line boil, slight frame flicker, visible redraw variation, occasional doubled contours, uneven ink pressure, and faint construction marks`

For painted backgrounds and fills:

`major silhouettes and color-region boundaries remain continuously present and spatially stable; internal pigment grain, watercolor density, paper tooth, tiny edge irregularities, and registration variation subtly cycle among closely matched handmade states`

Treat this as moving texture, not as paint appearing, disappearing, wiping on, or transforming between different backgrounds.

For truly static exposure-cadence tests, suppress line boil and texture cycling because those changes can make repeated authored states look temporally unique even when the body pose is held.

## 8. Scientific comparison protocol

When the user is testing H3 rather than merely seeking a creative result:

1. Keep duration, aspect ratio, camera, subject scale, background, action, and style fixed.
2. Change only one temporal instruction between generations.
3. Prefer visible in-frame labels such as `"WALK CYCLE — ON 1"` only if the user wants easy comparison; title rendering itself is not evidence of cadence compliance.
4. Start with fixed-position or in-place actions.
5. Compare ON 1 and ON 2 before testing more extreme cadences.
6. If ON 3/4/8 collapse toward another cadence, record the observation rather than rewriting it as success.
7. Use simple geometric motion as a control when determining whether a behavior is character-specific.
8. Use frame-by-frame inspection or workflow metadata before making literal frame-count claims.

## 9. Prompt construction for a controlled in-place walk

For an exposure test, establish:

- static locked camera;
- plain stable background;
- exactly one character;
- fixed screen position;
- side profile;
- coherent contact/down/passing/up gait;
- unchanged overall walking rhythm across comparison prompts;
- only cadence/in-between density changes.

For ON 2, a useful explicit core is:

`At 24 fps playback, target approximately 12 unique authored states per second. Every authored character state persists for two consecutive playback frames before the next state: A, A, B, B, C, C. Keep the cycle actively walking at normal speed; do not turn the paired exposure into long freeze frames.`

For ON 1, use:

`At 24 fps playback, target a newly advanced authored state on every successive frame: A, B, C, D. Do not intentionally repeat paired states.`

For ON 4, use:

`At 24 fps playback, target approximately 6 unique authored states per second with four exposures per state: A, A, A, A, B, B, B, B. Keep the action duration unchanged.`

These are target instructions, not verified guarantees.

## 10. When to return to translational locomotion

Only after an in-place cycle shows the desired temporal character should the prompt add left-to-right travel.

When adding travel:

- preserve the same gait;
- keep the camera locked;
- make the character advance along one continuous path;
- state that previous poses disappear rather than remaining as spatial copies;
- keep one opaque character visible;
- avoid describing every authored pose as a separate screen position unless the user explicitly wants a pose chart.

If H3 resumes smoothing, return to the in-place version rather than escalating to stronger freeze language.

## 11. Creative prompting versus diagnostic prompting

For a normal creative request such as `make this feel like classic limited animation`, use concise animation craft: held accents, sparse in-betweens, strong key poses, selective secondary motion.

For a research request such as `test whether H3 understands animation on fours`, use explicit playback patterns, controlled variables, and conservative claims.

Do not burden creative prompts with frame tables unless the user's actual goal is frame-cadence testing.

## 12. Reliability language

Good:

- `target animation on twos`
- `aim for paired playback exposures`
- `reduce unique in-between drawings`
- `use the temporal pattern A, A, B, B`
- `make the cadence visibly stepped`

Avoid unsupported certainty:

- `H3 will repeat every frame exactly twice`
- `this guarantees six unique drawings per second`
- `H3 internally renders at 12 fps`
- `humanoid motion automatically forces animation on twos`

The GPT may describe what the prompt is asking H3 to do. It must not present an unverified generation behavior as a known internal implementation fact.


## 16. Ref2VA cross-check — animation-language transfer versus literal production mechanics

Later controlled Ref2VA tests with video references produced a useful distinction.

Video references transferred **perceptual animation behavior** more reliably than literal production mechanics. Stronger results included:

- anticipation and pose rhythm;
- readable holds and burst timing;
- expressive transitional deformation and smear-like drawings;
- overshoot, follow-through, and secondary settling;
- nuanced acting hierarchy;
- graphic motion organization;
- cut and scale rhythm when explicitly reinforced.

Weaker or higher-variance results included:

- exact line-boil frequency;
- literal whole-frame redraw on every output frame;
- exact exposure cadence;
- physical wet-paint redistribution or similar destructive material reconstruction under complex semantic content.

Therefore, for Ref2VA prompts, do not equate a handmade-looking reference with guaranteed transfer of the reference's actual authored-drawing cadence. Describe visible motion grammar separately from literal frame mechanics.

When exact physical process matters, see `16_H3_REF2VA_STYLE_TRANSFER_LAB.md`. When cadence itself matters, retain the diagnostic and honesty rules in this file.

## 11. Cross-series finding: strong style transfer can coexist with weak cadence control

A later Ref2VA stress series held the action constant while changing visual style. Several visual treatments transferred strongly, while explicit cadence-driven looks such as papercraft-on-eights and flip-book/magazine timing were not among the preferred results.

This supports an important operational distinction:

**H3 can respond strongly to visual construction while still smoothing temporal exposure structure.**

Do not infer that a successful cut-paper, clay, comic, pencil, or painterly result has achieved the requested authored-frame cadence.

## 12. Updated cadence strategy

For production prompts:

1. describe the visual style without depending on cadence jargon;
2. describe readable pose rhythm, anticipation, compression, extension, overshoot, and recovery;
3. ask for visibly stepped or held key poses only where useful;
4. avoid claiming exact repeated playback frames;
5. if exact cadence matters, validate the output frame-by-frame or enforce it downstream in the workflow.

For stress tests, separate the questions:

- **Style test:** Can H3 make the frame look like this medium?
- **Motion-grammar test:** Can it reproduce these pose/deformation principles?
- **Cadence test:** Can it hold authored states for the requested number of playback frames?

Running all three as one prompt makes failure diagnosis ambiguous.

## 13. Repair ladder for failed "on fours / on eights" prompts

If the model remains smooth:

1. simplify the action;
2. reduce camera movement;
3. keep the subject in place;
4. use fewer major poses;
5. explicitly request `held key poses with abrupt drawing changes and minimal in-between states`;
6. provide an observable state sequence such as `A, A, A, A, B, B, B, B`;
7. if still unreliable, treat the cadence as a post-process/workflow problem rather than continuing to add style adjectives.

Do not use cadence failure as evidence that the visual style itself failed.
