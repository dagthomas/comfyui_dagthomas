# H3 Aesthetic, Motion, and Audio Craft Library

Version: 1.2
Purpose: Convert vague style requests into visible, audible, and temporally actionable language for MiniMax H3.

## 1. Injection budget

Use no more than:

- 1 visual-medium pack.
- 1 motion-behavior pack when motion character matters.
- 1 finish pack.
- 1 audio-treatment pack.

Omit any layer that adds nothing. A user-requested hybrid may combine two visual media, but state the division clearly, such as watercolor environments with inked cel characters. Never stack several near-synonymous packs.

Translate named films, shows, studios, games, eras, and cultural references into observable traits. The descriptive traits are the operative prompt language. Retain the proper name only when the user explicitly wants it included.

## 2. Mode-aware placement

- T2VA: put the visual medium and finish at the beginning of `[Shot 1]`; weave motion craft into actions; place audio treatment in the appropriate audio fields.
- I2VA: derive the baseline look from `<Picture 1>`. Add a different style only when the user asks for transformation; preserve first-frame identity, composition, and object anchors.
- FL2VA: use one consistent style path between both pictures. If the images differ in finish, describe a continuous, observable transformation that ends exactly on Picture 2.
- L2VA: make the inferred opening compatible with the final image and converge toward its medium, lighting, texture, and composition.
- Ref2VA: define referenced style or motion as `<Subject N>` when it must be tracked. State source and role in `subject_definitions`, classify its retention accurately, and cite it where applied in `detailed_description`.

## 3. Visual-medium packs

### V01 — Classical hand-drawn cel

Use for expressive 2D acting and traditional cel craft.

Inject: `hand-drawn cel animation with clean key poses, confident contour lines, painted backgrounds, clear silhouette staging, and natural overlap in hair and clothing`

### V02 — Limited television animation

Use for economical dialogue scenes, graphic anime timing, and held poses.

Inject: `limited 2D animation with held poses, crisp key drawings, selective movement in eyes, mouth, hair tips, and small head turns, with stable readable staging`

### V03 — Rubber-hose cartoon

Use for rhythmic vintage comedy.

Inject: `high-contrast rubber-hose animation with elastic limbs, rhythmic bounce, rounded poses, musical movement accents, and simple graphic backgrounds`

### V04 — Silhouette cutout

Use for shadow-puppet or ornate theatrical imagery.

Inject: `ornate silhouette cutout animation with flat dark figures, patterned shapes, articulated paper joints, theatrical composition, and layered parallax depth`

### V05 — Stop-motion puppet miniature

Use for tactile constructed characters and sets.

Inject: `cinematic puppet stop-motion with handcrafted materials, miniature set depth, visible surface texture, stepped frame-to-frame movement, and subtle registration variation`

### V06 — Clay animation

Use for sculpted softness and visible handmade material.

Inject: `clay animation with softly sculpted forms, faint thumbprint texture, gentle surface wobble, tactile deformation, and practical miniature lighting`

### V07 — Paper collage and cutout

Use for editorial collage, scrapbook, or graphic montage.

Inject: `layered paper-collage animation with torn and cut edges, printed textures, shallow parallax, composited shadows, and intentionally handmade depth`

### V08 — Pencil and watercolor

Use for illustrated storybook movement.

Inject: `pencil construction lines with translucent watercolor washes, visible paper fiber, soft pigment pooling, restrained color bleed, and gently fluctuating hand-painted edges`

### V09 — Gouache paint-in-motion

Use for opaque painterly scenes with visible brushwork.

Inject: `opaque gouache animation with layered brush strokes, matte pigment, simplified painted shapes, visible bristle texture, and controlled paint-like edge movement`

### V10 — Rotoscoped painterly realism

Use for realistic human motion under illustrated treatment.

Inject: `rotoscoped motion with lifelike timing, preserved body-weight shifts, inked contours, painterly fill, and subtle frame-to-frame drawing variation`

### V11 — Flat mid-century graphic

Use for modernist posters, simplified shapes, and restrained motion.

Inject: `mid-century flat graphic animation with asymmetric composition, simplified silhouettes, bold color blocks, sparse linework, and selective limited movement`

### V12 — Comic print hybrid

Use for halftone, ink, and graphic action frames.

Inject: `comic-print animation with inked contours, halftone dots, offset color registration, graphic impact frames, stylized motion streaks, and punchy panel-like composition`

### V13 — Psychedelic pop collage

Use for surreal 1960s-inspired visual transformation.

Inject: `psychedelic pop-art collage with saturated flat color, drifting graphic layers, playful scale changes, decorative pattern, and fluid surreal transitions`

### V14 — High-detail cinematic cel realism

Use for dense urban animation, weighty action, and dramatic light.

Inject: `high-detail cel animation with precise mechanical and architectural drawing, weighty body motion, atmospheric perspective, dramatic practical light sources, and cinematic depth`

### V15 — Painterly magical realism

Use for warm illustrated environments and grounded wonder.

Inject: `painterly hand-drawn magical realism with richly observed environments, warm natural light, grounded character acting, soft secondary motion, and restrained fantastical detail`

### V16 — Atmospheric cyber-noir animation

Use for contemplative technological environments.

Inject: `atmospheric cyber-noir animation with cool rain reflections, dense urban haze, precise sparse character movement, slow observational framing, and luminous practical signage`

### V17 — Stylized feature CG

Use for polished animated-character performance.

Inject: `stylized feature-quality CG with clean topology, expressive but controlled facial animation, smooth easing, natural secondary motion, cinematic lensing, and soft physically based lighting`

### V18 — Stylized CG comedy

Use for bold expressions and energetic readable staging.

Inject: `stylized CG comedy with bold facial shapes, snappy readable poses, controlled exaggeration, bright character separation, and energetic but coherent camera movement`

### V19 — Photoreal cinematic live action

Use for grounded filmic footage.

Inject: `photoreal live-action cinematography with physically plausible materials, motivated practical lighting, natural skin and fabric response, restrained depth of field, and coherent motion blur`

### V20 — Premium product commercial

Use for luxury objects, packaging, cosmetics, or technology.

Inject: `premium product cinematography with exact surface detail, controlled specular highlights, deliberate negative space, precise camera motion, clean reflections, and disciplined brand presentation`

### V21 — Observational documentary

Use for natural locations and unpolished human behavior.

Inject: `observational documentary realism with available light, responsive framing, natural exposure variation, unforced blocking, environmental detail, and restrained handheld movement`

### V22 — Archival newsreel

Use for period documentary or found-footage treatment.

Inject: `archival newsreel photography with period-appropriate framing, limited tonal range, intermittent exposure variation, mechanical camera steadiness, and aged photographic texture`

### V23 — Vector motion design

Use for explainers, title sequences, interfaces, and graphic advertising.

Inject: `vector-clean motion design with geometric shapes, strict alignment, controlled easing curves, layered 2.5D parallax, readable typography zones, and precise graphic transitions`

### V24 — Game-cinematic rendering

Use for real-time cutscenes or gameplay-adjacent visuals.

Inject: `high-end real-time game rendering with stable world geometry, readable character silhouettes, controlled depth of field, responsive animation, and coherent physically based materials`

## 4. Motion-behavior packs

### M01 — Naturalistic acting

`small anticipatory weight shifts, clean arcs, restrained gestures, breathing and eye focus preceding larger action, with secondary motion settling after the body`

### M02 — Limited held timing

`held key poses with selective facial and hair movement, sparse in-between action, brief stepped transitions, and clear readable changes between poses`

Use this for the visible character of limited animation. It is not a guarantee of literal playback-frame repetition. For `on ones/twos/fours/eights`, exposure cadence, or reduced in-between tests, consult `15_H3_TEMPORAL_ANIMATION_TECHNIQUES.md`.

### M03 — Snappy cartoon timing

`strong anticipation, rapid pose change, controlled squash and stretch, brief overshoot, and a clean held settle`

### M04 — Tactile stop-motion timing

`stepped stop-motion cadence, slight frame-to-frame registration variation, minimal conventional motion blur, tiny material shifts, and deliberate pose increments`

Use this for stop-motion character, not as a generic solution for literal `on twos` exposure. For exact cadence intent, consult file 15.

### M05 — Weighty cinematic action

`clear preparation, believable momentum and resistance, strong contact points, trailing secondary motion, and a gradual deceleration after impact`

### M06 — Rhythmic performance

`body accents and camera reframes land on audible beats, repeated movement motifs evolve over time, and the final pose resolves on the closing beat`

### M07 — Graphic morphing

`shapes transform through legible intermediate silhouettes, edges flow continuously, color regions exchange positions, and each metamorphosis reaches a stable graphic state`

### M08 — Product precision

`one controlled object action at a time, exact contact and release, smooth constant-speed movement, minimal vibration, and a clean final alignment`

### Temporal cadence note

Motion packs describe the perceptual character of movement. They do not establish verified playback-frame exposure counts. When the user explicitly cares about authored drawings versus playback frames, route the request through file 15 and keep the medium/motion pack separate from the temporal experiment.

## 5. Finish packs

- F01 Clean digital: `stable exposure, crisp controlled edges, minimal grain, neutral highlight rolloff, and clean color separation`
- F02 35 mm film: `fine film grain, mild highlight halation, soft contrast rolloff, subtle gate weave, and restrained lens flare`
- F03 16 mm reversal: `pronounced organic grain, compact highlight latitude, slight color drift, mild flicker, and documentary film texture`
- F04 VHS tape: `soft analog detail, light chroma bleed, faint scanline structure, intermittent tracking wobble, and low-level tape noise`
- F05 Paper and ink: `visible paper tooth, uneven ink density, restrained line boil, and handmade registration variation`
- F06 Watercolor/gouache: `paper fiber, pigment pooling, matte painted texture, soft edge variation, and restrained color bleed`
- F07 Print/collage: `halftone or screen-print texture, cut edges, layered shadows, imperfect registration, and tactile compositing`
- F08 Noir monochrome: `black-and-white tonal separation, deep controlled shadows, selective highlights, fine grain, and minimal midtone haze`

## 6. Audio-treatment packs

Audio packs guide content placement; they do not replace the three/six required H3 fields.

### A01 — Natural synchronized realism

Integrated/detailed description: synchronize footsteps, cloth, prop contacts, breaths, and impacts to visible actions.  
Soundscape: coherent room tone, environmental depth, and physically plausible distance.  
Music: `N/A` unless requested.

### A02 — Tactile miniature foley

Use small dry contacts, material creaks, tiny servo/armature-like clicks only when visibly justified, close-set room tone, and restrained dynamics. Do not add generic cinematic booms.

### A03 — Vintage cartoon orchestration

Use tightly synchronized instrumental accents for major visible actions. Describe exact instruments, rhythm, and dynamics in `non_diegetic_music`; keep physical impacts in the timeline/soundscape.

### A04 — Anime action sound design

Use short air displacements, cloth snaps, mechanical impacts, and brief tonal accents synchronized to decisive poses. Avoid continuous loud effects that obscure dialogue.

### A05 — Product ASMR

Use close, clean handling sounds: cap clicks, fabric glide, glass contact, packaging folds, and controlled room silence. Keep music sparse or absent.

### A06 — Documentary location sound

Use stable location ambience, realistic off-axis voices, environmental occlusion, natural microphone perspective, and no score unless requested.

### A07 — Analog media audio

Use low tape hiss, limited bandwidth, slight wow/flutter, and occasional mechanical transport noise. Apply only when the visual/source context supports it.

### A08 — Graphic rhythm bed

Use concise electronic percussion or acoustic clicks at a stated tempo, aligned with graphic transformations. Keep audience-only rhythm in `non_diegetic_music`.

## 7. Failure repair

If the style is weak: repeat one short descriptive anchor once in the appropriate main description; do not paste the whole pack twice.

If the image identity drifts: remove style traits that alter anatomy or wardrobe; restate exact identity/clothing anchors; begin with micro-motion.

If the clip is overdesigned: remove the finish or motion pack before removing user content.

If limited animation remains too smooth: consult file 15, simplify to one fixed-position action or in-place cycle, state the repeated-state pattern explicitly, and avoid stacking freeze/hold language.

If audio feels detached: bind 1–3 specific sound events to visible actions and simplify the remaining soundscape.

If FL2VA misses the last frame: reduce decorative transformation, keep one shot, and state the final convergence attributes explicitly.

If Ref2VA roles conflict: split source traits into separately defined subjects or narrow each source to one declared job.


## Ref2VA video-reference extraction packs — empirical

These are not named styles to impose over a reference. They are compact attribute families for describing reusable `<Subject N>` style/performance content sourced from a connected `<Video N>`, or whole-video temporal relationships kept on `<Video N>`. Use only the families supported by the visible reference and user intent.

### RVS1 — Surface language

Inject selectively: `transfer the reference video's line behavior, palette relationships, fill density, texture, edge treatment, graphic density, and background-rendering economy continuously across the generated frame`

Use when the user wants the reference look but controls action independently.

### RVM1 — Motion grammar

Inject selectively: `transfer the reference video's pose rhythm, readable holds, anticipation, acceleration and deceleration, expressive deformation, transitional smear-like drawings, overshoot, follow-through, secondary motion, and recovery`

Use when the user wants the reference's animation behavior.

### RVA1 — Acting grammar

Inject selectively: `transfer the reference video's thought-to-pose progression, eye-before-head hierarchy, gesture timing, changes of balance, facial reaction staging, expressive holds, and restrained recovery`

Use for nuanced character performance.

### RVT1 — Temporal/compositional grammar

Inject selectively: `transfer the reference video's cut rhythm, shot-scale contrast, viewpoint changes, graphic reframing, and relative duration of visual ideas`

When cuts matter, explicit target cut times are more reliable than leaving all editing decisions implicit.

### RVP1 — Physical material process

Inject only when visibly present and requested: `transfer the physical material behavior visible in the reference, including how marks are redistributed, erased, rebuilt, smeared, resculpted, or otherwise materially transformed between readable states`

Do not confuse material appearance with material process. Physical-process fidelity is high variance and often decreases as subject/anatomy/environment complexity increases.

### Reference-first rule

When a strong video-derived `<Subject N>` already supplies the style prior, do not automatically combine these packs with V01–V22 or a named cultural style. Reinforce only the attributes that need textual emphasis. Competing medium labels can pull H3 toward a generic learned style instead of the connected video reference.

See `16_H3_REF2VA_STYLE_TRANSFER_LAB.md`.

## 8. Empirical style-force ladder

For aggressive style transfer or stress testing, choose cues in descending order of observed leverage:

### Tier A — structural medium cues

Highest leverage when clearly described:

- silhouette cutout;
- articulated paper;
- clay/plasticine;
- liquid/slime;
- marble/prismatic fracture;
- geometric/Bauhaus construction;
- bold cel/comic shadow masses;
- screenprint/risograph separations.

These change the geometry or organization of the image, not merely its surface.

### Tier B — shape and palette systems

Strong when combined:

- carved black shadow masses;
- broken contour drawing;
- flat poster shapes;
- posterized value steps;
- restricted two- or three-color systems;
- fluorescent blacklight palette;
- chromatic misregistration;
- exaggerated graphic proportions.

### Tier C — animated surface systems

Useful as reinforcement:

- line boil;
- moving hatch groups;
- watercolor bleed;
- paint crawl;
- pigment redistribution;
- paper fiber movement;
- xerographic breakup;
- drifting registration.

These are more reliable when attached to a strong Tier A or Tier B system.

### Tier D — timing-only vocabulary

Lower leverage by itself:

- on twos;
- on fours;
- on eights;
- limited frame rate;
- flip-book timing.

Use these as temporal modifiers, not as the sole style descriptor.

## 9. Whole-frame style pack template

When maximum coherence is requested, phrase the visual pack so it governs every layer:

`Apply the same [medium/shape/palette] logic to the astronaut, unicorn, crowd, vehicles, architecture, signs, pavement, reflections, atmosphere, and transitional drawings. Do not leave the environment photographic while stylizing only the central subject.`

## 10. Selected-result lesson

Across the astronaut/unicorn Times Square stress series, visibly successful styles tended to include at least two high-leverage controls:

- material construction + shape grammar;
- shape grammar + restricted palette;
- medium + animated surface behavior;
- palette + compositing/registration behavior;
- geometry + motion grammar.

Styles that depended mainly on subtle painterliness, exact cadence, or a single post-process effect were more likely to collapse toward H3's baseline rendering.
