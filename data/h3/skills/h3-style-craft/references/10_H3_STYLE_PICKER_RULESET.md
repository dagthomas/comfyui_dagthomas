# H3 Automatic Style Picker Ruleset

Version: 1.3
Goal: Select a coherent H3 visual, motion, finish, and audio treatment from user language, reference media, and shot intent.

## 1. Hard rules

1. Select exactly one visual-medium pack when style treatment is requested or necessary.
2. Select at most one motion pack, one finish pack, and one audio pack.
3. Do not add a pack merely to fill a category.
4. Use a two-medium hybrid only when explicitly requested; state which layer each medium controls.
5. Reference assets outrank inferred style unless the user explicitly requests transformation.
6. A style choice must never alter exact dialogue, visible text, keyframe alignment, identity anchors, product geometry, or declared Ref2VA roles.
7. Translate named cultural references into concrete visible/audible traits. Proper names are discovery aliases, not sufficient final prompt language.

## 2. Evidence priority

Use this order:

1. Explicit user request to preserve or transform a reference.
2. Visible medium, rendering, lighting, texture, and grade in boundary-frame images.
3. Explicit user style or cultural reference.
4. Ref2VA source-role assignment.
5. Genre and shot intent.
6. Conservative fallback matching the apparent source medium.

If evidence conflicts, preserve fixed frames and identity. Ask one question only when choosing incorrectly would materially change the result.

## 3. Mode gates

### T2VA

Select freely from user intent. If no style is stated, choose the most conservative medium supported by the brief; do not default every request to cinematic live action.

### I2VA

Analyze Picture 1. Preserve its medium and finish unless the user explicitly requests transformation. If transformed, keep identity, wardrobe, composition, and scene anchors while describing the new treatment as developing from the first frame.

### FL2VA

If both images share a style, preserve it. If styles differ and the user expects interpolation, describe visible intermediate changes in medium, line, texture, light, and color until Picture 2 is reached. Never use a finish pack that contradicts either endpoint.

### L2VA

Choose an opening state compatible with the final image. The style path must converge on Picture 1's exact final medium and finish.

### Ref2VA

Determine whether style comes from source media or user text. When a reference video supplies abstract animation style or visible performance, define that reusable content as `<Subject N>` sourced from `<Video N>` and normally mark it `attribute_transfer`. Keep standalone `<Video N>` for whole-video camera, cuts, rhythm, temporal structure, editing, or continuation. One source video may support both a derived `<Subject N>` and a `<Video N>` temporal role.

For video-reference style transfer, apply the empirical rule **reference controls HOW; user controls WHAT**. Extract only the relevant attribute families: visual surface, motion grammar, acting grammar, temporal/compositional grammar, and—only when requested—physical material process. Explicitly protect the user's subject, anatomy, clothing, props, setting, composition, action, dialogue, and story from source-content leakage.

Do not automatically add a named visual-medium pack over a strong video style reference. First let the reference be the dominant style prior; add only minimal corrective traits. For multiple videos, prefer role separation over vague equal blending when sources contribute different strengths. See `16_H3_REF2VA_STYLE_TRANSFER_LAB.md`.

## 4. Keyword and cue routing

- Hand-drawn, cel, classic cartoon, traditional animation → V01.
- Anime, held frames, mouth flaps, television cel → V02.
- Rubber hose, 1920s cartoon, bouncy vintage → V03.
- Silhouette, shadow puppet, ornate cutout → V04.
- Puppet, miniature, tactile stop-motion → V05.
- Clay, plasticine, sculpted stop-motion → V06.
- Cut paper, collage, scrapbook → V07.
- Pencil, watercolor, illustrated storybook → V08.
- Gouache, opaque paint, brushy matte illustration → V09.
- Rotoscope, traced live movement → V10.
- Mid-century, UPA, flat poster → V11.
- Comic, halftone, ink splatter → V12.
- Psychedelic, pop-art, 1960s collage → V13.
- Detailed cel, urban anime realism, mechanical anime → V14.
- Warm painterly magic realism → V15.
- Cyber-noir, contemplative rain city → V16.
- Polished family CG, feature animation → V17.
- Exaggerated CG comedy → V18.
- Photoreal, live action, film still → V19.
- Luxury product, packaging, macro commercial → V20.
- Documentary, candid, available light → V21.
- Newsreel, archival footage → V22.
- Motion graphics, vector explainer, title design → V23.
- Gameplay, real-time cutscene, game trailer → V24.

When a user names a film/show/studio, consult `09_H3_STYLE_REFERENCE_ANCHORS.md` and use its descriptive mapping.

## 5. Motion selection

- Dialogue/performance → M01, or M02 when the medium is limited animation.
- Comedy/slapstick → M03.
- Stop-motion/clay → M04.
- Action/stunts/heavy objects → M05.
- Dance/music video → M06.
- Surreal shape change/motion graphics → M07.
- Product handling/assembly → M08.
- Exposure cadence, animation on ones/twos/fours/eights, reduced in-betweens, frame holds, or limited-frame diagnostics → consult file 15 first. Use M02 only when limited held timing is also the desired motion character; use M04 only for actual stop-motion aesthetics.

Skip a motion pack when the user's action direction is already precise.

## 6. Finish selection

- `clean`, `crisp`, `modern`, `stable exposure` → F01.
- `35 mm`, `filmic`, `halation`, `anamorphic` → F02.
- `16 mm`, `reversal`, `documentary grain` → F03.
- `VHS`, `camcorder`, `tape` → F04.
- `paper`, `ink`, `line boil` → F05.
- `watercolor`, `gouache`, `pigment` → F06.
- `halftone`, `screen print`, `collage` → F07.
- `black and white`, `monochrome noir` → F08.

Skip finish when the reference already fixes it or a finish would obscure critical text/product detail.

## 7. Audio selection

- Grounded live action or feature CG → A01.
- Stop-motion/clay/miniature → A02.
- Vintage cartoon comedy → A03.
- Anime action → A04.
- Product/beauty/food macro → A05.
- Documentary/verité → A06.
- VHS/archival tape → A07.
- Motion graphics/title sequence → A08.

Audio treatment cannot invent music. If no non-diegetic score is requested or implied, write `N/A`. Keep character-audible music in the main timeline.

## 8. Injection procedure

1. Identify the mode and immutable anchors.
2. Select packs within the budget.
3. Convert each pack into only the traits relevant to the shot.
4. Place medium/finish at the required style opening.
5. Integrate motion phrases into specific actions, not a detached tag stack.
6. Bind 1–3 sound cues to visible events; summarize remaining ambience separately.
7. Validate that the style survives across shots without repeated full-pack text.

## 9. Cultural-reference handling

If the user asks for “X style”:

1. Find X in the anchor library or infer its medium, design, motion, camera, finish, and audio traits.
2. Preserve the user's intended recognizable characteristics.
3. Write concrete traits into the prompt.
4. Avoid unsupported claims about exact lens, pipeline, frame rate, or production technique.
5. If several eras of a franchise differ materially, ask which era only when it changes the result; otherwise choose the cues most consistent with the attached reference.

## 10. Failure repair

- Weak style adherence → repeat one short anchor once; front-load medium traits.
- Identity drift → remove anatomy-altering style cues; reinforce reference identity and wardrobe.
- Motion chaos → reduce to one primary action and one camera move; keep one motion pack.
- Excessive smoothness → if the user wants general snappiness, add held accents or stepped timing when the medium supports it; if the user wants literal exposure cadence or reduced in-betweens, use file 15's controlled repair ladder rather than piling on stronger hold/freeze language.
- Audio drift → simplify sound bed and bind fewer, clearer synchronized events.
- Ref2VA confusion → redefine source roles and retention markers before expanding description.
- FL2VA endpoint miss → remove decorative changes and narrate the final two seconds as progressive convergence.

## 11. Selection audit

Before output, silently confirm:

- Pack choices agree with mode and references.
- No more than one pack per allowed category.
- Descriptive traits appear in the prompt, not only a proper name.
- Style does not override identity, text, product shape, dialogue, or endpoints.
- Motion instructions are observable and duration-appropriate.
- Audio treatment is placed in the correct H3 field.


## 8. Ref2VA video-style decision rules — empirical

When the user says "use this video as the style":

1. Default derived style/performance subjects to `attribute_transfer`, not literal source preservation.
2. Silently inspect which attributes actually define the style.
3. Reinforce motion, acting, or cut grammar when those are important; these often transfer less implicitly than broad surface appearance.
4. Add a content firewall when the new subject or story must remain independent.
5. Avoid unnecessary medium labels that compete with the reference.
6. If the style depends on destructive physical manipulation such as wet paint redistribution or erase/rebuild, describe that process explicitly but treat exact replication as high variance.
7. If the reference's style depends on cuts or viewpoint changes, write those temporal relationships into the target timeline rather than assuming H3 will infer them.
8. Never claim an IP-Adapter mechanism exists. "Adapter-like" describes the desired separation of style from content, not an H3 implementation detail.

## 9. Empirical style-strength routing

When the user's goal is ordinary coherence, keep the existing one-medium/one-motion/one-finish budget.

When the user's goal is explicitly to **push**, **stress-test**, **force**, or **maximize** a style, use a style-strength stack rather than adding more unrelated packs.

Choose:

1. one medium/construction rule;
2. one shape/value rule;
3. one palette rule;
4. one motion rule;
5. optionally one animated-surface rule.

All five must point toward the same aesthetic.

### Style-strength score

Before finalizing a stress-test prompt, silently check whether it contains:

- **G** geometry/material change;
- **S** shape/edge/shadow grammar;
- **P** palette/value system;
- **M** motion grammar;
- **T** animated texture/process.

A strong style prompt normally has at least three of `G/S/P/M/T`, with `G` or `S` preferred. A prompt containing only `T` and cadence language is likely to be weak.

### Baseline-collapse warning

Increase explicit style pressure when:

- the requested look is mostly a surface filter;
- the source action is complex and realistic;
- the environment is dense;
- the style depends on sparse linework or subtle painterliness;
- the request uses exact cadence terminology as its main differentiator.

Do not solve baseline collapse by stacking unrelated named styles.

### Cadence is not a style pack

`on twos`, `on fours`, `on eights`, `limited frame rate`, and `flip-book` are temporal requests. Route them through the temporal-animation rules even when they accompany a strong visual anchor.

If cadence is important but style is the test target, prioritize style coherence first and describe cadence second.
