# H3 T2VA Gold Examples

Use these examples only for text-only generation. T2VA has no alignment line and always uses the three-field format.

## T01 — Quiet live-action dialogue

**Brief:** Six seconds. A night baker opens her shop before dawn and says exactly, “First batch.” No score.  
**Mode:** T2VA · 6.00 seconds · one shot  
**Style recipe:** V19 photoreal live action + M01 naturalistic acting + F02 restrained 35 mm + A01 synchronized realism

```text
integrated_multimodal_description: [Shot 1] Photoreal live-action cinematography with motivated tungsten light, fine film grain, and restrained depth of field, a medium-wide shot frames a middle-aged baker lifting the wooden shutters of a narrow street bakery before dawn. The camera pushes in with small amplitude at slow speed as cool blue street light gives way to the warm interior. The baker with a low, slightly raspy voice (S1) sets a steaming tray on the counter, looks at the first loaf, and says: <d>[English] First batch.</d> She closes her lips, exhales through her nose, and begins arranging the loaves in one neat row.

overall_soundscape: Wooden shutters scrape upward over a nearly silent street. A metal tray touches the counter, bread crust crackles faintly, and the baker's apron rustles as she works.

non_diegetic_music: N/A
```

**Decision note:** One causal action and one camera move fit the short duration; silence means no score, not absence of physical sound.

## T02 — Limited-animation comedy

**Brief:** Eight-second flat TV animation. An office worker hides a birthday cake when the boss enters; the coworker whispers, “Act normal.”  
**Mode:** T2VA · 8.00 seconds · two shots  
**Style recipe:** V02 limited television animation + M03 snappy cartoon timing + F01 clean digital + A03 cartoon accents

```text
integrated_multimodal_description: [Shot 1] Limited 2D television animation with clean flat color, held key poses, selective eye and mouth movement, and stable sitcom staging, a medium-wide shot frames two office workers beside a desk with a brightly iced cake. The elevator bell sounds. One worker snaps upright, slides the cake into a lower cabinet with a fast anticipation and clean held settle, while the other keeps watch. The camera holds static as the cabinet door almost closes, then rebounds one inch. The young coworker with a quick, dry alto voice (S1) leans closer and whispers: <d>[English] Act normal.</d> [Shot 2] At 00:04.500, the shot cuts to the boss entering from the elevator while both workers hold exaggeratedly neutral poses. Only their pupils shift toward the cabinet as a single candle flame rises behind the desktop.

overall_soundscape: Low office ventilation and distant keyboard taps continue beneath the elevator bell. The cabinet slides, bumps softly, and squeaks once as it rebounds.

non_diegetic_music: Short muted clarinet and pizzicato-string notes at a moderate tempo punctuate the cabinet movement, then stop under the final held pose.
```

**Decision note:** Limited movement is concentrated in eyes, mouth, cabinet, and candle; the music is described instrumentally rather than as “comedic.”

## T03 — Premium product reveal

**Brief:** Five-second macro commercial for a cobalt perfume bottle; no people, no text, close handling sounds.  
**Mode:** T2VA · 5.00 seconds · one shot  
**Style recipe:** V20 premium product + M08 product precision + F01 clean digital + A05 product ASMR

```text
integrated_multimodal_description: [Shot 1] Premium product cinematography with exact glass detail, controlled specular highlights, clean reflections, and deep negative space, a macro shot frames a cobalt perfume bottle standing on black glass. The camera arcs clockwise with small amplitude at slow speed as a narrow white highlight travels across the bottle. A hand wearing a plain silver ring enters from the right, grips the cap without shifting the bottle, lifts it vertically with constant speed, and sets it beside the base. A fine mist releases once from the exposed atomizer and catches the moving light before dissipating; the hand withdraws and the camera settles on the bottle and cap in precise alignment.

overall_soundscape: Fingertips make a soft dry contact with the cap, followed by a snug release click, a delicate glass-on-glass tap, and the short pressurized hiss of the atomizer. The studio remains otherwise nearly silent.

non_diegetic_music: N/A
```

**Decision note:** Product geometry and contact order dominate; no decorative boom or brand text is invented.

## T04 — Documentary micro-story

**Brief:** Ten-second observational documentary at a community garden. An older gardener shows a child how to plant a seed. No dialogue and no score.  
**Mode:** T2VA · 10.00 seconds · two shots  
**Style recipe:** V21 observational documentary + M01 naturalistic acting + A06 location sound

```text
integrated_multimodal_description: [Shot 1] Observational documentary realism with available morning light, natural exposure variation, and restrained handheld framing, a medium shot follows an older gardener kneeling beside a raised bed with a child. The gardener presses one finger into the dark soil, pauses for the child to copy the motion, and opens a paper seed packet over the child's palm. The child studies the tiny seed, places it carefully in the hole, and glances up for confirmation. [Shot 2] At 00:06.000, the shot cuts to a close view of their hands as the gardener guides soil over the seed without taking over. The child pats the surface twice, and both hands withdraw while a small watering can enters the edge of frame.

overall_soundscape: Birds, distant neighborhood traffic, and light wind through leaves establish the garden location. Soil crumbles under fingertips, the paper packet rustles, and the child gives one quiet breath of concentration.

non_diegetic_music: N/A
```

**Decision note:** The camera observes rather than beautifies; the final watering can suggests the next action without overcrowding the clip.

## T05 — Comic action with visible text

**Brief:** Seven-second comic-print superhero landing. A rooftop sign must read exactly “EAST TERMINAL.” Add one short line: “Too late.”  
**Mode:** T2VA · 7.00 seconds · two shots  
**Style recipe:** V12 comic print + M05 weighty action + F07 print finish + A04 action sound design

```text
integrated_multimodal_description: [Shot 1] Comic-print animation with inked contours, halftone shading, offset red-and-cyan accents, and graphic panel-like composition, a low wide shot frames a rain-dark rooftop beneath a metal sign reading "EAST TERMINAL". A caped hero drops from above frame; her knees bend before impact, the cape trails behind, and a circular ink-splatter impact frame flashes as both boots strike the roof. The camera shakes slightly, then stabilizes while she rises through a believable weight shift. [Shot 2] At 00:04.200, the shot cuts to a close low angle as rain streaks across her halftone face. The hero with a steady contralto voice (S1) looks beyond the lens and says: <d>[English] Too late.</d> Her lips close and her cape settles one beat after her shoulders.

overall_soundscape: Rain strikes sheet metal and rooftop gravel. A short air displacement precedes the heavy two-foot landing, followed by a low roof-panel rattle and wet cape movement.

non_diegetic_music: Two low taiko strikes and a brief distorted electric-guitar chord accent the descent and landing, then cut to silence beneath the spoken line.
```

**Decision note:** Visible text remains exact; physical impact stays in soundscape while score accents remain audience-only.
