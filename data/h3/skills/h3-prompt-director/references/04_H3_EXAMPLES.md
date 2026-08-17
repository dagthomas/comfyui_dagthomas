# H3 Prompt Examples

These examples teach structure. Details should always be adapted to the user's media and intent.

## T2VA — 6 seconds, two shots

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a night-shift florist alone inside a narrow shop as rain streaks the front window. The camera trucks left with small amplitude at slow speed while the middle-aged florist with a warm, slightly husky voice (S1) wraps one white lily in brown paper and says: <d>[English] This one is for the last train.</d> [Shot 2] At 00:04.000, the camera cuts to a close-up of her hands tying dark green twine around the parcel as the final words carry naturally into the new shot; she places the flower beneath a small lamp.

overall_soundscape: Rain taps steadily against the glass beneath a low refrigerator hum. Paper folds crisply, scissors click once, and twine pulls taut against the parcel.

non_diegetic_music: Sparse felt-piano notes at a slow tempo with a low sustained cello tone, fading during the final second.
```

## I2VA — 6 seconds, first-frame anchor

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the young man in <Picture 1> begins with the same face, charcoal coat, body position, umbrella angle, wet alley composition, teal shop light, and red bicycle placement shown in the image. The camera pushes in with small amplitude at slow speed as he lifts his eyes toward a flickering sign, shifts the umbrella to his left hand, and reaches into his coat. The young man with a quiet baritone voice (S1) says: <d>[English] I knew you'd come back.</d> He removes a brass key, turns it once between his fingers, and looks past the camera while his coat and the umbrella edge move lightly in the wind.

overall_soundscape: Fine rain patters on the umbrella and pavement while a distant ventilation fan turns unevenly. Fabric rustles, the key lightly clicks against a metal zipper, and bicycle spokes tick in the wind.

non_diegetic_music: A slow three-note electric-piano figure repeats beneath a sustained bowed-bass note, then drops in volume at the end.
```

## FL2VA — 8 seconds, one continuous shot

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 8.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Photoreal product cinematography, the cobalt perfume bottle begins in the exact position, closed state, camera angle, black-glass surface, and narrow rim lighting established by Picture 1. The camera arcs clockwise with small amplitude at slow speed as a hand wearing the same silver ring enters from the right, grips the cap, and lifts it vertically. Condensation beads slide down the bottle while the hand rotates the cap, places it on the left side of the base, and withdraws. During the final two seconds, the camera completes its arc, the bottle's reflection lengthens, the moving highlights settle, and every object, hand absence, cap position, shadow, and compositional edge converges on Picture 2 exactly at the end.

overall_soundscape: A soft fingertip contact is followed by the snug release of the cap and a quiet glass-on-glass tap as it is set down. The room remains otherwise still with a faint studio ventilation tone.

non_diegetic_music: N/A
```

## L2VA — 6 seconds, final-frame anchor

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 6.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action macro cinematography, a clean ceramic plate sits beneath the same warm window light and at the same camera angle visible in <Picture 1>; only a few dark crumbs initially rest near its center. The camera holds a static shot as a chocolate biscuit drops from just above frame, strikes the plate, and breaks into several pieces. The fragments bounce once, rotate, and slide outward while fine crumbs scatter. Their movement progressively slows until the fragment shapes, crumb pattern, lighting, focus plane, and exact final composition match <Picture 1> at the end.

overall_soundscape: The biscuit strikes the ceramic plate with a dry snap, followed by several light taps as fragments bounce and scrape to a stop. A faint room tone continues underneath.

non_diegetic_music: A single low plucked-string note sounds at impact and decays naturally to silence.
```

## Ref2VA — image identity, video camera, audio voice

Assumed connection order: the portrait is `<Picture 1>`, the camera-motion clip is `<Video 1>`, and the voice clip is `<Audio 1>`.

```text
subject_definitions:
<Subject 1> is the woman whose facial identity, copper hair, green tailored suit, and gold earrings come from <Picture 1>.
<Video 1> is the camera-motion reference for the target video, providing only its smooth backward tracking movement and pacing.
<Audio 1> is the voice-timbre and delivery reference for <Subject 1> (S1).

summary:
[reference generation + audio reference] The target video presents <Subject 1> walking through a stone gallery while following <Video 1>'s backward tracking movement and using <Audio 1> as the reference for her spoken performance.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - her facial identity, copper hair, green tailored suit, and gold earrings from <Picture 1> remain consistent throughout the shot.
<Video 1> (camera movement and pacing): fully_preserved - the target camera follows the same smooth backward tracking pattern and measured pace.
<Audio 1>: reference - its voice timbre, cadence, and delivery guide <Subject 1> (S1) without copying the original signal or words.

detailed_description:
The target video uses polished live-action luxury-ad cinematography with clean natural highlights, restrained contrast, and precise full-body framing.
[Shot 1] A full-body shot frames <Subject 1> walking through a bright stone gallery toward the camera. Her facial identity, copper hair, green tailored suit, and gold earrings remain fully consistent with <Picture 1>. The camera tracks backward with the smooth, measured movement and pacing established by <Video 1>, keeping her centered while pale columns pass evenly along both sides of the frame. Her stride is controlled and natural; each step produces a small movement through the jacket fabric without changing its tailored silhouette. As she approaches the center of the gallery, a band of sunlight advances across the floor and catches the edge of her right earring. <Subject 1> (S1), using the voice timbre, cadence, and assured delivery referenced from <Audio 1>, looks directly toward the lens and says: <d>[English] Design should move with you.</d> She closes her lips at the end of the line and continues walking. The camera maintains the referenced backward motion as she turns her right wrist inward, then gradually outward, revealing a slim gold watch without interrupting her pace. Her left arm continues its natural counter-swing. The gallery opens into a sunlit atrium behind the camera; reflected light grows across her suit while the original green color remains stable. During the final second, she slows by half a step, holds the watch clearly within the lower-right portion of her silhouette, and keeps calm eye contact with the lens as the tracking movement eases to a stop.

overall_soundscape: Firm heels echo across polished stone beneath a spacious interior room tone. Fabric moves softly with each step, and a distant glass door closes once.

non_diegetic_music: A precise mid-tempo pattern of muted electronic percussion and short marimba notes adds a low synth bass during the final two seconds.
```

## Voiceover across a cut

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, an older fisherman with a weathered face and low, gravelly voice (S1) stands motionless at the harbor railing before sunrise. He says in an off-screen voiceover: <d>[English] The sea remembers <scenetrans></d> while his lips remain completely closed. [Shot 2] At 00:03.000, the camera cuts beneath the surface to pale ropes drifting beside the pier as his voice continues seamlessly across the cut: <d>[English] <scenetrans>every name.</d> The fisherman remains off-screen.

overall_soundscape: Small waves knock against wooden pilings while rope fibers creak under tension. Underwater bubbles and a distant engine rumble become audible after the cut.

non_diegetic_music: Long bowed harmonics at a very slow tempo rise slightly through the cut and then fade.
```
