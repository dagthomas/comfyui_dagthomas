# H3 Edge-Case Gold Examples

These cases teach decision behavior and prevent format contamination. Examples marked “Do not do this” are negative examples only.

## E01 — Ambiguous single image

**User:** Attaches one image and says, “Make a six-second H3 video using this.”

**Correct response:**

`Should this image be the exact first frame, the exact last frame, or a general identity/style reference?`

**Do not do this:** Silently assume I2VA and emit the first-frame alignment line.

**Why:** A single image can select I2VA, L2VA, or Ref2VA. The choice materially changes both workflow and output grammar.

## E02 — Dialogue crossing a cut

**Brief:** Six-second T2VA. A woman says exactly, “I thought the road would take us home.” The line begins before the cut and ends after it.  
**Mode:** T2VA · two shots

```text
integrated_multimodal_description: [Shot 1] Live-action cinematic realism, a medium shot follows a woman walking beside an empty rural road at dusk. The camera tracks backward at slow speed while the woman with a quiet, tired alto voice (S1) looks toward the fading horizon and says: <d>[English] I thought the road <scenetrans></d> Her voice continues seamlessly across the cut. [Shot 2] At 00:03.000, the shot cuts to her boots slowing beside a weathered road marker as the same voice carries over uninterrupted from the previous shot: <d>[English] <scenetrans>would take us home.</d> She stops, closes her lips, and lowers her gaze toward the marker.

overall_soundscape: Evening insects, light wind through dry grass, measured footsteps, and soft coat movement continue across both shots.

non_diegetic_music: Sparse acoustic-guitar harmonics at a slow tempo sustain through the cut and fade after the line ends.
```

**Why:** Both dialogue fragments contain `<scenetrans>`, and continuity is stated on each side of the cut.

## E03 — Reused song lyric cue without a new speaker

**Brief:** Ref2VA. Audio 1 is a complete copied song. When its lyric “Turn around.” occurs, Subject 1 turns. No character performs the lyric.  
**Mode:** Ref2VA · audio reuse

```text
subject_definitions:
<Subject 1> is the dancer whose identity, yellow jacket, black trousers, and short curls come from <Picture 1>.
<Audio 1> is the complete copied song used as the target video's full audience-only music track, including its existing vocal layer.

summary:
[reference generation + audio reuse] The target video shows <Subject 1> responding physically to a lyric cue within the fully copied <Audio 1> track.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - identity, hair, jacket, trousers, and silhouette remain consistent.
<Audio 1>: fully_copy - the complete source song is reused without added, removed, or replaced audio layers.

detailed_description:
The target video uses clean graphic live-action styling with a yellow-and-black palette, stable full-body staging, and restrained camera motion.
[Shot 1] <Subject 1> stands centered against a matte black background, preserving the identity, short curls, yellow jacket, and black trousers from <Picture 1>. The camera holds static while <Audio 1> plays as the complete copied soundtrack. The dancer remains still through the opening instrumental phrase, moving only through visible breathing. When <Audio 1> reaches the existing lyric <d>[English] Turn around.</d>, <Subject 1> shifts weight to the left foot, rotates through one clean half-turn, and lets the jacket hem trail and settle. The copied lyric remains sourced only from <Audio 1>; <Subject 1> does not sing, speak, or lip-sync it. The dancer finishes facing away from the camera and holds the final pose through the remaining copied music.

overall_soundscape:
No separately generated ambience or physical effects are added; all audible content comes from the complete copied <Audio 1> track.

non_diegetic_music:
<Audio 1> is directly reused as the complete audience-only song, including its existing vocal layer.
```

**Do not do this:** Assign `(S1)` to the dancer or describe the dancer as singing.

**Why:** A verbal cue inside a directly reused soundtrack uses `<Audio 1>` as its source unless a concrete character or narrator produces it.

## E04 — Conflicting style sources resolved by explicit roles

**Brief:** Ref2VA. Picture 1 supplies a photoreal woman's identity; Picture 2 supplies only a watercolor finish; Video 1 supplies only stop-motion timing. The user explicitly wants this hybrid.  
**Mode:** Ref2VA · explicit hybrid

```text
subject_definitions:
<Subject 1> is the woman whose facial identity, long dark hair, green dress, and silver necklace come from <Picture 1>.
<Subject 2> is the watercolor surface treatment from <Picture 2>, including translucent washes, paper fiber, pigment pooling, and soft edge variation.
<Video 1> is the motion-timing reference, providing only stepped pose increments, held accents, and minimal conventional motion blur.

summary:
[reference generation] The target video preserves <Subject 1>'s identity while transferring <Subject 2>'s watercolor treatment and following <Video 1>'s stepped timing.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - facial identity, hair, dress, and necklace remain stable beneath the new surface treatment.
<Subject 2> (applies throughout [Shot 1]): attribute_transfer - the watercolor finish is transferred to the newly generated woman and garden environment.
<Video 1> (motion timing): fully_preserved - stepped increments, held accents, and minimal blur guide movement without importing its subjects or setting.

detailed_description:
The target video uses translucent watercolor washes over clearly preserved human features, visible paper fiber, controlled pigment pooling, soft painted edges, and intentionally stepped motion timing.
[Shot 1] A medium-wide garden composition frames <Subject 1>, preserving her facial identity, long dark hair, green dress, and silver necklace from <Picture 1>. <Subject 2>'s watercolor surface treatment covers the woman, flowers, path, and sky without changing her proportions, wardrobe construction, or jewelry. Following <Video 1>'s motion timing only, she raises a watering can through discrete pose increments, holds briefly at the highest point, and tilts it forward. Painted water forms a continuous blue-grey arc while the hand and body retain the referenced stepped cadence. The camera trucks left slowly and smoothly, deliberately separating camera movement from the character's held animation timing. Her hair shifts in two small increments after the body stops, then settles. Pigment edges fluctuate gently without causing identity drift. She returns the can upright, lowers it through the final referenced pose sequence, and holds a clean ending silhouette among the flowers.

overall_soundscape:
Water pours steadily onto soil beneath light garden wind, distant insects, and small dress-fabric movement. The can handle gives one soft contact as it returns upright.

non_diegetic_music:
N/A
```

**Do not do this:** Describe the subject as simultaneously photoreal, watercolor, and physically miniature stop-motion.

**Why:** The roles are separated: identity from Picture 1, surface treatment from Picture 2, and timing only from Video 1.

## E05 — Do not mix Ref2VA and three-field formats

**Ref2VA brief:** “Use Picture 1 for identity and Audio 1 for voice.”

**Incorrect opening:**

```text
integrated_multimodal_description: [Shot 1] Use <Picture 1> for identity...
```

**Correct opening:**

```text
subject_definitions:
<Subject 1> is the person whose identity comes from <Picture 1>.
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).

summary:
[reference generation + audio reference] ...
```

**Why:** Full-reference mode always uses six sections; `integrated_multimodal_description` belongs to T2VA/I2VA/FL2VA/L2VA.

## E06 — Requested versus rendered duration

**User:** “Create an FL2VA prompt for five seconds. I do not know the snapped frame count.”

**Correct behavior:** Use `5.00-second mark` as the requested duration and do not claim it was calculated from ComfyUI's frame grid. If the user later provides the rendered frame count, update the alignment line using `frames ÷ 24`.

**Do not do this:** Invent a snapped duration or silently change the requested duration.
