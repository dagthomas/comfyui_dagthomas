# H3 Ref2VA Gold Examples

Use only with the full-reference workflow. Every example uses the official six-section structure. Connection order stated in each brief controls `<Picture N>`, `<Video N>`, and `<Audio N>` tags.

## R01 — Portrait identity plus voice reference

**Brief:** Picture 1 is a chef portrait; Audio 1 is her voice. Create a restaurant-kitchen scene where she says exactly, “Service begins now.”  
**Mode:** Ref2VA · 8.00 seconds  
**Style recipe:** V19 live action + M01 acting + A01 realism

```text
subject_definitions:
<Subject 1> is the chef whose facial identity, short dark curls, white jacket, blue neckerchief, and small silver earrings come from <Picture 1>.
<Audio 1> is the voice-timbre and measured-delivery reference for <Subject 1> (S1).

summary:
[reference generation + audio reference] The target video shows <Subject 1> preparing the kitchen for dinner service while <Audio 1> guides her spoken voice without copying its original signal or words.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - facial identity, hair, uniform, neckerchief, and earrings remain consistent throughout.
<Audio 1>: reference - its timbre and measured delivery guide <Subject 1> (S1), while the requested target line is newly performed.

detailed_description:
The target video uses photoreal live-action restaurant cinematography with warm overhead practicals, controlled stainless-steel reflections, and restrained handheld movement.
[Shot 1] A medium-wide shot frames <Subject 1> at the center pass of a working restaurant kitchen. Her facial identity, short dark curls, white chef's jacket, blue neckerchief, and silver earrings remain consistent with <Picture 1>. She checks three plated dishes from left to right, wipes one small sauce mark from the nearest rim, and turns her head toward the line cooks off-screen. The camera tracks slowly beside the pass, preserving her profile and the orderly row of heat lamps. Steam rises behind her while cooks move as soft background shapes without blocking her face. She rotates the center plate by a few degrees, compares its garnish height with the plate beside it, and uses the back of a spoon to restore one precise sauce edge. A cook crosses behind her carrying a pan; its brief reflection passes through the steel backsplash without changing the key light on her face. <Subject 1> (S1), using the stable voice timbre and measured delivery referenced from <Audio 1>, places both hands on the counter and says: <d>[English] Service begins now.</d> She closes her lips and gives one concise nod. Her left hand moves to the folded service cloth while her right hand reaches toward the bell, establishing the next action before the cut. [Shot 2] At 00:05.000, the shot cuts to a close view of her right hand ringing the service bell once. The bell cap depresses and returns, the cloth remains folded beneath her opposite hand, and the nearest plate stays aligned under the heat lamp. The camera tilts up to her face as she lifts the first plate, pivots toward the pickup side, and passes it out of frame. A server's hands receive the plate without covering its garnish. Her uniform, jewelry, and identity remain unchanged while the kitchen activity increases behind her. She immediately checks the second plate, lifts it with both hands, and pauses just long enough for steam to clear her face. The final composition holds her in three-quarter view beneath the warm lamps, shoulders squared, gaze directed toward the pickup area, and the second plate stable at counter height.

overall_soundscape:
Kitchen ventilation, low burner flame, restrained utensil contact, and distant staff movement continue throughout. A plate rim is wiped, the service bell rings once, and ceramic slides lightly across the pass.

non_diegetic_music:
N/A
```

**Decision note:** Audio is a timbre reference, not a copied signal; the supplied line replaces any source words.

## R02 — Identity, wardrobe, and style from separate pictures

**Brief:** Picture 1 provides a woman's identity, Picture 2 provides a red coat, Picture 3 provides a flat mid-century graphic style. She waits at a bus stop.  
**Mode:** Ref2VA · 8.00 seconds  
**Style recipe:** V11 mid-century flat graphic + M02 held timing + F01 clean digital

```text
subject_definitions:
<Subject 1> is the woman whose facial identity and short black bob come from <Picture 1>, wearing the red coat and black buttons from <Picture 2>.
<Subject 2> is the mid-century flat graphic treatment from <Picture 3>, including asymmetric layouts, simplified silhouettes, bold color blocks, sparse linework, and selective limited movement.

summary:
[reference generation] The target video shows <Subject 1> waiting at a stylized bus stop while the full scene adopts <Subject 2>'s graphic treatment.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - the identity, bob haircut, coat color, cut, and button placement remain stable.
<Subject 2> (applies to [Shot 1], [Shot 2]): attribute_transfer - the visual treatment from <Picture 3> is transferred across the newly generated woman, street, bus shelter, and passing vehicle.

detailed_description:
The target video uses <Subject 2>, a mid-century flat graphic animation style with asymmetric composition, simplified silhouettes, bold restrained color blocks, sparse linework, and clean digital edges.
[Shot 1] A wide graphic composition places <Subject 1> on the far right beneath a simple bus-stop canopy, preserving her facial identity, short black bob, red coat, and black buttons from the two source pictures. Large cream, teal, and charcoal shapes define the street and buildings. The camera holds static while only the coat hem, her eyes, and one loose strand of hair move. She looks down the empty road, checks a small wristwatch through one economical wrist rotation, and returns to a held pose. A paper timetable on the shelter reads "ROUTE 8" in clean black letters. Three simplified leaves cross the empty upper-left space one after another; her eyes follow the final leaf, but her shoulders remain in the same held drawing. A narrow charcoal shadow advances beneath the canopy while the red coat stays the dominant color accent. She shifts a rectangular tote bag from one hand to the other through two crisp key poses, then restores a clean silhouette with the bag hanging parallel to her leg. [Shot 2] At 00:04.500, the shot changes to a frontal medium view as a simplified teal bus slides into frame behind her at constant speed. Its windows pass as an evenly spaced dark-blue rhythm. The woman takes one small step forward; her red coat separates clearly from the bus color. The coat hem lags by one held frame and snaps gently into place. The doors fold open through two crisp graphic shapes. A cream interior rectangle appears behind them while a dark handrail resolves as one simple vertical line. She turns her head, gives a restrained smile, and reaches toward the doorway. Her wrist, elbow, and shoulder change through distinct economical poses rather than fluid full animation. She steps onto the first stair, keeps the tote clear of the door, and pauses as the flat background shapes lock into a balanced final layout. The final frame holds her red silhouette against the open cream-colored bus entrance.

overall_soundscape:
Light city ambience and a restrained electric-bus hum continue under the scene. The wristwatch clasp gives one small click, tires hiss briefly, and the folding doors open with two clean mechanical sounds.

non_diegetic_music:
Short vibraphone notes at a moderate tempo alternate with muted upright-bass plucks, decreasing in volume as the doors open.
```

**Decision note:** Style is modeled as a reusable subject and correctly marked as attribute transfer to newly generated content.

## R03 — Camera movement from reference video

**Brief:** Picture 1 provides the runner; Video 1 provides only a low backward tracking move. Create a nighttime sportswear shot.  
**Mode:** Ref2VA · 7.00 seconds  
**Style recipe:** V19 live action + M05 weighty action + F02 35 mm + A01 realism

```text
subject_definitions:
<Subject 1> is the runner whose identity, braided hair, black-and-orange sportswear, and shoes come from <Picture 1>.
<Video 1> is the camera-motion reference, providing only its low backward tracking path, stable horizon, acceleration, and framing response.

summary:
[reference generation] The target video follows <Subject 1> sprinting through a wet nighttime underpass while its camera reproduces the movement structure of <Video 1>.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - identity, hair, clothing colors, garment design, and shoes remain consistent.
<Video 1> (camera movement and pacing): fully_preserved - the low backward tracking path, stable horizon, acceleration, and responsive framing guide the full shot without importing visible source-video content.

detailed_description:
The target video uses photoreal sports-commercial cinematography with wet practical reflections, fine film grain, coherent motion blur, and strong but physically plausible contrast throughout.
[Shot 1] A low full-body shot frames <Subject 1> at the entrance of a concrete underpass after rain. Her facial identity, braided hair, black-and-orange sportswear, and shoes remain fully consistent with <Picture 1>. The camera begins the same low backward tracking movement established by <Video 1>, holding a stable horizon while responding to her acceleration. She shifts weight forward, drives off the rear foot, and enters a sprint through clear, powerful stride cycles. Each shoe compresses at contact and releases water from the pavement in a narrow spray. Her arms counter the legs, while braids and loose garment edges trail and settle naturally after each change in direction. The camera increases backward speed in the same progression as <Video 1>, maintaining the referenced distance and low angle instead of orbiting or zooming. Overhead practical lights pass rhythmically across her face and clothing. Reflections of the orange garment panels stretch across shallow puddles and break apart under each footfall. The underpass columns move past at the same visual rate demonstrated in <Video 1>, reinforcing the borrowed camera pacing without copying its location. A bicycle silhouette crosses a distant side opening but never enters her running path. At mid-shot she glances briefly toward that opening, plants the outside foot, and changes lane through one readable lateral push while the camera reframes in the referenced manner. Her inside shoulder drops slightly, the outside arm widens for balance, and both return to the normal sprint pattern after the lane change. She passes beneath a flickering practical light; exposure responds naturally without hiding her face or shifting clothing color. She returns her gaze forward and accelerates through the final pool of orange light. The camera preserves the same tracking relationship until the exit opens into cool blue night. Both runner and camera decelerate slightly, her stride length shortens through two visible contacts, and the shot ends with her identity, orange garment panels, braided hair movement, and wet shoe contacts clearly readable.

overall_soundscape:
Rapid shoe impacts and water spray move through the underpass reverb beneath controlled breathing and fabric movement. Distant traffic and a low ventilation fan establish the location.

non_diegetic_music:
A 120 BPM pattern of low electronic percussion and short processed claps follows the stride rhythm, adding a sustained bass tone during the lane change and stopping cleanly at the end.
```

**Decision note:** `<Video 1>` controls camera structure only; its characters, location, and sound are not imported.

## R04 — Direct video edit with complete audio reuse

**Brief:** Video 1 is a six-second café clip. Keep its timing and complete soundtrack but replace every visible cup with the branded cup from Picture 1.  
**Mode:** Ref2VA · 6.00 seconds

```text
subject_definitions:
<Video 1> is the source video for the target video edit, including its shots, timing, camera movement, people, café environment, and object interactions.
<Subject 1> is the branded ceramic cup from <Picture 1>, including its exact white shape, blue handle, printed "NORTH" wordmark, and logo placement.
<Audio 1> is the enabled synchronized soundtrack from <Video 1>, reused as the complete target audio track.

summary:
[video editing + reference generation + audio reuse] The target video is an edited version of <Video 1>. Every visible cup is replaced by <Subject 1> while the source timing, camera, actions, and complete <Audio 1> soundtrack remain unchanged.

retention_analysis:
<Video 1> (complete visual structure except cups): partially_preserved - all shots, people, environment, timing, camera motion, and interactions remain, while the original cup appearances are replaced.
<Subject 1> (appears wherever a cup appears in [Shot 1] and [Shot 2]): fully_preserved - exact shape, white ceramic, blue handle, "NORTH" wordmark, and logo placement are retained on every replacement.
<Audio 1>: fully_copy - the complete synchronized source soundtrack is reused 1:1 without added, removed, or regenerated layers.

detailed_description:
The target video preserves the original live-action café photography, lighting, cut structure, and camera behavior of <Video 1>.
[Shot 1] The shot follows <Video 1> exactly: the same customer sits at the same table, performs the same gestures, and lifts a cup at the same moment while the camera maintains the source framing and motion. Replace the original cup with <Subject 1>, preserving its white ceramic body, blue handle, exact printed "NORTH" text, and logo placement. The replacement follows the source object's position, rotation, grip, occlusion, reflections, and contact shadows frame by frame. The person's fingers wrap naturally around the blue handle without covering or distorting the wordmark. All other people, furniture, window light, table objects, and background motion remain unchanged from <Video 1>. [Shot 2] At the original source cut time, preserve the source close-up of a second cup being set on the counter. Replace that cup with the same <Subject 1> design, matching the source object's approach, downward movement, ceramic contact, final rotation, focus, and shadow. Keep the "NORTH" wordmark legible whenever its surface faces the camera; do not rotate the logo independently of the cup. Preserve the barista's hands, counter, espresso machine, steam, source camera movement, and exact shot duration. <Audio 1> remains synchronized to every original event and is copied without modification through the final frame.

overall_soundscape:
The ambience, dialogue, machine sounds, cup contacts, and all other sound layers from <Audio 1> are directly reused as the complete target soundtrack.

non_diegetic_music:
Any audience-only music contained in <Audio 1> is directly reused as part of the complete copied soundtrack; no additional music is generated.
```

**Decision note:** `partially_preserved` correctly reflects the cup replacement, while the soundtrack is `fully_copy`.

## R05 — Video continuation with audio reference

**Brief:** Continue Video 1 for eight seconds from its final forest shot. Use its enabled audio only as ambience and music-continuity reference, not a copied signal.  
**Mode:** Ref2VA · 8.00 seconds

```text
subject_definitions:
<Video 1> is the source video whose final frame, forest environment, walking character, camera direction, and temporal state begin the continuation.
<Subject 1> is the traveler from <Video 1>, including visible identity, brown coat, canvas backpack, walking rhythm, and direction of travel.
<Audio 1> is the enabled source soundtrack used as a reference for forest ambience, footstep perspective, and the instrumentation and tempo of the continuing score without copying the signal.

summary:
[video continuation + reference generation + audio reference] The target video continues directly from <Video 1>, following <Subject 1> deeper into the forest while new sound preserves the audible characteristics of <Audio 1> without copying it.

retention_analysis:
<Video 1> (continuation state and camera direction): fully_preserved - the new first frame, environment, direction, and camera trajectory continue seamlessly from the source ending.
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - identity, coat, backpack, gait, and direction remain consistent.
<Audio 1>: reference - newly generated ambience, footsteps, and score follow its perspective, instrumentation, tempo, and continuity without reusing the source signal.

detailed_description:
The target video continues the naturalistic forest cinematography, soft overcast light, subdued color palette, and measured pacing established by <Video 1>.
[Shot 1] The opening begins directly from the final frame of <Video 1>. <Subject 1> remains in the same walking phase, screen position, brown coat, canvas backpack, forest path, and direction of travel. The camera continues the source's slow forward tracking path without a jump in height or lens behavior. The traveler completes the interrupted step, passes a moss-covered trunk, and brushes one low branch aside with the left hand. The branch bends, trails behind the sleeve, and returns gradually. Newly generated footsteps and forest ambience follow the distance and acoustic character referenced from <Audio 1>. The backpack shifts downward with the next foot contact and settles against the coat. Small beads of water fall from the disturbed branch and darken the traveler's left shoulder without altering the garment. The path narrows between ferns; the camera moves slightly closer in the same direction, allowing the foreground leaves to occlude the lower frame for a moment while the traveler's identity and route remain clear. A pale marker tied around a distant tree becomes visible ahead, motivating the traveler's brief change of gaze. [Shot 2] At 00:04.800, the shot cuts to a side view as the traveler reaches a narrow stream crossing the path. Identity, clothing, backpack, and walking rhythm remain stable. The traveler slows through two shortening steps, tests one wet stone with the front foot, and shifts weight cautiously before committing. The supporting knee bends, the rear heel lifts, and one hand steadies the backpack strap. The traveler steps across while the camera trucks right at the same measured pace. Water moves around the stones, a boot displaces a small splash, and the coat hem responds to the turn. A new score continues the instrumentation and tempo of <Audio 1> without copying any waveform, melody, or verbal content. On the middle stone, the traveler pauses and looks toward the pale trail marker through the trees. The final step reaches the far bank; soil compresses under the boot and the free hand releases the strap. The traveler looks toward a brighter opening between the trees, resumes the original direction, and passes behind a foreground trunk while the camera settles into the same trailing relationship established by the source video.

overall_soundscape:
Newly generated forest air, distant birds, leaf movement, footsteps, branch contact, and running water reference the acoustic perspective and restraint of <Audio 1> without copying it.

non_diegetic_music:
A newly generated continuation of <Audio 1>'s instrumental palette and slow tempo uses soft plucked strings and sustained low woodwinds, maintaining similar dynamics without copying the source signal or melody.
```

**Decision note:** Continuation begins from the source ending; audio is `reference`, not `partially_copy`.

## R06 — Full-reference keyframe completion

**Brief:** Picture 1 provides a character identity; Picture 2 is the exact final frame of Shot 2. Create a seven-second bookstore scene.  
**Mode:** Ref2VA · 7.00 seconds

```text
subject_definitions:
<Subject 1> is the bookseller whose facial identity, round glasses, grey cardigan, and braided hair come from <Picture 1>.
<Picture 2> is the last frame of [Shot 2], showing <Subject 1> on a ladder holding a blue book beneath the brass reading lamp.

summary:
[reference generation + keyframe completion] The target video follows <Subject 1> locating a book in a quiet shop and ends exactly on the concrete frame anchor <Picture 2>.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - identity, glasses, cardigan, and braided hair remain consistent.
<Picture 2> ([Shot 2] last frame): fully_preserved - final ladder position, blue book, body pose, brass lamp, lighting, camera angle, and composition are matched exactly.

detailed_description:
The target video uses warm live-action bookstore cinematography with soft practical light, wooden shelf texture, restrained depth of field, and quiet naturalistic acting.
[Shot 1] A medium-wide shot establishes <Subject 1> behind a wooden counter, preserving the facial identity, round glasses, grey cardigan, and braided hair from <Picture 1>. She reads a handwritten request card, looks toward the upper shelves, and traces one shelf row with her eyes before stepping around the counter. The camera pans right slowly as she passes a small stack of returns and pulls a rolling ladder along its rail. Its brass wheels turn visibly, cross two shelf divisions, and stop beneath a reading lamp. She checks the shelf number against the card, folds the card once, and places it in the cardigan pocket without changing the garment. Dust moves gently through the lamp beam. She steadies the ladder with one hand, tests the first rung with the front of her shoe, and places her weight onto it while the free hand reaches for the rail. [Shot 2] At 00:03.800, the shot cuts to the angle that will become <Picture 2>. <Subject 1> climbs two rungs through careful weight shifts, preserving the glasses, braid, cardigan folds, and exact identity. The ladder remains fixed on its rail while the camera begins a restrained push. She reaches toward a cluster of books, touches two spines with her index finger, and stops on the blue volume. The book resists slightly; she braces with the other hand and slides it halfway free. The neighboring books compress and return as the volume clears. A thin layer of dust releases from the top edge and drifts through the brass light. She holds the rail, turns the cover toward herself, confirms it against the card partly visible in her pocket, and adjusts the book into the exact hand position shown in <Picture 2>. During the final two seconds, her movement becomes progressively smaller. The camera completes its push while her foot placement, shoulder angle, braid position, glasses, blue book, brass lamp, shelf spacing, hand grip, lighting, focus, and every compositional boundary converge on <Picture 2>. The shot ends exactly on that frame without a late blink or camera drift.

overall_soundscape:
Quiet room tone, faint street noise through glass, rolling ladder wheels, wooden rung creaks, cardigan movement, and a book sliding against paper fill the shop.

non_diegetic_music:
N/A
```

**Decision note:** A standalone `<Picture 2>` is valid because it is a concrete final-frame anchor.

## R07 — Product identity plus surface style reference

**Brief:** Picture 1 is the exact sneaker product; Picture 2 supplies only a black-and-white ink illustration treatment. Create a rotating launch film with no dialogue.  
**Mode:** Ref2VA · 8.00 seconds  
**Style recipe:** V12 comic print + M08 precision + F05 paper/ink + A05 product detail

```text
subject_definitions:
<Subject 1> is the sneaker from <Picture 1>, including its exact silhouette, panel geometry, sole profile, laces, stitching, and logo placement.
<Subject 2> is the monochrome ink-illustration treatment from <Picture 2>, including variable contour weight, cross-hatched shadow, visible paper tooth, and sparse ink splatter.

summary:
[reference generation] The target video presents <Subject 1> in a controlled product rotation while transferring <Subject 2>'s monochrome ink treatment across the product and environment.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - all product geometry, panels, sole, laces, stitching, and logo placement remain exact.
<Subject 2> (applies throughout): attribute_transfer - the ink and paper treatment is transferred from <Picture 2> to the newly staged sneaker film without transferring its original subject or composition.

detailed_description:
The target video uses monochrome comic-ink product animation with variable contour weight, cross-hatched shadow, visible paper tooth, restrained ink splatter, and clean negative space.
[Shot 1] A three-quarter product view frames <Subject 1> floating above a white paper-textured ground. The exact silhouette, panel geometry, sole profile, laces, stitching, and logo placement from <Picture 1> remain unchanged while <Subject 2>'s ink treatment defines every visible surface. The camera arcs clockwise with small amplitude at slow speed as the sneaker rotates in the opposite direction at constant speed, revealing side, heel, and outsole without distortion. Cross-hatching changes density according to the moving light but never changes the product construction. The lace tips remain aligned with gravity and respond with only a slight delayed swing. As the heel rotates toward camera, the outsole grooves become visible one by one and keep their exact depth and spacing. Fine ink flecks trail briefly behind the heel and settle into the paper background. Two broader brush marks enter from opposite frame edges, pass behind the shoe without occluding its logo, and flatten into graphic shadow bands. The sneaker completes three quarters of its rotation and pauses with the lateral profile readable. [Shot 2] At 00:04.500, the shot cuts to a macro view traveling along the sole edge. The camera tracks from heel to toe while panel seams, stitch count, lace routing, and logo remain exact. Paper texture passes softly through the white negative space, but the product edges stay stable. A narrow band of cross-hatching moves across the midsole to reveal its curvature, then clears before reaching the logo. The camera reaches the toe, tilts upward through a small controlled angle, and reveals the lace bed and tongue without changing proportions. A single bold ink line sweeps beneath the shoe and resolves into a stable shadow. The sneaker lowers at constant speed until the sole makes controlled contact with the illustrated ground, creating one compact ink-ring accent. The outsole compresses only as allowed by its material, then returns without bounce. All linework, ink flecks, shadow bands, laces, stitching, and camera movement stabilize while the logo remains unobstructed, legible, and correctly proportioned.

overall_soundscape:
Close, clean lace movement, a soft rubber-like surface pass, and one restrained contact sound accompany the product actions within a nearly silent studio bed.

non_diegetic_music:
A sparse 90 BPM pattern of dry rim clicks and low plucked bass notes marks the rotation and stops on the final contact.
```

**Decision note:** Style transfer does not alter product geometry; no unsupported slogan or brand copy is invented.

## R08 — Dance motion and partial music reuse

**Brief:** Picture 1 gives the dancer's identity and costume. Video 1 supplies choreography. Audio 1 is a music clip; reuse only its first eight seconds and add synchronized footfalls.  
**Mode:** Ref2VA · 8.00 seconds

```text
subject_definitions:
<Subject 1> is the dancer whose facial identity, silver braided hair, blue asymmetrical costume, and white boots come from <Picture 1>, performing the choreography from <Video 1>.
<Video 1> is the motion reference for the full eight-second dance phrase, including pose order, body pathways, turns, and timing.
<Audio 1> is the music source whose first eight seconds are copied while newly generated footfalls and fabric movement are added.

summary:
[reference generation + audio reuse] The target video shows <Subject 1> performing <Video 1>'s choreography in a mirrored studio while partially copying <Audio 1> and adding synchronized physical sounds.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - identity, hair, costume construction, color, and boots remain stable while the referenced movement is performed.
<Video 1> (choreography and timing): fully_preserved - pose sequence, turns, pathways, and beat alignment are reproduced across the target shots.
<Audio 1>: partially_copy - its first eight seconds are copied, while new footfalls, breaths, and costume movement are mixed with the reused music.

detailed_description:
The target video uses polished stylized live-action dance cinematography with cool studio light, clean mirror reflections, and clear full-body staging.
[Shot 1] A wide shot frames <Subject 1> in a mirrored rehearsal studio, preserving her facial identity, silver braided hair, blue asymmetrical costume, and white boots from <Picture 1>. She begins the eight-second choreography from <Video 1> in its exact opening pose. On the first beat of <Audio 1>, her right shoulder releases, the torso follows through a curved pathway, and the left foot crosses according to the source sequence. The camera pushes in slowly while keeping her boots and complete silhouette visible. She performs the same arm sweep and directional change from <Video 1>; hair and costume fabric trail naturally without changing the core pose timing. The mirror reflects the same costume construction and movement with correct spatial correspondence rather than generating a second independent performer. Newly generated shoe contacts align precisely with the copied music. On the third beat, her hands pass close to the torso and open outward in the exact source order. The white boots slide through the referenced diagonal pathway, stop without foot drift, and prepare the turn. The camera reaches a medium-wide distance but preserves both hands and feet inside frame. [Shot 2] At 00:04.000, the shot cuts to a three-quarter side view while preserving the choreography's continuous phase and exact beat position. <Subject 1> completes the referenced turn, lands on the correct supporting foot, and moves through the source video's final pose sequence. The reflected dancer remains synchronized and spatially accurate. The camera trucks right to maintain full-body readability as she lowers one shoulder, extends the opposite arm, and travels through two quick steps from <Video 1>. Her braids continue rotating after the head stops, then fall back across the costume. A controlled breath becomes audible after the turn, and the costume gives one fabric snap during the last arm extension. The music from <Audio 1> remains copied continuously across the cut, while the added footfalls occupy the same physical studio perspective. The final pose lands on the eighth-second musical beat, with feet, hands, gaze, body angle, costume panels, braid placement, and mirror alignment matching the referenced choreography while identity and clothing remain stable.

overall_soundscape:
Newly generated boot contacts, one controlled breath, and costume movement are synchronized to the visible dance and mixed beneath the copied portion of <Audio 1>.

non_diegetic_music:
The first eight seconds of <Audio 1> are directly reused as the audience-only score, with no extension or newly generated musical layers.
```

**Decision note:** Added physical sounds make the audio `partially_copy`, not `fully_copy`; choreography is a video-structure relationship.


## R09 — Single-video abstract style + performance transfer

**Brief:** Video 1 is an animation reference. Use its drawing style and performance grammar, but create a completely new compact woman opening a paper fan in a sparse room. Do not copy source characters, props, setting, exact poses, or story.  
**Mode:** Ref2VA · 5.00 seconds  
**Empirical strategy:** derived style/performance subject + whole-video temporal role + content firewall

```text
subject_definitions:
<Subject 1> is the abstract drawing and performance language sourced from <Video 1>, including line behavior, graphic density, expressive deformation, follow-through, and recovery. It does not include source character identity, anatomy, clothing, props, setting, exact composition, exact poses, exact actions, dialogue, or story content.
<Video 1> is the whole-video temporal reference for pose rhythm, anticipation, and the relative timing of holds, acceleration, landing, and recovery.

summary:
[reference generation] The target video applies <Subject 1>'s abstract drawing/performance language and <Video 1>'s temporal rhythm to a completely new compact woman and a new paper-fan action.

retention_analysis:
<Subject 1>: attribute_transfer - transfer non-identifying visual surface and performance grammar while preserving the user-defined woman, prop, setting, action, and narrative.
<Video 1>: attribute_transfer - transfer only whole-video temporal relationships such as hold-to-burst timing and recovery rhythm.

detailed_description:
Use <Subject 1> continuously as the dominant source for HOW the animation is drawn and performed, while <Video 1> guides the relative timing of held ideas and action beats. Do not impose another named animation medium over the reference. The prompt controls WHAT appears.
[Shot 1] A locked medium composition shows one entirely original short, compact woman with a broad round face, small nose, short tightly curled hair, broad shoulders, and plain loose clothing beside a simple table. She holds one folded paper fan. She studies it during a readable hold, opens it once, and reacts when it snaps fully open. Her eyes lead the head; the body follows. At peak speed, allow a coherent expressive transitional drawing if supported by <Subject 1>, then land on a clean readable reaction pose. Her clothing and fan continue after the torso stops and settle afterward. Keep her compact design stable even when action drawings deform. No recognizable source character, costume, prop, location, composition, pose, or story element appears.

overall_soundscape:
Quiet room tone, soft paper movement, one crisp fan-opening snap, light clothing movement, and subtle settling.

non_diegetic_music:
N/A
```

**Decision note:** Visual style/performance is reusable content derived as `<Subject 1>`; `<Video 1>` remains the whole-video temporal source.

## R10 — Three videos with separated style responsibilities

**Brief:** Video 1 supplies line/surface treatment, Video 2 supplies visible acting/deformation, Video 3 supplies cut and graphic motion rhythm. Make an original woman flick a small paper windmill.  
**Mode:** Ref2VA · 5.00 seconds  
**Empirical strategy:** modular multi-reference role separation

```text
subject_definitions:
<Subject 1> is the sparse line/surface treatment sourced from <Video 1>, including redraw character, contour variation, and graphic economy.
<Subject 2> is the visible performance language sourced from <Video 2>, including anticipation, expressive deformation, overshoot, follow-through, and recovery.
<Video 3> is the whole-video temporal/compositional reference for motion punctuation, rotational rhythm, and cut/scale relationships. None of the sources supplies character identity, anatomy, clothing, props, setting, exact composition, exact poses, exact actions, dialogue, or story content.

summary:
[reference generation] The target video combines two derived abstract subjects with <Video 3>'s temporal/compositional grammar while keeping the woman, windmill, setting, and action completely original.

retention_analysis:
<Subject 1>: attribute_transfer - transfer only sparse line surface and redraw character from <Video 1>.
<Subject 2>: attribute_transfer - transfer only performance timing, deformation, follow-through, and recovery from <Video 2>.
<Video 3>: attribute_transfer - transfer only whole-video graphic motion organization and cut/scale rhythm.

detailed_description:
Use all three source roles continuously with separated responsibilities rather than averaging them into a generic style. <Subject 1> controls HOW EACH DRAWING IS BUILT. <Subject 2> controls HOW THE CHARACTER PERFORMS. <Video 3> controls HOW the windmill's graphic rotation and any requested reframing are organized through time.
[Shot 1] A locked medium composition shows one entirely original young woman with a compact build, broad round face, small nose, short tightly curled hair, and plain loose jacket at a minimal tabletop. One small paper windmill is mounted on a thin stick. She touches a blade and the windmill begins to rotate. One blade catches. She notices, anticipates briefly, and gives it one quick flick. Use <Subject 2> for the sharp hand acceleration and recovery; use <Video 3> for the clear graphic rotational rhythm; use <Subject 1> so the line construction stays economical and reference-derived throughout. The woman's hand may pass through one coherent transitional deformation, but her identity anchors remain stable. Exactly one woman and one windmill remain visible. No source prop, mascot, logo, setting, or narrative element appears.

overall_soundscape:
Quiet ambience, light paper movement, one fingertip tap, a quick flick, rapidly rotating paper blades, and progressively softer rotation.

non_diegetic_music:
N/A
```

**Decision note:** More references do not automatically mean more fidelity; visual/performance roles become `<Subject N>`, while `<Video 3>` is reserved for whole-video temporal structure.

## R11 — Physical material process with explicit limitation-aware prompting

**Brief:** Video 1 shows a physical-media animation process. Transfer the visible material behavior to one simple original abstract leaf-like form.  
**Mode:** Ref2VA · 5.00 seconds  
**Empirical strategy:** low semantic load + explicit physical process

```text
subject_definitions:
<Subject 1> is the physical material behavior sourced from <Video 1>: thick luminous pigment, visible brush pressure, pooled paint, dragged edges, soft backlighting, smeared intermediate states, and reconstruction of readable form through redistribution of existing paint. No source subject, prop, setting, composition, exact action, or story content is retained.

summary:
[reference generation] The target video applies <Subject 1>'s physical-material behavior to one simple original abstract form with minimal competing semantic demands.

retention_analysis:
<Subject 1>: attribute_transfer - transfer the visible wet-paint material behavior and temporal redistribution of marks while excluding recognizable source content.

detailed_description:
Use <Subject 1> as the dominant reference for the PHYSICAL MATERIAL BEHAVIOR. The subject is not a rigid finished illustration with a paint filter. It exists as one continuously manipulated paint field. Treat material-process fidelity as the priority and keep content deliberately simple.
[Shot 1] A locked macro composition shows one original asymmetrical leaf-like painted form against a softly illuminated field. It holds briefly, then bends toward screen right. The bend occurs by visibly redistributing the existing paint: pigment on the compressed side thickens and pools; pigment on the stretched side drags into longer thinner marks; the previous contour partially smears away before a new readable contour reforms. The form returns toward center through another visibly repainted transition. No character, anatomy, dialogue, secondary prop, or detailed environment competes with the material process.

overall_soundscape:
N/A

non_diegetic_music:
N/A
```

**Decision note:** Physical-process transfer is higher variance than surface or performance transfer. Low semantic complexity improves the chance of visible process behavior, but the prompt must not promise exact frame-by-frame replication.
