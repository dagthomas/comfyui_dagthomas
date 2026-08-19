# MiniMax-H3 Crossover Scene Rules

Production-tested rules for writing multi-character *crossover* scenes (characters from
different shows and films sharing one story) as independent MiniMax-H3 T2VA clips. These
sit on top of the official base guide: the official guide owns field names and grammar;
this document owns the four-section crossover layout and the anti-artifact rules learned
from rendering hundreds of these scenes.

## 1. The four-section scene layout

Every scene is one plain-text prompt in exactly this order, sections separated by one blank
line, no markdown, no bold, no bullets:

```
subject_definitions:
<Subject 1> [Character] (played by [Actor]) from [Show]
<Subject 2> [Character] (played by [Actor]) from [Show]

integrated_multimodal_description:
[Shot 1] [Visual style, lighting, time of day, setting]. A single medium close-up of <Subject 1> [Character] (S1) alone in frame, wearing [wardrobe anchors]. <Subject 2> is not in frame. The shot opens with <Subject 1> [Character] (S1) already speaking, with no silent establishing beat. The camera holds a static shot as <Subject 1> [Character] (S1) says: <d>[English in [Character]'s voice from [Show]] Dialogue.</d>

[Shot 2] At 00:05.500, the shot cuts to a single medium close-up of <Subject 2> [Character] (S2) alone in frame, wearing [wardrobe anchors]. <Subject 1> is not in frame. The camera pushes in with small amplitude at slow speed as <Subject 2> [Character] (S2) [physical action] and says: <d>[English in [Character]'s voice from [Show]] Dialogue.</d>

overall_soundscape:
[Room tone, physical sounds, ambient environment, non-verbal human sounds. Never voices.]

non_diegetic_music:
N/A
```

- `subject_definitions:` lists every character who appears on screen in this scene, one per
  line, in the exact form `<Subject N> Character (played by Actor) from Show`. Use the
  character, actor and show strings you were given verbatim - never rename, never
  substitute another actor. If the show string carries extra look notes after a comma,
  keep them as wardrobe/physical anchors in the shots but put only the show name after
  `from`.
- Never bold section headers. `subject_definitions:` not `**subject_definitions:**`.
- `[Shot 1]` never carries a timecode. `[Shot 2]` and later always open with
  `At MM:SS.mmm, the shot cuts to ...`.
- Scenes with no people (never the case here unless asked) write `subject_definitions:` then
  `N/A` on its own line.

## 2. Speaker binding - the rules that stop face and voice blending

1. **The primary speaker of the scene is always `<Subject 1>`.** H3 binds dialogue audio
   to `<Subject 1>`. Reorder the cast per scene so the character with the most spoken
   dialogue is `<Subject 1>`.
2. **Every spoken line is wrapped**: `<d>[English in Character's voice from Show] words</d>`.
   Speaker phrase, ID `(S1)`, action and delivery stay outside the tag; only the language
   tag and the words go inside.
3. **Stable IDs** `(S1)`, `(S2)`, `(S3)` across every shot of the scene. `(S1,S2)` only for a
   single word or short phrase spoken in unison.
4. **Silence mandates for everyone on screen who is not speaking**:
   `<Subject 2> Name (S2) remains completely silent with his mouth closed and lips sealed,
   speaking no dialogue.` Add it whenever a non-speaker is in frame, and after a speaker
   finishes early in a shot: `then falls silent with her mouth closed and lips sealed,
   speaking no further dialogue for the remainder of the shot.`
5. **Isolated single-subject shots are the safe default for dialogue exchanges**: `a single
   medium close-up of <Subject 1> Name (S1) alone in frame ... <Subject 2> is not in
   frame.` Say who is *not* in frame explicitly.
6. **Shared frames are allowed and encouraged for variety** (establishing shots,
   walk-and-talks, handoffs, toasts, reactions to the same event) with safeguards: state
   each subject's position (`<Subject 1> stands screen-left, <Subject 2> stands roughly
   three feet to his right at screen-right, separated by a clear gap with no overlap`), one
   active speaker at a time, silence mandate on the others, restate wardrobe anchors, and
   never more than 3 people visible in one shot (prefer 2).
7. **After an action beat, cut to the victor alone** before they deliver a line, with an
   off-camera eye-line (`looking down at the fallen guard off-camera`).
8. **Off-screen voiceover**: `says in an off-screen voiceover: <d>...</d> while his lips
   remain completely closed.`
9. **Never name a voice or dialogue in `overall_soundscape:`** (`and Saul's fast-talking
   voice` is a known double-voice bug). Non-verbal sounds (a cough, a laugh) are fine.
   Never put musical score in the soundscape; it belongs in `non_diegetic_music:`.

## 3. Identity anchoring

- Restate key wardrobe / physical anchors for every subject in every shot they appear in
  (`brown leather jacket over a dark flannel shirt, silver pendant necklace`; `bald head,
  facial scar over his right eye`). Generic terms (`the agent`, `in a shirt`) drift.
- Anchor era/age for long-career actors when it matters:
  `John McClane (played by 35-year-old Bruce Willis, 1988 Die Hard era) from Die Hard`, and
  restate the age-defining traits in the shots.
- Keep costume, hair, props and demographics identical across every scene of the story.
- Visible on-screen text goes in English double quotes: `a red neon sign reading "QUARANTINE"`.

## 4. Pacing, duration and dead air

- Each scene lands between 5 and 20 seconds; prefer 12-15 s for dialogue scenes, 5-8 s for
  transitions, punches and arrivals. Never exceed 20 s.
- **Fill the whole duration** with dialogue and physical action. No multi-second silent
  stares, no frozen smiles at the end. If dialogue ends 1-2 s early, end on concrete
  movement: pocketing a prop, turning and walking away, stepping out of frame, closing a
  case, sliding into a car.
- Proven structures for a 15 s scene:
  - A: S1 speaks (00:00-00:07) -> S2 speaks and acts to the end (00:07-00:15)
  - B: S1 speaks (00:00-00:06) -> S2 speaks (00:06-00:11) -> S2 pockets prop, turns, walks (00:11-00:15)
  - C: S1 (00:00-00:05) -> S2 (00:05-00:10) -> S1 quick closer while stepping forward (00:10-00:15)
- Spread cut timecodes across the actual scene duration (`00:05.500`, `00:10.500` for 15 s;
  `00:04.000`, `00:08.000` for 12 s). Vary the shot pattern between scenes.
- Open T2V clips on the speaker already talking: `The shot opens with <Subject 1> Name (S1)
  already speaking, with no silent establishing beat.`

## 5. Staging, story and flow across scenes

- **Show, don't tell.** Characters physically do things: uncork a flask, unlatch a case,
  tap a glowing interface, unhinge a trunk, aim a device, lean on a hood, push through a
  door. Never a static monologue into the lens.
- **Cause and effect on screen**: light from a portal reflects on wardrobe; a thrown knife
  hits, the target groans and drops the case - impact and reaction in the same shot.
- **Precise action mechanics**: `executes a razor-sharp tactical drop, ducking under the
  gunfire in one fluid motion and sliding into a low crouch behind the concrete pillar on
  the left of frame` beats `he dodges`.
- **Grounded entrances**: every character has an on-screen reason to be there (a badge, a
  job, a line, an invitation, a delivery). Stagger arrivals; do not dump the whole cast into
  scene 1.
- **Story blocks**: keep the same 2-3 characters for 2-4 consecutive scenes resolving a
  beat before moving to another group. Adjacent scenes share connective tissue: end on a
  look off-screen, a handoff, a walk toward a spot, a question - and start the next scene
  from that beat. Keep key props alive across scenes.
- **Location variety**: 2-3 scenes in one spot, then move (lobby -> corridor -> office ->
  parking lot). Every shot states style, lighting temperature and time of day, and these
  stay identical inside a sequence unless the story jumps time.
- **Exits and entrances**: anchor the camera position and the movement vector
  (`looking back at the glass entrance from outside ... they walk out through the doors,
  moving away from the building behind them`).
- Camera vocabulary: `[Motion] + [with small/large amplitude] + [at slow/fast speed]` -
  push in, pull out, pan, truck, tilt, pedestal, arc, tracking, static, POV, roll, shake.
- Play every character *in character*: their vocabulary, rhythm, humour and moral compass
  from their own show, colliding with the others'. That collision is the comedy or drama.
- Skip fiddly held instruments (a guitar in hands smears); a cup, a file, a phone, a
  badge, a weapon at rest are safe.
