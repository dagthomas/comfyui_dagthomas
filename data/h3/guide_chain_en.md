# MiniMax-H3 Continuous Chain Rules (C2V / motion-context continuation)

Use these when the scenes will be rendered as a **continuous chain**: the last frames (and
audio) of scene N are pinned to the head of scene N+1 (ComfyUI `MiniMaxH3AddGuide`, H3
Motion Context, or the Contex Loop pack). The model starts scene N+1 *on the previous
ending* and trims that head off the delivered file. Independent-clip habits break here;
these rules replace them for every scene after the first.

## 1. Scene 01 is a normal opener; every later scene inherits its start

- Scene 01 follows the ordinary T2V rules (state the style, light and setting; open on the
  speaker `already speaking, with no silent establishing beat`).
- Scene 02 onward: `[Shot 1]` **opens on the last frame of the previous scene**. Its first
  sentence names that: `Opens on the last frame of the previous scene: <Subject 2> Name (S2)
  still in close-up at screen-right, glass at her lips, warm amber lamplight.` Then describe
  **what changes** - who enters, who speaks, a small camera move. Do not re-describe a fresh
  establishing shot of the location; do not paste the same wide "establishing boilerplate"
  at the start and end of every scene (that is what produces the "sandwich" loop).
- Every scene still repeats the style words, the lighting lock and the wardrobe anchors in
  its own text - the chain inherits pixels, not the prompt.

## 2. One continuous take, never a hard cut

- A pinned head fights editorial cuts: the model wanders after `the shot cuts to` and then
  snaps back to the pinned ending for the last third. So inside a chained scene write **one
  unbroken camera move** from the inherited frame to a new closer.
- `[Shot 2]` / `[Shot 3]` may still mark timing beats, but say `without cutting` /
  `the camera keeps moving` / `in the same continuous take`. Never `the shot cuts to`.
- The last beat is a **moving closer**, not a hold: a sip in progress, a turn, a truck, a
  step out of frame, a hand lowering a prop. Do not end on a frozen tableau or a still stare.

## 3. Rotate the closer - it is the next scene's opening frame

- The final framing of scene N **is** `[Shot 1]` of scene N+1. Do not close on the same
  framing you opened with, and do not close on the same class of shot two scenes running.
- Rotate closer classes: face-into-lens close-up, profile, hands + prop filling the frame,
  looking down, eyes only, a walk-away over the shoulder, a two-shot only once in a while
  and never as the default. A held two-shot of two people looking at each other is illegal
  as a default closer.

## 4. Speaker hand-off: never open a subject change on dialogue

Before writing a chained scene, decide: *is the person who speaks first already the person on
screen at the end of the previous scene?*

- **Same speaker on screen** - continuation opener; the line can start at once.
- **Different person speaks first** - mandatory three-part hand-off:
  1. **Inherited silent beat, 00:00.000-00:02.000.** Name the previous scene's on-screen
     subject by `<Subject N>` tag, state what they are finishing, and mark them silent:
     `mouth closed, completely silent, speaking no dialogue. No dialogue, narration or voice
     of any kind is heard during this opening beat, and no lips move.`
  2. **A named transition out of them, without cutting** - a truck, whip pan, arc or rack
     focus that leaves them out of frame and settles on the new speaker.
  3. **The new speaker's first `<d>` at 00:02.000 or later, said out loud**: `At 00:02.000,
     now that the camera has settled on <Subject 1> Name (S1) alone in frame and only now,
     he says: <d>...</d>`.
- The outgoing person is on screen during that beat, so they **must be declared** in
  `subject_definitions:` (next free number after the scene's real cast) and referred to by
  tag, never bare name. After the transition add `<Subject N> is not in frame.` to every
  later beat.
- Budget the hand-off: a subject-change scene needs ~2 s of head before its first line, so
  such a scene carries less dialogue than a same-speaker scene of equal length. Group
  same-speaker scenes deliberately - two scenes on one person cost no hand-off.

## 5. Latent identity persistence

- If a person other than the speaker was visible at the end of the previous scene (a
  panel neighbour, someone at the table), they are still in the inherited pixels. Keep them
  in `subject_definitions:` (speaker stays `<Subject 1>`, the neighbour `<Subject 2>`) and
  describe their passive presence with a silence mandate. Dropping the tag turns them into
  a hallucinated stranger.

## 6. Positions, lighting, lines

- State `screen-left` / `screen-center` / `screen-right` for everyone in every multi-person
  beat and keep the seats unless someone clearly moves.
- One lighting string for the whole chain (`bright daylight on skin`, `warm amber
  lamplight`); never drift morning -> sunset -> night unless the story jumps time.
- New dialogue in every scene; never recycle a line already spoken in the chain.
- One new person or one new action per scene is plenty; the plot must advance (a new
  person, a new action, a new closer), not remix the previous two-shot.
