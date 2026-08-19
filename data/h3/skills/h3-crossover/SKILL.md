---
name: h3-crossover
description: The four-section MiniMax H3 crossover-scene contract the APNext H3 Crossover Writer node emits - subject_definitions / integrated_multimodal_description / overall_soundscape / non_diegetic_music per scene, the scene envelope the node parses, how the cast list maps to <Subject N>, speaker binding, silence mandates and shared-frame safeguards, and how a run of scenes hangs together as one story. Load with h3-prompt-director for crossover work.
---

# H3 crossover scenes

This is what the `APNext H3 Crossover Writer` node expects back: a short synopsis, then
one envelope per scene, each envelope holding a complete four-section T2VA prompt. The node
splits on the envelope markers, so they must be exact and nothing else may sit outside them.

```
=== SYNOPSIS ===
Title: <a title for this crossover>
Logline: <one or two sentences>
Cast: <who is in it and why each is there, one line per character>
=== END SYNOPSIS ===

=== SCENE 01 | duration: 15.0 ===
subject_definitions:
<Subject 1> Character (played by Actor) from Show
<Subject 2> Character (played by Actor) from Show

integrated_multimodal_description:
[Shot 1] ...

[Shot 2] At 00:05.500, the shot cuts to ...

overall_soundscape:
...

non_diegetic_music:
N/A
=== END SCENE 01 ===
```

- Scene numbers are two digits and count up from 01. `duration:` is the length in seconds
  the node will render that scene at; respect the duration the node asks for, or if it lets
  you vary, keep every scene between 5 and 20 s and prefer 12-15 s for dialogue.
- Inside an envelope: exactly the four section labels, plain text, one blank line between
  sections and between shots. No markdown fences, no bold, no commentary.
- No title cards, credit cards or logo cards unless the node explicitly asks for one. The
  scenes are the story itself.

## Cast

The node hands you a cast list, one `Character (played by Actor) from Show` per line, and
sometimes a free-text steer from the user. Use every listed character at least once
across the run unless the steer says otherwise; do not invent extra named characters (an
unnamed off-camera guard or waiter is fine). Per scene, order `subject_definitions:` so
the character with the most dialogue is `<Subject 1>` - H3 binds the audio to that slot.
Copy character, actor and show strings verbatim.

## What makes a crossover work

- Each character behaves exactly as they do in their own show - vocabulary, rhythm,
  attitude - and the story is the collision of those worlds. Sheldon Cooper explaining
  the rules to Jack Sparrow *is* the scene.
- Everyone has an on-screen reason to be there, revealed by a line, a badge, an action.
- Something physical happens in every scene. Props travel. Doors open. Someone leaves.
- Consecutive scenes hand off to each other (a look off-screen, a question, an object)
  and the location moves every 2-3 scenes.

## Reference library

| Read this | When |
|---|---|
| `17_CROSSOVER_GOLD_EXAMPLES.md` | Every run - verified rendered scenes to match for density, opener, isolation, silence mandates and closers. |
