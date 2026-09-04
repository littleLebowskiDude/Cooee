# Cooee — brand

## The name

A **cooee** is a call. Specifically, the long, carrying "coo-eee!" used in the
Australian bush to reach someone out of sight — from the Dharug language of the
Sydney region, and long since naturalised into Australian English.

The idiom is **"within cooee"**: close enough to hear.

It was chosen over the alternatives because it *is* the product rather than
describing it. Every competitor in this category mines the quiet-speech vein —
Whisper, Wispr Flow, Sotto — which is exhausted and crowded. A cooee is the
opposite: not a whisper, but a voice thrown deliberately at a target, arriving
intact. That is exactly what push-to-talk dictation does.

Practical reasons it holds up:

- Two syllables, six letters, unambiguous to type.
- Unmistakably Australian without reaching for a cliché (`Bonza`, `Crikey`).
- No collision in the dictation category — verified against the app stores and
  general search.
- Sound and distance are its native metaphors, so the identity draws itself.

**Ruled out:** `Sotto` (four existing apps in this exact category), `Yarn`
(collides with the JavaScript package manager — fatal for a dev-adjacent tool),
`Murmur` (quiet-speech vein, plus "murmur" carries grumble and heart-defect
senses).

## The mark

A source point with arcs travelling outward, opening to the right.

The arcs are **directional, not concentric**. Concentric rings read as a target;
a cooee travels. They also **cool from ochre to terracotta** as they move out,
and **taper at the tips** so each arc dissolves rather than stopping dead —
sound attenuating with distance.

Regenerate at any size (no dependencies):

```bash
node tools/make-icon.cjs src-tauri/icons/icon.png 512
```

## Palette

Australian earth, on warm near-black. Defined once as custom properties in
`src/theme.css`; nothing downstream hardcodes a hex value.

| Token | Hex | Role |
|---|---|---|
| `--bg` | `#100d0a` | Warm near-black. Never a neutral grey — the warmth is the point. |
| `--ochre` | `#e9a13b` | Primary. The call itself. Capture state. |
| `--terracotta` | `#cf6a3f` | Secondary. Distance and attenuation. Transcribing state. |
| `--eucalypt` | `#8aa87c` | Success. Text delivered. |
| `--redearth` | `#a8442a` | Reserved for depth and accent. |
| `--bone` | `#f4ece1` | Type. Warm off-white, never pure `#fff`. |

Two rules: **no pure black and no pure white** — every neutral carries warmth;
and **state is colour, not text** — the overlay is legible at a glance without
being read.

## Motion

The overlay pill animates only what is happening:

- **Capturing** — rings propagate outward from the core on a 1.6 s cycle, offset
  so a second ring launches as the first fades. This is the mark, alive.
- **Transcribing** — the level bars pulse in sequence, ochre shifting to
  terracotta.
- **Injecting** — a single eucalypt flash. Brief; the work is done.
- **Idle** — nothing moves, and the overlay hides itself after 1.2 s.

Animation runs *only* in the states that need it, so a resident idle overlay
costs no compositing work. `prefers-reduced-motion` disables all of it.

## Sound

Two notes, 70 ms each, quiet: A5 rising into capture, D5 falling out of it.
Enough to confirm the hotkey registered without looking. Optional, on by
default.

## Voice

Plain and unfussy. Australian by construction, not by costume — the name carries
the accent, so the copy does not need to. No "g'day", no exclamation marks.

- Tagline: **Within earshot.**
- Empty dictionary: "No corrections yet."
- Error: say what failed and what to do, in one line.
