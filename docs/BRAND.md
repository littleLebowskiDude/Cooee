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
a cooee travels. They **step through the ramp** as they move out, blue at the
core to pink at the far arc, and **taper at the tips** so each arc dissolves
rather than stopping dead — sound attenuating with distance.

Regenerate every size (no dependencies):

```bash
node tools/make-icon.cjs src-tauri/icons
```

## Palette

Windows 11 dark neutrals, so the pill and the settings window sit beside
Teams and Edge as if they shipped with them. The Copilot ramp — blue through
violet to pink — is spent on **one moment only**: the listening bars, one stop
per bar, so a voice lights it left to right. Everywhere else is flat.

Defined once as custom properties in `src/theme.css`; nothing downstream
hardcodes a hex value.

| Token | Hex | Role |
|---|---|---|
| `--bg` | `#1c1c22` | Near-black with a touch of blue. Windows 11 dark, not pure black. |
| `--text` | `#f5f5f7` | Type. Off-white, never pure `#fff`. |
| `--muted` | `#9a9aa6` | Labels and hints. |
| `--ramp-1` … `--ramp-5` | `#4cc2ff` `#7a7dff` `#a06cff` `#d16bd6` `#ff6bb5` | The Copilot ramp. Listening bars only. |
| `--listening` | `#4cc2ff` | Capture state: the ring, the core, the accent. |
| `--thinking` | `#7a7dff` | Transcribing. |
| `--injecting` | `#6ccb5f` | Delivered. Fluent green. |
| `--error` | `#ff7b7b` | Something failed. |
| `--accent` / `--on-accent` | `#4cc2ff` / `#0b1a24` | Controls: the Save button, focus rings, checkboxes. |

Two rules: **no pure black and no pure white**; and **state is colour, not
text** — the overlay is legible at a glance without being read. The ramp is
the third rule: it appears in the listening bars and the icon, and nowhere
else, so it keeps meaning.

Earlier the identity was Australian earth — ochre and terracotta on warm
near-black. It was retired in favour of a palette that reads as native on a
work machine; the name and the mark carry the Australian thread on their own.

## Motion

The overlay pill animates only what is happening:

- **Capturing** — rings propagate outward from the core on a 1.6 s cycle, offset
  so a second ring launches as the first fades. This is the mark, alive. The
  level bars follow the microphone: the capture callback publishes a level,
  the pill reads it twenty times a second on a decibel scale with a fast
  attack and slow release, and each bar carries its own weight and a little
  jitter so one number still reads as a spectrum. Until the first level lands
  a CSS equaliser on five drifting durations stands in. The label says only
  "Listening" — no device name, which was long enough to be cut off.
- **Transcribing** — the level bars pulse in sequence, in the ramp's violet.
- **Injecting** — a single green flash. Brief; the work is done.
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
