# Changelog

## 1.1.0 — 2026-09-14

Three additions, each closing a gap the first release left open.

### Dictating without holding anything

- **Latch mode.** Tap the chord to start, tap it again to insert. Hold
  remains the default and is unchanged. Push-to-talk assumes a hand that
  can hold two keys steady for the length of a sentence, which is the one
  assumption that rules the app out for RSI, tremor, limited dexterity,
  one-handed use and switch devices — the people dictation exists for.
- **Any capture now stops itself at the five-minute ceiling**, rather than
  holding the microphone open with nothing left to record — past the ceiling
  every sample is discarded anyway. This was added for latch, which has no
  key-up to end it, but it applies to hold as well: a key stuck down leaves a
  capture running just as effectively as a latch the user walked away from.
- The HUD says **"tap to stop"** while latched. With no key held down, a
  latched capture would otherwise look exactly like a jammed hotkey.

### Knowing when nothing landed

- **A third tone**, lower and longer than the other two, plays whenever an
  utterance inserts nothing: no speech, an engine error, or text that
  polished away to nothing. Until now failure was *silent* and sounded
  exactly like success to anyone not watching the pill — and the overlay is
  a window that never takes focus, so it is not a surface a screen reader
  can announce. Sound is the channel that works regardless of focus.

### Per-app insertion

- **Profiles keyed by executable** override the insertion strategy for one
  app. Some Electron apps drop synthesised keystrokes and want a paste;
  fields that refuse a paste want the keystrokes. No single global setting
  is right everywhere.
- The strategy is now resolved **at the moment of insertion**, not before
  transcription, because focus can move while the model is working.
- **Detect** in settings names the foreground app for you, so nobody has to
  know that Windows Terminal ships as `WindowsTerminal.exe`.
- Ships with **no profiles configured**, so an upgraded install behaves
  exactly as it did before.

### Your data, and proving the claim

- **A data panel** in settings naming the engine in use, the model, and
  every file the app writes, with its size and entry count.
- **Export history**, **delete all history**, and **open the folder**.
- **`tools/no-network.ps1`** walks the real dependency tree and fails if
  anything able to reach the internet is linked in. The precise claim it
  checks: no HTTP client and no TLS stack, without which nothing in the
  binary can originate a request. It also *names* the network-adjacent
  crates that are present — `http` is types-only, `tokio` is there for the
  Tauri event loop and a local named pipe — because a claim that hides its
  awkward parts is worth less than one that explains them.

### Fixed

- Dictionary entries were rendered into the settings list with `innerHTML`.
  They come from a file people edit by hand, so they are now built as text
  nodes.
- A capture now keeps the mode it began under. Changing the setting
  mid-utterance previously reinterpreted an in-flight hold capture under latch
  rules, which ignores the release — and since applying a settings change
  clears the held-chord flag, that release was the only one the capture would
  ever have received.

## 1.0.1 — 2026-09-05

- Capture up to five minutes (was one, and the cut was silent). The HUD
  says "cut at 5:00" if it is ever hit.
- The NPU engine cuts long audio into 30 s windows at the quietest moment
  before each limit instead of at the limit, so no word is halved.

## 1.0.0 — 2026-09-05

The first release. Push-to-talk dictation for Windows on ARM64, entirely on
the machine.

### Transcription

- **Whisper on the Hexagon NPU.** The ONNX Runtime engine runs whisper's
  encoder and decoder on the NPU through the QNN execution provider.
  `small.en`, the model that hears made-up words like "Cooee" as one word,
  transcribes a 13.6 s utterance in 0.8 s where whisper.cpp took 5.2 s on the
  CPU. The first load compiles the model for the NPU and caches the result;
  every later launch loads in a few seconds.
- **whisper.cpp stays as the CPU fallback**, for a machine without an NPU.
  A `.bin` model selects it; a model folder selects the NPU engine.
- **The dictionary prompts the model.** Target spellings from the personal
  dictionary go to whisper as context before it listens, on both engines,
  so names come out right the first time; the replacements still apply
  afterwards.
- **Runtime bundled.** The installers carry ONNX Runtime and the Qualcomm
  QNN libraries (under Qualcomm's AI Stack licence, as part of the app).

### Dictation

- **Hold a chord, speak, release.** Default Ctrl+Win; presets and custom
  chords in settings. The keyboard hook never blocks, so a slow model can
  never drop keys.
- **Clean text.** Filler words removed, spoken punctuation applied,
  sentences capitalised, dictionary corrections applied.
- **Spacing around the caret.** Before inserting, the app reads the
  character on either side of the caret in the focused field and adds a
  space where the dictation would otherwise fuse with a word.
- **Insertion that fits the app.** Typed for short text, pasted for long,
  with the clipboard restored afterwards; configurable.
- **A HUD that follows you**: a small pill on the monitor with the focused
  window, with level bars while listening, and start and stop tones.

### The window

- **History first.** The main window lists everything dictated, newest
  first, with Copy and Delete, so text that landed in the wrong place is
  never lost. Kept in `%APPDATA%\cooee\history.json`, capped at 500.
- **Settings behind the cog**: model, hotkey, insertion strategy, tones,
  dictionary. Closing the window hides it; it opens centred on the monitor
  under the cursor.
- **Tray icon** with Open and Quit; a second launch focuses the running app.

### Known limits

- The NPU decoder runs fp16 and can spell an uncertain word differently
  from the CPU; the dictionary prompt is the guard, and `COOEE_NPU_DECODER=0`
  keeps the decoder on the CPU.
- Fields that expose no UI Automation text pattern (terminals, some
  Electron apps) get the text without the caret spacing.
- The installers are unsigned; Defender has quarantined an installed build
  before (see the README).
- `large-v3-turbo` does not compile for the NPU in reasonable time and is
  not offered.
