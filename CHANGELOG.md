# Changelog

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
