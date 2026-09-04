# Cooee

**Within earshot.**

Push-to-talk dictation for Windows. Hold a key, speak, and clean text lands in
whatever app has focus. All transcription runs locally — no network, no account.

A *cooee* is the long carrying call used in the Australian bush to reach someone
out of sight. "Within cooee" means close enough to hear. See
[docs/BRAND.md](docs/BRAND.md).

Built as a clone of [Wispr Flow](https://wisprflow.ai)'s core loop.

## Status

**Working.** Hold Ctrl+Win, speak, release — polished text lands in whatever
app has focus. Transcription is local whisper.cpp on ARM64 NEON; no network, no
account, no telemetry.

| Check | Result |
|---|---|
| `cargo test` | 36/36 |
| `cargo clippy --all-targets` | 0 warnings (with and without `whisper`) |
| `cargo fmt --check` | clean |
| `tsc --noEmit` | clean |
| `vite build` | clean |

Measured on a Snapdragon X Elite: 2.2 s for 13.6 s of speech with `base.en` at
4 threads on an idle machine, so ~6x realtime. Bigger models and more threads
were both measured and both lose. See [Performance](#performance).

The polish pass is rule-based by decision, not by omission: an LLM pass was built
and benchmarked, then rejected. See
[docs/PHI-SILICA.md](docs/PHI-SILICA.md).

## Target

**Windows 11 on ARM64** (Snapdragon X Elite). Build from a **native ARM64 shell**
(`pwsh`) — Git Bash and PowerShell 5.1 run under x64 emulation on this machine,
and building there silently produces an emulated binary that transcribes ~30-40%
slower.

```powershell
[System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture  # must print Arm64
```

## Run

```powershell
npm install
npm run tauri dev          # mock engine, no C++ toolchain needed
```

Hold **Ctrl+Win**, speak, release. The mock engine inserts a line reporting
the captured duration and peak amplitude — enough to confirm the hook, the mic,
and injection all work before adding a model.

## Hotkey

The default is **Ctrl+Win**, the same chord Wispr Flow uses on Windows. It was
chosen over the original Right Ctrl because laptop keyboards routinely omit
Right Ctrl, and over Caps Lock because the muscle memory transfers.

Settings offers Ctrl+Win, Caps Lock, Right Ctrl, Right Shift, or a custom chord
captured by holding the keys together. Changes apply on save, no restart.

Two details make a chord usable as *hold*-to-talk:

- **Either side of a modifier counts** in a chord, so Ctrl+Win works from the
  right Ctrl on a full keyboard. A lone modifier keeps its side — a single
  "Right Ctrl" must not become "any Ctrl" and fire on every Ctrl+C.
- **The key that completes the chord is swallowed if it has a solo action.**
  Win opens Start on release, Alt focuses the menu bar, Caps Lock toggles.
  Press Ctrl then Win and Windows never sees the Win key; press Win then Ctrl
  and Windows sees a second key during the Win hold, which suppresses Start on
  its own. Either order leaves the Start menu closed.

Keys to avoid: Win+H (Windows voice typing), Ctrl+Space (IME switching and
editor completion), Alt+Space (window menu), Right Alt (AltGr on many layouts),
and F-keys (behind Fn on most laptops).

`hotkey` in `config.json` is an array of virtual-key codes; a 0.1.0 `hotkey_vk`
is migrated to a one-key array on load.

## Dictionary

Settings → Dictionary maps what whisper hears to what you meant (`kui` →
`Cooee`). It does two things:

1. **Corrects after the fact.** Whole-word, case-insensitive replacement in the
   polish pass, so a short entry cannot corrupt a longer word.
2. **Primes the model.** The target spellings are handed to whisper as its
   initial prompt (`Claude, Cooee.`), which it treats as text that came just
   before the audio. That is usually enough to tip a name it would otherwise
   spell phonetically, so the correction rarely has to fire. The prompt is
   capped at 200 characters: a long one costs decoder context, and on a
   near-silent clip whisper is prone to echoing its prompt back.

## Real transcription

Needs MSVC ARM64 + clang + CMake + ninja (see [Prerequisites](#prerequisites)).

```powershell
# 141 MB, fast, English-only. Good daily driver.
curl.exe -L -o models/ggml-base.en.bin `
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

npm run tauri dev -- --features whisper
```

Choose it under **Settings → Model** (or set `model_path` in
`%APPDATA%\cooee\config.json`). The model loads in the background and swaps in
when ready; the settings header shows which engine and file are live. Until a
model is chosen the mock engine is active and nothing is transcribed.

### Whisper on the NPU (ONNX Runtime)

The `onnx` feature adds a second engine: a Hugging Face ONNX export of
whisper with the encoder on the Hexagon NPU through ONNX Runtime's QNN
execution provider, the decoder on the CPU. `small.en` becomes affordable
this way (see [Performance](#performance)). Measurements, the plan and the
gotchas are in [docs/NPU.md](docs/NPU.md).

```powershell
# Runtime DLLs, for now from the pip packages (the engine finds them there)
python -m pip install onnxruntime-qnn

# Model: the export as downloaded. 290 MB for base.en, 970 MB for small.en.
# onnx/encoder_model.onnx, onnx/decoder_model_merged.onnx, config.json,
# generation_config.json, added_tokens.json, vocab.json, merges.txt
# -> models/whisper-small.en-onnx/ from huggingface.co/onnx-community/whisper-small.en

npm run tauri dev -- --features onnx,whisper
```

The features are independent: `onnx` alone builds an app that can only load
model folders, and a `.bin` model then fails with "this build has no whisper
engine". Build both so either kind of model works.

Choose the model **folder** under **Settings > Model > Folder...**. A folder
selects this engine, a `.bin` file selects whisper.cpp. The first load
compiles the encoder for the NPU (7 s for `base.en`, 20 s for `small.en`)
and caches the result under `%LOCALAPPDATA%\cooee\qnn`; later loads take
under two seconds. Without an NPU the encoder runs on ONNX Runtime's CPU
kernels, which are still 2x faster than whisper.cpp here.

The dictionary prompt works as it does with whisper.cpp (`merges.txt` is what
makes that possible; without it the engine runs unprompted and the
dictionary's replacements still apply).

To ship the DLLs, collect them from the pip package and bundle with the
second config, which adds them as resources; the installers put them in
`runtime\` under the install directory, where the engine looks first:

```powershell
.\tools\collect-onnx-runtime.ps1     # 15 files, 134 MB -> src-tauri/runtime/
npm run tauri build -- --features onnx,whisper --config src-tauri/tauri.onnx.conf.json
```

The Qualcomm files are under Qualcomm's AI Stack licence, which allows
redistribution in object form as part of an application and not on their
own; the licence and notices are bundled alongside. The default build has no
resources entry because a resource glob that matches nothing fails the build.

### Building whisper.cpp on Windows ARM64

Handled for you by [`.cargo/config.toml`](.cargo/config.toml) — no shell setup,
no env vars, works the same from PowerShell, bash, `tauri dev`, or an IDE. Only
`ninja` and `cmake` need to be on PATH (pip puts them there).

The three things it sets, and why each is load-bearing:

1. **`CC`/`CXX` = clang-cl.** ggml refuses MSVC on ARM — `MSVC is not supported
   for ARM, use clang`. Its check is
   `if (MSVC AND NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")`, which clang-cl
   satisfies.
2. **`CMAKE_GENERATOR = Ninja`.** The `cmake` crate defaults to the Visual Studio
   generator, which resolves its compiler through MSBuild and ignores
   `CMAKE_C_COMPILER` — so setting `CC` alone silently does nothing.
3. **`LIBCLANG_PATH`.** `whisper-rs-sys` runs bindgen, which needs libclang at
   runtime. `python -m pip install libclang` ships an ARM64 build.

If you change any of these, run `cargo clean -p whisper-rs-sys` first.
`CMakeCache.txt` pins both the generator and the compiler on first configure, so
a stale cache reports either `Does not match the generator used previously` or
re-runs the *old* compiler and makes a correct fix look like it failed.

## Performance

**Four threads, `base.en`.** Measured on a Snapdragon X Elite (12 cores) with
everything closed, 13.6 s of TTS speech, three models, `asr_threads` swept:

| Model | Size | 2 threads | 4 threads | 8 threads | 12 threads |
|---|---|---|---|---|---|
| `base.en` | 141 MB | 3.4 s | **2.2 s** | 69 s | 379 s |
| `small.en-q5_1` | 190 MB | 9.9 s | 5.2 s | | |
| `large-v3-turbo-q5_0` | 574 MB | 46 s | 24 s | | |

Two findings, both against the plan in [ARCHITECTURE.md](ARCHITECTURE.md):

1. **More than 4 threads collapses, even on an idle machine.** 8 threads is 30x
   slower than 4; 12 threads is 170x slower, and only kept ~6 cores busy while
   it ran. ggml's workers spin-wait at barriers, and the X Elite's 12 cores are
   three clusters of four — synchronising across clusters is where the time
   goes. The default is capped at 4 and `asr_threads` overrides it, but there
   is no idle-machine case where raising it helps. Under load (Teams, Edge, a
   security agent), 2 threads beat 4 by a wide margin, so drop to 2 if the
   machine is busy.
2. **Bigger models are not worth it here.** `large-v3-turbo` runs slower than
   realtime (a 10 s utterance would sit transcribing for ~18 s) and `small.en`
   is 2.4x slower than `base.en`. All three produced the same words on the
   sample; they differed only on how to spell "Cooee", which the dictionary
   prompt settles anyway.

**On the NPU** the picture changes. The ONNX engine, same clip, machine idle,
encoder on the Hexagon NPU and decoder on 4 CPU threads:

| Model | whisper.cpp, 4 threads | ONNX engine, NPU encoder |
|---|---|---|
| `base.en` | 2.2 s | **0.42 s** |
| `small.en` | 5.2 s | **1.2 s** |

`small.en` on the NPU is faster than `base.en` on the CPU, and it is the
model that hears "Cooee" as one word. Details in [docs/NPU.md](docs/NPU.md).

Re-run the sweep after a toolchain or model change:

```powershell
cargo run --release --features whisper --example bench -- `
  models/ggml-base.en.bin models/sample.wav "2,4"
```

Generate a test sample with Windows TTS (real speech, not noise — whisper's
decoder hallucinates on noise and the timings become meaningless):

```powershell
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, 'Sixteen', 'Mono')
$s.SetOutputToWaveFile("models\sample.wav", $fmt)
$s.Speak("The quick brown fox jumps over the lazy dog.")
$s.Dispose()
```

## Prerequisites

| Tool | Why | Install |
|---|---|---|
| Node 18+ | Frontend build | present |
| Rust (`aarch64-pc-windows-msvc`) | Everything native | `winget install Rustlang.Rustup` |
| MSVC ARM64 + Windows SDK | Links the Rust binary; compiles whisper.cpp | see below |
| CMake | whisper.cpp build | `python -m pip install cmake` |
| Ninja | so CMake honours `CC` (the VS generator does not) | `python -m pip install ninja` |
| clang (ARM64) | ggml refuses MSVC on ARM | VS component `VC.Llvm.Clang` |
| libclang | `whisper-rs-sys` bindgen | `python -m pip install libclang` |

Only the first two are needed for the default (mock) build.

### Notes for a locked-down / Intune-managed machine

This was set up on a managed device without local admin, and the obvious routes
failed. What actually worked:

- **winget is unreliable here.** Every MSI install returned `1602` (cancelled) —
  elevation requests get declined by policy. `Rustlang.Rustup` also left a
  `rustup-init` hung forever on an interactive prompt that a non-interactive
  shell can never answer; it held the `.rustup` lock until killed.
- **MSVC: use the bootstrapper directly**, not winget. It self-elevates
  successfully where the MSI path does not:
  ```powershell
  curl.exe -L -o vs_BuildTools.exe https://aka.ms/vs/17/release/vs_BuildTools.exe
  .s_BuildTools.exe --quiet --wait --norestart `
    --add Microsoft.VisualStudio.Workload.VCTools `
    --add Microsoft.VisualStudio.Component.VC.Tools.ARM64 `
    --add Microsoft.VisualStudio.Component.Windows11SDK.26100
  ```
- **CMake: use pip**, not the MSI. Installs to user space, needs no elevation.
- **Never interrupt `rustup` mid-unpack.** It leaves `lib/rustlib/components`
  claiming `rust-std` is installed while its directory is empty, after which
  `rustup target add` reports "up to date" and every build fails with
  `can't find crate for std`. The only fix is
  `rustup toolchain uninstall stable-aarch64-pc-windows-msvc` then reinstall.
- Expect slow downloads: TLS inspection on the corporate network made a 2 KB
  file from `static.rust-lang.org` take 14 s. Builds are fine; fetches crawl.

- **Defender for Endpoint flags the installed app.** The NSIS-installed
  `cooee.exe` was quarantined as `Trojan:Win32/Bearfoos.A!ml` — the `!ml`
  suffix is a machine-learning heuristic, not a signature. An unsigned binary
  that installs a `WH_KEYBOARD_LL` hook, calls `SendInput`, touches the
  clipboard, and arrives with Start Menu shortcuts and an uninstall key is
  what a keylogger looks like to that model. The detection lists the install
  context (shortcuts, uninstall key) as resources; the byte-identical
  `src-tauri\target\release\cooee.exe` was left alone and runs fine from
  there, which is the workaround on a machine with no admin for exclusions.
  Report the false positive at https://www.microsoft.com/wdsi/filesubmission
  with the file's SHA-256. Code signing is the real fix.
## Layout

```
src-tauri/src/
  pipeline.rs      state machine — owns all transitions
  hotkey.rs        WH_KEYBOARD_LL push-to-talk chord  ← read the header comment
  audio.rs         WASAPI capture → 16 kHz mono
  vad.rs           silence trimming
  asr/
    mod.rs         AsrEngine trait — the swap point
    mock.rs        canned output; no toolchain required
    whisper_cpp.rs real inference (feature = "whisper")
  polish.rs        fillers, spoken punctuation, dictionary, capitalisation
  inject.rs        SendInput / clipboard paste
  overlay.rs       places the HUD bottom-centre of the focused window's monitor
  tone.rs          start/stop cues on the default output device
  config.rs        settings + personal dictionary
src/
  overlay/         floating HUD pill
  settings/        settings window
```

Cooee has **no primary window**. The overlay is transient and settings starts
hidden, so the **tray icon is the only visible UI** — left-click opens settings,
right-click gives a menu. Without it the app is invisible once installed and
looks like it failed to launch.

Only one instance may run: two would each install a global keyboard hook and
every dictation would be typed twice. Launching again focuses the existing one.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the reasoning behind each choice.

## Tests

```powershell
cargo test --manifest-path src-tauri/Cargo.toml
```

Covers VAD trimming, the polish rules, and whole-word dictionary replacement —
the three places where a subtle bug quietly corrupts the user's text.

## Branding

Name, mark, palette, motion and voice: [docs/BRAND.md](docs/BRAND.md).

The identity is confined to `src/theme.css` (tokens) and `tools/make-icon.cjs`
(the mark). Nothing downstream hardcodes a colour.

Regenerate every icon asset — the multi-size `.ico` Tauri needs for the Windows
resource, plus the PNG set:

```bash
node tools/make-icon.cjs src-tauri/icons
```

## Not yet built

- LLM polish pass and Command Mode ("make this more formal")
- `large-v3-turbo` on the NPU: parked, its encoder did not finish compiling
  in an hour. The decoder on the NPU needs a static-length export. See
  [docs/NPU.md](docs/NPU.md).
- Per-app injection profiles

## Tried and removed

- **Adreno GPU via ggml's Vulkan backend.** Built (whisper-rs `vulkan`
  feature, Vulkan SDK 1.4.357 ARM64) and measured on the same 13.6 s sample:
  **35 s** against 2.2 s on the CPU, plus a 15 s shader compile at first load.
  Same words out, so the backend works; the Adreno X1-85 driver reports no
  matrix cores and ggml's shaders are not tuned for it. Not worth a config
  knob. To reproduce: `CXXFLAGS=/EHsc` (ggml-vulkan uses exceptions), link
  with `lld-link` (MSVC `link.exe` fails with LNK1322, a Cortex-A53 erratum
  workaround it cannot apply to ggml-vulkan's largest function), and add
  `vulkan = ["whisper", "whisper-rs/vulkan"]` to the features.
- **Streaming partial transcripts.** Built (a preview thread re-transcribing
  the audio so far, aborted via whisper's callback on release) and taken out
  the same day: the pill grew large and busy, and the text that matters is the
  one that lands after release, which is also how Wispr Flow behaves. The
  overlay stays a small status pill.
