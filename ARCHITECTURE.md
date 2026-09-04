# Architecture

Name: **Cooee** — see [docs/BRAND.md](docs/BRAND.md).
Target: **Windows 11 on ARM64** (Snapdragon X Elite X1E80100, 12 cores, 64 GB RAM).
Transcription: **fully local**, no network required.

## Why this stack

| Concern | Choice | Rationale |
|---|---|---|
| Core runtime | **Rust** | The four hard parts — global keyboard hook, WASAPI capture, `SendInput` injection, and whisper.cpp FFI — are all native. Rust does them without a marshalling layer. |
| Shell / UI | **Tauri v2** | ~10 MB binary vs Electron's ~150 MB, and the process is idle-cheap for an always-resident tray app. The overlay HUD is HTML/CSS, so the branding work is cheap to iterate. |
| Win32 bindings | **`windows` crate** | Microsoft's own generated bindings. Full ARM64 support. |
| Audio | **`cpal`** | WASAPI backend, shared mode, low latency. |
| ASR | **whisper.cpp** via `whisper-rs` | Compiles to ARM64 NEON. Behind a Cargo feature so the app builds and runs without it. |
| Text out | **`SendInput` + clipboard fallback** | Two strategies; see Injection. |

### Why not the alternatives

- **Electron** — a resident tray app that must respond to a keypress in <50 ms shouldn't carry a Chromium main process. Node native addons for the keyboard hook add a marshalling hop on the latency-critical path.
- **C# / WinUI 3** — viable, and P/Invoke handles the Win32 surface fine. But whisper.cpp interop is worse than Rust's, and custom overlay chrome is harder than CSS.
- **Pure Win32 C++** — fastest, but the branding and settings UI would cost 10x the effort.

## The ARM64 constraint

`pwsh` and Node report `Arm64`; Git Bash and PowerShell 5.1 report `X64` because they run under **Prism** x64 emulation. Emulated whisper inference costs roughly 30-40%.

**Always build from a native ARM64 shell** (`pwsh`), targeting `aarch64-pc-windows-msvc`. Do not build from Git Bash.

## Pipeline

```
                  ┌──────────── hotkey DOWN ────────────┐
                  │                                     ▼
   Idle ──────────┴──────────►  Capturing  ──── hotkey UP ────►  Transcribing
    ▲                              │                                  │
    │                         (ring buffer,                     (whisper.cpp,
    │                          16 kHz mono)                      NEON, local)
    │                                                                 │
    └──── Injecting ◄──── Polishing ◄─────────────────────────────────┘
           (SendInput)     (cleanup, dictionary, formatting)
```

Every transition emits an event to the overlay HUD so the pill can animate.

## Modules

| File | Responsibility |
|---|---|
| `pipeline.rs` | The state machine above. Owns transitions; everything else is a leaf it calls. |
| `hotkey.rs` | `WH_KEYBOARD_LL` low-level hook on a dedicated thread with its own message pump. Matches a chord (default Ctrl+Win) against system key state, swallows keys with solo side effects (Win, Alt, Caps Lock) so they never fire, and takes a new chord at runtime. |
| `audio.rs` | `cpal` WASAPI capture → resample to 16 kHz mono f32 → lock-free ring buffer. |
| `vad.rs` | Energy gate + hangover to trim leading/trailing silence before ASR. |
| `asr/mod.rs` | `AsrEngine` trait. One seam, two implementations. Also `EngineSlot`, the swappable handle the pipeline reads from: models load on a background thread (startup and on change in settings) and swap in when ready, so the tray never waits on a load and a failed load keeps the previous engine. |
| `asr/whisper_cpp.rs` | Real engine. Feature-gated on `whisper`. |
| `asr/mock.rs` | Returns canned text. Lets the *whole* pipeline run before the C++ toolchain works. |
| `polish.rs` | Raw transcript → clean text. Filler removal, dictionary, capitalisation. The dictionary's targets also go to the engine as whisper's initial prompt, so most corrections never need to fire. |
| `inject.rs` | Text → focused window. |
| `overlay.rs` | Moves the HUD to the bottom-centre of the work area of the monitor holding the focused window, just before each show. |
| `tone.rs` | Two short faded sine bursts (rising start, falling stop) via `cpal`, on their own thread so a slow output device never delays capture. Off by `audio_feedback`. |
| `config.rs` | Persisted settings, hotkey binding, personal dictionary. |
| `tray.rs` | Tray icon, menu, lifecycle. |

### Why `AsrEngine` is a trait

whisper.cpp on Windows ARM64 is not a well-trodden path — it needs CMake, MSVC ARM64, and NEON codegen that few people exercise on this target. Putting it behind a trait plus a Cargo feature means:

1. The skeleton compiles and the pipeline is testable **today**, with no C++ toolchain.
2. If whisper.cpp fights the ARM64 build, swapping in `sherpa-onnx` or ONNX Runtime (both ship prebuilt Windows ARM64 binaries) touches one file.
3. The Hexagon NPU path (45 TOPS, via ONNX Runtime QNN EP) can be added later as a third impl without disturbing anything.

## Two hard problems

### 1. The keyboard hook must never block

`WH_KEYBOARD_LL` callbacks run on the thread that installed the hook, and Windows silently removes a hook whose callback exceeds `LowLevelHooksTimeout` (~300 ms default). So the hook thread does exactly one thing: push an event into a channel and return. All real work happens elsewhere. This is the single most common way dictation apps break.

### 2. Injection has no universally correct strategy

| Strategy | Works | Fails |
|---|---|---|
| `SendInput` with `KEYEVENTF_UNICODE` | Most apps, preserves clipboard | Slow for long text; some Electron apps drop chars |
| Clipboard + `Ctrl+V` | Fast, reliable, handles any length | Clobbers clipboard (we save/restore); blocked in some secure fields |

Default: clipboard-paste for text over ~120 chars, `SendInput` below. Save and restore the prior clipboard contents either way. This is configurable per-app later.

## Latency budget

Target: hotkey release → text on screen, for ~10 s of speech.

| Stage | Budget |
|---|---|
| Capture stop + VAD trim | 10 ms |
| whisper `large-v3-turbo` q5, 12 ARM cores | 600-1200 ms |
| Polish pass (rule-based) | 5 ms |
| Injection | 20 ms |
| **Total** | **~0.7-1.3 s** |

The polish pass is rule-based to start. An LLM pass is a much better product but needs either a network call (rejected: local-only) or a local small model — deferred until the core loop is solid.

## Model choice

`ggml-large-v3-turbo-q5_0.bin` (~570 MB) is the sweet spot: near-large accuracy, ~8x faster than large-v3, fits trivially in 64 GB. Fall back to `small.en-q5_1` (~180 MB) if ARM64 throughput disappoints.

## Deferred

- LLM polish pass, Command Mode ("make this more formal")
- Hexagon NPU via ONNX QNN
- Per-app injection profiles

Rejected, not deferred: streaming partial transcripts. Built and removed — the
HUD is a status pill, not a text view, and the text after release is the product.
