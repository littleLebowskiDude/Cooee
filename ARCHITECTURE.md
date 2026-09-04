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
| `audio.rs` | `cpal` WASAPI capture → resample to 16 kHz mono f32 → lock-free ring buffer. The callback also publishes the chunk's RMS through an atomic, which a meter thread in `pipeline.rs` samples at 20 Hz for the HUD bars. |
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

| Stage | Planned | Measured |
|---|---|---|
| Capture stop + VAD trim | 10 ms | |
| whisper | `large-v3-turbo` q5, 12 cores: 600-1200 ms | `base.en`, 4 threads: ~1.6 s |
| Polish pass (rule-based) | 5 ms | 3 µs |
| Injection | 20 ms | |
| **Total** | **~0.7-1.3 s** | **~1.6 s** |

The plan assumed whisper would scale across all 12 cores. It does not — see
Model choice. The measured figure is the idle-machine rate of 6.2x realtime
applied to 10 s of speech; the README has the full sweep.

The polish pass is rule-based by decision, not to start: an LLM pass was built, benchmarked and rejected for over-editing. See [docs/PHI-SILICA.md](docs/PHI-SILICA.md).

## Model choice

**`base.en`**, 4 threads. The plan was `ggml-large-v3-turbo-q5_0.bin` (~570 MB,
near-large accuracy, ~8x faster than large-v3) with `small.en-q5_1` as the
fallback. Measured on the X Elite with the machine idle, turbo runs slower than
realtime (24 s for 13.6 s of speech at 4 threads) and `small.en` is 2.4x slower
than `base.en`, for no difference in the words produced on the test sample.

The reason the plan was wrong: ggml's thread pool spin-waits at barriers, and
the X Elite's 12 cores are three clusters of four. Past 4 threads the
cross-cluster synchronisation dominates — 8 threads is 30x slower than 4, 12
threads is 170x slower. So the whole budget has to fit on one cluster, and on
one cluster only `base.en` is fast enough. This is what makes the Hexagon NPU
path (below) interesting: it is the only route to a bigger model without going
through that thread pool. The Adreno GPU is not that route — ggml's Vulkan
backend runs on it but 16x slower than the CPU (README, "Tried and removed").
The NPU is: measured at 59 ms for the `base.en` encoder against 460 ms on the
CPU, and 165 ms against 1.7 s for `small.en`, transcripts unchanged
([docs/NPU.md](docs/NPU.md)). **The plan is now `small.en` on the NPU**
through a third `AsrEngine` on ONNX Runtime; turbo's encoder would not
compile for the NPU in over an hour and would be slower than `small.en` even
if it did.

## Deferred

- LLM polish pass, Command Mode ("make this more formal")
- Hexagon NPU via ONNX Runtime QNN — feasibility measured, see
  [docs/NPU.md](docs/NPU.md): encoder 59 ms on the NPU vs 460 ms on the CPU,
  and ORT's CPU decoder alone is faster than ggml. The third `AsrEngine`.
- Per-app injection profiles

Rejected, not deferred: streaming partial transcripts. Built and removed — the
HUD is a status pill, not a text view, and the text after release is the product.
