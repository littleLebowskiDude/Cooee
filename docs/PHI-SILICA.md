# Phi Silica vs the rule-based polish pass

Status: **rules measured, Phi Silica blocked on this machine.** The harness is
built and ready; it needs one policy change to run.

## What the rules cost and what they fix

Measured with `cargo run --release --example polish_bench`, 1000 iterations per
case on the shared corpus in [`bench/transcripts.json`](../bench/transcripts.json):

**3 microseconds** mean per transcript. That is effectively free — four orders of
magnitude below the ASR step, and six below anything involving a model.

| Case | Rules handle it? |
|---|---|
| Standalone fillers (`um`, `uh`) | ✅ removed |
| Orphaned comma after a filler | ✅ fixed (was a bug — see below) |
| Sentence capitalisation | ✅ |
| Personal dictionary | ✅ |
| Already-clean text | ✅ left untouched — the important control |
| Missing sentence breaks | ❌ cannot insert punctuation |
| Doubled words (`the the`) | ❌ |
| Filler *phrases* (`you know`, `sort of`) | ❌ single words only |
| Spoken self-correction (`Friday, no wait, Thursday`) | ❌ |
| List formatting (`first… second… third…`) | ❌ |

The first five are most of the everyday value. The last five are exactly the
class of problem a language model is for — they need meaning, not pattern
matching.

### A bug this comparison found

Building the baseline exposed a real defect. Speech puts fillers between commas,
so removing the word alone orphaned its punctuation:

```
"we should probably, uh, move"  ->  "we should probably, move"   ✗
```

Fixed, with a regression test that also asserts a comma doing real work survives
(`"first, um, we ship, then we test"` -> `"First we ship, then we test"`).

## Why Phi Silica could not be measured here

**The model is present and loads. Text generation is gated by a Limited Access
Feature that reports `Unavailable` on this Windows build.**

Final harness output (`tools/phi-winui`, WinUI 3, packaged, with identity):

```
LAF status: Unavailable
ready state: Ready
model created in 97 ms
FATAL: UnauthorizedAccessException
  Limited Access Feature is not available:
  com.microsoft.windows.ai.languagemodel. Status: 0
```

`GetReadyState()` returns **Ready** and `LanguageModel.CreateAsync()` succeeds in
**97 ms** — Phi Silica is genuinely installed and initialisable. Only
`GenerateResponseAsync()` fails.

`LimitedAccessFeatures.TryUnlockFeature` reports **`Unavailable` (0)**, not
`Available` (1, needs a token) or `AvailableWithoutToken` (2). That distinction
matters: **no unlock token would help.** The feature is not offered to
third-party apps on this build at all.

Windows 11 25H2, build 26200. There is an open Windows App SDK issue
([#5580](https://github.com/microsoft/WindowsAppSDK/issues/5580)) reporting
Windows AI API failures on 26200 Insider builds, though its symptom is
"Not declared by app" rather than this LAF gate.

### Three gates, in the order we hit them

| Gate | Symptom | Resolution |
|---|---|---|
| **Package identity** | `UnauthorizedAccessException` at `GetReadyState()` | MSIX package + `systemAIModels` capability |
| **Sideloading** | `0x80073CFF` on `Add-AppxPackage -Register` | Developer Mode |
| **App shape** | Console app created then destroyed in <1 s, before `Main` | **WinUI 3** — a console `Windows.FullTrustApplication` never starts inside the AppX container. Every Microsoft sample is WinUI for a reason. |
| **Limited Access Feature** | `Status: 0 (Unavailable)` at generation | **unresolved** — platform-side |

The first three are solved and reproducible. The fourth is not something an app
can fix.

### What would unblock it

- A Windows build where `com.microsoft.windows.ai.languagemodel` is provisioned
  for third-party apps. Re-run the harness after a Windows update — it is a
  one-command check now.
- Internal Microsoft channels, if a LAF token is issuable for this feature. Note
  the status is `Unavailable` rather than `Available`, which suggests the device
  is not entitled regardless of token.

Re-test at any time:

```powershell
Add-AppxPackage -Register .	ools\phi-winui\pkg\AppxManifest.xml
Start-Job { Invoke-CommandInDesktopPackage -PackageFamilyName 'Cooee.PhiWinUI_kep965c3h3gg2' `
  -AppId 'PhiWinUI' -Command '<repo>	ools\phi-winui\pkg\phi-winui.exe' } | Wait-Job
Get-Content "$env:LOCALAPPDATA\Packages\Cooee.PhiWinUI_kep965c3h3gg2\LocalCache\Local\phi-winui-log.txt"
```

## What to expect, and how to judge it

The honest answer is that **latency is not the interesting question** — the rules
are 3 µs, so Phi Silica will be thousands of times slower no matter what. The
question is whether the quality gain is worth adding to the ~2.7 s already spent
on ASR. A polish pass that costs another second is a materially different product
from one that costs 50 ms.

The prompt in the harness is deliberately conservative — *"do not add, remove, or
reword any content"* — because for dictation, an LLM that improves your wording
is **worse**, not better. Judge it on:

1. **Does it leave clean text alone?** (`already-clean` is the control; any edit
   there is a regression.)
2. **Does it preserve meaning?** Especially `self-correction` and
   `technical-terms` — mangling `500` or resolving to the wrong day is
   disqualifying.
3. **Only then**: does it fix the five things rules cannot?

## Recommended design regardless of outcome

Keep the rules as the default and treat Phi Silica as an enhancement:

- Phi Silica has a real lifecycle — `GetReadyState` can report the model absent
  and `EnsureReadyAsync` pulls it via Windows Update. Dictation must keep working
  during that window.
- It needs MSIX packaging, which Tauri does not currently produce (it builds MSI
  and NSIS). That is a distribution change, not just a code change.
- The seam already exists: `polish::polish()` is a single call in `pipeline.rs`
  between transcription and injection.

---

# Measured: Phi-4-mini vs rules

Phi Silica's API is gated, but a **Phi model** can be run directly — no MSIX, no
identity, no LAF. `onnxruntime-genai` ships a native `win_arm64` wheel, so this
needed no compilation at all.

**Model:** `microsoft/Phi-4-mini-instruct-onnx`, int4 CPU (4.6 GB on disk).
**Load:** 19.8 s. **Throughput:** ~2 tokens/sec on the Snapdragon X CPU.

| | Rules | Phi-4-mini |
|---|---|---|
| Mean per transcript | **0.003 ms** | **10,071 ms** |
| Ratio | — | ~3.4 million× slower |

## Quality, case by case

| Case | Rules | Phi-4-mini | Verdict |
|---|---|---|---|
| `already-clean` (control) | untouched | **untouched** | ✅ both — Phi did not over-edit |
| `no-punctuation` | capitalise only | `"Hey, just following up on my last email. Did you get a chance to look at the proposal?"` | ✅ **Phi wins clearly** |
| `self-correction` | unchanged | `"Let's ship it on Thursday because Friday is a public holiday."` | ✅ **Phi wins clearly** |
| `stutter-repeat` | `the the` remains | `the the` → `the`, `i'm` → `I'm` | ✅ Phi (run-on still not split) |
| `fillers-simple` | removes `um`/`uh` | removes them, but **also drops "so"** | ⚠️ |
| `technical-terms` | removes `um`, keeps `500` | keeps `500`, but **`probably` → `likely`** | ⚠️ reworded |
| `discourse-markers` | leaves `you know` | removes `you know`, but **drops "I think" and "about"** | ❌ meaning changed |
| `list-dictation` | no list | no list | ➖ neither |

## The problem is not speed

10 s is disqualifying on its own, but this is 3.8B int4 on CPU — **Phi Silica on
the NPU would be far faster**, so latency alone would not settle it.

The real finding is the **over-editing**. The prompt says *"do NOT add, remove,
or reword any content"* and Phi-4-mini violated it in three of eight cases:

- `"probably"` → `"likely"` — a gratuitous reword.
- `"I think we need to…"` → `"We need to…"` — **it deleted a hedge**. The speaker
  chose to soften that sentence; the model made it assertive.
- `"past about ten thousand users"` → `"past ten thousand users"` — dropped an
  approximation, turning an estimate into a figure.

For dictation this is the cardinal sin. You said a thing; the app's job is to
type it. A polish pass that quietly changes your tone or firms up your estimates
is worse than one that leaves a filler in — because you will not notice, and the
text goes out under your name.

That failure mode is a property of instruction-following in a small model, not of
this particular runtime, so it likely transfers to Phi Silica.

## Recommendation

**Ship on rules.** They are free, correct, and predictable — and predictability
is the whole point here.

If revisiting when Phi Silica's LAF opens up, the design worth trying is
**selective invocation** rather than always-on: run rules always (3 µs), and call
the model only when rules detect something they cannot fix — no terminal
punctuation, or a detected doubled word. That bounds both the latency cost and
the blast radius of over-editing to the cases where the model demonstrably wins
(`no-punctuation`, `self-correction`).

A stricter prompt is also worth testing — the current one is already explicit,
which is itself evidence about how much instruction-following to expect.

## Reproduce

```bash
python -m pip install onnxruntime-genai huggingface_hub
python bench/phi_bench.py models/phi/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4
cargo run --release --example polish_bench    # the 3 us baseline
```
