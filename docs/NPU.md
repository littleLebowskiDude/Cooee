# Whisper on the Hexagon NPU

Status: **feasible, measured, not yet in the app. Target model: `small.en`.**
The encoder runs on the NPU through ONNX Runtime's QNN execution provider at
8-10x the CPU speed; `small.en` becomes a ~1.3 s pipeline instead of 5.2 s.
Reproduce with [`bench/npu_whisper.py`](../bench/npu_whisper.py).

## Why this matters

The CPU path is boxed in. ggml's thread pool collapses past one 4-core cluster
on the Snapdragon X Elite, so `base.en` at 4 threads (2.2 s for a 13.6 s clip)
is the ceiling, and bigger models are out (README, Performance). The Adreno GPU
is slower still (README, Tried and removed). The NPU is the only compute on
the chip that the thread pool problem does not touch.

## Measured

Snapdragon X Elite, machine idle, `base.en`, the 13.6 s TTS sample, ONNX
Runtime 1.29 + onnxruntime-qnn 2.5.0 (QNN SDK 2.49):

| Stage | CPU (ORT, 4 threads) | NPU (QNN HTP, fp16) |
|---|---|---|
| log-mel (numpy) | 126 ms | — |
| encoder | 460 ms | **59 ms** |
| decoder, 38 tokens, KV cache | 380 ms (10 ms/token) | not tried |
| **pipeline** | **~970 ms** | **~570 ms** |
| whisper.cpp, same clip | 2.2 s | |

All 353 encoder nodes were placed on the NPU. Its output differs from the CPU
by 0.5% relative (fp16), and the decoded transcript is identical to
whisper.cpp's, down to "Coo E".

Two things the table does not show:

- **First-session compile is 6.8 s.** The HTP compiler builds the graph on
  every process start unless the QNN context binary is cached to disk
  (`ep.context_enable` in the session options). That cache is the difference
  between a usable tray app and a 7 s startup stall.
- **ORT's CPU path already beats ggml here.** Even with nothing on the NPU the
  ORT pipeline is 2.3x faster than whisper.cpp for the same model. ORT's CPU
  kernels are not fighting the cluster boundary the way ggml's spin-waiting
  thread pool does.

## What it took

1. **`onnxruntime-qnn` is a plugin EP**, not a separate runtime. Version 2.x
   installs the QNN libraries beside the standard `onnxruntime` package and is
   registered at run time:
   ```python
   ort.register_execution_provider_library(onnxruntime_qnn.get_ep_name(), onnxruntime_qnn.get_library_path())
   dev = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider" and d.device.type is NPU]
   so.add_provider_for_devices(dev, {"backend_type": "htp"})
   ```
   Having a stray plain `onnxruntime` install from the Phi bench alongside it
   hid the provider entirely; uninstall both and reinstall to get the pair.
2. **The graph must be fully static.** The Optimum export has symbolic batch
   and sequence dims, and the QNN partitioner took only 66 of 437 nodes,
   leaving the rest on the CPU and making the NPU *slower* than CPU alone
   (1045 ms). Pinning the input to `(1, 80, 3000)` with
   `make_dim_param_fixed` and saving an ORT basic-optimised copy (so the
   `Shape`/`Gather` chains constant-fold away) put all 353 nodes on the NPU.
3. **fp32 in, fp16 on the HTP.** No quantisation step: the QNN EP runs fp32
   graphs at fp16 precision on the HTP by default. 0.5% relative error, no
   change to the transcript.
4. **Qualcomm's own whisper exports are no longer on Hugging Face** (their
   repos are deprecation stubs pointing at AI Hub, which needs an account). The
   community Optimum export `onnx-community/whisper-base.en` works fine.
5. **The merged decoder's cache branch returns the encoder K/V as empty
   batch-0 tensors** (shape `(0, 8, 1, 64)`), not as pass-throughs. A loop that
   copies every `present.*` into `past.*` silently blinds cross-attention after
   the first cached step, and the model emits end-of-text after two words.
   Keep the encoder K/V from the first (no-cache) step.

## Bigger models

Same clip, same method. `small.en` is the fp32 Optimum export; turbo is the
fp16 export (its fp32 encoder is a 2.5 GB external-data file).

| Model | Encoder CPU (ORT) | Encoder NPU | NPU compile | Decoder CPU | Pipeline | whisper.cpp |
|---|---|---|---|---|---|---|
| `base.en` | 460 ms | **59 ms** | 6.8 s | 380 ms (10 ms/tok) | ~0.57 s | 2.2 s |
| `small.en` | 1708 ms | **165 ms** | 21 s | 1006 ms (26 ms/tok) | ~1.3 s | 5.2 s |
| `large-v3-turbo` fp16 | no ARM64 CPU kernel for fp16 Gelu | see below | **> 1 h** | | | 24 s |

`small.en` on the NPU is a 10x encoder gain and the transcript matches
whisper.cpp's `small.en` word for word (it hears "KUI", base hears "Coo E"; both
are the dictionary's job). Its encoder output differs from the CPU by a large
max-abs on a handful of elements but the decoded tokens are identical; the
fp16 outliers sit in activations the decoder does not care about.

**Turbo's encoder did not finish compiling, and was killed after 70 minutes.**
1561 nodes, 32 layers, 1.3 GB of fp16 weights: the HTP graph-preparation stage
ran at ~2 cores and 22 GB of working set, and its CPU consumption per minute
fell away towards the end, which looks like thrashing rather than converging.
`small.en`'s 683 nodes took 21 s, so this is not linear scaling. Whether the
compiled graph would then be fast is unknown, but the arithmetic says it would
not be *quick*: 32 layers at width 1280 is ~7.5x `small.en`'s encoder work, so
~1.2 s on the NPU if it scaled linearly, plus a decoder no cheaper per token
than `small.en`'s. That is a ~2.5 s pipeline: slower than `small.en` on the NPU
and no better than `base.en` on whisper.cpp today, for a 2 GB model that has to
ship as a precompiled QNN context binary (`ep.context_enable`) per HTP
generation.

**Decision: `small.en` on the NPU is the target.** It is the model that only
becomes affordable with the NPU, and it beats every CPU option measured.

## What is still open

- **Decoder on the NPU.** Left on the CPU here. It is 10-26 ms/token, so most
  of the remaining time for `small.en`. Putting it on the HTP needs a static
  maximum sequence length with an attention mask, which is a different export,
  not a session option.
- **Turbo.** Parked, see above. If revisited: the int8 export (645 MB) may
  compile where fp16 did not, but the speed ceiling stays below `small.en`.
- **The Rust side.** The app uses whisper.cpp through `whisper-rs`. This path
  needs a third `AsrEngine` on ONNX Runtime — the `ort` crate with dynamic
  loading of Microsoft's `onnxruntime.dll`, plus the QNN plugin registered
  through the C API — and the mel front end in Rust. Ship the ORT and QNN
  DLLs beside the binary.
- **Context cache**, as above, before this goes near the tray app.

## Rust reproduction

[`src-tauri/examples/ort_bench.rs`](../src-tauri/examples/ort_bench.rs) is the
Python script ported to the `ort` crate, stage for stage. Same machine, same
clip, 2026-09-04:

```powershell
cargo build --release --features onnx --example ort_bench
.\src-tauri	argetelease\examples\ort_bench.exe models\whisper-small.en-onnx models\sample.wav
```

| Stage | `base.en` Python | `base.en` Rust | `small.en` Python | `small.en` Rust |
|---|---|---|---|---|
| log-mel | 126 ms | **11 ms** | 126 ms | 9 ms |
| encoder CPU (ORT, 4 thr) | 460 ms | 474 ms | 1708 ms | 1741 ms |
| encoder NPU compile | 6.8 s | 6.9 s | 21 s | 19.9 s |
| encoder NPU, from context cache | — | **640 ms** | | |
| encoder NPU run | 59 ms | 60 ms | 165 ms | 162 ms |
| decoder CPU, per token | 10 ms | 8.7 ms | 26 ms | 26.6 ms |

Transcripts are identical to the Python run ("Coo E" for `base.en`, "KUI" for
`small.en`). Three things the port settled:

- **No pre-fixed encoder file is needed.** `SessionBuilder::with_dimension_override`
  (ORT's `AddFreeDimensionOverrideByName`) on `batch_size`, `feature_size`
  and `encoder_sequence_length` puts the as-exported encoder wholly on the
  NPU: 60 ms, the same as the `_static_opt` file. The model directory ships
  as downloaded.
- **The context cache is two session config entries**, `ep.context_enable=1`
  and `ep.context_file_path=<file>.onnx`. ORT writes an 871-byte wrapper
  `.onnx` plus `<stem>_qnn.bin` (44 MB for `base.en`) beside it. To reload,
  open the wrapper as an ordinary model *without* those entries: with
  `ep.context_enable` set and the file present, session creation fails with
  "exists already".
- **The plugin EP is named by its registration.** `register_ep_library(name,
  path)` sets what `Device::ep()` reports and what prefixes the
  `with_devices` options (`<name>.backend_type`). Register it as
  `QNNExecutionProvider`, the name the pip package uses.

## Implementation plan

Research done 2026-09-04. Step 1 (the Rust bench) and the context cache part
of step 3 are done, above; the rest is not built yet.

### The crate

`ort` 2.0.0-rc.13 (2026-07-28) binds ONNX Runtime 1.28 and has the plugin-EP
API behind the `api-22` feature:

- `Environment::current()?.register_ep_library(name, path)` loads
  `onnxruntime_providers_qnn.dll`.
- `env.devices()` enumerates `Device`s; filter on `ep() == "QNNExecutionProvider"`
  and the NPU hardware type.
- `SessionBuilder::with_devices([dev], Some(&[("backend_type", "htp")]))`
  binds a session to it; `with_config_entry(k, v)` for session config.
- `with_intra_threads(4)`, `with_optimization_level`, `commit_from_file`.

Use it with `default-features = false` and `features = ["std", "api-22",
"load-dynamic", "half"]`: `load-dynamic` so the app loads whatever
`onnxruntime.dll` is beside it (the pip package's 1.29 works; the API is
backward compatible) instead of `download-binaries` pulling one at build
time; `half` for the fp16 tensors. `Device` is `!Send`, so enumerate and
build the session on the loader thread.

Verified in the bench: `ort::init_from(path)` loads the runtime (the
`ORT_DYLIB_PATH` variable is the fallback), and
`with_dimension_override(name, size)` pins the three symbolic dims at session
creation, so nothing needs to be pre-fixed.

### The engine

`src-tauri/src/asr/onnx_npu.rs`, behind a `onnx` Cargo feature, mirroring
[`bench/npu_whisper.py`](../bench/npu_whisper.py) step for step:

1. **Log-mel** in Rust: 400-point periodic Hann, hop 160, reflect-padded,
   `realfft` (already in the tree via `rubato`), slaney filterbank computed at
   load from `config.json`'s `num_mel_bins`, log10, clamp to max−8, (x+4)/4,
   padded to 3000 frames. Port the numpy version; it matched whisper.cpp.
2. **Encoder session** on the QNN device with `ep.context_enable=1` and
   `ep.context_file_path=<app data>/qnn/<model>.onnx` so the 21 s compile
   happens once per machine; load the context file when it exists. Check the
   exact key names in ORT 1.29's QNN EP docs before relying on them.
3. **Decoder session** on the CPU EP, 4 intra-op threads, merged model with
   `use_cache_branch`. Greedy loop; keep the encoder K/V from the first step
   (the cache branch returns batch-0 empties). Stop at EOT or 224 tokens.
4. **Tokenizer decode** from `vocab.json` + `added_tokens.json`: GPT-2
   byte-level BPE, decode only. `generation_config.json` gives
   `decoder_start_token_id`, `eos_token_id`, `forced_decoder_ids`.
5. **Prompting.** whisper.cpp takes the dictionary prompt as text; this
   engine would need BPE *encoding* to do the same. Skip in v1 — the
   dictionary's post-hoc replacement still runs — and note it in settings.

### Plumbing

- `Config::model_path` pointing at a **directory** selects this engine; a
  `.bin` file stays whisper.cpp. `build_engine` in `asr/mod.rs` branches on
  that. Settings needs a folder picker beside the file picker.
- Model directory layout is the Hugging Face export as downloaded:
  `onnx/encoder_model.onnx`, `onnx/decoder_model_merged.onnx`, and the four
  json files. `small.en` is 970 MB; `base.en` 290 MB as the small option.
- **DLLs beside the exe**: `onnxruntime.dll`; from the `onnxruntime_qnn`
  package `onnxruntime_providers_qnn.dll`, `QnnHtp.dll`, `QnnHtpPrepare.dll`,
  `QnnHtpV73Stub.dll`, `QnnSystem.dll`, `libQnnHtpV73Skel.so`,
  `libqnnhtpv73.cat` (V73 is the X Elite's HTP generation). For dev, point
  at the pip site-packages copies. Before bundling, read
  `Qualcomm_LICENSE.pdf` in that package for redistribution terms.

### Order

1. ~~`--example ort_bench`~~: done, matches (Rust reproduction, above).
2. Engine behind `AsrEngine` + config branch + `cargo test` for mel and the
   tokenizer against fixtures dumped from the Python script.
3. ~~Context cache~~: mechanism verified in the bench (640 ms reload); the
   engine needs to place the file under app data, keyed by model.
4. Settings folder picker and engine name in the header.
5. Bundling the DLLs; Defender will have opinions about a new unsigned
   binary loading Qualcomm DLLs, see README.

## Reproduce

```powershell
python -m pip install onnxruntime-qnn onnx numpy
# onnx-community/whisper-base.en: onnx/encoder_model.onnx, onnx/decoder_model_merged.onnx,
# vocab.json, added_tokens.json, config.json, generation_config.json -> models/whisper-base.en-onnx/
python bench/qnn_smoke.py                                                   # NPU alive?
python bench/npu_whisper.py models/whisper-base.en-onnx models/sample.wav   # the numbers above
```
