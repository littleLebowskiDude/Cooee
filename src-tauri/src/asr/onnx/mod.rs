//! Whisper on ONNX Runtime, encoder on the Hexagon NPU. Built only with
//! `--features onnx`. See docs/NPU.md for the measurements and the plan.
//!
//! The model is a Hugging Face Optimum export as downloaded (for example
//! `onnx-community/whisper-small.en`): `onnx/encoder_model.onnx`,
//! `onnx/decoder_model_merged.onnx`, `config.json`, `generation_config.json`,
//! `vocab.json`. The encoder runs on the QNN EP when an NPU is present and
//! on the CPU EP otherwise; the merged decoder always runs on the CPU with
//! its KV cache. Greedy decoding, English, no timestamps.
//!
//! The first NPU session on a machine compiles the encoder for the HTP (7 s
//! for base.en, 20 s for small.en); the compiled context is cached under the
//! user's cache directory and reloads in well under a second.

pub mod mel;
pub mod runtime;
pub mod static_dec;
pub mod tokenizer;

use super::{AsrEngine, Transcript};
use anyhow::{anyhow, bail, Context, Result};
use mel::MelSpectrogram;
use ort::environment::Environment;
use ort::logging::LogLevel;
use ort::session::builder::{BuilderResult, SessionBuilder};
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor, ValueType};
use parking_lot::Mutex;
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizer::Tokenizer;

/// Whisper's decoder context is 448 positions, shared by the prompt and
/// the output.
pub(crate) const N_CTX: usize = 448;
/// The most prompt tokens whisper keeps: half the context, less one.
const MAX_PROMPT_TOKENS: usize = N_CTX / 2 - 1;

/// Token ids that start every decode, and the one that ends it.
pub struct Generation {
    pub sot_ids: Vec<u32>,
    pub eot: u32,
    /// `<|startofprev|>`, which introduces prompt text. Absent from some
    /// exports' `added_tokens.json`; then prompting is off.
    pub start_of_prev: Option<u32>,
}

impl Generation {
    /// The full start sequence for a decode: the prompt (if any) after
    /// `<|startofprev|>`, then the usual start tokens. Whisper keeps only
    /// the last `MAX_PROMPT_TOKENS` of a long prompt.
    pub fn prefix(&self, prompt: &[u32]) -> Vec<u32> {
        let mut ids = Vec::with_capacity(prompt.len() + self.sot_ids.len() + 1);
        if let (Some(prev), false) = (self.start_of_prev, prompt.is_empty()) {
            ids.push(prev);
            let keep = prompt.len().saturating_sub(MAX_PROMPT_TOKENS);
            ids.extend_from_slice(&prompt[keep..]);
        }
        ids.extend_from_slice(&self.sot_ids);
        ids
    }
}

/// What `config.json` and `generation_config.json` say about the export.
pub struct ModelInfo {
    pub n_mels: usize,
    pub d_model: usize,
    pub decoder_layers: usize,
    pub decoder_heads: usize,
    pub vocab: usize,
    pub generation: Generation,
}

pub fn read_model_info(dir: &Path) -> Result<ModelInfo> {
    let read = |name: &str| -> Result<serde_json::Value> {
        let path = dir.join(name);
        let raw = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))
    };
    let cfg = read("config.json")?;
    let gen = read("generation_config.json")?;
    let pick = |k: &str| gen[k].as_u64().or_else(|| cfg[k].as_u64());
    let sot = pick("decoder_start_token_id").context("decoder_start_token_id")? as u32;
    let eot = pick("eos_token_id").context("eos_token_id")? as u32;
    let mut sot_ids = vec![sot];
    if let Some(forced) = gen["forced_decoder_ids"].as_array() {
        sot_ids.extend(forced.iter().filter_map(|p| p[1].as_u64()).map(|t| t as u32));
    }
    let start_of_prev = read("added_tokens.json")
        .ok()
        .and_then(|added| added["<|startofprev|>"].as_u64())
        .map(|t| t as u32);
    Ok(ModelInfo {
        n_mels: cfg["num_mel_bins"].as_u64().unwrap_or(80) as usize,
        d_model: cfg["d_model"].as_u64().context("d_model")? as usize,
        decoder_layers: cfg["decoder_layers"].as_u64().context("decoder_layers")? as usize,
        decoder_heads: cfg["decoder_attention_heads"].as_u64().context("decoder_attention_heads")? as usize,
        vocab: cfg["vocab_size"].as_u64().context("vocab_size")? as usize,
        generation: Generation { sot_ids, eot, start_of_prev },
    })
}

/// Builder errors carry the builder back for recovery, which makes them
/// `!Send`; keep the message only.
pub(crate) fn sb(r: BuilderResult) -> Result<SessionBuilder> {
    r.map_err(|e| anyhow!("{e}"))
}

pub(crate) fn quiet_builder() -> Result<SessionBuilder> {
    sb(Session::builder()?.with_log_level(LogLevel::Error))
}

/// A builder pinned to the CPU EP. With the QNN plugin registered, a session
/// left to ORT's default device choice ran the same graph 60% slower.
pub(crate) fn cpu_builder(threads: usize) -> Result<SessionBuilder> {
    let env = Environment::current()?;
    let cpu = env.devices().filter(|d| d.ep().ok() == Some("CPUExecutionProvider"));
    let b = sb(quiet_builder()?.with_intra_threads(threads))?;
    sb(b.with_devices(cpu, None))
}

/// Where the encoder ended up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Npu,
    Cpu,
}

/// The encoder session: on the NPU through a cached QNN context when there
/// is one, on the CPU EP otherwise. `cache_dir` is where compiled contexts
/// live; `None` compiles every time.
pub fn build_encoder(
    encoder: &Path,
    n_mels: usize,
    threads: usize,
    cache_dir: Option<&Path>,
) -> Result<(Session, Placement)> {
    let rt = runtime::init()?;
    if rt.npu {
        match build_npu_encoder(encoder, n_mels, cache_dir) {
            Ok(session) => return Ok((session, Placement::Npu)),
            Err(e) => tracing::warn!("NPU encoder failed ({e:#}); falling back to the CPU"),
        }
    }
    let session = cpu_builder(threads)?
        .commit_from_file(encoder)
        .with_context(|| format!("load encoder {}", encoder.display()))?;
    Ok((session, Placement::Cpu))
}

fn build_npu_encoder(encoder: &Path, n_mels: usize, cache_dir: Option<&Path>) -> Result<Session> {
    // The QNN compiler needs static shapes. Pinning the Optimum export's
    // symbolic dims here puts the whole graph on the NPU (docs/NPU.md).
    let dims = [("batch_size", 1), ("feature_size", n_mels as i64), ("encoder_sequence_length", mel::N_FRAMES as i64)];
    build_npu_session(encoder, cache_dir, &dims, &[])
}

/// A session on the QNN NPU for `model`, through its cached compiled context
/// when there is one. `dim_overrides` pin symbolic dims; `extra` are further
/// QNN EP options (without the EP-name prefix).
pub(crate) fn build_npu_session(
    model: &Path,
    cache_dir: Option<&Path>,
    dim_overrides: &[(&str, i64)],
    extra: &[(&str, &str)],
) -> Result<Session> {
    let env = Environment::current()?;
    let mut opts = vec![(format!("{}.backend_type", runtime::QNN_EP), "htp".to_string())];
    opts.extend(extra.iter().map(|(k, v)| (format!("{}.{k}", runtime::QNN_EP), v.to_string())));
    let cache = cache_dir.map(|dir| context_path(dir, model));

    // A session that generates the context skips some of the EP's setup (the
    // shared-memory allocator, for one), so when there is a cache directory
    // the compile is its own pass with no extra options, and the session that
    // is used comes from the cache like every later launch.
    if let (Some(ctx), false) = (cache.as_ref(), extra.is_empty()) {
        if !ctx.exists() {
            let compile = build_npu_session(model, cache_dir, dim_overrides, &[])?;
            drop(compile);
            if !ctx.exists() {
                bail!("compiling {} left no context at {}", model.display(), ctx.display());
            }
        }
    }

    // A cached context loads as an ordinary model. If it is stale or damaged
    // the session fails; delete it and compile again below.
    if let Some(ctx) = cache.as_ref().filter(|p| p.exists()) {
        let t = Instant::now();
        let mut b = sb(quiet_builder()?.with_devices(runtime::npu_devices(&env), Some(&opts)))?;
        let session = b.commit_from_file(ctx);
        match session {
            Ok(s) => {
                tracing::info!(ms = t.elapsed().as_millis() as u64, ctx = %ctx.display(), "NPU session from cached context");
                return Ok(s);
            }
            Err(e) => {
                tracing::warn!("cached QNN context {} unusable ({e}); recompiling", ctx.display());
                remove_context(ctx);
            }
        }
    }

    let mut b = quiet_builder()?;
    for (sym, size) in dim_overrides {
        b = sb(b.with_dimension_override(sym, *size))?;
    }
    if let Some(ctx) = &cache {
        if let Some(parent) = ctx.parent() {
            std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
        b = sb(b.with_config_entry("ep.context_enable", "1"))?;
        b = sb(b.with_config_entry("ep.context_file_path", ctx.to_string_lossy()))?;
    }
    let mut b = sb(b.with_devices(runtime::npu_devices(&env), Some(&opts)))?;
    let t = Instant::now();
    let session = b
        .commit_from_file(model)
        .with_context(|| format!("compile {} for the NPU", model.display()))?;
    tracing::info!(ms = t.elapsed().as_millis() as u64, model = %model.display(), "compiled for the NPU");
    Ok(session)
}

/// One context per graph file, keyed by the model directory, the file stem
/// and its size so a swapped model never picks up the old compile.
fn context_path(cache_dir: &Path, graph: &Path) -> PathBuf {
    let model = graph
        .parent()
        .and_then(Path::parent)
        .and_then(Path::file_name)
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    let stem = graph.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let size = std::fs::metadata(graph).map(|m| m.len()).unwrap_or(0);
    let name = if stem == "encoder_model" { format!("{model}-{size}.onnx") } else { format!("{model}-{stem}-{size}.onnx") };
    cache_dir.join("qnn").join(name)
}

/// ORT writes the wrapper `.onnx` plus `<stem>_qnn.bin` beside it.
fn remove_context(ctx: &Path) {
    let _ = std::fs::remove_file(ctx);
    if let Some(stem) = ctx.file_stem() {
        let _ = std::fs::remove_file(ctx.with_file_name(format!("{}_qnn.bin", stem.to_string_lossy())));
    }
}

pub fn build_decoder(decoder: &Path, threads: usize) -> Result<Session> {
    runtime::init()?;
    cpu_builder(threads)?
        .commit_from_file(decoder)
        .with_context(|| format!("load decoder {}", decoder.display()))
}

fn tensor_dims(ty: &ValueType) -> Vec<i64> {
    match ty {
        ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
        _ => Vec::new(),
    }
}

/// Greedy decoding through the Optimum merged decoder: the no-cache branch
/// on the first step, the cache branch after. The cache branch hands the
/// encoder K/V back as empty batch-0 tensors, so those are kept from the
/// first step. `prefix` is the start sequence (see [`Generation::prefix`]);
/// the result includes it, so callers decode `ids[prefix.len()..]`.
pub fn greedy_decode(
    dec: &mut Session,
    enc_out: &Tensor<f32>,
    prefix: &[u32],
    eot: u32,
) -> Result<Vec<u32>> {
    let past_names: Vec<String> = dec
        .inputs()
        .iter()
        .map(|o| o.name().to_string())
        .filter(|n| n.starts_with("past_key_values."))
        .collect();
    let first = dec
        .inputs()
        .iter()
        .find(|o| Some(o.name()) == past_names.first().map(String::as_str))
        .context("decoder has no past_key_values inputs")?;
    let dims = tensor_dims(first.dtype());
    if dims.len() != 4 {
        bail!("unexpected KV cache shape {dims:?}");
    }
    let (heads, head_dim) = (dims[1], dims[3]);
    let has_branch_flag = dec.inputs().iter().any(|o| o.name() == "use_cache_branch");

    let mut past: Vec<DynValue> = past_names
        .iter()
        .map(|_| Tensor::<f32>::from_array(([1, heads, 0, head_dim], Vec::new())).map(|t| t.into_dyn()))
        .collect::<ort::Result<_>>()?;

    let mut tokens = prefix.to_vec();
    let mut use_cache = false;
    while tokens.len() < N_CTX {
        let ids: Vec<i64> = if use_cache {
            vec![*tokens.last().unwrap() as i64]
        } else {
            tokens.iter().map(|&t| t as i64).collect()
        };
        let mut feed: Vec<(Cow<str>, SessionInputValue)> = Vec::with_capacity(past.len() + 3);
        feed.push(("input_ids".into(), Tensor::from_array(([1, ids.len() as i64], ids))?.into()));
        feed.push(("encoder_hidden_states".into(), enc_out.into()));
        for (name, v) in past_names.iter().zip(past.iter()) {
            feed.push((name.as_str().into(), v.into()));
        }
        if has_branch_flag {
            feed.push(("use_cache_branch".into(), Tensor::from_array(([1], vec![use_cache]))?.into()));
        }
        let mut outs = dec.run(feed)?;
        let (shape, logits) = outs.get("logits").context("decoder has no logits output")?.try_extract_tensor::<f32>()?;
        let vocab = shape[2] as usize;
        let last = &logits[logits.len() - vocab..];
        let next = last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(eot);

        let mut fresh = Vec::with_capacity(past_names.len());
        for name in &past_names {
            let present = format!("present.{}", &name["past_key_values.".len()..]);
            fresh.push(outs.remove(&present).filter(|v| {
                v.try_extract_tensor::<f32>()
                    .map(|(s, _)| s.num_elements() > 0)
                    .unwrap_or(false)
            }));
        }
        drop(outs);
        for (slot, v) in past.iter_mut().zip(fresh) {
            if let Some(v) = v {
                *slot = v;
            }
        }
        use_cache = true;

        tokens.push(next);
        if next == eot {
            break;
        }
        if let Some(period) = repeating(&tokens[prefix.len()..]) {
            // Greedy whisper can lock into a loop on noise. Keep one copy.
            tracing::debug!(period, "decoder looping; stopping");
            tokens.truncate(tokens.len() - 2 * period);
            tokens.push(eot);
            break;
        }
    }
    Ok(tokens)
}

/// A short cycle repeated three times at the end of the sequence.
pub(crate) fn repeating(tokens: &[u32]) -> Option<usize> {
    (1..=8).find(|&p| {
        tokens.len() >= 3 * p && {
            let n = tokens.len();
            tokens[n - p..] == tokens[n - 2 * p..n - p] && tokens[n - p..] == tokens[n - 3 * p..n - 2 * p]
        }
    })
}

/// The decoder: the merged export on the CPU, or the static-shape graphs
/// (`static_dec`), which also run on the NPU.
enum Decoder {
    Merged(Mutex<Session>),
    Static(static_dec::StaticDecoder),
}

pub struct OnnxEngine {
    name: String,
    encoder: Mutex<Session>,
    decoder: Decoder,
    encoder_input: String,
    d_model: usize,
    mel: MelSpectrogram,
    tokenizer: Tokenizer,
    generation: Generation,
}

impl OnnxEngine {
    /// `dir` is the model directory. `threads: None` picks up to 4 for the
    /// CPU sessions, as the whisper.cpp engine does.
    pub fn load(dir: &Path, threads: Option<usize>) -> Result<Self> {
        let started = Instant::now();
        if !dir.is_dir() {
            bail!("{} is not a directory", dir.display());
        }
        let encoder_path = dir.join("onnx").join("encoder_model.onnx");
        let decoder_path = dir.join("onnx").join("decoder_model_merged.onnx");
        for p in [&encoder_path, &decoder_path] {
            if !p.exists() {
                bail!("{} is missing; expected the Hugging Face ONNX export layout", p.display());
            }
        }
        let info = read_model_info(dir)?;
        let tokenizer = Tokenizer::load(dir)?;

        const DEFAULT_THREADS: usize = 4;
        let threads = threads
            .or_else(|| {
                std::thread::available_parallelism()
                    .ok()
                    .map(|n| n.get().saturating_sub(1).min(DEFAULT_THREADS))
            })
            .unwrap_or(2)
            .max(1);

        let cache_dir = dirs::cache_dir().map(|d| d.join("cooee"));
        let (encoder, placement) = build_encoder(&encoder_path, info.n_mels, threads, cache_dir.as_deref())?;
        let encoder_input = encoder.inputs().first().context("encoder has no inputs")?.name().to_string();
        // The static decoder's cache length: the largest `decoder_step_<N>`
        // graph in the folder up to whisper's 448, or COOEE_DEC_MAX to pick.
        let max_len = std::env::var("COOEE_DEC_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(N_CTX)
            .min(N_CTX);
        let dims = static_dec::Dims {
            layers: info.decoder_layers,
            heads: info.decoder_heads,
            head_dim: info.d_model / info.decoder_heads,
            max_len,
            vocab: info.vocab,
        };
        let (decoder, dec_placement) = if static_dec::files(dir, max_len).is_some() {
            let d = static_dec::StaticDecoder::build(dir, dims, threads, cache_dir.as_deref())?;
            let p = d.placement;
            (Decoder::Static(d), Some(p))
        } else {
            (Decoder::Merged(Mutex::new(build_decoder(&decoder_path, threads)?)), None)
        };
        let name = match (placement, dec_placement) {
            (Placement::Npu, Some(Placement::Npu)) => "ONNX Runtime (NPU)",
            (Placement::Npu, _) => "ONNX Runtime (NPU encoder)",
            (Placement::Cpu, _) => "ONNX Runtime (CPU)",
        };
        tracing::info!(
            model = %dir.display(),
            ms = started.elapsed().as_millis() as u64,
            encoder = ?placement,
            decoder = ?dec_placement,
            threads,
            "loaded onnx whisper"
        );
        Ok(Self {
            name: name.into(),
            encoder: Mutex::new(encoder),
            decoder,
            encoder_input,
            d_model: info.d_model,
            mel: MelSpectrogram::new(info.n_mels),
            tokenizer,
            generation: info.generation,
        })
    }

    /// Log-mel, encoder, decoder for one 30 s window.
    fn transcribe_window(&self, pcm: &[f32], prefix: &[u32]) -> Result<String> {
        let t = Instant::now();
        let feats = self.mel.compute(pcm);
        let mel_ms = t.elapsed().as_millis();

        let t = Instant::now();
        let (hidden_shape, hidden) = {
            let mut enc = self.encoder.lock();
            let input = Tensor::from_array(([1, self.mel.n_mels() as i64, mel::N_FRAMES as i64], feats))?;
            let outs = enc.run(ort::inputs![self.encoder_input.as_str() => input])?;
            let (shape, data) = outs[0].try_extract_tensor::<f32>()?;
            (shape.iter().copied().collect::<Vec<i64>>(), data.to_vec())
        };
        let enc_ms = t.elapsed().as_millis();

        let t = Instant::now();
        let ids = match &self.decoder {
            Decoder::Static(d) => d.decode(&hidden, self.d_model, prefix, self.generation.eot)?,
            Decoder::Merged(m) => {
                let hidden = Tensor::from_array((hidden_shape, hidden))?;
                greedy_decode(&mut m.lock(), &hidden, prefix, self.generation.eot)?
            }
        };
        let dec_ms = t.elapsed().as_millis();
        let n = ids.len().saturating_sub(prefix.len());
        tracing::debug!(mel_ms, enc_ms, dec_ms, tokens = n, "onnx window");
        Ok(self.tokenizer.decode(&ids[prefix.len()..], self.generation.eot))
    }
}

impl AsrEngine for OnnxEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn transcribe(&self, pcm: &[f32], prompt: Option<&str>) -> Result<Transcript> {
        let started = Instant::now();
        // Whisper decodes as if the prompt preceded the audio; like
        // whisper.cpp, the text gets a leading space. Without merges.txt
        // there is no encoder, and the dictionary's replacements still run.
        let prompt_ids = match prompt.map(str::trim).filter(|p| !p.is_empty()) {
            Some(p) if self.generation.start_of_prev.is_some() => match self.tokenizer.encode(&format!(" {p}")) {
                Some(ids) => ids,
                None => {
                    tracing::debug!("no merges.txt beside the model; prompt ignored");
                    Vec::new()
                }
            },
            _ => Vec::new(),
        };
        let prefix = self.generation.prefix(&prompt_ids);
        let mut text = String::new();
        // 30 s windows, each cut where it is quietest so no word is halved.
        for window in crate::vad::windows(pcm, mel::CHUNK) {
            let piece = self.transcribe_window(window, &prefix)?;
            let piece = piece.trim();
            if piece.is_empty() {
                continue;
            }
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(piece);
        }
        Ok(Transcript {
            text,
            inference_ms: started.elapsed().as_millis() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repetition_guard_sees_short_cycles_only_when_they_recur() {
        assert_eq!(repeating(&[1, 2, 3, 4, 5, 6]), None);
        assert_eq!(repeating(&[9, 9, 9]), Some(1));
        assert_eq!(repeating(&[1, 2, 3, 1, 2, 3, 1, 2, 3]), Some(3));
        assert_eq!(repeating(&[1, 2, 3, 1, 2, 3]), None);
        assert_eq!(repeating(&[7, 1, 2, 1, 2, 1, 2]), Some(2));
    }

    #[test]
    fn prefix_puts_the_prompt_after_start_of_prev_and_keeps_its_tail() {
        let gen = Generation { sot_ids: vec![50257, 50362], eot: 50256, start_of_prev: Some(50360) };
        assert_eq!(gen.prefix(&[]), vec![50257, 50362]);
        assert_eq!(gen.prefix(&[1, 2]), vec![50360, 1, 2, 50257, 50362]);
        let long: Vec<u32> = (0..300).collect();
        let p = gen.prefix(&long);
        assert_eq!(p.len(), 1 + MAX_PROMPT_TOKENS + 2);
        assert_eq!(p[1], 300 - MAX_PROMPT_TOKENS as u32);
        let no_prev = Generation { sot_ids: vec![50257], eot: 50256, start_of_prev: None };
        assert_eq!(no_prev.prefix(&[1, 2]), vec![50257]);
    }

    #[test]
    fn context_path_is_keyed_by_model_directory_and_size() {
        let dir = std::env::temp_dir().join("cooee-onnx-test").join("whisper-x").join("onnx");
        std::fs::create_dir_all(&dir).unwrap();
        let enc = dir.join("encoder_model.onnx");
        std::fs::write(&enc, b"12345").unwrap();
        let p = context_path(Path::new("cache"), &enc);
        assert_eq!(p, Path::new("cache").join("qnn").join("whisper-x-5.onnx"));
    }
}
