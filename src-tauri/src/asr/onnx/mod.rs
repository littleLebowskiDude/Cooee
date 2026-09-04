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

/// Whisper's decoder context is 448 positions; the prompt takes two.
const MAX_TOKENS: usize = 446;

/// Token ids that start every decode, and the one that ends it.
pub struct Generation {
    pub sot_ids: Vec<u32>,
    pub eot: u32,
}

/// What `config.json` and `generation_config.json` say about the export.
pub struct ModelInfo {
    pub n_mels: usize,
    pub d_model: usize,
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
    Ok(ModelInfo {
        n_mels: cfg["num_mel_bins"].as_u64().unwrap_or(80) as usize,
        d_model: cfg["d_model"].as_u64().context("d_model")? as usize,
        generation: Generation { sot_ids, eot },
    })
}

/// Builder errors carry the builder back for recovery, which makes them
/// `!Send`; keep the message only.
fn sb(r: BuilderResult) -> Result<SessionBuilder> {
    r.map_err(|e| anyhow!("{e}"))
}

fn quiet_builder() -> Result<SessionBuilder> {
    sb(Session::builder()?.with_log_level(LogLevel::Error))
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
    let session = sb(quiet_builder()?.with_intra_threads(threads))?
        .commit_from_file(encoder)
        .with_context(|| format!("load encoder {}", encoder.display()))?;
    Ok((session, Placement::Cpu))
}

fn build_npu_encoder(encoder: &Path, n_mels: usize, cache_dir: Option<&Path>) -> Result<Session> {
    let env = Environment::current()?;
    let opts = [(format!("{}.backend_type", runtime::QNN_EP), "htp".to_string())];
    let cache = cache_dir.map(|dir| context_path(dir, encoder));

    // A cached context loads as an ordinary model. If it is stale or damaged
    // the session fails; delete it and compile again below.
    if let Some(ctx) = cache.as_ref().filter(|p| p.exists()) {
        let t = Instant::now();
        let mut b = sb(quiet_builder()?.with_devices(runtime::npu_devices(&env), Some(&opts)))?;
        let session = b.commit_from_file(ctx);
        match session {
            Ok(s) => {
                tracing::info!(ms = t.elapsed().as_millis() as u64, ctx = %ctx.display(), "NPU encoder from cached context");
                return Ok(s);
            }
            Err(e) => {
                tracing::warn!("cached QNN context {} unusable ({e}); recompiling", ctx.display());
                remove_context(ctx);
            }
        }
    }

    // The QNN compiler needs static shapes. Pinning the Optimum export's
    // symbolic dims here puts the whole graph on the NPU (docs/NPU.md).
    let mut b = quiet_builder()?;
    for (sym, size) in [("batch_size", 1), ("feature_size", n_mels as i64), ("encoder_sequence_length", mel::N_FRAMES as i64)] {
        b = sb(b.with_dimension_override(sym, size))?;
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
        .commit_from_file(encoder)
        .with_context(|| format!("compile {} for the NPU", encoder.display()))?;
    tracing::info!(ms = t.elapsed().as_millis() as u64, "NPU encoder compiled");
    Ok(session)
}

/// One context per encoder file, keyed by its directory name and size so a
/// swapped model never picks up the old compile.
fn context_path(cache_dir: &Path, encoder: &Path) -> PathBuf {
    let model = encoder
        .parent()
        .and_then(Path::parent)
        .and_then(Path::file_name)
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    let size = std::fs::metadata(encoder).map(|m| m.len()).unwrap_or(0);
    cache_dir.join("qnn").join(format!("{model}-{size}.onnx"))
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
    sb(quiet_builder()?.with_intra_threads(threads))?
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
/// first step. Returns every token including the start sequence.
pub fn greedy_decode(
    dec: &mut Session,
    enc_out: &Tensor<f32>,
    gen: &Generation,
    max_tokens: usize,
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

    let mut tokens = gen.sot_ids.clone();
    let mut use_cache = false;
    while tokens.len() < max_tokens {
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
            .unwrap_or(gen.eot);

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
        if next == gen.eot {
            break;
        }
        if let Some(period) = repeating(&tokens) {
            // Greedy whisper can lock into a loop on noise. Keep one copy.
            tracing::debug!(period, "decoder looping; stopping");
            tokens.truncate(tokens.len() - 2 * period);
            tokens.push(gen.eot);
            break;
        }
    }
    Ok(tokens)
}

/// A short cycle repeated three times at the end of the sequence.
fn repeating(tokens: &[u32]) -> Option<usize> {
    (1..=8).find(|&p| {
        tokens.len() >= 3 * p && {
            let n = tokens.len();
            tokens[n - p..] == tokens[n - 2 * p..n - p] && tokens[n - p..] == tokens[n - 3 * p..n - 2 * p]
        }
    })
}

pub struct OnnxEngine {
    name: String,
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    encoder_input: String,
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
        let decoder = build_decoder(&decoder_path, threads)?;
        let name = match placement {
            Placement::Npu => "ONNX Runtime (NPU)",
            Placement::Cpu => "ONNX Runtime (CPU)",
        };
        tracing::info!(
            model = %dir.display(),
            ms = started.elapsed().as_millis() as u64,
            ?placement,
            threads,
            "loaded onnx whisper"
        );
        Ok(Self {
            name: name.into(),
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            encoder_input,
            mel: MelSpectrogram::new(info.n_mels),
            tokenizer,
            generation: info.generation,
        })
    }

    /// Log-mel, encoder, decoder for one 30 s window.
    fn transcribe_window(&self, pcm: &[f32]) -> Result<String> {
        let t = Instant::now();
        let feats = self.mel.compute(pcm);
        let mel_ms = t.elapsed().as_millis();

        let t = Instant::now();
        let hidden = {
            let mut enc = self.encoder.lock();
            let input = Tensor::from_array(([1, self.mel.n_mels() as i64, mel::N_FRAMES as i64], feats))?;
            let outs = enc.run(ort::inputs![self.encoder_input.as_str() => input])?;
            let (shape, data) = outs[0].try_extract_tensor::<f32>()?;
            Tensor::from_array((shape.iter().copied().collect::<Vec<i64>>(), data.to_vec()))?
        };
        let enc_ms = t.elapsed().as_millis();

        let t = Instant::now();
        let ids = greedy_decode(&mut self.decoder.lock(), &hidden, &self.generation, MAX_TOKENS)?;
        let dec_ms = t.elapsed().as_millis();
        let n = ids.len().saturating_sub(self.generation.sot_ids.len());
        tracing::debug!(mel_ms, enc_ms, dec_ms, tokens = n, "onnx window");
        Ok(self.tokenizer.decode(&ids, self.generation.eot))
    }
}

impl AsrEngine for OnnxEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn transcribe(&self, pcm: &[f32], prompt: Option<&str>) -> Result<Transcript> {
        let started = Instant::now();
        if prompt.is_some() {
            // Needs BPE encoding, which this tokenizer does not have yet. The
            // dictionary's replacements still run on the output.
            tracing::debug!("initial prompt not supported by the onnx engine; ignored");
        }
        let mut text = String::new();
        for window in pcm.chunks(mel::CHUNK) {
            let piece = self.transcribe_window(window)?;
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
    fn context_path_is_keyed_by_model_directory_and_size() {
        let dir = std::env::temp_dir().join("cooee-onnx-test").join("whisper-x").join("onnx");
        std::fs::create_dir_all(&dir).unwrap();
        let enc = dir.join("encoder_model.onnx");
        std::fs::write(&enc, b"12345").unwrap();
        let p = context_path(Path::new("cache"), &enc);
        assert_eq!(p, Path::new("cache").join("qnn").join("whisper-x-5.onnx"));
    }
}
