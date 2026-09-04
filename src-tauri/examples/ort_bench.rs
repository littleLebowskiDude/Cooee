//! Rust reproduction of `bench/npu_whisper.py`: whisper encoder on the Hexagon
//! NPU through ONNX Runtime's QNN plugin EP, decoder on the CPU EP with a KV
//! cache. Prints per-stage timings and the transcript, so the numbers can be
//! set beside the Python ones before any of this goes into the app.
//!
//!     cargo run --release --features onnx --example ort_bench -- models/whisper-base.en-onnx models/sample.wav
//!     cargo run --release --features onnx --example ort_bench -- models/whisper-small.en-onnx models/sample.wav [encoder.onnx] [decoder.onnx]
//!
//! Runtime libraries come from the pip packages unless overridden:
//!   ORT_DYLIB_PATH   onnxruntime.dll                 (ort's own variable)
//!   QNN_EP_PATH      onnxruntime_providers_qnn.dll   (its siblings: QnnHtp.dll etc.)
//!   QNN_CONTEXT      path to write/read the compiled QNN context, to test the cache
//!
//! The encoder is fed as exported (symbolic dims) with ORT's free-dimension
//! overrides pinning them at session creation. If that leaves nodes off the
//! NPU, pass the `_static_opt.onnx` file the Python script produced as the
//! third argument; the bench skips the overrides for a file with fixed dims.

use anyhow::{anyhow, bail, Context, Result};
use ort::environment::Environment;
use ort::logging::LogLevel;
use ort::memory::DeviceType;
use ort::session::builder::{BuilderResult, SessionBuilder};
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor, ValueType};
use realfft::RealFftPlanner;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

const SAMPLE_RATE: usize = 16_000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const CHUNK: usize = 30 * SAMPLE_RATE; // whisper always encodes a 30 s window
const N_FRAMES: usize = CHUNK / HOP; // 3000
/// The plugin EP takes its name from the registration; the options prefix and
/// the device filter both key on it, so it is the name the pip package uses.
const QNN_EP: &str = "QNNExecutionProvider";

// --- audio ---------------------------------------------------------------------

fn read_wav_16k_mono(path: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if b.len() < 12 || &b[..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        bail!("{} is not a WAV file", path.display());
    }
    let (mut pos, mut fmt, mut data) = (12usize, None, None);
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let size = u32::from_le_bytes(b[pos + 4..pos + 8].try_into()?) as usize;
        let body = &b[pos + 8..(pos + 8 + size).min(b.len())];
        match id {
            b"fmt " => {
                fmt = Some((
                    u16::from_le_bytes(body[2..4].try_into()?) as usize, // channels
                    u32::from_le_bytes(body[4..8].try_into()?) as usize, // rate
                    u16::from_le_bytes(body[14..16].try_into()?),        // bits
                ))
            }
            b"data" => data = Some(body),
            _ => {}
        }
        pos += 8 + size + (size & 1);
    }
    let (channels, rate, bits) = fmt.context("no fmt chunk")?;
    let data = data.context("no data chunk")?;
    if rate != SAMPLE_RATE || bits != 16 {
        bail!("need 16 kHz 16-bit, got {rate} Hz {bits}-bit");
    }
    let samples: Vec<f32> = data
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();
    Ok(if channels > 1 {
        samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    })
}

/// librosa.filters.mel(sr, n_fft, n_mels) with slaney scale and norm: what
/// whisper uses. Row-major (n_mels, n_fft/2 + 1).
fn mel_filterbank(n_mels: usize) -> Vec<f32> {
    let (min_log_hz, min_log_mel, logstep) = (1000.0f64, 15.0f64, (6.4f64).ln() / 27.0);
    let hz_to_mel = |f: f64| {
        if f >= min_log_hz {
            min_log_mel + (f.max(1e-9) / min_log_hz).ln() / logstep
        } else {
            f / (200.0 / 3.0)
        }
    };
    let mel_to_hz = |m: f64| {
        if m >= min_log_mel {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        } else {
            (200.0 / 3.0) * m
        }
    };
    let n_bins = 1 + N_FFT / 2;
    let fft_freqs: Vec<f64> = (0..n_bins)
        .map(|i| i as f64 * (SAMPLE_RATE as f64 / 2.0) / (n_bins - 1) as f64)
        .collect();
    let (m_lo, m_hi) = (hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE as f64 / 2.0));
    let mel_pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(m_lo + (m_hi - m_lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut fb = vec![0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let (lo, mid, hi) = (mel_pts[m], mel_pts[m + 1], mel_pts[m + 2]);
        let norm = 2.0 / (hi - lo);
        for (k, &f) in fft_freqs.iter().enumerate() {
            let lower = (f - lo) / (mid - lo);
            let upper = (hi - f) / (hi - mid);
            fb[m * n_bins + k] = (lower.min(upper).max(0.0) * norm) as f32;
        }
    }
    fb
}

/// whisper.audio.log_mel_spectrogram. Returns (n_mels, 3000) row-major.
fn log_mel(pcm: &[f32], fb: &[f32], n_mels: usize) -> Vec<f32> {
    let mut x = pcm[..pcm.len().min(CHUNK)].to_vec();
    x.resize(CHUNK, 0.0);
    // reflect pad N_FFT/2 each side, like torch.nn.functional.pad(mode="reflect")
    let half = N_FFT / 2;
    let mut padded = Vec::with_capacity(CHUNK + N_FFT);
    padded.extend((1..=half).rev().map(|i| x[i]));
    padded.extend_from_slice(&x);
    padded.extend((1..=half).map(|i| x[CHUNK - 1 - i]));

    let window: Vec<f32> = (0..N_FFT)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / N_FFT as f32).cos())
        .collect();
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut frame = fft.make_input_vec();
    let mut spec = fft.make_output_vec();
    let n_bins = spec.len(); // 201

    // The last of the 3001 frames is dropped, as whisper does.
    let mut mag = vec![0f32; n_bins * N_FRAMES]; // (n_bins, n_frames)
    for t in 0..N_FRAMES {
        for i in 0..N_FFT {
            frame[i] = padded[t * HOP + i] * window[i];
        }
        fft.process(&mut frame, &mut spec).expect("fft");
        for (k, c) in spec.iter().enumerate() {
            mag[k * N_FRAMES + t] = c.re * c.re + c.im * c.im;
        }
    }
    let mut mel = vec![0f32; n_mels * N_FRAMES];
    for m in 0..n_mels {
        let row = &fb[m * n_bins..(m + 1) * n_bins];
        let out = &mut mel[m * N_FRAMES..(m + 1) * N_FRAMES];
        for (k, &w) in row.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let src = &mag[k * N_FRAMES..(k + 1) * N_FRAMES];
            for t in 0..N_FRAMES {
                out[t] += w * src[t];
            }
        }
    }
    let mut max = f32::MIN;
    for v in mel.iter_mut() {
        *v = v.max(1e-10).log10();
        max = max.max(*v);
    }
    for v in mel.iter_mut() {
        *v = (v.max(max - 8.0) + 4.0) / 4.0;
    }
    mel
}

// --- tokenizer (decode only) ---------------------------------------------------

struct TokenDecoder {
    id_to_tok: HashMap<u32, String>,
    special: HashSet<u32>,
    u2b: HashMap<char, u8>,
}

impl TokenDecoder {
    fn load(dir: &Path) -> Result<Self> {
        let vocab: HashMap<String, u32> =
            serde_json::from_str(&std::fs::read_to_string(dir.join("vocab.json"))?)?;
        let added: HashMap<String, u32> =
            serde_json::from_str(&std::fs::read_to_string(dir.join("added_tokens.json"))?)?;
        // GPT-2's byte<->unicode table
        let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
            .chain(0xA1..=0xAC)
            .chain(0xAE..=0xFF)
            .collect();
        let mut cs = bs.clone();
        let mut n = 0;
        for b in 0..256u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let u2b = cs
            .iter()
            .zip(bs.iter())
            .map(|(&c, &b)| (char::from_u32(c).unwrap(), b as u8))
            .collect();
        Ok(Self {
            id_to_tok: vocab.into_iter().map(|(k, v)| (v, k)).collect(),
            special: added.into_values().collect(),
            u2b,
        })
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut out = Vec::new();
        for id in ids {
            if self.special.contains(id) {
                continue;
            }
            if let Some(tok) = self.id_to_tok.get(id) {
                out.extend(tok.chars().filter_map(|c| self.u2b.get(&c).copied()));
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }
}

// --- runtime -------------------------------------------------------------------

/// The pip packages are the dev-time source of the DLLs. Bundling is later.
fn site_packages() -> Option<PathBuf> {
    let local = std::env::var_os("LOCALAPPDATA")?;
    let programs = PathBuf::from(local).join("Programs").join("Python");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(programs)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join("Lib").join("site-packages"))
        .filter(|p| p.join("onnxruntime").is_dir())
        .collect();
    dirs.sort();
    dirs.pop()
}

fn runtime_paths() -> Result<(PathBuf, PathBuf)> {
    let sp = site_packages();
    let from = |var: &str, rel: &[&str]| -> Result<PathBuf> {
        if let Some(v) = std::env::var_os(var) {
            return Ok(PathBuf::from(v));
        }
        let mut p = sp
            .clone()
            .ok_or_else(|| anyhow!("{var} not set and no Python site-packages found"))?;
        p.extend(rel);
        Ok(p)
    };
    Ok((
        from("ORT_DYLIB_PATH", &["onnxruntime", "capi", "onnxruntime.dll"])?,
        from("QNN_EP_PATH", &["onnxruntime_qnn", "onnxruntime_providers_qnn.dll"])?,
    ))
}

fn timed<T>(n: usize, mut f: impl FnMut() -> Result<T>) -> Result<(T, f64)> {
    let mut best = f64::MAX;
    let mut out = None;
    for _ in 0..n {
        let t = Instant::now();
        out = Some(f()?);
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    Ok((out.unwrap(), best))
}

/// Builder errors carry the builder back for recovery, which makes them !Send;
/// keep the message only.
fn sb(r: BuilderResult) -> Result<SessionBuilder> {
    r.map_err(|e| anyhow!("{e}"))
}

fn tensor_dims(ty: &ValueType) -> Vec<i64> {
    match ty {
        ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
        _ => Vec::new(),
    }
}

fn compare(label: &str, out: &[f32], reference: &[f32]) {
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut ref_max = 0f32;
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&a, &b) in out.iter().zip(reference) {
        max_abs = max_abs.max((a - b).abs());
        sum_abs += (a - b).abs() as f64;
        ref_max = ref_max.max(b.abs());
        dot += (a as f64) * (b as f64);
        na += (a as f64) * (a as f64);
        nb += (b as f64) * (b as f64);
    }
    println!(
        "{label}: vs CPU max abs diff {max_abs:.3} (rel {:.2}%), mean abs diff {:.3e}, cosine {:.5}",
        100.0 * max_abs / (ref_max + 1e-9),
        sum_abs / out.len() as f64,
        dot / (na.sqrt() * nb.sqrt() + 1e-9)
    );
}

// --- decoder loop --------------------------------------------------------------

struct Generation {
    sot_ids: Vec<u32>,
    eot: u32,
}

/// Optimum merged decoder: no-cache branch first, cache branch after. The
/// cache branch hands the encoder K/V back as empty batch-0 tensors, so those
/// are kept from the first step.
fn greedy_decode(
    dec: &mut Session,
    enc_out: &Tensor<f32>,
    gen: &Generation,
    max_tokens: usize,
    use_kv_cache: bool,
) -> Result<Vec<u32>> {
    let input_names: Vec<String> = dec.inputs().iter().map(|o| o.name().to_string()).collect();
    let past_names: Vec<String> = input_names
        .iter()
        .filter(|n| n.starts_with("past_key_values."))
        .cloned()
        .collect();
    let dims = tensor_dims(
        dec.inputs()
            .iter()
            .find(|o| o.name() == past_names[0])
            .unwrap()
            .dtype(),
    );
    let (heads, head_dim) = (dims[1], dims[3]);
    let has_branch_flag = input_names.iter().any(|n| n == "use_cache_branch");

    let mut past: Vec<DynValue> = past_names
        .iter()
        .map(|_| Tensor::<f32>::from_array(([1, heads, 0, head_dim], Vec::new())).map(|t| t.into_dyn()))
        .collect::<ort::Result<_>>()?;

    let mut tokens = gen.sot_ids.clone();
    let mut use_cache = false;
    while tokens.len() < max_tokens {
        let ids: Vec<i64> = if use_kv_cache && use_cache {
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
        let (shape, logits) = outs.get("logits").context("no logits")?.try_extract_tensor::<f32>()?;
        let vocab = shape[2] as usize;
        let last = &logits[logits.len() - vocab..];
        let next = last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap();
        if next == gen.eot {
            let mut top: Vec<(usize, f32)> = last.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            let top: Vec<String> = top[..5].iter().map(|(i, l)| format!("({i}, {l:.1})")).collect();
            println!("  (stopped: top-5 at EOT step [{}])", top.join(", "));
        }
        if use_kv_cache {
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
        }
        tokens.push(next);
        if next == gen.eot {
            break;
        }
    }
    Ok(tokens)
}

// --- main ----------------------------------------------------------------------

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: ort_bench <model-dir> <wav> [encoder.onnx] [decoder.onnx]");
    }
    let model_dir = PathBuf::from(&args[1]);
    let wav = PathBuf::from(&args[2]);
    let enc_name = args.get(3).map(String::as_str).unwrap_or("encoder_model.onnx");
    let dec_name = args.get(4).map(String::as_str).unwrap_or("decoder_model_merged.onnx");
    let enc_path = model_dir.join("onnx").join(enc_name);
    let dec_path = model_dir.join("onnx").join(dec_name);

    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
    let gcfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("generation_config.json"))?)?;
    let n_mels = cfg["num_mel_bins"].as_u64().unwrap_or(80) as usize;
    let d_model = cfg["d_model"].as_u64().context("d_model")? as usize;
    let pick = |k: &str| gcfg[k].as_u64().or_else(|| cfg[k].as_u64()).map(|v| v as u32);
    let sot = pick("decoder_start_token_id").context("decoder_start_token_id")?;
    let eot = pick("eos_token_id").context("eos_token_id")?;
    let mut sot_ids = vec![sot];
    if let Some(forced) = gcfg["forced_decoder_ids"].as_array() {
        sot_ids.extend(forced.iter().filter_map(|p| p[1].as_u64()).map(|t| t as u32));
    }
    let gen = Generation { sot_ids, eot };
    let tok = TokenDecoder::load(&model_dir)?;
    println!(
        "model: {}  encoder={enc_name}  decoder={dec_name}  n_mels={n_mels}",
        model_dir.display()
    );

    // Runtime: ORT via load-dynamic, QNN as a plugin EP registered at run time.
    let (ort_dll, qnn_dll) = runtime_paths()?;
    println!("onnxruntime: {}\nqnn ep:      {}", ort_dll.display(), qnn_dll.display());
    let committed = ort::init_from(&ort_dll)
        .map_err(|e| anyhow!("load {}: {e}", ort_dll.display()))?
        .with_name("cooee")
        .commit();
    if !committed {
        bail!("ort environment was already committed");
    }
    let env = Environment::current()?;
    env.set_log_level(LogLevel::Error);
    let _qnn = env
        .register_ep_library(QNN_EP, &qnn_dll)
        .with_context(|| format!("register QNN EP from {}", qnn_dll.display()))?;
    for d in env.devices() {
        let hw = d.hardware_device();
        println!("  device: {} {:?} {} ({})", d.ep()?, hw.ty(), hw.vendor().unwrap_or("?"), hw.id());
    }
    let npu_devices = || {
        env.devices().filter(|d| {
            d.ep().ok() == Some(QNN_EP) && d.hardware_device().ty() == DeviceType::NPU
        })
    };
    if npu_devices().next().is_none() {
        bail!("no QNN NPU device");
    }

    // Audio -> log-mel
    let pcm = read_wav_16k_mono(&wav)?;
    println!("audio: {:.1}s from {}", pcm.len() as f64 / SAMPLE_RATE as f64, wav.display());
    let t = Instant::now();
    let fb = mel_filterbank(n_mels);
    let feats = log_mel(&pcm, &fb, n_mels);
    println!("log-mel: {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    let feats_shape = [1i64, n_mels as i64, N_FRAMES as i64];

    // Encoder: CPU reference
    let t = Instant::now();
    let mut enc_cpu = sb(sb(Session::builder()?.with_log_level(LogLevel::Error))?.with_intra_threads(4))?
        .commit_from_file(&enc_path)?;
    let load = t.elapsed().as_secs_f64() * 1000.0;
    let in_name = enc_cpu.inputs()[0].name().to_string();
    let enc_dims = tensor_dims(enc_cpu.inputs()[0].dtype());
    let is_static = enc_dims.iter().all(|&d| d > 0);
    let syms: Vec<String> = match enc_cpu.inputs()[0].dtype() {
        ValueType::Tensor { dimension_symbols, .. } => {
            dimension_symbols.iter().map(|s| s.to_string()).collect()
        }
        _ => Vec::new(),
    };
    let (reference, ms) = timed(3, || {
        let outs = enc_cpu.run(ort::inputs![in_name.as_str() => Tensor::from_array((feats_shape, feats.clone()))?])?;
        let (_, v) = outs[0].try_extract_tensor::<f32>()?;
        Ok(v.to_vec())
    })?;
    println!("\nencoder CPU (4 thr): load {load:.0} ms, run {ms:.0} ms");
    drop(enc_cpu);

    // Encoder: NPU
    let mut b = sb(Session::builder()?.with_log_level(LogLevel::Error))?;
    if is_static {
        println!("encoder has fixed dims; no overrides");
    } else {
        println!("encoder dims {syms:?} pinned to {feats_shape:?} by free-dimension override");
        for (sym, size) in syms.iter().zip(feats_shape) {
            if !sym.is_empty() {
                b = sb(b.with_dimension_override(sym, size))?;
            }
        }
    }
    // Context cache: ORT writes a small wrapper .onnx plus <stem>_qnn.bin beside
    // it. With `ep.context_enable` set and the file already present it refuses
    // to run, so the entries are only set for the generating run; the reload
    // is a plain session over the wrapper file.
    let context = std::env::var_os("QNN_CONTEXT").map(PathBuf::from);
    match &context {
        Some(ctx) if !ctx.exists() => {
            b = sb(b.with_config_entry("ep.context_enable", "1"))?;
            b = sb(b.with_config_entry("ep.context_file_path", ctx.to_string_lossy()))?;
        }
        _ => {}
    }
    let opts = [(format!("{QNN_EP}.backend_type"), "htp".to_string())];
    let mut b = sb(b.with_devices(npu_devices(), Some(&opts)))?;
    let t = Instant::now();
    let load_from = match &context {
        Some(ctx) if ctx.exists() => {
            println!("encoder NPU: loading cached context {}", ctx.display());
            ctx.clone()
        }
        _ => enc_path.clone(),
    };
    let mut enc_npu = b.commit_from_file(&load_from).context("build NPU encoder session")?;
    println!("encoder NPU: load/compile {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    let (npu_out, ms) = timed(3, || {
        let outs = enc_npu.run(ort::inputs![in_name.as_str() => Tensor::from_array((feats_shape, feats.clone()))?])?;
        let (_, v) = outs[0].try_extract_tensor::<f32>()?;
        Ok(v.to_vec())
    })?;
    println!("encoder NPU: run {ms:.0} ms");
    compare("encoder NPU", &npu_out, &reference);

    // Decoder on the CPU, fed by each encoder output
    let t = Instant::now();
    let mut dec = sb(sb(Session::builder()?.with_log_level(LogLevel::Error))?.with_intra_threads(4))?
        .commit_from_file(&dec_path)?;
    println!("\ndecoder CPU: load {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    let enc_len = reference.len() / d_model;
    let enc_shape = [1i64, enc_len as i64, d_model as i64];
    let runs = [
        ("CPU encoder", &reference, true),
        ("NPU encoder", &npu_out, true),
        ("NPU encoder, no KV cache", &npu_out, false),
    ];
    for (label, enc, kv) in runs {
        let enc_value = Tensor::from_array((enc_shape, enc.clone()))?;
        let t = Instant::now();
        let ids = greedy_decode(&mut dec, &enc_value, &gen, 224, kv)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let n = ids.len() - gen.sot_ids.len();
        println!(
            "decode from {label}: {n} tokens in {ms:.0} ms ({:.1} ms/token)",
            ms / n.max(1) as f64
        );
        let head: Vec<String> = ids.iter().take(12).map(|i| i.to_string()).collect();
        println!("  ids: [{}]{}", head.join(", "), if ids.len() > 12 { " ..." } else { "" });
        println!("  -> {}", tok.decode(&ids).trim());
    }
    Ok(())
}
