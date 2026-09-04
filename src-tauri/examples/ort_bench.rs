//! Rust reproduction of `bench/npu_whisper.py`: whisper encoder on the Hexagon
//! NPU through ONNX Runtime's QNN plugin EP, decoder on the CPU EP with a KV
//! cache. Prints per-stage timings and the transcript, so the numbers can be
//! set beside the Python ones. Runs the same code as the app's engine
//! (`cooee_lib::asr::onnx`), plus a CPU encoder for reference.
//!
//!     cargo run --release --features onnx --example ort_bench -- models/whisper-base.en-onnx models/sample.wav
//!     cargo run --release --features onnx --example ort_bench -- models/whisper-small.en-onnx models/sample.wav
//!
//! Runtime libraries come from the pip packages unless overridden (see
//! `asr::onnx::runtime`): `ORT_DYLIB_PATH` for onnxruntime.dll, `QNN_EP_PATH`
//! for onnxruntime_providers_qnn.dll. Set `QNN_CACHE=<dir>` to exercise the
//! compiled-context cache (a second run then loads in under a second).

use anyhow::{bail, Context, Result};
use cooee_lib::asr::onnx::{self, mel, runtime, tokenizer::Tokenizer, Placement};
use ort::logging::LogLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::time::Instant;

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
    if rate != mel::SAMPLE_RATE || bits != 16 {
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

fn run_encoder(session: &mut Session, input: &str, feats: &[f32], shape: [i64; 3]) -> Result<Vec<f32>> {
    let outs = session.run(ort::inputs![input => Tensor::from_array((shape, feats.to_vec()))?])?;
    let (_, v) = outs[0].try_extract_tensor::<f32>()?;
    Ok(v.to_vec())
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

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "cooee=info".into()),
        )
        .init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: ort_bench <model-dir> <wav>");
    }
    let model_dir = PathBuf::from(&args[1]);
    let wav = PathBuf::from(&args[2]);
    let enc_path = model_dir.join("onnx").join("encoder_model.onnx");
    let dec_path = model_dir.join("onnx").join("decoder_model_merged.onnx");

    let info = onnx::read_model_info(&model_dir)?;
    let tok = Tokenizer::load(&model_dir)?;
    println!("model: {}  n_mels={}  d_model={}", model_dir.display(), info.n_mels, info.d_model);

    let rt = runtime::init()?;
    println!(
        "onnxruntime: {}\nqnn ep:      {}\nnpu:         {}",
        rt.ort_dll.display(),
        rt.qnn_ep.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "none".into()),
        rt.npu
    );
    if !rt.npu {
        bail!("no QNN NPU device");
    }

    // Audio -> log-mel
    let pcm = read_wav_16k_mono(&wav)?;
    println!("audio: {:.1}s from {}", pcm.len() as f64 / mel::SAMPLE_RATE as f64, wav.display());
    let t = Instant::now();
    let spectrogram = mel::MelSpectrogram::new(info.n_mels);
    let feats = spectrogram.compute(&pcm);
    println!("log-mel: {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    let feats_shape = [1i64, info.n_mels as i64, mel::N_FRAMES as i64];

    // Encoder: CPU reference
    let t = Instant::now();
    let mut enc_cpu = Session::builder()?
        .with_log_level(LogLevel::Error)
        .and_then(|b| b.with_intra_threads(4))
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .commit_from_file(&enc_path)?;
    let load = t.elapsed().as_secs_f64() * 1000.0;
    let in_name = enc_cpu.inputs()[0].name().to_string();
    let (reference, ms) = timed(3, || run_encoder(&mut enc_cpu, &in_name, &feats, feats_shape))?;
    println!("\nencoder CPU (4 thr): load {load:.0} ms, run {ms:.0} ms");
    drop(enc_cpu);

    // Encoder: NPU, through the engine's builder (context cache optional)
    let cache = std::env::var_os("QNN_CACHE").map(PathBuf::from);
    if let Some(c) = &cache {
        println!("encoder NPU: context cache under {}", c.display());
    }
    let t = Instant::now();
    let (mut enc_npu, placement) = onnx::build_encoder(&enc_path, info.n_mels, 4, cache.as_deref())?;
    println!("encoder NPU: load/compile {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    if placement != Placement::Npu {
        bail!("encoder did not land on the NPU");
    }
    let (npu_out, ms) = timed(3, || run_encoder(&mut enc_npu, &in_name, &feats, feats_shape))?;
    println!("encoder NPU: run {ms:.0} ms");
    compare("encoder NPU", &npu_out, &reference);

    // Decoder on the CPU, fed by each encoder output
    let t = Instant::now();
    let mut dec = onnx::build_decoder(&dec_path, 4)?;
    println!("\ndecoder CPU: load {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    let enc_len = reference.len() / info.d_model;
    let enc_shape = [1i64, enc_len as i64, info.d_model as i64];
    for (label, enc) in [("CPU encoder", &reference), ("NPU encoder", &npu_out)] {
        let enc_value = Tensor::from_array((enc_shape, enc.clone()))?;
        let t = Instant::now();
        let ids = onnx::greedy_decode(&mut dec, &enc_value, &info.generation, 446)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let n = ids.len() - info.generation.sot_ids.len();
        println!(
            "decode from {label}: {n} tokens in {ms:.0} ms ({:.1} ms/token)",
            ms / n.max(1) as f64
        );
        let head: Vec<String> = ids.iter().take(12).map(|i| i.to_string()).collect();
        println!("  ids: [{}]{}", head.join(", "), if ids.len() > 12 { " ..." } else { "" });
        println!("  -> {}", tok.decode(&ids, info.generation.eot).trim());
    }
    drop(dec);
    drop(enc_npu);

    // The app's path: load the directory as an engine and transcribe.
    use cooee_lib::asr::AsrEngine;
    let t = Instant::now();
    let engine = onnx::OnnxEngine::load(&model_dir, Some(4))?;
    println!("
engine {}: load {:.0} ms", engine.name(), t.elapsed().as_secs_f64() * 1000.0);
    let out = engine.transcribe(&pcm, None)?;
    println!("engine transcribe: {} ms
  -> {}", out.inference_ms, out.text);
    Ok(())
}
