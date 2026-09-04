//! The static-shape decoder: whisper's decoder as two fixed-shape graphs so
//! it compiles for the Hexagon NPU. Built from the export's weights by
//! `bench/static_decoder.py`, which writes into the model's `onnx/` folder:
//!
//! - `cross_kv.onnx`: encoder output -> per-layer K/V for cross-attention,
//!   once per utterance.
//! - `decoder_step_<MAX>.onnx` (fp32 inputs) and `..._f16.onnx` (fp16 cache
//!   and cross inputs): one token + its position + an additive mask + the
//!   self-attention cache (`MAX` slots) + the cross K/V -> logits and this
//!   token's K/V for every layer.
//!
//! The host owns the cache. Each step writes the returned K/V into slot
//! `position` and unmasks it. Every tensor has the same shape every step, so
//! nothing is reallocated: inputs and outputs are bound once per utterance
//! through an `IoBinding`, and on the NPU the big inputs live in HTP shared
//! memory so a step copies nothing but a token id.
//!
//! Why the mask is -1e4 and not -inf: the HTP runs fp16, and (-inf) + x
//! arithmetic there yields NaN in the softmax.

use super::{repeating, runtime, Placement, N_CTX};
use anyhow::{bail, Context, Result};
use half::f16;
use half::slice::HalfFloatSliceExt;
use ort::environment::Environment;
use ort::memory::Allocator;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use ort::{sys, AsPointer};
use parking_lot::Mutex;
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Encoder frames for a 30 s window: whisper always attends over all of them.
const ENC_LEN: usize = 1500;
const MASKED: f32 = -1e4;

#[derive(Debug, Clone, Copy)]
pub struct Dims {
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub max_len: usize,
    pub vocab: usize,
}

/// A cache or cross buffer: fp32 on the CPU path, fp16 on the NPU path,
/// where it lives in HTP shared memory when the EP provides it.
enum Buf {
    F32(Tensor<f32>),
    F16(Tensor<f16>),
}

impl Buf {
    /// Writes `src` (f32) at element offset `at`. Raw pointers because the NPU
    /// buffers live in HTP shared memory, which ort will not hand out as a
    /// slice; the memory is host-mapped and the session is not running.
    fn write(&mut self, at: usize, src: &[f32]) {
        match self {
            Buf::F32(t) => unsafe {
                let p = (t.data_ptr_mut() as *mut f32).add(at);
                std::ptr::copy_nonoverlapping(src.as_ptr(), p, src.len());
            },
            Buf::F16(t) => unsafe {
                let p = (t.data_ptr_mut() as *mut f16).add(at);
                std::slice::from_raw_parts_mut(p, src.len()).convert_from_f32_slice(src);
            },
        }
    }

    fn zero(&mut self, len: usize) {
        match self {
            Buf::F32(t) => unsafe { std::ptr::write_bytes(t.data_ptr_mut() as *mut f32, 0, len) },
            Buf::F16(t) => unsafe { std::ptr::write_bytes(t.data_ptr_mut() as *mut f16, 0, len) },
        }
    }

    fn input(&self) -> SessionInputValue<'_> {
        match self {
            Buf::F32(t) => t.into(),
            Buf::F16(t) => t.into(),
        }
    }
}

/// Memory from the NPU's own allocator: rpcmem the HTP reads directly, so an
/// input placed there is not copied per run. The QNN EP registers it with a
/// session created with `enable_htp_shared_memory_allocator`, under the
/// device's host-accessible memory descriptor.
fn shared_allocator(session: &Session, env: &Environment) -> Option<Allocator> {
    let api = ort::api();
    let dev = runtime::npu_devices(env).next()?;
    let raw = unsafe { (api.EpDevice_MemoryInfo)(dev.ptr(), sys::OrtDeviceMemoryType::OrtDeviceMemoryType_HOST_ACCESSIBLE) };
    if raw.is_null() {
        tracing::debug!("NPU device reports no host-accessible memory");
        return None;
    }
    let mem_info = dev.memory_info(true);
    tracing::debug!("NPU device has host-accessible memory; asking the session for its allocator");
    match Allocator::new(session, mem_info) {
        Ok(a) => {
            tracing::info!("NPU shared-memory allocator ready");
            Some(a)
        }
        Err(e) => {
            tracing::debug!("session has no allocator for the NPU shared memory: {e}");
            None
        }
    }
}

/// Field order matters: the buffers are freed through `allocator`, which
/// belongs to `session`, so they must drop first (Rust drops in order).
struct StepSession {
    /// `past.{l}.key`, `past.{l}.value` in layer order: `[1, H, MAX, D]`.
    past: Vec<Buf>,
    /// `cross.{l}.key`, `cross.{l}.value`: `[1, H, ENC_LEN, D]`.
    cross: Vec<Buf>,
    ids: Tensor<i32>,
    pos: Tensor<i32>,
    mask: Tensor<f32>,
    /// The NPU shared-memory allocator the buffers came from, if any. Held
    /// so the buffers can be freed through it.
    _allocator: Option<Allocator>,
    /// Set when the buffers are in HTP shared memory (zero-copy).
    shared: bool,
    session: Session,
}

pub struct StaticDecoder {
    cross_kv: Mutex<Session>,
    step: Mutex<StepSession>,
    dims: Dims,
    pub placement: Placement,
}

fn kv_names(prefix: &str, layers: usize) -> Vec<String> {
    (0..layers)
        .flat_map(|l| [format!("{prefix}.{l}.key"), format!("{prefix}.{l}.value")])
        .collect()
}

/// The static graphs' paths if the model folder has them.
pub fn files(model_dir: &Path, max_len: usize) -> Option<(PathBuf, PathBuf, PathBuf)> {
    let onnx = model_dir.join("onnx");
    let cross = onnx.join("cross_kv.onnx");
    let step32 = onnx.join(format!("decoder_step_{max_len}.onnx"));
    let step16 = onnx.join(format!("decoder_step_{max_len}_f16.onnx"));
    (cross.exists() && step32.exists() && step16.exists()).then_some((cross, step32, step16))
}

impl StaticDecoder {
    /// Builds the two sessions. On the NPU when the runtime has one (falling
    /// back to the CPU graph if the compile or the shared allocator fails).
    pub fn build(model_dir: &Path, dims: Dims, threads: usize, cache_dir: Option<&Path>) -> Result<Self> {
        let (cross_path, step32, step16) = files(model_dir, dims.max_len).context("static decoder graphs missing")?;
        let rt = runtime::init()?;
        // COOEE_NPU_DECODER=0 keeps the decoder on the CPU (diagnostics).
        let want_npu = std::env::var("COOEE_NPU_DECODER").map(|v| v != "0").unwrap_or(true);
        if rt.npu && want_npu {
            match Self::build_npu(&cross_path, &step16, dims, cache_dir) {
                Ok(d) => return Ok(d),
                Err(e) => tracing::warn!("NPU decoder failed ({e:#}); decoder falls back to the CPU"),
            }
        }
        Self::build_cpu(&cross_path, &step32, dims, threads)
    }

    fn build_cpu(cross_path: &Path, step_path: &Path, dims: Dims, threads: usize) -> Result<Self> {
        let cross_kv = super::cpu_builder(threads)?
            .commit_from_file(cross_path)
            .with_context(|| format!("load {}", cross_path.display()))?;
        let session = super::cpu_builder(threads)?
            .commit_from_file(step_path)
            .with_context(|| format!("load {}", step_path.display()))?;
        let step = StepSession::new(session, dims, false, None)?;
        Ok(Self { cross_kv: Mutex::new(cross_kv), step: Mutex::new(step), dims, placement: Placement::Cpu })
    }

    fn build_npu(cross_path: &Path, step_path: &Path, dims: Dims, cache_dir: Option<&Path>) -> Result<Self> {
        let t = Instant::now();
        let cross_kv = super::build_npu_session(cross_path, cache_dir, &[], &[])?;
        // The shared memory only exists for a session created with this option.
        let shared_opt = [("enable_htp_shared_memory_allocator", "1")];
        let session = super::build_npu_session(step_path, cache_dir, &[], &shared_opt)?;
        tracing::info!(ms = t.elapsed().as_millis() as u64, "NPU decoder sessions ready");
        // Without shared memory every step would copy the cache and cross K/V
        // (68 MB for small.en) into the NPU, which measured no faster than
        // the CPU graph; then the CPU is the better decoder.
        let env = Environment::current()?;
        let Some(allocator) = shared_allocator(&session, &env) else {
            // Dropping a session created with that option has crashed the
            // process later on; leak the two rather than risk it.
            std::mem::forget(session);
            std::mem::forget(cross_kv);
            bail!("the QNN EP exposes no host-accessible shared memory for the NPU");
        };
        let step = StepSession::new(session, dims, true, Some(allocator))?;
        Ok(Self { cross_kv: Mutex::new(cross_kv), step: Mutex::new(step), dims, placement: Placement::Npu })
    }

    /// Greedy decoding for one 30 s window. `hidden` is the encoder output
    /// `[1, ENC_LEN, d_model]`; the result includes `prefix`.
    pub fn decode(&self, hidden: &[f32], d_model: usize, prefix: &[u32], eot: u32) -> Result<Vec<u32>> {
        let Dims { layers, heads, head_dim, max_len, vocab } = self.dims;
        if hidden.len() != ENC_LEN * d_model {
            bail!("encoder output has {} values, expected {}", hidden.len(), ENC_LEN * d_model);
        }
        let mut guard = self.step.lock();
        let StepSession { session, shared, ids, pos: pos_t, mask, past, cross: cross_bufs, .. } = &mut *guard;

        // Cross-attention K/V for this utterance, into the step's buffers.
        let t = Instant::now();
        {
            let mut cross = self.cross_kv.lock();
            let input = Tensor::from_array(([1, ENC_LEN as i64, d_model as i64], hidden.to_vec()))?;
            let outs = cross.run(ort::inputs!["encoder_hidden_states" => input])?;
            for (i, name) in kv_names("cross", layers).iter().enumerate() {
                let (_, data) = outs.get(name.as_str()).context("cross_kv output missing")?.try_extract_tensor::<f32>()?;
                cross_bufs[i].write(0, data);
            }
        }
        let cross_ms = t.elapsed().as_millis();

        // Inputs are borrowed, not copied: the session reads the buffers in
        // place, on the NPU straight from shared memory. (An IoBinding did
        // the same job 3 ms/step slower on the CPU EP.)
        let past_names = kv_names("past", layers);
        let cross_names = kv_names("cross", layers);
        let new_names = kv_names("new", layers);
        let _ = vocab;

        // mask: slots [0, pos) valid, [pos, MAX) hidden, MAX (this token) valid
        {
            let (_, m) = mask.extract_tensor_mut();
            m.fill(MASKED);
            m[max_len] = 0.0;
        }

        let mut tokens = prefix.to_vec();
        let mut pos = 0usize;
        let slot = heads * max_len * head_dim; // elements per past buffer
        let mut new_kv = vec![0f32; heads * head_dim];
        let t = Instant::now();
        let mut run_secs = 0f64;
        while pos < max_len.min(N_CTX) {
            let tok = tokens[pos];
            ids.extract_tensor_mut().1[0] = tok as i32;
            pos_t.extract_tensor_mut().1[0] = pos as i32;
            let t_run = Instant::now();
            let mut feed: Vec<(Cow<str>, SessionInputValue)> = Vec::with_capacity(3 + past.len() + cross_bufs.len());
            feed.push(("input_ids".into(), (&*ids).into()));
            feed.push(("position".into(), (&*pos_t).into()));
            feed.push(("mask".into(), (&*mask).into()));
            for (name, buf) in past_names.iter().zip(past.iter()) {
                feed.push((name.as_str().into(), buf.input()));
            }
            for (name, buf) in cross_names.iter().zip(cross_bufs.iter()) {
                feed.push((name.as_str().into(), buf.input()));
            }
            let outs = session.run(feed)?;
            run_secs += t_run.elapsed().as_secs_f64();
            let want_next = pos + 1 >= tokens.len();
            let next = if want_next {
                let (_, logits) = outs.get("logits").context("no logits")?.try_extract_tensor::<f32>()?;
                Some(argmax(logits))
            } else {
                None
            };
            // This token's K/V into slot `pos`: [1,H,1,D] -> rows of [H][MAX][D]
            for (i, name) in new_names.iter().enumerate() {
                let (_, kv) = outs.get(name.as_str()).context("new K/V missing")?.try_extract_tensor::<f32>()?;
                new_kv.copy_from_slice(kv);
                for h in 0..heads {
                    let at = h * max_len * head_dim + pos * head_dim;
                    debug_assert!(at + head_dim <= slot);
                    past[i].write(at, &new_kv[h * head_dim..(h + 1) * head_dim]);
                }
            }
            drop(outs);
            mask.extract_tensor_mut().1[pos] = 0.0;
            pos += 1;
            if let Some(next) = next {
                tokens.push(next);
                if next == eot {
                    break;
                }
                if let Some(period) = repeating(&tokens[prefix.len()..]) {
                    tracing::debug!(period, "decoder looping; stopping");
                    tokens.truncate(tokens.len() - 2 * period);
                    tokens.push(eot);
                    break;
                }
            }
        }
        let n = tokens.len().saturating_sub(prefix.len());
        tracing::debug!(
            cross_ms,
            steps = pos,
            per_step_ms = t.elapsed().as_secs_f64() * 1000.0 / pos.max(1) as f64,
            run_ms = run_secs * 1000.0 / pos.max(1) as f64,
            tokens = n,
            shared = *shared,
            "static decode"
        );
        Ok(tokens)
    }
}

impl StepSession {
    fn new(session: Session, dims: Dims, f16_io: bool, allocator: Option<Allocator>) -> Result<Self> {
        let Dims { layers, heads, head_dim, max_len, .. } = dims;
        let past_shape = [1usize, heads, max_len, head_dim];
        let cross_shape = [1usize, heads, ENC_LEN, head_dim];
        // ORT's allocators hand out 64-byte aligned memory, which its kernels
        // prefer; a Vec does not.
        let cpu = Allocator::default();
        let make = |shape: [usize; 4]| -> Result<Buf> {
            let len: usize = shape.iter().product();
            let mut buf = match (f16_io, allocator.as_ref()) {
                (true, Some(a)) => Buf::F16(Tensor::<f16>::new(a, shape)?),
                (true, None) => Buf::F16(Tensor::<f16>::new(&cpu, shape)?),
                (false, _) => Buf::F32(Tensor::<f32>::new(&cpu, shape)?),
            };
            buf.zero(len);
            Ok(buf)
        };
        let past = (0..2 * layers).map(|_| make(past_shape)).collect::<Result<Vec<_>>>()?;
        let cross = (0..2 * layers).map(|_| make(cross_shape)).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            past,
            cross,
            ids: Tensor::from_array(([1usize], vec![0i32]))?,
            pos: Tensor::from_array(([1usize], vec![0i32]))?,
            mask: Tensor::from_array(([1usize, 1, 1, max_len + 1], vec![MASKED; max_len + 1]))?,
            shared: allocator.is_some(),
            _allocator: allocator,
            session,
        })
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
