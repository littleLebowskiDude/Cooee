//! The one seam that keeps the rest of the app independent of *how* we transcribe.
//!
//! whisper.cpp on Windows ARM64 is not a well-trodden path (CMake + MSVC ARM64 +
//! NEON codegen). Keeping it behind this trait means the pipeline is testable
//! today, and swapping in sherpa-onnx or an NPU backend later touches one file.

use anyhow::Result;
use parking_lot::RwLock;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub mod mock;
#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(feature = "whisper")]
pub mod whisper_cpp;

/// Audio handed to an engine is always 16 kHz, mono, f32 in [-1.0, 1.0].
pub const SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone, Default)]
pub struct Transcript {
    pub text: String,
    /// Wall-clock time the engine spent, for the latency HUD.
    pub inference_ms: u64,
}

pub trait AsrEngine: Send + Sync {
    /// Human-readable name, shown in settings and logs.
    fn name(&self) -> &str;

    /// Transcribe a complete utterance. Blocking: callers run this off the
    /// hotkey and audio threads.
    ///
    /// `prompt` is text the engine treats as having come just before the
    /// audio. Whisper uses it to bias decoding, so listing names and jargon
    /// there makes them come out spelled right rather than patched up after.
    /// Engines without the concept ignore it.
    fn transcribe(&self, pcm_16k_mono: &[f32], prompt: Option<&str>) -> Result<Transcript>;
}

/// The engine the pipeline uses, replaceable at runtime.
///
/// `None` means a model is loading and there is nothing to transcribe with
/// yet: the pipeline reports that rather than typing mock text into the
/// user's document. Loading a large model takes seconds, so this is also what
/// keeps the tray icon appearing at once on startup.
pub type EngineSlot = Arc<RwLock<Option<Arc<dyn AsrEngine>>>>;

/// What the settings window shows about the engine. Pushed as an `engine`
/// event on every change and available on demand via `engine_info`.
#[derive(Debug, Clone, Serialize)]
pub struct EngineInfo {
    pub state: EngineState,
    /// Name of the engine currently answering, if any.
    pub engine: Option<String>,
    /// The model being loaded, loaded, or that failed to load.
    pub model: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineState {
    Loading,
    Ready,
    Failed,
}

/// Builds an engine for the given model. No model means the mock engine, by
/// design; a model that cannot be loaded is an error the UI should show,
/// not something to paper over with mock output.
///
/// A directory is an ONNX export (encoder on the NPU); a file is a
/// whisper.cpp GGML model.
pub fn build_engine(
    model_path: Option<&Path>,
    threads: Option<usize>,
) -> Result<Box<dyn AsrEngine>, String> {
    let Some(path) = model_path else {
        tracing::warn!("no model file configured; using mock engine");
        return Ok(Box::new(mock::MockEngine));
    };

    if path.is_dir() {
        #[cfg(feature = "onnx")]
        {
            let engine = onnx::OnnxEngine::load(path, threads).map_err(|e| format!("{e:#}"))?;
            tracing::info!(model = %path.display(), engine = engine.name(), "loaded onnx engine");
            return Ok(Box::new(engine));
        }
        #[cfg(not(feature = "onnx"))]
        {
            return Err(format!(
                "this build has no ONNX engine (built without the `onnx` feature), so the model folder {} cannot be used",
                path.display()
            ));
        }
    }

    #[cfg(feature = "whisper")]
    {
        let engine = whisper_cpp::WhisperCpp::load_with_threads(path, threads)
            .map_err(|e| format!("{e:#}"))?;
        tracing::info!(model = %path.display(), "loaded whisper.cpp");
        Ok(Box::new(engine))
    }
    #[cfg(not(feature = "whisper"))]
    {
        let _ = threads;
        Err(format!(
            "this build has no whisper engine (built without the `whisper` feature), so {} cannot be used",
            path.display()
        ))
    }
}
