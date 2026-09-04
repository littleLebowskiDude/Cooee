//! Real local inference. Built only with `--features whisper`.
//!
//! On Snapdragon X Elite this runs on 12 ARM cores with NEON. Build from a
//! native ARM64 shell (`pwsh`) or it silently lands under Prism emulation.

use super::{AsrEngine, Transcript, SAMPLE_RATE};
use anyhow::{Context, Result};
use parking_lot::Mutex;
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub struct WhisperCpp {
    ctx: WhisperContext,
    /// whisper.cpp state is not re-entrant; serialise access.
    lock: Mutex<()>,
    threads: i32,
}

impl WhisperCpp {
    pub fn load(model: &Path) -> Result<Self> {
        Self::load_with_threads(model, None)
    }

    /// `threads: None` leaves one core for the UI and the audio callback.
    pub fn load_with_threads(model: &Path, threads: Option<usize>) -> Result<Self> {
        let path = model.to_str().context("model path is not valid UTF-8")?;
        let ctx = WhisperContext::new_with_params(path, WhisperContextParameters::default())
            .with_context(|| format!("failed to load model at {}", model.display()))?;

        // Measured on a Snapdragon X Elite (12 cores) under normal corporate
        // load — Teams, Edge, and a security agent already saturating the CPU:
        //
        //   threads=2   2.7 s     threads=4   7.2 s     threads=8   42 s
        //
        // ggml's workers spin-wait, so oversubscribing a busy machine collapses
        // throughput rather than improving it. `cores - 1` was ~15x slower than
        // a small fixed pool. Stay conservative and let the user opt into more
        // via `asr_threads` if their machine is actually idle.
        const DEFAULT_THREADS: usize = 4;
        let threads = threads
            .or_else(|| {
                std::thread::available_parallelism()
                    .ok()
                    .map(|n| n.get().saturating_sub(1).min(DEFAULT_THREADS))
            })
            .unwrap_or(2)
            .max(1) as i32;

        Ok(Self {
            ctx,
            lock: Mutex::new(()),
            threads,
        })
    }
}

impl AsrEngine for WhisperCpp {
    fn name(&self) -> &str {
        "whisper.cpp"
    }

    fn transcribe(&self, pcm: &[f32], prompt: Option<&str>) -> Result<Transcript> {
        let _guard = self.lock.lock();
        let started = std::time::Instant::now();

        let mut state = self.ctx.create_state().context("create whisper state")?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.threads);
        params.set_translate(false);
        params.set_language(Some("en"));
        // Dictation is a one-shot utterance; suppress all console chatter.
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        // Non-speech tokens like (wind blowing) are noise for dictation.
        params.set_suppress_blank(true);
        // Vocabulary from the personal dictionary. Whisper decodes as if this
        // text preceded the audio, which is enough to tip a name it would
        // otherwise spell phonetically. Kept short: a long prompt costs
        // decoder context and, on near-silent clips, invites the model to
        // echo it back.
        if let Some(prompt) = prompt {
            params.set_initial_prompt(prompt);
        }

        debug_assert_eq!(SAMPLE_RATE, 16_000);
        state
            .full(params, pcm)
            .context("whisper inference failed")?;

        let segments = state.full_n_segments().context("segment count")?;
        let mut text = String::new();
        for i in 0..segments {
            text.push_str(&state.full_get_segment_text(i).context("segment text")?);
        }

        Ok(Transcript {
            text: text.trim().to_string(),
            inference_ms: started.elapsed().as_millis() as u64,
        })
    }
}
