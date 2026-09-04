//! Returns canned text so the full hotkey -> audio -> VAD -> inject loop can be
//! exercised with no C++ toolchain. This is not a placeholder to delete later:
//! it stays as the fixture backend for integration tests.

use super::{AsrEngine, Transcript, SAMPLE_RATE};
use anyhow::Result;

#[derive(Default)]
pub struct MockEngine;

impl AsrEngine for MockEngine {
    fn name(&self) -> &str {
        "mock"
    }

    fn transcribe(&self, pcm: &[f32], _prompt: Option<&str>) -> Result<Transcript> {
        let seconds = pcm.len() as f32 / SAMPLE_RATE as f32;
        // This is the *post-normalisation* peak, so it sits near the target for
        // any real speech. The pre-normalisation level — the number that tells
        // you whether the mic gain is right — is logged by the pipeline.
        let peak = pcm.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        Ok(Transcript {
            text: format!("[mock transcript: {seconds:.1}s captured, normalised peak {peak:.2}]"),
            inference_ms: 0,
        })
    }
}
