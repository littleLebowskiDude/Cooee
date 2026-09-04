//! The state machine that ties everything together.
//!
//! ```text
//!   Idle --press--> Capturing --release--> Transcribing --> Polishing --> Injecting --> Idle
//! ```
//!
//! Runs on its own thread. The hotkey hook only ever pushes events into the
//! channel this drains — see the comment at the top of `hotkey.rs` for why that
//! separation is load-bearing.
//!
//! Nothing is transcribed until the key comes up. A live preview was built and
//! removed: it made the pill large and busy, and the text that matters is the
//! one that lands after release — which is also how Wispr Flow behaves.

use crate::asr::EngineSlot;
use crate::audio::Capture;
use crate::config::Config;
use crate::tone::{self, Tone};
use crate::{inject, polish, vad};
use crossbeam_channel::Receiver;
use parking_lot::RwLock;
use serde::Serialize;
use std::sync::Arc;
use std::time::Instant;

use crate::hotkey::HotkeyEvent;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum State {
    Idle,
    Capturing,
    Transcribing,
    Injecting,
    Error,
}

/// Emitted to the overlay on every transition so the HUD can animate.
#[derive(Debug, Clone, Serialize)]
pub struct StatusEvent {
    pub state: State,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Anything that wants to observe the pipeline (the Tauri window, tests, a CLI).
pub trait Observer: Send + Sync {
    fn on_status(&self, event: StatusEvent);
}

/// No-op observer, used by headless tests.
pub struct Silent;
impl Observer for Silent {
    fn on_status(&self, _: StatusEvent) {}
}

pub struct Pipeline {
    pub config: Arc<RwLock<Config>>,
    pub engine: EngineSlot,
    pub observer: Arc<dyn Observer>,
}

impl Pipeline {
    /// Drains hotkey events until the channel closes. Blocking.
    pub fn run(self, rx: Receiver<HotkeyEvent>) {
        let mut capture: Option<Capture> = None;

        for event in rx.iter() {
            match event {
                HotkeyEvent::Pressed => {
                    if capture.is_some() {
                        continue; // already recording; ignore duplicate press
                    }
                    match Capture::start() {
                        Ok(c) => {
                            self.cue(Tone::Start);
                            self.emit(State::Capturing, Some(format!("mic: {}", c.device_name)));
                            capture = Some(c);
                        }
                        Err(e) => {
                            tracing::error!("could not start capture: {e:#}");
                            self.emit(State::Error, Some(e.to_string()));
                        }
                    }
                }

                HotkeyEvent::Released => {
                    let Some(c) = capture.take() else { continue };
                    if let Err(e) = self.finish(c) {
                        tracing::error!("dictation failed: {e:#}");
                        self.emit(State::Error, Some(e.to_string()));
                    }
                }
            }
        }
    }

    /// Everything after the key comes up: stop, trim, transcribe, polish, inject.
    fn finish(&self, capture: Capture) -> anyhow::Result<()> {
        let started = Instant::now();
        let pcm = capture.take()?;
        // After the mic is closed, so the stop tone is never in the recording.
        self.cue(Tone::Stop);

        let Some(speech) = vad::trim(&pcm) else {
            tracing::info!("no speech detected; nothing to do");
            self.emit(State::Idle, Some("no speech detected".into()));
            return Ok(());
        };

        // Peak-normalise before inference: a quiet mic measurably hurts whisper.
        // Read the level first — after normalising, every clip reads ~0.7.
        let input_peak = vad::peak(speech);
        let mut audio = speech.to_vec();
        let gain = vad::normalise(&mut audio);
        if gain > 4.0 {
            tracing::warn!(
                input_peak,
                gain,
                "microphone level is low; raise it in Windows sound settings for better accuracy"
            );
        }

        // Clone the Arc out so a model swap mid-utterance cannot block or
        // invalidate this transcription; the old engine simply finishes.
        let Some(engine) = self.engine.read().clone() else {
            tracing::warn!("dictation arrived while the model was still loading");
            self.emit(
                State::Error,
                Some("model is still loading; try again in a moment".into()),
            );
            return Ok(());
        };

        self.emit(State::Transcribing, None);
        let transcript = engine.transcribe(&audio)?;

        let (text, strategy) = {
            let cfg = self.config.read();
            (
                polish::polish(&transcript.text, &cfg.dictionary),
                cfg.injection,
            )
        };

        if text.is_empty() {
            self.emit(State::Idle, Some("nothing to insert".into()));
            return Ok(());
        }

        self.emit(State::Injecting, None);
        inject::inject(&text, strategy)?;

        let total = started.elapsed().as_millis();
        tracing::info!(
            chars = text.len(),
            input_peak,
            gain,
            inference_ms = transcript.inference_ms,
            total_ms = total as u64,
            "dictated"
        );
        self.emit(State::Idle, Some(format!("{total} ms")));
        Ok(())
    }

    fn emit(&self, state: State, detail: Option<String>) {
        self.observer.on_status(StatusEvent { state, detail });
    }

    fn cue(&self, tone: Tone) {
        if self.config.read().audio_feedback {
            tone::play(tone);
        }
    }
}
