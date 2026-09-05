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
use crate::audio::{Capture, Meter};
use crate::config::Config;
use crate::tone::{self, Tone};
use crate::{inject, polish, vad};
use crossbeam_channel::Receiver;
use parking_lot::RwLock;
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

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

/// A finished dictation, reported before insertion so it is kept even when
/// the insertion fails or lands in the wrong place.
#[derive(Debug, Clone)]
pub struct Dictated {
    pub text: String,
    pub inference_ms: u64,
    /// Release to text ready.
    pub elapsed_ms: u64,
}

/// Anything that wants to observe the pipeline (the Tauri window, tests, a CLI).
pub trait Observer: Send + Sync {
    fn on_status(&self, event: StatusEvent);

    /// Text is ready and about to be inserted.
    fn on_dictated(&self, _dictated: Dictated) {}

    /// Input level in `0.0..=1.0`, about twenty times a second while
    /// capturing. Drives the HUD bars; nothing else depends on it.
    fn on_level(&self, _level: f32) {}
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

/// Forwards the microphone level to the observer for one utterance.
struct MeterThread {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

impl MeterThread {
    fn finish(self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.handle.join();
    }
}

/// Maps a raw RMS to a bar height in `0..=1` on a decibel scale, since
/// perceived loudness is logarithmic: -50 dBFS (room tone on a quiet mic)
/// reads as nothing, -10 dBFS (speaking up, close to the mic) as full.
pub fn level_from_rms(rms: f32) -> f32 {
    const FLOOR_DB: f32 = -50.0;
    const CEIL_DB: f32 = -10.0;
    if rms <= 1e-5 {
        return 0.0;
    }
    let db = 20.0 * rms.log10();
    ((db - FLOOR_DB) / (CEIL_DB - FLOOR_DB)).clamp(0.0, 1.0)
}

/// Fast attack, slow release: a syllable lifts the bars at once and they
/// settle over a few ticks, which reads as a meter rather than a flicker.
fn meter_loop(meter: Meter, observer: Arc<dyn Observer>, stop: Arc<AtomicBool>) {
    const TICK: Duration = Duration::from_millis(50);
    const RELEASE: f32 = 0.7;
    let mut smoothed = 0.0f32;
    while !stop.load(Ordering::Relaxed) {
        let level = level_from_rms(meter.rms());
        smoothed = if level > smoothed {
            level
        } else {
            smoothed * RELEASE + level * (1.0 - RELEASE)
        };
        observer.on_level(smoothed);
        std::thread::sleep(TICK);
    }
}

impl Pipeline {
    /// Drains hotkey events until the channel closes. Blocking.
    pub fn run(self, rx: Receiver<HotkeyEvent>) {
        let mut capture: Option<(Capture, Option<MeterThread>)> = None;

        for event in rx.iter() {
            match event {
                HotkeyEvent::Pressed => {
                    if capture.is_some() {
                        continue; // already recording; ignore duplicate press
                    }
                    match Capture::start() {
                        Ok(c) => {
                            self.cue(Tone::Start);
                            // The device name is logged by audio.rs; on the
                            // pill it is truncated noise ("Microphone Array
                            // (Qualcomm ...)") and the bars say "recording".
                            self.emit(State::Capturing, None);
                            let meter = self.start_meter(c.meter());
                            capture = Some((c, meter));
                        }
                        Err(e) => {
                            tracing::error!("could not start capture: {e:#}");
                            self.emit(State::Error, Some(e.to_string()));
                        }
                    }
                }

                HotkeyEvent::Released => {
                    let Some((c, meter)) = capture.take() else {
                        continue;
                    };
                    if let Some(m) = meter {
                        m.finish();
                    }
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
        let (pcm, cut) = capture.take()?;
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
        let prompt = self.config.read().dictionary.prompt();
        let transcript = engine.transcribe(&audio, prompt.as_deref())?;

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

        self.observer.on_dictated(Dictated {
            text: text.clone(),
            inference_ms: transcript.inference_ms,
            elapsed_ms: started.elapsed().as_millis() as u64,
        });

        self.emit(State::Injecting, None);
        // A space either side when the caret is against a word, so a
        // mid-sentence dictation does not fuse with what is already there.
        let text = match crate::caret::neighbours() {
            Some(n) => crate::caret::pad(&text, &n),
            None => text,
        };
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
        let detail = if cut {
            format!("cut at {}:{:02} · {total} ms", crate::audio::MAX_SECONDS / 60, crate::audio::MAX_SECONDS % 60)
        } else {
            format!("{total} ms")
        };
        self.emit(State::Idle, Some(detail));
        Ok(())
    }

    fn start_meter(&self, meter: Meter) -> Option<MeterThread> {
        let stop = Arc::new(AtomicBool::new(false));
        let observer = self.observer.clone();
        let flag = stop.clone();
        let handle = std::thread::Builder::new()
            .name("cooee-meter".into())
            .spawn(move || meter_loop(meter, observer, flag))
            .map_err(|e| tracing::debug!("could not start meter thread: {e}"))
            .ok()?;
        Some(MeterThread { stop, handle })
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

#[cfg(test)]
mod tests {
    use super::level_from_rms;

    #[test]
    fn silence_reads_as_nothing() {
        assert_eq!(level_from_rms(0.0), 0.0);
        assert_eq!(level_from_rms(0.001), 0.0); // -60 dBFS, below the floor
    }

    #[test]
    fn loud_speech_pegs_the_meter() {
        assert_eq!(level_from_rms(0.5), 1.0); // -6 dBFS, above the ceiling
    }

    #[test]
    fn level_rises_monotonically_on_a_log_scale() {
        let quiet = level_from_rms(0.01); // -40 dBFS
        let normal = level_from_rms(0.05); // -26 dBFS
        let loud = level_from_rms(0.2); // -14 dBFS
        assert!(0.0 < quiet && quiet < normal && normal < loud && loud < 1.0);
        assert!((quiet - 0.25).abs() < 0.01, "{quiet}");
    }
}
