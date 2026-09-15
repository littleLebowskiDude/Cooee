//! The state machine that ties everything together.
//!
//! ```text
//!   Idle --press--> Capturing --release--> Transcribing --> Polishing --> Injecting --> Idle
//! ```
//!
//! That is `CaptureMode::Hold`. Under `CaptureMode::Latch` the release is
//! ignored and a second press ends the capture, so nothing has to be held
//! down; see [`action`], which is the whole of the difference.
//!
//! Runs on its own thread. The hotkey hook only ever pushes events into the
//! channel this drains — see the comment at the top of `hotkey.rs` for why that
//! separation is load-bearing. Latch is decided here and not in the hook for
//! the same reason: the hook has a hard deadline and this does not.
//!
//! Nothing is transcribed until the key comes up. A live preview was built and
//! removed: it made the pill large and busy, and the text that matters is the
//! one that lands after release — which is also how Wispr Flow behaves.

use crate::asr::EngineSlot;
use crate::audio::{Capture, Meter};
use crate::config::Config;
use crate::tone::{self, Tone};
use crate::{inject, polish, vad};
use crossbeam_channel::{Receiver, RecvTimeoutError};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
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

/// How the hotkey drives capture.
///
/// `Hold` is push-to-talk: the chord stays down for the whole utterance. That
/// assumes a hand which can hold two keys steady for the length of a sentence,
/// which is exactly the assumption that rules the app out for RSI, tremor,
/// limited dexterity, one-handed use and switch devices — the people dictation
/// exists for. `Latch` costs two taps instead and asks nothing to be held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CaptureMode {
    #[default]
    Hold,
    Latch,
}

/// Emitted to the overlay on every transition so the HUD can animate.
#[derive(Debug, Clone, Serialize)]
pub struct StatusEvent {
    pub state: State,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Set while a latched capture is running. The HUD has to say so: with no
    /// key held down, a latched capture is otherwise indistinguishable from a
    /// hotkey that has jammed.
    pub latched: bool,
}

/// What a hotkey event means, given the mode and whether a capture is running.
///
/// Split out as a pure function because this is the whole of the latch
/// behaviour, and a state machine that types into the user's apps is worth
/// testing without a keyboard hook in the way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    Start,
    Stop,
    Ignore,
}

fn action(mode: CaptureMode, capturing: bool, event: HotkeyEvent) -> Action {
    match (mode, event, capturing) {
        // Hold: the key is the capture. Press starts, release ends.
        (CaptureMode::Hold, HotkeyEvent::Pressed, false) => Action::Start,
        (CaptureMode::Hold, HotkeyEvent::Released, true) => Action::Stop,
        // Latch: releases mean nothing, and the second press is the stop.
        (CaptureMode::Latch, HotkeyEvent::Pressed, false) => Action::Start,
        (CaptureMode::Latch, HotkeyEvent::Pressed, true) => Action::Stop,
        _ => Action::Ignore,
    }
}

/// The capture in flight, and the mode it began under.
///
/// The mode is carried here rather than re-read from config on each event.
/// Hold and latch disagree about what ends a capture, so a setting changed
/// mid-utterance could leave a hold capture whose release is now interpreted
/// under latch rules and therefore ignored — and `hotkey::set` clears the
/// held-chord flag as it applies the change, so the synthetic release it
/// emits is the *only* one that capture would ever see. Keeping the mode
/// beside the capture makes the two unable to disagree.
struct Active {
    capture: Capture,
    meter: Option<MeterThread>,
    mode: CaptureMode,
}

/// Which mode governs the next decision: the one a running capture began
/// under, or the configured one when the pipeline is idle. A mode change
/// therefore takes effect from the next utterance, never during one.
fn governing_mode(configured: CaptureMode, in_flight: Option<CaptureMode>) -> CaptureMode {
    in_flight.unwrap_or(configured)
}

/// How often an open capture checks whether it has hit the ceiling. Coarse on
/// purpose: the only deadline it has to catch is a five-minute one.
const CEILING_POLL: Duration = Duration::from_millis(500);

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
        let mut active: Option<Active> = None;

        loop {
            // Polled whenever a capture is open, not only a latched one.
            // MAX_SECONDS exists to stop a capture that never ends, and a key
            // stuck down in hold mode does that just as well as a latch the
            // user walked away from.
            let event = if active.is_some() {
                match rx.recv_timeout(CEILING_POLL) {
                    Ok(ev) => Some(ev),
                    Err(RecvTimeoutError::Timeout) => None,
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            } else {
                match rx.recv() {
                    Ok(ev) => Some(ev),
                    Err(_) => break,
                }
            };

            // Woken by the timeout rather than a key: end the capture if it
            // has filled. Without this the microphone stays open with nothing
            // left to record, since past the ceiling every sample is dropped.
            let Some(event) = event else {
                if active.as_ref().is_some_and(|a| a.capture.truncated()) {
                    tracing::info!("capture hit the ceiling; stopping it");
                    self.stop(&mut active);
                }
                continue;
            };

            let mode = governing_mode(
                self.config.read().capture_mode,
                active.as_ref().map(|a| a.mode),
            );
            match action(mode, active.is_some(), event) {
                Action::Ignore => {}
                Action::Start => match Capture::start() {
                    Ok(capture) => {
                        self.cue(Tone::Start);
                        // The device name is logged by audio.rs; on the
                        // pill it is truncated noise ("Microphone Array
                        // (Qualcomm ...)") and the bars say "recording".
                        self.emit_capturing(mode == CaptureMode::Latch);
                        let meter = self.start_meter(capture.meter());
                        active = Some(Active {
                            capture,
                            meter,
                            mode,
                        });
                    }
                    Err(e) => {
                        tracing::error!("could not start capture: {e:#}");
                        self.failed(State::Error, Some(e.to_string()));
                    }
                },
                Action::Stop => self.stop(&mut active),
            }
        }
    }

    /// Closes the microphone and runs everything downstream of it.
    fn stop(&self, active: &mut Option<Active>) {
        let Some(active) = active.take() else {
            return;
        };
        if let Some(meter) = active.meter {
            meter.finish();
        }
        if let Err(e) = self.finish(active.capture) {
            tracing::error!("dictation failed: {e:#}");
            self.failed(State::Error, Some(e.to_string()));
        }
    }

    /// Everything after the key comes up: stop, trim, transcribe, polish, inject.
    fn finish(&self, capture: Capture) -> anyhow::Result<()> {
        let started = Instant::now();
        let (pcm, cut) = capture.take()?;

        let Some(speech) = vad::trim(&pcm) else {
            // The failure cue replaces the stop cue here rather than following
            // it. Nothing has been transcribed yet, so the two would start
            // within a millisecond of each other and overlap into one
            // indistinct sound; of the two, "nothing was heard" is the one
            // worth hearing.
            tracing::info!("no speech detected; nothing to do");
            self.failed(State::Idle, Some("no speech detected".into()));
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
            // Also ahead of the stop cue, and for the same reason as the
            // no-speech branch: everything between the two is arithmetic, so
            // the cues would sound together. This is the failure a first run
            // is most likely to hit, while the model is still loading.
            tracing::warn!("dictation arrived while the model was still loading");
            self.failed(
                State::Error,
                Some("model is still loading; try again in a moment".into()),
            );
            return Ok(());
        };

        // Past every branch that ends in silence, so the stop cue only ever
        // plays for an utterance that is actually going to be transcribed.
        // The microphone closed at `take` above, so it is never in the audio.
        self.cue(Tone::Stop);

        self.emit(State::Transcribing, None);
        let prompt = self.config.read().dictionary.prompt();
        let transcript = engine.transcribe(&audio, prompt.as_deref())?;

        let text = {
            let cfg = self.config.read();
            polish::polish(&transcript.text, &cfg.dictionary)
        };

        if text.is_empty() {
            self.failed(State::Idle, Some("nothing to insert".into()));
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
        // Resolved here rather than alongside the polish above: focus can move
        // while the model is working, and the only window that matters is the
        // one that is about to receive the text.
        let app = crate::focus::foreground_exe();
        let strategy = self.config.read().strategy_for(app.as_deref());
        inject::inject(&text, strategy)?;

        let total = started.elapsed().as_millis();
        tracing::info!(
            chars = text.len(),
            app = app.as_deref().unwrap_or("unknown"),
            ?strategy,
            input_peak,
            gain,
            inference_ms = transcript.inference_ms,
            total_ms = total as u64,
            "dictated"
        );
        let detail = if cut {
            format!(
                "cut at {}:{:02} · {total} ms",
                crate::audio::MAX_SECONDS / 60,
                crate::audio::MAX_SECONDS % 60
            )
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
        self.observer.on_status(StatusEvent {
            state,
            detail,
            latched: false,
        });
    }

    /// Capture start carries whether the session is latched, so the HUD can
    /// offer a way out of it.
    fn emit_capturing(&self, latched: bool) {
        self.observer.on_status(StatusEvent {
            state: State::Capturing,
            detail: None,
            latched,
        });
    }

    /// An utterance that inserted nothing, whether through an error or a clip
    /// with no speech in it.
    ///
    /// The cue matters more than the state here. The overlay is a transparent
    /// window that never takes focus, so it is not a surface a screen reader
    /// can announce; without a sound, a user who is not watching the pill has
    /// no way to tell this apart from a dictation that worked.
    fn failed(&self, state: State, detail: Option<String>) {
        self.cue(Tone::Failed);
        self.emit(state, detail);
    }

    fn cue(&self, tone: Tone) {
        if self.config.read().audio_feedback {
            tone::play(tone);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn hold_is_press_to_start_and_release_to_stop() {
        use CaptureMode::Hold;
        assert_eq!(action(Hold, false, HotkeyEvent::Pressed), Action::Start);
        assert_eq!(action(Hold, true, HotkeyEvent::Released), Action::Stop);
    }

    #[test]
    fn hold_ignores_repeats_and_stray_releases() {
        use CaptureMode::Hold;
        // Auto-repeat is collapsed in the hook, but a second Pressed must
        // never restart a capture that is already running.
        assert_eq!(action(Hold, true, HotkeyEvent::Pressed), Action::Ignore);
        // A release with nothing running: e.g. hotkey::set releasing a stale
        // chord, which must not run the pipeline on an empty buffer.
        assert_eq!(action(Hold, false, HotkeyEvent::Released), Action::Ignore);
    }

    #[test]
    fn latch_starts_and_stops_on_successive_presses() {
        use CaptureMode::Latch;
        assert_eq!(action(Latch, false, HotkeyEvent::Pressed), Action::Start);
        assert_eq!(action(Latch, true, HotkeyEvent::Pressed), Action::Stop);
    }

    #[test]
    fn latch_never_stops_on_a_release() {
        use CaptureMode::Latch;
        // The whole point: letting go of the chord is not the end of the
        // utterance, so the user never has to hold anything.
        assert_eq!(action(Latch, true, HotkeyEvent::Released), Action::Ignore);
        assert_eq!(action(Latch, false, HotkeyEvent::Released), Action::Ignore);
    }

    #[test]
    fn a_full_latched_utterance_is_two_taps() {
        // A tap is Pressed then Released; the hook emits both either way.
        let taps = [
            HotkeyEvent::Pressed,
            HotkeyEvent::Released,
            HotkeyEvent::Pressed,
            HotkeyEvent::Released,
        ];
        let mut capturing = false;
        let mut actions = Vec::new();
        for ev in taps {
            let a = action(CaptureMode::Latch, capturing, ev);
            match a {
                Action::Start => capturing = true,
                Action::Stop => capturing = false,
                Action::Ignore => {}
            }
            actions.push(a);
        }
        assert_eq!(
            actions,
            [Action::Start, Action::Ignore, Action::Stop, Action::Ignore]
        );
        assert!(!capturing, "the second tap must have ended the capture");
    }

    #[test]
    fn capture_mode_defaults_to_hold_and_round_trips() {
        assert_eq!(CaptureMode::default(), CaptureMode::Hold);
        assert_eq!(
            serde_json::to_string(&CaptureMode::Latch).unwrap(),
            "\"latch\""
        );
        let back: CaptureMode = serde_json::from_str("\"hold\"").unwrap();
        assert_eq!(back, CaptureMode::Hold);
    }

    #[test]
    fn a_mode_change_cannot_strand_a_running_capture() {
        // The setting flips to Latch while a Hold capture is in flight.
        // `hotkey::set` emits a synthetic Released as it applies the change
        // and clears the held-chord flag, so that release is the only one
        // this capture will ever see: reading the new mode here would return
        // Ignore and nothing would ever stop the microphone.
        let mode = governing_mode(CaptureMode::Latch, Some(CaptureMode::Hold));
        assert_eq!(mode, CaptureMode::Hold, "a running capture keeps its mode");
        assert_eq!(action(mode, true, HotkeyEvent::Released), Action::Stop);
    }

    #[test]
    fn the_reverse_switch_is_also_honoured_by_the_running_capture() {
        // Latch capture in flight, setting flips to Hold. The capture must
        // still end on the next press, not on the release of that press.
        let mode = governing_mode(CaptureMode::Hold, Some(CaptureMode::Latch));
        assert_eq!(mode, CaptureMode::Latch);
        assert_eq!(action(mode, true, HotkeyEvent::Pressed), Action::Stop);
    }

    #[test]
    fn a_mode_change_applies_from_the_next_utterance() {
        // Nothing in flight: the configured mode is the one that counts.
        assert_eq!(governing_mode(CaptureMode::Latch, None), CaptureMode::Latch);
        assert_eq!(governing_mode(CaptureMode::Hold, None), CaptureMode::Hold);
    }
}
