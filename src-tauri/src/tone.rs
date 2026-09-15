//! Short audible cues for capture start, capture stop, and a dictation that
//! produced nothing.
//!
//! Useful when the overlay is out of sight — a second monitor, a full-screen
//! app — and as confirmation that the hotkey registered at all. Two notes,
//! rising for start and falling for stop, each short enough to finish before
//! the first word.
//!
//! The third cue exists because the overlay is not an accessible surface: it
//! is a transparent window that deliberately never takes focus, so a screen
//! reader has nothing to announce and a live region would not be read
//! reliably. Sound is the channel that works regardless of focus, and until
//! this existed a failed dictation was *silent* — identical, to anyone not
//! watching the pill, to one that worked.
//!
//! Best-effort and off the pipeline thread: opening a WASAPI output stream
//! costs tens of milliseconds, and a missing cue must never delay capture.
//! The start tone is quiet and brief because the microphone hears it too.

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat};
use std::f32::consts::PI;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    Start,
    Stop,
    /// The utterance ended with nothing inserted: no speech, an engine error,
    /// or text that polished away to nothing.
    Failed,
}

impl Tone {
    fn hz(self) -> f32 {
        match self {
            Tone::Start => 880.0,   // A5
            Tone::Stop => 587.33,   // D5
            Tone::Failed => 329.63, // E4, nearly an octave under the stop cue
        }
    }

    /// The failure cue is longer as well as lower. Someone who cannot see the
    /// pill should not have to compare two pitches from memory to work out
    /// whether their words landed.
    fn duration_ms(self) -> u32 {
        match self {
            Tone::Failed => 180,
            _ => DURATION_MS,
        }
    }
}

const DURATION_MS: u32 = 70;
/// Well below full scale: a cue, not an alarm, and less for the mic to hear.
const AMPLITUDE: f32 = 0.12;
/// Linear ramp either end so the tone starts and stops without a click.
const FADE_MS: u32 = 6;

/// Plays the tone on the default output device. Returns immediately.
pub fn play(tone: Tone) {
    let spawned = std::thread::Builder::new()
        .name("cooee-tone".into())
        .spawn(move || {
            if let Err(e) = play_blocking(tone) {
                tracing::debug!("audio feedback unavailable: {e:#}");
            }
        });
    if let Err(e) = spawned {
        tracing::debug!("could not spawn tone thread: {e}");
    }
}

/// Mono samples for a faded sine burst.
pub fn samples(hz: f32, rate: u32, duration_ms: u32) -> Vec<f32> {
    let n = (rate as u64 * duration_ms as u64 / 1000) as usize;
    let fade = ((rate as u64 * FADE_MS as u64 / 1000) as usize).max(1);
    (0..n)
        .map(|i| {
            let envelope = if i < fade {
                i as f32 / fade as f32
            } else if i + fade >= n {
                (n - i) as f32 / fade as f32
            } else {
                1.0
            };
            let t = i as f32 / rate as f32;
            AMPLITUDE * envelope * (2.0 * PI * hz * t).sin()
        })
        .collect()
}

fn play_blocking(tone: Tone) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow!("no default output device"))?;
    let config = device
        .default_output_config()
        .context("failed to read default output config")?;
    let rate = config.sample_rate().0;
    let channels = config.channels() as usize;
    if channels == 0 {
        return Err(anyhow!("output device reported zero channels"));
    }

    let pcm = samples(tone.hz(), rate, tone.duration_ms());
    let on_error = |e| tracing::debug!("tone stream error: {e}");

    // The callback walks `pcm` once and then emits silence until dropped.
    macro_rules! build {
        ($t:ty) => {{
            let mut pos = 0usize;
            device.build_output_stream(
                &config.clone().into(),
                move |data: &mut [$t], _: &cpal::OutputCallbackInfo| {
                    for frame in data.chunks_mut(channels) {
                        let s = pcm.get(pos).copied().unwrap_or(0.0);
                        pos += 1;
                        for out in frame {
                            *out = <$t>::from_sample(s);
                        }
                    }
                },
                on_error,
                None,
            )
        }};
    }

    let stream = match config.sample_format() {
        SampleFormat::F32 => build!(f32),
        SampleFormat::I16 => build!(i16),
        SampleFormat::U16 => build!(u16),
        other => return Err(anyhow!("unsupported sample format: {other:?}")),
    }
    .context("failed to build output stream")?;

    stream.play().context("failed to start output stream")?;
    // Hold the stream open past the tone so the tail is not cut off.
    std::thread::sleep(Duration::from_millis(tone.duration_ms() as u64 + 60));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_is_the_requested_length() {
        assert_eq!(samples(880.0, 48_000, 70).len(), 3_360);
    }

    #[test]
    fn tone_stays_quiet_and_never_clips() {
        let pcm = samples(880.0, 48_000, 70);
        let peak = pcm.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(peak <= AMPLITUDE + 1e-6);
        assert!(
            peak > AMPLITUDE * 0.9,
            "the sustained section should reach full amplitude"
        );
    }

    #[test]
    fn tone_fades_in_and_out_to_avoid_clicks() {
        let pcm = samples(880.0, 48_000, 70);
        assert_eq!(pcm[0], 0.0);
        assert!(pcm.last().unwrap().abs() < 0.01);
    }

    #[test]
    fn start_and_stop_are_distinguishable() {
        assert!(Tone::Start.hz() > Tone::Stop.hz());
    }

    #[test]
    fn failure_is_unmistakable_by_ear() {
        // Lower than both, so it cannot be mistaken for a normal stop...
        assert!(Tone::Failed.hz() < Tone::Stop.hz());
        assert!(Tone::Failed.hz() < Tone::Start.hz());
        // ...and longer, so the two do not have to be told apart by pitch.
        assert!(Tone::Failed.duration_ms() > Tone::Stop.duration_ms());
    }

    #[test]
    fn every_cue_stays_within_the_amplitude_ceiling() {
        for tone in [Tone::Start, Tone::Stop, Tone::Failed] {
            let pcm = samples(tone.hz(), 48_000, tone.duration_ms());
            let peak = pcm.iter().fold(0.0f32, |m, s| m.max(s.abs()));
            assert!(peak <= AMPLITUDE + 1e-6, "{tone:?} peaked at {peak}");
            assert_eq!(pcm[0], 0.0, "{tone:?} must fade in");
        }
    }
}
