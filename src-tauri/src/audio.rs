//! Microphone capture via WASAPI (through `cpal`).
//!
//! Push-to-talk utterances are short, so we capture at the device's native rate
//! into a plain buffer and resample once on stop. Ten seconds at 48 kHz is ~2 MB
//! — not worth the complexity of a real-time resampler on the audio callback,
//! which must stay allocation-free.

use crate::asr::SAMPLE_RATE;
use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, Stream};
use parking_lot::Mutex;
use rubato::{FftFixedIn, Resampler};
use std::sync::Arc;

/// Guards against a stuck hotkey eating memory. 60 s at 48 kHz mono.
const MAX_SAMPLES: usize = 48_000 * 60;

#[derive(Default)]
struct Buffer {
    samples: Vec<f32>,
    /// Set when we hit MAX_SAMPLES so the UI can explain the truncation.
    truncated: bool,
}

pub struct Capture {
    stream: Stream,
    buffer: Arc<Mutex<Buffer>>,
    src_rate: u32,
    pub device_name: String,
}

impl Capture {
    /// Opens the default input device and starts streaming into the buffer.
    /// Nothing is retained until [`Capture::take`] is called.
    pub fn start() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device — check microphone permissions"))?;
        let device_name = device.name().unwrap_or_else(|_| "unknown".into());

        let config = device
            .default_input_config()
            .context("failed to read default input config")?;
        let src_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        if channels == 0 {
            return Err(anyhow!("input device reported zero channels"));
        }
        let format = config.sample_format();

        let buffer = Arc::new(Mutex::new(Buffer::default()));
        let sink = buffer.clone();

        let on_error = |e| tracing::error!("audio stream error: {e}");

        // Downmix to mono in the callback: cheap, and it halves what we store.
        macro_rules! build {
            ($t:ty) => {
                device.build_input_stream(
                    &config.clone().into(),
                    move |data: &[$t], _: &cpal::InputCallbackInfo| {
                        let mut buf = sink.lock();
                        if buf.samples.len() >= MAX_SAMPLES {
                            buf.truncated = true;
                            return;
                        }
                        for frame in data.chunks(channels) {
                            let sum: f32 = frame.iter().map(|s| f32::from_sample(*s)).sum();
                            buf.samples.push(sum / channels as f32);
                        }
                    },
                    on_error,
                    None,
                )
            };
        }

        let stream = match format {
            SampleFormat::F32 => build!(f32),
            SampleFormat::I16 => build!(i16),
            SampleFormat::U16 => build!(u16),
            other => return Err(anyhow!("unsupported sample format: {other:?}")),
        }
        .context("failed to build input stream")?;

        stream.play().context("failed to start input stream")?;
        tracing::info!(device = %device_name, rate = src_rate, channels, "capture started");

        Ok(Self {
            stream,
            buffer,
            src_rate,
            device_name,
        })
    }

    /// Stops the stream and returns 16 kHz mono f32, ready for the ASR engine.
    pub fn take(self) -> Result<Vec<f32>> {
        drop(self.stream); // stops capture
        let buf = std::mem::take(&mut *self.buffer.lock());
        if buf.truncated {
            tracing::warn!("capture hit the {MAX_SAMPLES}-sample ceiling and was truncated");
        }
        resample_to_16k(&buf.samples, self.src_rate)
    }
}

/// Whole-buffer resample. Returns the input untouched when already at 16 kHz.
fn resample_to_16k(input: &[f32], src_rate: u32) -> Result<Vec<f32>> {
    if src_rate == SAMPLE_RATE {
        return Ok(input.to_vec());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut resampler = FftFixedIn::<f32>::new(src_rate as usize, SAMPLE_RATE as usize, 1024, 2, 1)
        .context("failed to construct resampler")?;

    let estimated = input.len() * SAMPLE_RATE as usize / src_rate as usize + 1024;
    let mut out = Vec::with_capacity(estimated);
    let mut pos = 0usize;

    while pos < input.len() {
        let need = resampler.input_frames_next();
        let end = (pos + need).min(input.len());
        let frame = &input[pos..end];

        // `process_partial` handles the short final chunk and flushes the tail.
        let produced = if frame.len() == need {
            resampler.process(&[frame], None)
        } else {
            resampler.process_partial(Some(&[frame]), None)
        }
        .context("resampling failed")?;

        out.extend_from_slice(&produced[0]);
        pos = end;
    }

    tracing::debug!(
        "resampled {} samples @ {src_rate} Hz -> {} @ {SAMPLE_RATE} Hz",
        input.len(),
        out.len()
    );
    Ok(out)
}
