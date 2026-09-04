//! Whisper's log-mel front end: `whisper.audio.log_mel_spectrogram`, in Rust.
//!
//! 400-point periodic Hann window, hop 160, reflect padding, slaney-scaled
//! mel filterbank, log10 floored at max-8 and scaled to roughly [-1, 1].
//! The output is what the ONNX encoder was exported to take: one 30 s
//! window of `n_mels` x 3000 frames, silence-padded.

use realfft::RealFftPlanner;

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
/// Whisper always encodes a 30 s window.
pub const CHUNK: usize = 30 * SAMPLE_RATE;
pub const N_FRAMES: usize = CHUNK / HOP; // 3000
const N_BINS: usize = 1 + N_FFT / 2; // 201

/// The mel filterbank plus the FFT plan, built once per engine.
pub struct MelSpectrogram {
    n_mels: usize,
    /// Row-major (n_mels, N_BINS).
    filters: Vec<f32>,
    window: Vec<f32>,
    fft: std::sync::Arc<dyn realfft::RealToComplex<f32>>,
}

impl MelSpectrogram {
    pub fn new(n_mels: usize) -> Self {
        let window = (0..N_FFT)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / N_FFT as f32).cos())
            .collect();
        Self {
            n_mels,
            filters: mel_filterbank(n_mels),
            window,
            fft: RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT),
        }
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Log-mel features for up to 30 s of 16 kHz mono audio, row-major
    /// `(n_mels, N_FRAMES)`. Longer input is truncated; the caller chunks.
    pub fn compute(&self, pcm: &[f32]) -> Vec<f32> {
        let mut x = pcm[..pcm.len().min(CHUNK)].to_vec();
        x.resize(CHUNK, 0.0);
        // Reflect-pad N_FFT/2 each side, as torch's pad(mode="reflect") does.
        let half = N_FFT / 2;
        let mut padded = Vec::with_capacity(CHUNK + N_FFT);
        padded.extend((1..=half).rev().map(|i| x[i]));
        padded.extend_from_slice(&x);
        padded.extend((1..=half).map(|i| x[CHUNK - 1 - i]));

        let mut frame = self.fft.make_input_vec();
        let mut spec = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();

        // Power spectrum as (N_BINS, N_FRAMES); the 3001st frame is dropped,
        // as whisper does.
        let mut power = vec![0f32; N_BINS * N_FRAMES];
        for t in 0..N_FRAMES {
            for i in 0..N_FFT {
                frame[i] = padded[t * HOP + i] * self.window[i];
            }
            self.fft
                .process_with_scratch(&mut frame, &mut spec, &mut scratch)
                .expect("fft sizes match the plan");
            for (k, c) in spec.iter().enumerate() {
                power[k * N_FRAMES + t] = c.re * c.re + c.im * c.im;
            }
        }

        let mut mel = vec![0f32; self.n_mels * N_FRAMES];
        for m in 0..self.n_mels {
            let weights = &self.filters[m * N_BINS..(m + 1) * N_BINS];
            let out = &mut mel[m * N_FRAMES..(m + 1) * N_FRAMES];
            for (k, &w) in weights.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                let src = &power[k * N_FRAMES..(k + 1) * N_FRAMES];
                for t in 0..N_FRAMES {
                    out[t] += w * src[t];
                }
            }
        }
        let mut max = f32::MIN;
        for v in mel.iter_mut() {
            *v = v.max(1e-10).log10();
            max = max.max(*v);
        }
        for v in mel.iter_mut() {
            *v = (v.max(max - 8.0) + 4.0) / 4.0;
        }
        mel
    }
}

/// `librosa.filters.mel(sr, n_fft, n_mels)` with the slaney scale and slaney
/// normalisation, which is what whisper's exported `mel_filters.npz` holds.
/// Row-major `(n_mels, N_BINS)`.
fn mel_filterbank(n_mels: usize) -> Vec<f32> {
    let (min_log_hz, min_log_mel, logstep) = (1000.0f64, 15.0f64, (6.4f64).ln() / 27.0);
    let hz_to_mel = |f: f64| {
        if f >= min_log_hz {
            min_log_mel + (f.max(1e-9) / min_log_hz).ln() / logstep
        } else {
            f / (200.0 / 3.0)
        }
    };
    let mel_to_hz = |m: f64| {
        if m >= min_log_mel {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        } else {
            (200.0 / 3.0) * m
        }
    };
    let nyquist = SAMPLE_RATE as f64 / 2.0;
    let fft_freqs: Vec<f64> = (0..N_BINS)
        .map(|i| i as f64 * nyquist / (N_BINS - 1) as f64)
        .collect();
    let (m_lo, m_hi) = (hz_to_mel(0.0), hz_to_mel(nyquist));
    let mel_pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(m_lo + (m_hi - m_lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut fb = vec![0f32; n_mels * N_BINS];
    for m in 0..n_mels {
        let (lo, mid, hi) = (mel_pts[m], mel_pts[m + 1], mel_pts[m + 2]);
        let norm = 2.0 / (hi - lo);
        for (k, &f) in fft_freqs.iter().enumerate() {
            let lower = (f - lo) / (mid - lo);
            let upper = (hi - f) / (hi - mid);
            fb[m * N_BINS + k] = (lower.min(upper).max(0.0) * norm) as f32;
        }
    }
    fb
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference: `bench/npu_whisper.py`'s numpy `log_mel` on one second
    /// of two tones, first 100 frames. Regenerate with `bench/mel_fixture.py`.
    const FIXTURE: &[u8] = include_bytes!("fixtures/mel_two_tones_100.f32");

    fn two_tones() -> Vec<f32> {
        (0..SAMPLE_RATE)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.25 * (2.0 * std::f32::consts::PI * 3000.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn matches_the_numpy_reference() {
        let mel = MelSpectrogram::new(80).compute(&two_tones());
        let expected: Vec<f32> = FIXTURE
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(expected.len(), 80 * 100);
        let mut worst = 0f32;
        for m in 0..80 {
            for t in 0..100 {
                let diff = (mel[m * N_FRAMES + t] - expected[m * 100 + t]).abs();
                worst = worst.max(diff);
            }
        }
        assert!(worst < 2e-3, "max abs difference {worst}");
    }

    #[test]
    fn silence_is_the_floor() {
        // All-zero input: every bin is 1e-10, log10 = -10, so (−10 + 4) / 4.
        let mel = MelSpectrogram::new(80).compute(&[]);
        assert_eq!(mel.len(), 80 * N_FRAMES);
        assert!(mel.iter().all(|&v| (v - (-1.5)).abs() < 1e-6));
    }

    #[test]
    fn filterbank_rows_are_normalised_triangles() {
        let fb = mel_filterbank(80);
        for m in 0..80 {
            let row = &fb[m * N_BINS..(m + 1) * N_BINS];
            assert!(row.iter().any(|&w| w > 0.0), "mel {m} is empty");
            assert!(row.iter().all(|&w| w >= 0.0));
        }
    }
}
