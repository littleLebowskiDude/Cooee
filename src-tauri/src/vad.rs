//! Silence trimming.
//!
//! Push-to-talk gives us clean utterance boundaries already, so this is not a
//! speech *detector* — it just trims the dead air either side of the press so
//! whisper spends no time on it. A Silero ONNX model can replace this behind the
//! same function signature if the energy gate proves too blunt.

use crate::asr::SAMPLE_RATE;

/// 20 ms analysis window.
const FRAME: usize = SAMPLE_RATE as usize / 50;
/// RMS below this counts as silence. Roughly -46 dBFS.
const THRESHOLD: f32 = 0.005;
/// Keep this much either side so words aren't clipped at the edges.
const HANGOVER_FRAMES: usize = 8; // 160 ms

fn rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt()
}

/// Returns the speech-bearing slice, or `None` if the clip is all silence.
pub fn trim(pcm: &[f32]) -> Option<&[f32]> {
    if pcm.len() < FRAME {
        return None;
    }

    let voiced: Vec<bool> = pcm.chunks(FRAME).map(|f| rms(f) > THRESHOLD).collect();
    let first = voiced.iter().position(|&v| v)?;
    let last = voiced.iter().rposition(|&v| v)?;

    let start = first.saturating_sub(HANGOVER_FRAMES) * FRAME;
    let end = ((last + 1 + HANGOVER_FRAMES) * FRAME).min(pcm.len());

    Some(&pcm[start..end])
}

/// Peak amplitude, for the live level meter in the overlay.
pub fn peak(pcm: &[f32]) -> f32 {
    pcm.iter().fold(0.0f32, |m, s| m.max(s.abs()))
}

/// Peak-normalise toward [`TARGET_PEAK`], returning the gain applied.
///
/// Whisper is trained on normalised audio, and a quiet microphone measurably
/// hurts accuracy — a headset at low gain can land near 0.05 peak where speech
/// should sit around 0.3-0.9. Gain is capped so a near-silent clip is not
/// amplified into pure noise, and loud input is scaled *down* to avoid clipping.
pub fn normalise(pcm: &mut [f32]) -> f32 {
    const TARGET_PEAK: f32 = 0.7;
    const MAX_GAIN: f32 = 15.0;
    /// Below this the clip is noise, not quiet speech; leave it alone.
    const NOISE_FLOOR: f32 = 1e-4;

    let current = peak(pcm);
    if current < NOISE_FLOOR {
        return 1.0;
    }
    let gain = (TARGET_PEAK / current).min(MAX_GAIN);
    for sample in pcm.iter_mut() {
        *sample *= gain;
    }
    gain
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_silence_yields_nothing() {
        assert!(trim(&vec![0.0; SAMPLE_RATE as usize]).is_none());
    }

    #[test]
    fn normalise_lifts_quiet_audio_to_target() {
        let mut pcm = vec![0.05f32; 100];
        let gain = normalise(&mut pcm);
        assert!((peak(&pcm) - 0.7).abs() < 1e-3, "expected peak near target");
        assert!(gain > 1.0, "quiet input should be amplified");
    }

    #[test]
    fn normalise_leaves_silence_alone() {
        let mut pcm = vec![0.0f32; 100];
        assert_eq!(normalise(&mut pcm), 1.0);
        assert_eq!(peak(&pcm), 0.0);
    }

    #[test]
    fn normalise_attenuates_clipping_input() {
        let mut pcm = vec![0.99f32; 100];
        normalise(&mut pcm);
        assert!(peak(&pcm) <= 0.71, "loud input should be scaled down");
    }

    #[test]
    fn trims_leading_and_trailing_silence() {
        let mut pcm = vec![0.0f32; SAMPLE_RATE as usize];
        // 200 ms of tone in the middle.
        let mid = pcm.len() / 2;
        for s in &mut pcm[mid..mid + SAMPLE_RATE as usize / 5] {
            *s = 0.5;
        }
        let trimmed = trim(&pcm).expect("should find speech");
        assert!(
            trimmed.len() < pcm.len(),
            "expected trimming to shorten the clip"
        );
        assert!(
            trimmed.len() > SAMPLE_RATE as usize / 5,
            "kept the speech plus hangover"
        );
    }
}
