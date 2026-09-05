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

/// How far before a window's limit the quietest cut is looked for.
const CUT_SEARCH_SECS: usize = 5;

/// Splits `pcm` into windows no longer than `max_len` samples, each cut at
/// the quietest 20 ms in the last [`CUT_SEARCH_SECS`] before the limit rather
/// than at the limit itself, so a word is never halved. A clip within the
/// limit comes back whole.
pub fn windows(pcm: &[f32], max_len: usize) -> Vec<&[f32]> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let search = CUT_SEARCH_SECS * SAMPLE_RATE as usize;
    while pcm.len() - start > max_len {
        let limit = start + max_len;
        let from = limit.saturating_sub(search).max(start + FRAME);
        let mut best = (f32::MAX, limit);
        let mut at = from;
        while at + FRAME <= limit {
            let e = rms(&pcm[at..at + FRAME]);
            if e < best.0 {
                best = (e, at + FRAME / 2);
            }
            at += FRAME;
        }
        out.push(&pcm[start..best.1]);
        start = best.1;
    }
    out.push(&pcm[start..]);
    out
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
    fn windows_cut_at_the_quietest_frame_before_the_limit() {
        let sr = SAMPLE_RATE as usize;
        // 50 s of "speech" with one 200 ms silence at 27 s.
        let mut pcm = vec![0.3f32; 50 * sr];
        for s in &mut pcm[27 * sr..27 * sr + sr / 5] {
            *s = 0.0;
        }
        let w = windows(&pcm, 30 * sr);
        assert_eq!(w.len(), 2);
        let cut = w[0].len();
        assert!(cut > 27 * sr && cut < 27 * sr + sr / 5, "cut inside the silence, got {cut}");
        assert_eq!(w[0].len() + w[1].len(), pcm.len());
    }

    #[test]
    fn windows_leave_a_short_clip_whole_and_cap_a_flat_one() {
        let sr = SAMPLE_RATE as usize;
        let short = vec![0.3f32; 10 * sr];
        assert_eq!(windows(&short, 30 * sr).len(), 1);
        let flat = vec![0.3f32; 65 * sr];
        let w = windows(&flat, 30 * sr);
        assert_eq!(w.len(), 3);
        assert!(w.iter().all(|x| x.len() <= 30 * sr));
        assert_eq!(w.iter().map(|x| x.len()).sum::<usize>(), flat.len());
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
