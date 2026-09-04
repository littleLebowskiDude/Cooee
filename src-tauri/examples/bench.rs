//! Measures real transcription latency and accuracy on this machine.
//!
//!   cargo run --release --features whisper --example bench -- <model.bin> <sample.wav>
//!
//! Uses a real speech WAV, not noise. That matters: whisper's decoder is
//! autoregressive, so on noise it hallucinates and rambles to the token limit,
//! which inflates timings by an order of magnitude and measures nothing useful.

#[cfg(feature = "whisper")]
fn main() -> anyhow::Result<()> {
    use cooee_lib::asr::{whisper_cpp::WhisperCpp, AsrEngine};

    let mut args = std::env::args().skip(1);
    let model = args.next().expect("usage: bench <model.bin> <sample.wav>");
    let wav = args.next().expect("usage: bench <model.bin> <sample.wav>");

    let pcm = read_wav_16k_mono(&wav)?;
    let seconds = pcm.len() as f32 / 16_000.0;
    println!("audio: {seconds:.1}s from {wav}");
    println!("threads: {}", std::thread::available_parallelism()?.get());

    // Sweep thread counts: on a contended machine more threads can be slower.
    let sweep: Vec<usize> = match args.next() {
        Some(list) => list.split(',').filter_map(|s| s.parse().ok()).collect(),
        None => vec![4],
    };

    for threads in sweep {
        print!("loading {model} (threads={threads}) ... ");
        let load = std::time::Instant::now();
        let engine = WhisperCpp::load_with_threads(std::path::Path::new(&model), Some(threads))?;
        println!("{} ms", load.elapsed().as_millis());

        for run in 1..=2 {
            let t = std::time::Instant::now();
            let out = engine.transcribe(&pcm)?;
            let ms = t.elapsed().as_millis().max(1);
            println!(
                "  threads={threads} run{run}: {ms} ms ({:.2}x realtime)",
                seconds * 1000.0 / ms as f32
            );
            if run == 1 {
                println!("    -> {}", out.text.trim());
            }
        }
    }
    Ok(())
}

/// Minimal 16-bit PCM WAV reader. Walks the chunk list rather than assuming a
/// 44-byte header, since writers routinely insert LIST/fact chunks.
#[cfg(feature = "whisper")]
fn read_wav_16k_mono(path: &str) -> anyhow::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    anyhow::ensure!(
        &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "not a WAV file"
    );

    let mut pos = 12;
    let mut channels = 1u16;
    let mut data: Option<&[u8]> = None;

    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into()?) as usize;
        let body = &bytes[pos + 8..(pos + 8 + size).min(bytes.len())];
        match id {
            b"fmt " => {
                channels = u16::from_le_bytes(body[2..4].try_into()?);
                let rate = u32::from_le_bytes(body[4..8].try_into()?);
                let bits = u16::from_le_bytes(body[14..16].try_into()?);
                anyhow::ensure!(rate == 16_000, "expected 16 kHz, got {rate}");
                anyhow::ensure!(bits == 16, "expected 16-bit, got {bits}");
            }
            b"data" => data = Some(body),
            _ => {}
        }
        pos += 8 + size + (size & 1); // chunks are word-aligned
    }

    let data = data.ok_or_else(|| anyhow::anyhow!("no data chunk"))?;
    let samples: Vec<f32> = data
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect();

    Ok(if channels == 1 {
        samples
    } else {
        samples
            .chunks(channels as usize)
            .map(|f| f.iter().sum::<f32>() / f.len() as f32)
            .collect()
    })
}

#[cfg(not(feature = "whisper"))]
fn main() {
    eprintln!("build with --features whisper");
}
