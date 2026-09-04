"""Writes the log-mel test fixture for src-tauri/src/asr/onnx/mel.rs.

One second of two tones (0.5 sin 440 Hz + 0.25 sin 3 kHz), through the same
numpy log_mel as npu_whisper.py, first 100 frames, as little-endian f32.

    python bench/mel_fixture.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from npu_whisper import log_mel, mel_filterbank  # noqa: E402

SR = 16_000
t = np.arange(SR, dtype=np.float32) / SR
pcm = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
mel = log_mel(pcm, mel_filterbank(80))[:, :100]
out = Path(__file__).parent.parent / "src-tauri" / "src" / "asr" / "onnx" / "fixtures" / "mel_two_tones_100.f32"
out.write_bytes(mel.astype("<f4").tobytes())
print(f"{out}: {mel.shape} min {mel.min():.3f} max {mel.max():.3f}")
