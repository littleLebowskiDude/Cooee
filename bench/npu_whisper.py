"""Feasibility: whisper base.en with the encoder on the Hexagon NPU via ONNX Runtime's QNN EP.

Uses the Optimum export from onnx-community/whisper-base.en. The encoder runs
on the NPU (HTP, fp16) and on the CPU EP for comparison; the decoder runs on
the CPU EP with a KV cache. Prints per-stage timings and the transcript.

    python bench/npu_whisper.py models/whisper-base.en-onnx models/sample.wav
    python bench/npu_whisper.py models/whisper-large-v3-turbo-onnx models/sample.wav encoder_model_fp16.onnx

Expected layout of the model dir:
    onnx/<encoder>.onnx  onnx/decoder_model_merged.onnx
    vocab.json  added_tokens.json  config.json  generation_config.json
Optional 3rd/4th args pick the encoder/decoder file names inside onnx/.
"""

import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_qnn
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed

ort.set_default_logger_severity(3)  # QNN fusion chatter is not useful here

SAMPLE_RATE = 16_000
N_FFT = 400
HOP = 160
N_MELS = 80
CHUNK = 30 * SAMPLE_RATE  # whisper always encodes a 30 s window


# --- audio ------------------------------------------------------------------

def read_wav_16k_mono(path):
    b = Path(path).read_bytes()
    assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", "not a WAV"
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(b):
        cid, size = b[pos:pos + 4], struct.unpack_from("<I", b, pos + 4)[0]
        body = b[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", body, 0)
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    channels, rate, bits = fmt[1], fmt[2], fmt[5]
    assert rate == SAMPLE_RATE and bits == 16, f"need 16 kHz 16-bit, got {rate} Hz {bits}-bit"
    pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm


def mel_filterbank(n_mels=N_MELS, sr=SAMPLE_RATE, n_fft=N_FFT):
    """librosa.filters.mel(sr, n_fft, n_mels) with slaney scale and norm — what whisper uses."""
    fmax = sr / 2

    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        mel = f / (200.0 / 3)
        min_log_hz, min_log_mel, logstep = 1000.0, 15.0, np.log(6.4) / 27.0
        return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-9) / min_log_hz) / logstep, mel)

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        f = (200.0 / 3) * m
        min_log_hz, min_log_mel, logstep = 1000.0, 15.0, np.log(6.4) / 27.0
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f)

    fft_freqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_pts = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_pts)
    ramps = mel_pts[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    fb = np.maximum(0, np.minimum(lower, upper))
    fb *= (2.0 / (mel_pts[2:n_mels + 2] - mel_pts[:n_mels]))[:, None]
    return fb.astype(np.float32)


def log_mel(pcm, fb):
    """whisper.audio.log_mel_spectrogram, numpy edition. Returns (n_mels, 3000)."""
    pcm = pcm[:CHUNK]
    pcm = np.pad(pcm, (0, CHUNK - len(pcm)))
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)  # periodic hann, like torch
    padded = np.pad(pcm, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (len(padded) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = padded[idx] * window
    spec = np.fft.rfft(frames, axis=1)
    mag = (spec.real ** 2 + spec.imag ** 2)[:-1].T  # drop last frame -> (201, 3000)
    mel = fb @ mag
    log = np.log10(np.maximum(mel, 1e-10))
    log = np.maximum(log, log.max() - 8.0)
    return ((log + 4.0) / 4.0).astype(np.float32)


# --- tokenizer (decode only) ----------------------------------------------------

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for c, b in zip(cs, bs)}


class Decoder:
    def __init__(self, model_dir):
        vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
        added = json.loads((model_dir / "added_tokens.json").read_text(encoding="utf-8"))
        self.id_to_tok = {v: k for k, v in vocab.items()}
        self.special = set(added.values())
        self.u2b = bytes_to_unicode()

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            if i in self.special:
                continue
            tok = self.id_to_tok.get(i)
            if tok is None:
                continue
            out.extend(self.u2b[c] for c in tok)
        return out.decode("utf-8", errors="replace")


# --- sessions -------------------------------------------------------------------

def static_encoder(enc_path, n_mels):
    """The QNN compiler needs fully static shapes. Pin the encoder's input dims
    (batch, n_mels, 3000), then let ORT's basic optimiser fold the Shape/Gather
    chains away so nothing dynamic reaches the NPU partitioner."""
    stem = enc_path.stem
    fixed = enc_path.with_name(f"{stem}_static.onnx")
    opt = enc_path.with_name(f"{stem}_static_opt.onnx")
    if opt.exists():
        return opt
    model = onnx.load(str(enc_path))
    inp = model.graph.input[0]
    for i, d in enumerate(inp.type.tensor_type.shape.dim):
        if d.dim_param:
            make_dim_param_fixed(model.graph, d.dim_param, (1, n_mels, 3000)[i])
    onnx.save(model, str(fixed))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(opt)
    try:
        ort.InferenceSession(str(fixed), so, providers=["CPUExecutionProvider"])
    except Exception as e:  # e.g. fp16 contrib ops with no CPU kernel on ARM64
        print(f"  (CPU optimise pass skipped: {str(e).splitlines()[0][:100]})")
        return fixed
    return opt


def npu_device():
    ort.register_execution_provider_library(onnxruntime_qnn.get_ep_name(), onnxruntime_qnn.get_library_path())
    for d in ort.get_ep_devices():
        if d.ep_name == onnxruntime_qnn.get_ep_name() and str(d.device.type).endswith("NPU"):
            return d
    raise SystemExit("no QNN NPU device")


def npu_session(path, dev, **qnn_opts):
    so = ort.SessionOptions()
    so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    so.add_provider_for_devices([dev], {"backend_type": "htp", **qnn_opts})
    t = time.perf_counter()
    sess = ort.InferenceSession(str(path), so)
    load_ms = (time.perf_counter() - t) * 1000
    assigned = {}
    for sub in sess.get_provider_graph_assignment_info():
        assigned[sub.ep_name] = assigned.get(sub.ep_name, 0) + len(list(sub.get_nodes()))
    return sess, load_ms, assigned


def cpu_session(path, threads=4):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    t = time.perf_counter()
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return sess, (time.perf_counter() - t) * 1000


def timed(fn, n=3):
    best = None
    for _ in range(n):
        t = time.perf_counter()
        out = fn()
        ms = (time.perf_counter() - t) * 1000
        best = ms if best is None else min(best, ms)
    return out, best


# --- decoder loop -------------------------------------------------------------------

def greedy_decode(dec, enc_out, sot_ids, eot, max_tokens=224, use_kv_cache=True):
    """Optimum merged decoder: no-cache branch on first call, cache branch after.
    With use_kv_cache=False the whole sequence is re-fed every step (slow, but
    a reference for checking the cache path)."""
    inputs = {i.name: i for i in dec.get_inputs()}
    out_names = [o.name for o in dec.get_outputs()]
    past_names = [n for n in inputs if n.startswith("past_key_values.")]
    # Shapes like [batch, heads, seq, head_dim]; seq is the dynamic axis.
    heads, head_dim = inputs[past_names[0]].shape[1], inputs[past_names[0]].shape[3]
    empty = {n: np.zeros((1, heads, 0, head_dim), dtype=np.float32) for n in past_names}

    tokens = list(sot_ids)
    past = dict(empty)
    use_cache = False
    while len(tokens) < max_tokens:
        if use_kv_cache and use_cache:
            feed_ids = np.array([[tokens[-1]]], dtype=np.int64)
        else:
            feed_ids = np.array([tokens], dtype=np.int64)
        feed = {"input_ids": feed_ids, "encoder_hidden_states": enc_out, **past}
        if "use_cache_branch" in inputs:
            feed["use_cache_branch"] = np.array([use_cache], dtype=bool)
        outs = dict(zip(out_names, dec.run(None, feed)))
        logits = outs["logits"][0, -1]
        next_id = int(logits.argmax())
        if use_kv_cache:
            for n in past_names:
                pres = "present." + n[len("past_key_values."):]
                # The cache branch returns the encoder K/V as empty (batch 0)
                # placeholders; keep the real ones from the first step.
                if pres in outs and outs[pres].size > 0:
                    past[n] = outs[pres]
            use_cache = True
        tokens.append(next_id)
        if next_id == eot:
            top = np.argsort(logits)[-5:][::-1]
            print(f"  (stopped: top-5 at EOT step {[(int(t), round(float(logits[t]), 1)) for t in top]})")
            break
    return tokens


# --- main --------------------------------------------------------------------------

def main():
    model_dir = Path(sys.argv[1])
    wav = sys.argv[2]
    enc_path = model_dir / "onnx" / (sys.argv[3] if len(sys.argv) > 3 else "encoder_model.onnx")
    dec_path = model_dir / "onnx" / (sys.argv[4] if len(sys.argv) > 4 else "decoder_model_merged.onnx")
    cfg = json.loads((model_dir / "config.json").read_text())
    n_mels = cfg.get("num_mel_bins", N_MELS)  # 80 for base/small, 128 for large-v3 family
    print(f"model: {model_dir.name}  encoder={enc_path.name}  decoder={dec_path.name}  n_mels={n_mels}")
    gen = json.loads((model_dir / "generation_config.json").read_text())
    eot = gen.get("eos_token_id", cfg.get("eos_token_id"))
    sot = gen.get("decoder_start_token_id", cfg.get("decoder_start_token_id"))
    forced = [t for _, t in gen.get("forced_decoder_ids", [])]  # e.g. <|notimestamps|>
    sot_ids = [sot] + forced
    tok = Decoder(model_dir)

    pcm = read_wav_16k_mono(wav)
    print(f"audio: {len(pcm) / SAMPLE_RATE:.1f}s from {wav}")
    t = time.perf_counter()
    feats = log_mel(pcm, mel_filterbank(n_mels))[None]  # (1, n_mels, 3000)
    print(f"log-mel: {(time.perf_counter() - t) * 1000:.0f} ms")

    # Encoder: CPU reference (best effort; fp16 exports may have no CPU kernels on ARM64)
    enc_in = onnx.load(str(enc_path), load_external_data=False).graph.input[0]
    in_name = enc_in.name
    if enc_in.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
        feats = feats.astype(np.float16)
    ref = None
    try:
        enc_cpu, load = cpu_session(enc_path)
        ref, ms = timed(lambda: enc_cpu.run(None, {in_name: feats})[0].astype(np.float32))
        print(f"\nencoder CPU (4 thr): load {load:.0f} ms, run {ms:.0f} ms")
    except Exception as e:
        print(f"\nencoder CPU: cannot run this export on the CPU EP ({str(e).splitlines()[0][:90]})")

    # Encoder: NPU, from a static-shape copy of the graph
    t = time.perf_counter()
    static_path = static_encoder(enc_path, n_mels)
    print(f"static encoder prepared in {(time.perf_counter() - t) * 1000:.0f} ms -> {static_path.name}")
    dev = npu_device()
    enc_npu, load, assigned = npu_session(static_path, dev)
    print(f"encoder NPU: load/compile {load:.0f} ms, nodes by EP {assigned}")
    out, ms = timed(lambda: enc_npu.run(None, {in_name: feats})[0].astype(np.float32))
    if ref is not None:
        err = np.abs(out - ref).max()
        rel = err / (np.abs(ref).max() + 1e-9)
        mean = np.abs(out - ref).mean()
        cos = float((out.ravel() @ ref.ravel()) / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-9))
        print(f"encoder NPU: run {ms:.0f} ms; vs CPU max abs diff {err:.3g} (rel {rel:.2%}), "
              f"mean abs diff {mean:.3g}, cosine {cos:.5f}")
    else:
        print(f"encoder NPU: run {ms:.0f} ms")

    # Decoder on CPU, fed by each encoder output
    dec, load = cpu_session(dec_path)
    print(f"\ndecoder CPU: load {load:.0f} ms")
    runs = [("NPU encoder", out, True), ("NPU encoder, no KV cache", out, False)]
    if ref is not None:
        runs.insert(0, ("CPU encoder", ref, True))
    for label, enc_out, kv in runs:
        t = time.perf_counter()
        ids = greedy_decode(dec, enc_out, sot_ids, eot, use_kv_cache=kv)
        ms = (time.perf_counter() - t) * 1000
        n = len(ids) - len(sot_ids)
        print(f"decode from {label}: {n} tokens in {ms:.0f} ms ({ms / max(n, 1):.1f} ms/token)")
        print(f"  ids: {ids[:12]}{' ...' if len(ids) > 12 else ''}")
        print(f"  -> {tok.decode(ids).strip()}")


if __name__ == "__main__":
    main()
