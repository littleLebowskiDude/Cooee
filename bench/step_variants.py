"""Where does an NPU decoder step's time go? Times the same step graph with
its per-step inputs varied: as built (fp32 cache + cross K/V inputs), with
the cross K/V baked in as constants (no 37 MB input), and with fp16 inputs.

    python bench/step_variants.py models/whisper-base.en-onnx [MAX]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).parent))
import npu_whisper as ref  # noqa: E402
import static_decoder as sd  # noqa: E402


def time_session(sess, feed, n=30):
    for _ in range(3):
        sess.run(None, feed)
    ms = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(None, feed)
        ms.append((time.perf_counter() - t) * 1000)
    return np.mean(ms), np.min(ms)


def bake_cross(model, cross_values):
    """Turn the cross.* inputs into initializers."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    keep = []
    for inp in m.graph.input:
        if inp.name.startswith("cross."):
            m.graph.initializer.append(numpy_helper.from_array(cross_values[inp.name], inp.name))
        else:
            keep.append(inp)
    del m.graph.input[:]
    m.graph.input.extend(keep)
    return m


def fp16_inputs(model, prefixes=("past.", "cross.")):
    """Declare the big inputs as float16 and cast to float32 inside the graph."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    casts = []
    for inp in m.graph.input:
        if inp.name.startswith(prefixes):
            inp.type.tensor_type.elem_type = TensorProto.FLOAT16
            orig = inp.name
            inp.name = orig + "_h"
            casts.append(helper.make_node("Cast", [orig + "_h"], [orig], to=TensorProto.FLOAT))
    nodes = casts + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    return m


def main():
    model_dir = Path(sys.argv[1])
    max_len = int(sys.argv[2]) if len(sys.argv) > 2 else 448
    cfg = json.loads((model_dir / "config.json").read_text())
    layers, d, heads = cfg["decoder_layers"], cfg["d_model"], cfg["decoder_attention_heads"]
    head_dim, vocab = d // heads, cfg["vocab_size"]
    W = sd.load_weights(onnx.load(str(model_dir / "onnx" / "decoder_model_merged.onnx")))
    step = sd.build_step(W, layers, d, heads, head_dim, max_len, vocab)
    L, H, D, M = layers, heads, head_dim, max_len
    rng = np.random.default_rng(0)
    cross = {f"cross.{l}.{n}": rng.standard_normal((1, H, sd.ENC_LEN, D)).astype(np.float32) for l in range(L) for n in ("key", "value")}
    past = {f"past.{l}.{n}": np.zeros((1, H, M, D), dtype=np.float32) for l in range(L) for n in ("key", "value")}
    mask = np.zeros((1, 1, 1, M + 1), dtype=np.float32)
    small = {"input_ids": np.array([50257], dtype=np.int32), "position": np.array([0], dtype=np.int32), "mask": mask}
    out_dir = model_dir / "onnx"
    dev = ref.npu_device()

    variants = [
        ("fp16 cache + cross inputs", fp16_inputs(step), {**small, **{k + "_h": v.astype(np.float16) for k, v in {**past, **cross}.items()}}),
        ("fp16 cache, cross baked", fp16_inputs(bake_cross(step, cross), ("past.",)), {**small, **{k + "_h": v.astype(np.float16) for k, v in past.items()}}),
    ]
    for label, model, feed in variants:
        path = out_dir / "_variant.onnx"
        onnx.save(model, str(path))
        mb = sum(v.nbytes for v in feed.values()) / 2**20
        sess, load, assigned = ref.npu_session(path, dev)
        mean, best = time_session(sess, feed)
        print(f"{label}: inputs {mb:.0f} MB/step, compile {load / 1000:.1f} s, EP {assigned}, step {mean:.1f} ms mean / {best:.1f} min")
        cpu, _ = ref.cpu_session(path)
        mean, best = time_session(cpu, feed)
        print(f"    same graph on CPU (4 thr): {mean:.1f} ms mean / {best:.1f} min")
        del sess
    path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
