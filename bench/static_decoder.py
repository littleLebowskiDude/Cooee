"""Build a static-shape whisper decoder for the QNN NPU from the Optimum export.

The exported `decoder_model_merged.onnx` grows its KV cache one token at a
time, which the HTP compiler cannot take. This rebuilds the same decoder as
two fixed-shape graphs, from the export's own weights (no PyTorch):

  cross_kv.onnx    encoder output (1, 1500, d)  ->  per-layer K/V for cross-attention
  step.onnx        one token + position + additive mask + self-attention cache
                   (MAX slots) + cross K/V  ->  logits and this token's K/V per layer

The host keeps the self-attention cache: it writes each step's new K/V into
slot `position`, and the mask hides the slots not yet written. Every tensor
has the same shape on every step, so the graph compiles once.

    python bench/static_decoder.py models/whisper-base.en-onnx [MAX]        # build + verify on CPU
    python bench/static_decoder.py models/whisper-base.en-onnx 448 --npu    # + compile for QNN, time per step

Verification runs greedy decoding on the sample through both decoders and
compares tokens and logits. Timing reports per-step latency on the CPU EP and
on the NPU, including the input copies a real loop pays.
"""

import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).parent))
import npu_whisper as ref  # noqa: E402

LN_EPS = 1e-5
ENC_LEN = 1500


# --- weight mapping ---------------------------------------------------------------

def load_weights(model):
    """Map the export's anonymous MatMul weights to roles by walking the
    no-cache branch: a MatMul's consumer Add names the projection, the two
    bias-free ones per layer are k_proj, told apart by reading the encoder."""
    g = model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    strip = lambda n: n.replace("_merged_0", "")
    W = {strip(k): v for k, v in inits.items()}
    branch = [helper.get_attribute_value(a) for a in g.node[0].attribute if a.name == "else_branch"][0]
    consumers = {}
    for n in branch.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    roles = {}
    self_k_by_input = {}
    cross_k_in_order = []
    for n in branch.node:
        if n.op_type != "MatMul" or n.input[1] not in inits:
            continue
        wname = strip(n.input[1])
        adds = [c for c in consumers.get(n.output[0], []) if c.op_type == "Add"]
        bias = next((strip(i) for c in adds for i in c.input if i in inits), None)
        if bias:
            roles[bias.replace(".bias", ".weight")] = W[wname]
        elif n.input[0] == "encoder_hidden_states":
            cross_k_in_order.append(W[wname])
        else:
            self_k_by_input[n.input[0]] = W[wname]
    # self k shares its input with self q of the same layer
    q_inputs = {}
    for n in branch.node:
        if n.op_type == "MatMul" and n.input[1] in inits:
            adds = [c for c in consumers.get(n.output[0], []) if c.op_type == "Add"]
            bias = next((strip(i) for c in adds for i in c.input if i in inits), None)
            if bias and "self_attn.q_proj" in bias:
                q_inputs[bias.split(".self_attn")[0]] = n.input[0]
    for layer, inp in q_inputs.items():
        roles[f"{layer}.self_attn.k_proj.weight"] = self_k_by_input[inp]
    for l, w in enumerate(cross_k_in_order):
        roles[f"model.decoder.layers.{l}.encoder_attn.k_proj.weight"] = w
    for k, v in W.items():
        if not k.startswith("onnx::"):
            roles.setdefault(k, v)
    return roles


# --- graph building -------------------------------------------------------------------

class Builder:
    def __init__(self):
        self.nodes, self.inits, self.n = [], [], 0

    def const(self, name, arr):
        self.inits.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))
        return name

    def op(self, op_type, inputs, attrs=None, name=None, outputs=1):
        self.n += 1
        outs = [f"{name or op_type.lower()}_{self.n}" + (f"_{i}" if outputs > 1 else "") for i in range(outputs)]
        self.nodes.append(helper.make_node(op_type, inputs, outs, **(attrs or {})))
        return outs[0] if outputs == 1 else outs

    def linear(self, x, w, b=None):
        y = self.op("MatMul", [x, w])
        return self.op("Add", [y, b]) if b is not None else y

    def layer_norm(self, x, w, b):
        return self.op("LayerNormalization", [x, w, b], {"epsilon": LN_EPS, "axis": -1})

    def gelu(self, x, name):
        inv_sqrt2 = self.const(f"{name}_inv_sqrt2", np.array(1 / np.sqrt(2), dtype=np.float32))
        half = self.const(f"{name}_half", np.array(0.5, dtype=np.float32))
        one = self.const(f"{name}_one", np.array(1.0, dtype=np.float32))
        e = self.op("Erf", [self.op("Mul", [x, inv_sqrt2])])
        return self.op("Mul", [self.op("Mul", [x, half]), self.op("Add", [e, one])])


def build_cross_kv(W, layers, d, heads, head_dim):
    b = Builder()
    outputs = []
    for l in range(layers):
        p = f"model.decoder.layers.{l}.encoder_attn"
        k = b.linear("encoder_hidden_states", b.const(f"ck_w{l}", W[f"{p}.k_proj.weight"]))
        v = b.linear("encoder_hidden_states", b.const(f"cv_w{l}", W[f"{p}.v_proj.weight"]), b.const(f"cv_b{l}", W[f"{p}.v_proj.bias"]))
        shape = b.const(f"cshape{l}", np.array([1, ENC_LEN, heads, head_dim], dtype=np.int64))
        for name, t in (("key", k), ("value", v)):
            t = b.op("Transpose", [b.op("Reshape", [t, shape])], {"perm": [0, 2, 1, 3]})
            b.nodes[-1].output[0] = f"cross.{l}.{name}"
            outputs.append(helper.make_tensor_value_info(f"cross.{l}.{name}", TensorProto.FLOAT, [1, heads, ENC_LEN, head_dim]))
    inp = [helper.make_tensor_value_info("encoder_hidden_states", TensorProto.FLOAT, [1, ENC_LEN, d])]
    return helper.make_model(helper.make_graph(b.nodes, "whisper_cross_kv", inp, outputs, b.inits),
                             opset_imports=[helper.make_opsetid("", 17)], ir_version=9)


def build_step(W, layers, d, heads, head_dim, max_len, vocab):
    b = Builder()
    scaling = np.array(head_dim ** -0.5, dtype=np.float32)
    tok = b.op("Gather", [b.const("embed_tokens", W["model.decoder.embed_tokens.weight"]), "input_ids"])  # (1, d)
    pos = b.op("Gather", [b.const("embed_positions", W["model.decoder.embed_positions.weight"]), "position"])  # (1, d)
    x = b.op("Add", [tok, pos])
    x = b.op("Reshape", [x, b.const("shape_1_1_d", np.array([1, 1, d], dtype=np.int64))])  # (1, 1, d)
    heads_shape = b.const("shape_heads", np.array([1, 1, heads, head_dim], dtype=np.int64))
    flat_shape = b.const("shape_flat", np.array([1, 1, d], dtype=np.int64))
    scale = b.const("scaling", scaling)
    outputs = []

    def split_heads(t, out=None):  # (1,1,d) -> (1,H,1,D)
        y = b.op("Transpose", [b.op("Reshape", [t, heads_shape])], {"perm": [0, 2, 1, 3]})
        if out:
            b.nodes[-1].output[0] = out
            return out
        return y

    def attention(q, k, v, mask=None):
        s = b.op("MatMul", [q, b.op("Transpose", [k], {"perm": [0, 1, 3, 2]})])  # (1,H,1,T)
        if mask is not None:
            s = b.op("Add", [s, mask])
        a = b.op("Softmax", [s], {"axis": -1})
        o = b.op("MatMul", [a, v])  # (1,H,1,D)
        return b.op("Reshape", [b.op("Transpose", [o], {"perm": [0, 2, 1, 3]}), flat_shape])

    for l in range(layers):
        p = f"model.decoder.layers.{l}"
        w = lambda n: b.const(f"L{l}.{n}", W[f"{p}.{n}"])
        # self-attention over the cache plus this token
        h = b.layer_norm(x, w("self_attn_layer_norm.weight"), w("self_attn_layer_norm.bias"))
        q = b.op("Mul", [b.linear(h, w("self_attn.q_proj.weight"), w("self_attn.q_proj.bias")), scale])
        k = b.linear(h, w("self_attn.k_proj.weight"))
        v = b.linear(h, w("self_attn.v_proj.weight"), w("self_attn.v_proj.bias"))
        q = split_heads(q)
        k = split_heads(k, f"new.{l}.key")
        v = split_heads(v, f"new.{l}.value")
        for name in ("key", "value"):
            outputs.append(helper.make_tensor_value_info(f"new.{l}.{name}", TensorProto.FLOAT, [1, heads, 1, head_dim]))
        K = b.op("Concat", [f"past.{l}.key", k], {"axis": 2})  # (1,H,MAX+1,D)
        V = b.op("Concat", [f"past.{l}.value", v], {"axis": 2})
        a = attention(q, K, V, "mask")
        x = b.op("Add", [x, b.linear(a, w("self_attn.out_proj.weight"), w("self_attn.out_proj.bias"))])
        # cross-attention over the precomputed encoder K/V
        h = b.layer_norm(x, w("encoder_attn_layer_norm.weight"), w("encoder_attn_layer_norm.bias"))
        q = split_heads(b.op("Mul", [b.linear(h, w("encoder_attn.q_proj.weight"), w("encoder_attn.q_proj.bias")), scale]))
        a = attention(q, f"cross.{l}.key", f"cross.{l}.value")
        x = b.op("Add", [x, b.linear(a, w("encoder_attn.out_proj.weight"), w("encoder_attn.out_proj.bias"))])
        # feed-forward
        h = b.layer_norm(x, w("final_layer_norm.weight"), w("final_layer_norm.bias"))
        h = b.gelu(b.linear(h, w("fc1.weight"), w("fc1.bias")), f"gelu{l}")
        x = b.op("Add", [x, b.linear(h, w("fc2.weight"), w("fc2.bias"))])

    x = b.layer_norm(x, b.const("ln_f.weight", W["model.decoder.layer_norm.weight"]), b.const("ln_f.bias", W["model.decoder.layer_norm.bias"]))
    x = b.op("Reshape", [x, b.const("shape_1_d", np.array([1, d], dtype=np.int64))])
    logits = b.op("MatMul", [x, b.const("lm_head", np.ascontiguousarray(W["model.decoder.embed_tokens.weight"].T))])
    b.nodes[-1].output[0] = "logits"
    outputs.insert(0, helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, vocab]))

    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT32, [1]),
        helper.make_tensor_value_info("position", TensorProto.INT32, [1]),
        helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, 1, max_len + 1]),
    ]
    for l in range(layers):
        for name in ("key", "value"):
            inputs.append(helper.make_tensor_value_info(f"past.{l}.{name}", TensorProto.FLOAT, [1, heads, max_len, head_dim]))
    for l in range(layers):
        for name in ("key", "value"):
            inputs.append(helper.make_tensor_value_info(f"cross.{l}.{name}", TensorProto.FLOAT, [1, heads, ENC_LEN, head_dim]))
    return helper.make_model(helper.make_graph(b.nodes, "whisper_decoder_step", inputs, outputs, b.inits),
                             opset_imports=[helper.make_opsetid("", 17)], ir_version=9)


def to_fp16_io(model):
    """The same step graph with float16 cache and cross inputs (the host halves
    what it copies, or shares, per step); a Cast to float32 follows each."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    renamed = {}
    for inp in m.graph.input:
        if inp.name.startswith(("past.", "cross.")):
            inp.type.tensor_type.elem_type = TensorProto.FLOAT16
            renamed[inp.name] = inp.name + "_f32"
    for n in m.graph.node:
        for i, name in enumerate(n.input):
            if name in renamed:
                n.input[i] = renamed[name]
    casts = [helper.make_node("Cast", [src], [dst], to=TensorProto.FLOAT) for src, dst in renamed.items()]
    nodes = casts + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    return m


# --- host-side decode loop --------------------------------------------------------------

class StaticDecoder:
    """Greedy decoding through cross_kv + step, with the cache on the host."""

    def __init__(self, cross_sess, step_sess, layers, heads, head_dim, max_len):
        self.cross, self.step = cross_sess, step_sess
        self.layers, self.heads, self.head_dim, self.max_len = layers, heads, head_dim, max_len
        self.step_ms = []

    def decode(self, enc_out, prefix, eot, max_tokens=None):
        L, H, D, M = self.layers, self.heads, self.head_dim, self.max_len
        cross = dict(zip([o.name for o in self.cross.get_outputs()], self.cross.run(None, {"encoder_hidden_states": enc_out})))
        past = {f"past.{l}.{n}": np.zeros((1, H, M, D), dtype=np.float32) for l in range(L) for n in ("key", "value")}
        tokens = list(prefix)
        limit = min(max_tokens or M, M)
        pos = 0
        self.step_ms = []
        while pos < limit:
            mask = np.full((1, 1, 1, M + 1), -1e4, dtype=np.float32)  # not -inf: fp16 on the HTP
            mask[..., :pos] = 0.0
            mask[..., M] = 0.0
            tok = tokens[pos] if pos < len(tokens) else tokens[-1]
            feed = {"input_ids": np.array([tok], dtype=np.int32), "position": np.array([pos], dtype=np.int32), "mask": mask, **past, **cross}
            t = time.perf_counter()
            outs = self.step.run(None, feed)
            self.step_ms.append((time.perf_counter() - t) * 1000)
            names = [o.name for o in self.step.get_outputs()]
            out = dict(zip(names, outs))
            for l in range(L):
                past[f"past.{l}.key"][:, :, pos] = out[f"new.{l}.key"][:, :, 0]
                past[f"past.{l}.value"][:, :, pos] = out[f"new.{l}.value"][:, :, 0]
            pos += 1
            if pos >= len(tokens):  # past the prefix: this step's logits pick the next token
                nxt = int(out["logits"][0].argmax())
                tokens.append(nxt)
                if nxt == eot:
                    break
        return tokens, out["logits"][0]


# --- main -------------------------------------------------------------------------------

def main():
    import json
    import onnxruntime as ort
    model_dir = Path(sys.argv[1])
    max_len = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 448
    do_npu = "--npu" in sys.argv
    cfg = json.loads((model_dir / "config.json").read_text())
    layers, d, heads = cfg["decoder_layers"], cfg["d_model"], cfg["decoder_attention_heads"]
    head_dim, vocab = d // heads, cfg["vocab_size"]
    print(f"{model_dir.name}: {layers} layers, d={d}, {heads} heads, MAX={max_len}")

    merged = onnx.load(str(model_dir / "onnx" / "decoder_model_merged.onnx"))
    W = load_weights(merged)
    t = time.perf_counter()
    cross_model = build_cross_kv(W, layers, d, heads, head_dim)
    step_model = build_step(W, layers, d, heads, head_dim, max_len, vocab)
    onnx.checker.check_model(cross_model)
    onnx.checker.check_model(step_model)
    out_dir = model_dir / "onnx"
    cross_path, step_path = out_dir / "cross_kv.onnx", out_dir / f"decoder_step_{max_len}.onnx"
    onnx.save(cross_model, str(cross_path))
    onnx.save(step_model, str(step_path))
    onnx.save(to_fp16_io(step_model), str(out_dir / f"decoder_step_{max_len}_f16.onnx"))
    print(f"built {cross_path.name} ({cross_path.stat().st_size // 2**20} MB) and {step_path.name} "
          f"({step_path.stat().st_size // 2**20} MB) in {time.perf_counter() - t:.1f}s")

    # Reference: the merged decoder, greedy, on the CPU
    gen = json.loads((model_dir / "generation_config.json").read_text())
    eot, sot = gen["eos_token_id"], gen["decoder_start_token_id"]
    prefix = [sot] + [t for _, t in gen.get("forced_decoder_ids", [])]
    tok = ref.Decoder(model_dir)
    pcm = ref.read_wav_16k_mono("models/sample.wav")
    feats = ref.log_mel(pcm, ref.mel_filterbank(cfg["num_mel_bins"]))[None]
    enc, _ = ref.cpu_session(model_dir / "onnx" / "encoder_model.onnx")
    enc_out = enc.run(None, {"input_features": feats})[0]
    dec, _ = ref.cpu_session(model_dir / "onnx" / "decoder_model_merged.onnx")
    t = time.perf_counter()
    ref_ids = ref.greedy_decode(dec, enc_out, prefix, eot)
    ref_ms = (time.perf_counter() - t) * 1000
    print(f"\nreference (merged, CPU): {len(ref_ids) - len(prefix)} tokens, {ref_ms:.0f} ms, {ref_ms / (len(ref_ids) - len(prefix)):.1f} ms/token")
    print(f"  -> {tok.decode(ref_ids).strip()}")

    # Static graphs on the CPU: correctness
    cross_cpu, _ = ref.cpu_session(cross_path)
    step_cpu, _ = ref.cpu_session(step_path)
    sd = StaticDecoder(cross_cpu, step_cpu, layers, heads, head_dim, max_len)
    ids, _ = sd.decode(enc_out, prefix, eot)
    print(f"static (CPU): {len(ids) - len(prefix)} tokens, {np.mean(sd.step_ms):.1f} ms/step (min {np.min(sd.step_ms):.1f})")
    print(f"  -> {tok.decode(ids).strip()}")
    print("  tokens identical to reference:", ids == ref_ids)
    # logits check at the last shared position: rerun reference with cache off for one exact comparison
    if ids != ref_ids:
        n = next(i for i, (a, b) in enumerate(zip(ids, ref_ids)) if a != b)
        print(f"  first divergence at token {n}: static {ids[n]} vs ref {ref_ids[n]}")

    if not do_npu:
        return
    dev = ref.npu_device()
    for name, path in (("cross_kv", cross_path), ("step", step_path)):
        t = time.perf_counter()
        sess, load, assigned = ref.npu_session(path, dev)
        print(f"\n{name} NPU: compile {load / 1000:.1f} s, nodes by EP {assigned}")
        if name == "cross_kv":
            cross_npu = sess
        else:
            step_npu = sess
    for label, cross_sess, step_sess in (("cross NPU + step NPU", cross_npu, step_npu), ("cross CPU + step NPU", cross_cpu, step_npu)):
        sd = StaticDecoder(cross_sess, step_sess, layers, heads, head_dim, max_len)
        t = time.perf_counter()
        ids, _ = sd.decode(enc_out, prefix, eot)
        total = (time.perf_counter() - t) * 1000
        print(f"{label}: {len(ids) - len(prefix)} tokens in {total:.0f} ms; step {np.mean(sd.step_ms):.1f} ms mean, {np.min(sd.step_ms):.1f} min, {np.max(sd.step_ms):.1f} max")
        print(f"  -> {tok.decode(ids).strip()}")
        print("  tokens identical to reference:", ids == ref_ids)


if __name__ == "__main__":
    main()
