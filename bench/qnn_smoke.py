"""Smoke test: can ONNX Runtime's QNN plugin EP run anything on the Hexagon NPU?

Builds a tiny fp32 matmul+relu graph in memory, runs it on the CPU EP and on
the QNN HTP backend, and compares the outputs. No downloads, no models.

    python bench/qnn_smoke.py
"""

import time

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_qnn
from onnx import TensorProto, helper


def build_model(n=512):
    rng = np.random.default_rng(0)
    w = rng.standard_normal((n, n), dtype=np.float32) * 0.05
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, n])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, n])
    w_init = helper.make_tensor("w", TensorProto.FLOAT, w.shape, w.flatten().tolist())
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["h"]),
        helper.make_node("Relu", ["h"], ["y"]),
    ]
    graph = helper.make_graph(nodes, "smoke", [x], [y], initializer=[w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    return model.SerializeToString()


def main():
    print("onnxruntime", ort.__version__, "| onnxruntime-qnn", onnxruntime_qnn.__version__,
          "| QNN SDK", onnxruntime_qnn.build_and_package_info.qnn_version)

    ort.register_execution_provider_library(
        onnxruntime_qnn.get_ep_name(), onnxruntime_qnn.get_library_path()
    )
    devices = ort.get_ep_devices()
    for d in devices:
        print(f"ep_device: ep={d.ep_name!r} vendor={d.ep_vendor!r} "
              f"type={d.device.type} vendor={d.device.vendor!r} id={d.device.device_id}")

    model = build_model()
    x = np.random.default_rng(1).standard_normal((1, 512), dtype=np.float32)

    cpu = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    ref = cpu.run(None, {"x": x})[0]

    qnn_devs = [d for d in devices if d.ep_name == onnxruntime_qnn.get_ep_name()]
    if not qnn_devs:
        raise SystemExit("QNN EP registered but exposes no devices")

    for dev in qnn_devs:
        so = ort.SessionOptions()
        so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
        so.add_provider_for_devices([dev], {"backend_type": "htp"})
        t = time.perf_counter()
        sess = ort.InferenceSession(model, so)
        print(f"\n[{dev.device.type}] session created in {(time.perf_counter()-t)*1000:.0f} ms; "
              f"providers={sess.get_providers()}")
        for sub in sess.get_provider_graph_assignment_info():
            ops = [n.op_type for n in sub.get_nodes()]
            print(f"  assigned to {sub.ep_name}: {len(ops)} nodes {ops}")
        out = sess.run(None, {"x": x})[0]
        t = time.perf_counter()
        for _ in range(50):
            sess.run(None, {"x": x})
        print(f"  50 runs: {(time.perf_counter()-t)*1000/50:.2f} ms each")
        err = np.abs(out - ref).max()
        print(f"  max abs diff vs CPU: {err:.4g}  (fp16 on HTP: expect ~1e-2)")


if __name__ == "__main__":
    main()
