"""Benchmark Microsoft Phi-4-mini against Cooee's rule-based polish pass.

    python bench/phi_bench.py models/phi/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4

Runs the model directly via onnxruntime-genai, which needs no MSIX packaging, no
package identity, and no Limited Access Feature entitlement — the three gates
that block the built-in Phi Silica on this machine.
"""

import json
import pathlib
import sys
import time

import onnxruntime_genai as og

# Identical to the C# harness. Deliberately conservative: for dictation, a model
# that "improves" your wording is worse than one that only cleans it up.
INSTRUCTION = (
    "Clean up this dictated text so it reads as if it were typed. "
    "Remove filler words and stutters, fix punctuation and capitalisation, "
    "and split run-on sentences. Do NOT add, remove, or reword any content, "
    "and do not answer or comment on it. Return only the cleaned text."
)

REPO = pathlib.Path(__file__).resolve().parent.parent


def build_prompt(raw: str) -> str:
    """Phi-4 chat template."""
    return (
        f"<|system|>{INSTRUCTION}<|end|>"
        f"<|user|>{raw}<|end|>"
        f"<|assistant|>"
    )


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "models" / "phi")
    corpus_path = sys.argv[2] if len(sys.argv) > 2 else str(REPO / "bench" / "transcripts.json")

    print(f"model: {model_dir}")
    t0 = time.perf_counter()
    model = og.Model(model_dir)
    tokenizer = og.Tokenizer(model)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"loaded in {load_ms:.0f} ms\n")

    cases = json.loads(pathlib.Path(corpus_path).read_text(encoding="utf-8"))
    results = []

    for case in cases:
        prompt = build_prompt(case["raw"])
        tokens = tokenizer.encode(prompt)

        times, output = [], ""
        # Warm-up excluded, then two measured runs.
        for run in range(3):
            params = og.GeneratorParams(model)
            # Greedy: dictation cleanup should be deterministic, not creative.
            params.set_search_options(do_sample=False, max_length=len(tokens) + 256)
            generator = og.Generator(model, params)
            generator.append_tokens(tokens)

            start = time.perf_counter()
            produced = []
            while not generator.is_done():
                generator.generate_next_token()
                produced.append(generator.get_next_tokens()[0])
            elapsed = (time.perf_counter() - start) * 1000

            if run > 0:
                times.append(elapsed)
            output = tokenizer.decode(produced).strip()
            del generator

        n_out = len(produced)
        mean = sum(times) / len(times)
        print(f"[{case['id']}] {times[0]:.0f} / {times[1]:.0f} ms  ({n_out} tokens, {n_out / (mean / 1000):.1f} tok/s)")
        print(f"  in : {case['raw']}")
        print(f"  out: {output}\n")
        results.append({
            "id": case["id"],
            "raw": case["raw"],
            "output": output,
            "ms": times,
            "tokens_out": n_out,
        })

    out_path = REPO / "bench" / "phi-results.json"
    out_path.write_text(json.dumps({"load_ms": load_ms, "results": results}, indent=2), encoding="utf-8")
    mean_all = sum(sum(r["ms"]) / len(r["ms"]) for r in results) / len(results)
    print(f"{'-' * 70}")
    print(f"Phi-4-mini: {mean_all:.0f} ms mean per transcript")
    print(f"rules     : 0.003 ms mean  ({mean_all / 0.003:.0f}x faster)")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
