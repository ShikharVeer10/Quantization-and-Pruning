"""Recover numeric grid records preserved in notebook cell outputs."""
from __future__ import annotations
import csv, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB = ROOT / "Comprehensive_Context_Quantization_and_Pruning.ipynb"
OUT = ROOT / "results.csv"
PAT = re.compile(
    r"\[(FP32|INT8)\] Sparsity: ([\d.]+)% \| Top-1: ([\d.]+)% \| "
    r"Top-5: ([\d.]+)% \| CosSim: ([\d.]+) \| Latency: ([\d.]+) ms"
    r"(?: \| Size: ([\d.]+) MB)?"
)
METHODS = {
    "Local L1": "Local L1",
    "Global L1": "Global L1",
    "Structured Ln": "Structured L2",
    "Random (Ablation)": "Random",
}


def main():
    notebook = json.loads(NB.read_text(encoding="utf-8"))
    output = "".join(notebook["cells"][8]["outputs"][0]["text"])
    method = None
    records = []
    for line in output.splitlines():
        m = re.search(r"Evaluating Pruning Method: (.*?) =+", line)
        if m:
            method = METHODS.get(m.group(1).strip(), m.group(1).strip())
            continue
        m = PAT.search(line)
        if not m or method is None:
            continue
        quant = "FP32" if m.group(1) == "FP32" else "Dynamic INT8"
        records.append({
            "regime": method,
            "sparsity_target_pct": m.group(2),
            "measured_sparsity_pct": m.group(2),
            "top1_pct": m.group(3),
            "top5_pct": m.group(4),
            "embedding_cosine": m.group(5),
            "latency_ms_original_protocol": m.group(6),
            "latency_p95_ms": "MISSING; saved output did not include P95",
            "size_mb": m.group(7) or "MISSING",
            "quantization": quant,
            "seed": "42",
            "provenance": "embedded notebook cell 8 output; not rerun",
        })

    records.insert(0, {
        "regime": "Dense baseline", "sparsity_target_pct": "0",
        "measured_sparsity_pct": "0", "top1_pct": "37.5", "top5_pct": "75.0",
        "embedding_cosine": "1.0000", "latency_ms_original_protocol": "221.35",
        "latency_p95_ms": "MISSING; saved output did not include P95",
        "size_mb": "255.57", "quantization": "FP32", "seed": "42",
        "provenance": "embedded notebook cell 7 output; not rerun",
    })
    for regime, quant in [
        ("Dense baseline", "FP16"),
        ("Dense baseline", "Dynamic INT8"),
        ("Dense baseline", "Simulated INT4"),
        ("Global L1 30%", "Simulated INT4"),
    ]:
        records.append({
            "regime": regime, "sparsity_target_pct": "0" if regime == "Dense baseline" else "30",
            "measured_sparsity_pct": "MISSING", "top1_pct": "MISSING", "top5_pct": "MISSING",
            "embedding_cosine": "MISSING", "latency_ms_original_protocol": "MISSING",
            "latency_p95_ms": "MISSING",
            "size_mb": "31.94 theoretical MiB" if quant == "Simulated INT4" and regime == "Dense baseline" else "MISSING", "quantization": quant, "seed": "42",
            "provenance": "requested configuration; output not preserved",
        })
    fields = list(records[0])
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Recovered {len(records)} rows ({sum(r['provenance'].startswith('embedded') for r in records)} with saved metrics): {OUT}")


if __name__ == "__main__":
    main()
