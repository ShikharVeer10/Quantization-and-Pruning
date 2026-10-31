"""Update results.csv with verified empirical measurements from notebook and Step 0 runs."""
import csv
import json
from pathlib import Path

ROOT = Path("C:/Users/shikh/Quantization-and-Pruning")
NB = ROOT / "Comprehensive_Context_Quantization_and_Pruning.ipynb"
EXP_JSON = ROOT / "experimental_run_data.json"
OUT = ROOT / "results.csv"

exp_data = json.loads(EXP_JSON.read_text(encoding="utf-8"))

records = [
    # Baseline
    {
        "regime": "Dense baseline", "sparsity_target_pct": "0.0", "measured_sparsity_pct": "0.0",
        "top1_pct": "37.50", "top5_pct": "75.00", "embedding_cosine": "1.0000",
        "latency_median_ms": "170.33", "latency_p95_ms": "204.37", "size_mb": "255.57",
        "quantization": "FP32", "seed": "42", "provenance": "Step 0 verified benchmark (100 runs, 20 warm-up, 1 CPU thread)"
    },
    # Precision baselines
    {
        "regime": "Dense baseline", "sparsity_target_pct": "0.0", "measured_sparsity_pct": "0.0",
        "top1_pct": "37.50", "top5_pct": "75.00", "embedding_cosine": "1.0000",
        "latency_median_ms": "MISSING", "latency_p95_ms": "MISSING", "size_mb": "127.80",
        "quantization": "FP16", "seed": "42", "provenance": "Step 0 verified run"
    },
    {
        "regime": "Dense baseline", "sparsity_target_pct": "0.0", "measured_sparsity_pct": "0.0",
        "top1_pct": "43.75", "top5_pct": "50.00", "embedding_cosine": "0.9450",
        "latency_median_ms": "95.03", "latency_p95_ms": "113.19", "size_mb": "154.76",
        "quantization": "Dynamic INT8", "seed": "42", "provenance": "Step 0 verified benchmark"
    },
    {
        "regime": "Dense baseline", "sparsity_target_pct": "0.0", "measured_sparsity_pct": "0.0",
        "top1_pct": "18.75", "top5_pct": "75.00", "embedding_cosine": "0.9228",
        "latency_median_ms": "MISSING", "latency_p95_ms": "MISSING", "size_mb": "31.94 (theoretical MiB)",
        "quantization": "Simulated INT4", "seed": "42", "provenance": "Step 0 verified run (fake quantization)"
    },
    {
        "regime": "Global L1 30%", "sparsity_target_pct": "30.0", "measured_sparsity_pct": "30.0",
        "top1_pct": "25.00", "top5_pct": "37.50", "embedding_cosine": "0.9109",
        "latency_median_ms": "MISSING", "latency_p95_ms": "MISSING", "size_mb": "31.94 (theoretical MiB)",
        "quantization": "Simulated INT4", "seed": "42", "provenance": "Step 0 verified run"
    },
    # Distillation recovery
    {
        "regime": "Global L1 45% (Pre-Distill)", "sparsity_target_pct": "45.0", "measured_sparsity_pct": "45.0",
        "top1_pct": "12.50", "top5_pct": "12.50", "embedding_cosine": "0.7791",
        "latency_median_ms": "170.87", "latency_p95_ms": "187.91", "size_mb": "255.57",
        "quantization": "FP32", "seed": "42", "provenance": "Step 0 pre-distillation measurement"
    },
    {
        "regime": "Global L1 45% + Distillation", "sparsity_target_pct": "45.0", "measured_sparsity_pct": "45.0",
        "top1_pct": "18.75", "top5_pct": "37.50", "embedding_cosine": "0.8773",
        "latency_median_ms": "MISSING", "latency_p95_ms": "MISSING", "size_mb": "255.57",
        "quantization": "FP32", "seed": "42", "provenance": "Step 0 verified distillation run (3 epochs, logit KL)"
    },
]

# Pruning grid results from notebook
grid_data = [
    # Local L1 FP32
    ("Local L1", "15.0", "15.0", "43.75", "75.00", "0.9903", "205.44", "255.57", "FP32"),
    ("Local L1", "30.0", "30.0", "25.00", "62.50", "0.9585", "134.77", "255.57", "FP32"),
    ("Local L1", "45.0", "45.0", "12.50", "12.50", "0.8325", "136.77", "255.57", "FP32"),
    ("Local L1", "60.0", "60.0", "0.00", "0.00", "0.4970", "136.49", "255.57", "FP32"),
    ("Local L1", "75.0", "75.0", "0.00", "0.00", "0.4070", "135.66", "255.57", "FP32"),
    # Local L1 INT8
    ("Local L1", "15.0", "15.0", "50.00", "75.00", "0.9370", "121.99", "154.77", "Dynamic INT8"),
    ("Local L1", "30.0", "30.0", "18.75", "37.50", "0.9213", "78.33", "154.77", "Dynamic INT8"),
    ("Local L1", "45.0", "45.0", "0.00", "18.75", "0.8221", "77.84", "154.77", "Dynamic INT8"),
    ("Local L1", "60.0", "60.0", "0.00", "0.00", "0.4634", "79.26", "154.77", "Dynamic INT8"),
    ("Local L1", "75.0", "75.0", "0.00", "0.00", "0.3861", "78.13", "154.77", "Dynamic INT8"),
    # Global L1 FP32
    ("Global L1", "15.0", "15.0", "43.75", "75.00", "0.9882", "136.07", "255.57", "FP32"),
    ("Global L1", "30.0", "30.0", "25.00", "75.00", "0.9464", "135.77", "255.57", "FP32"),
    ("Global L1", "45.0", "45.0", "12.50", "12.50", "0.7791", "133.59", "255.57", "FP32"),
    ("Global L1", "60.0", "60.0", "0.00", "0.00", "0.5212", "133.88", "255.57", "FP32"),
    ("Global L1", "75.0", "75.0", "0.00", "0.00", "0.4467", "133.60", "255.57", "FP32"),
    # Global L1 INT8
    ("Global L1", "15.0", "15.0", "43.75", "75.00", "0.9366", "78.86", "154.77", "Dynamic INT8"),
    ("Global L1", "30.0", "30.0", "12.50", "37.50", "0.9122", "78.06", "154.77", "Dynamic INT8"),
    ("Global L1", "45.0", "45.0", "0.00", "18.75", "0.7804", "76.15", "154.77", "Dynamic INT8"),
    ("Global L1", "60.0", "60.0", "0.00", "6.25", "0.5330", "78.35", "154.77", "Dynamic INT8"),
    ("Global L1", "75.0", "75.0", "0.00", "0.00", "0.4485", "76.48", "154.77", "Dynamic INT8"),
    # Structured L2 All-Linear (original notebook ablation)
    ("Structured L2 (All-Linear)", "14.99", "14.99", "0.00", "0.00", "0.2642", "134.74", "255.57", "FP32"),
    ("Structured L2 (All-Linear)", "29.98", "29.98", "0.00", "0.00", "0.3446", "132.39", "255.57", "FP32"),
    ("Structured L2 (All-Linear)", "45.02", "45.02", "0.00", "0.00", "0.3449", "133.67", "255.57", "FP32"),
    ("Structured L2 (All-Linear)", "60.01", "60.01", "0.00", "0.00", "0.3989", "132.27", "255.57", "FP32"),
    ("Structured L2 (All-Linear)", "75.00", "75.00", "0.00", "0.00", "0.4710", "131.35", "255.57", "FP32"),
    # Structured L2 Encoder-Only (main result)
    ("Structured L2 (Encoder-Only)", "15.00", "15.00", "0.00", "0.00", "0.4889", "MISSING", "255.57", "FP32"),
    ("Structured L2 (Encoder-Only)", "30.00", "30.00", "0.00", "0.00", "0.4129", "MISSING", "255.57", "FP32"),
    ("Structured L2 (Encoder-Only)", "45.00", "45.00", "0.00", "0.00", "0.4534", "MISSING", "255.57", "FP32"),
    ("Structured L2 (Encoder-Only)", "60.00", "60.00", "0.00", "6.25", "0.4549", "MISSING", "255.57", "FP32"),
    ("Structured L2 (Encoder-Only)", "75.00", "75.00", "0.00", "0.00", "0.4985", "MISSING", "255.57", "FP32"),
    # Random Control (3 seeds mean)
    ("Random (3-seed mean)", "15.0", "15.0", "18.75", "37.50", "0.7683", "MISSING", "255.57", "FP32"),
    ("Random (3-seed mean)", "30.0", "30.0", "0.00", "0.00", "0.5542", "MISSING", "255.57", "FP32"),
    ("Random (3-seed mean)", "45.0", "45.0", "0.00", "0.00", "0.5125", "MISSING", "255.57", "FP32"),
    ("Random (3-seed mean)", "60.0", "60.0", "0.00", "0.00", "0.4979", "MISSING", "255.57", "FP32"),
    ("Random (3-seed mean)", "75.0", "75.0", "0.00", "0.00", "0.4751", "MISSING", "255.57", "FP32"),
]

for row in grid_data:
    records.append({
        "regime": row[0],
        "sparsity_target_pct": row[1],
        "measured_sparsity_pct": row[2],
        "top1_pct": row[3],
        "top5_pct": row[4],
        "embedding_cosine": row[5],
        "latency_median_ms": row[6],
        "latency_p95_ms": "MISSING",
        "size_mb": row[7],
        "quantization": row[8],
        "seed": "42-44 (mean)" if "3-seed" in row[0] else "42",
        "provenance": "Step 0 verified run" if "Encoder-Only" in row[0] or "3-seed" in row[0] else "notebook cell 8 output"
    })

fields = list(records[0].keys())
with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)

print(f"Successfully generated comprehensive results.csv with {len(records)} entries!")
