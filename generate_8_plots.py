"""Generate all 8 publication-quality figures at 300 DPI into plots/ as specified in Step 12."""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path("C:/Users/shikh/Quantization-and-Pruning")
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set publication style
sns.set_theme(style="whitegrid", font="serif", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

# 1. Pruning Data from Notebook and Verification runs
methods = ["Local L1", "Global L1", "Structured Ln", "Random (Ablation)"]
sparsities = [0, 15, 30, 45, 60, 75]

# Top-5 Accuracy Data (%)
top5_data = {
    "Local L1": [75.00, 75.00, 62.50, 12.50, 0.00, 0.00],
    "Global L1": [75.00, 75.00, 75.00, 12.50, 0.00, 0.00],
    "Structured Ln (All-Linear)": [75.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "Structured Ln (Encoder-Only)": [75.00, 0.00, 0.00, 0.00, 6.25, 0.00],
    "Random (3-seed mean)": [75.00, 37.50, 0.00, 0.00, 0.00, 0.00],
}
random_top5_std = [0.0, 10.21, 0.0, 0.0, 0.0, 0.0]

# Cosine Similarity Data
cossim_data = {
    "Local L1": [1.0000, 0.9903, 0.9585, 0.8325, 0.4970, 0.4070],
    "Global L1": [1.0000, 0.9882, 0.9464, 0.7791, 0.5212, 0.4467],
    "Structured Ln (All-Linear)": [1.0000, 0.2642, 0.3446, 0.3449, 0.3989, 0.4710],
    "Structured Ln (Encoder-Only)": [1.0000, 0.4889, 0.4129, 0.4534, 0.4549, 0.4985],
    "Random (3-seed mean)": [1.0000, 0.7683, 0.5542, 0.5125, 0.4979, 0.4751],
}

# ── Fig 1: Context Accuracy vs. Sparsity across Pruning Regimes ──
plt.figure(figsize=(7, 4.5))
plt.plot(sparsities, top5_data["Local L1"], marker="o", color="#1f77b4", linewidth=2.0, label="Local L1 (Unstructured)")
plt.plot(sparsities, top5_data["Global L1"], marker="s", color="#2ca02c", linewidth=2.0, label="Global L1 (Adaptive)")
plt.plot(sparsities, top5_data["Structured Ln (Encoder-Only)"], marker="^", color="#d62728", linewidth=2.0, linestyle="--", label="Structured L2 (Encoder-Only)")
plt.errorbar(sparsities, top5_data["Random (3-seed mean)"], yerr=random_top5_std, marker="x", color="#9467bd", linewidth=1.8, linestyle=":", capsize=4, label="Random (3-Seed Mean ± Std)")
plt.title("Figure 1: Context Token Prediction (Top-5 Accuracy) vs. Sparsity", fontsize=12, fontweight="bold")
plt.xlabel("Linear Layer Sparsity (%)", fontsize=11)
plt.ylabel("Top-5 Contextual Accuracy (%)", fontsize=11)
plt.ylim(-5, 105)
plt.legend(frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig1_accuracy_vs_sparsity.png", dpi=300)
plt.close()

# ── Fig 2: Semantic Embedding Cosine Similarity Decay ──
plt.figure(figsize=(7, 4.5))
plt.plot(sparsities, cossim_data["Local L1"], marker="o", color="#1f77b4", linewidth=2.0, label="Local L1 CosSim")
plt.plot(sparsities, cossim_data["Global L1"], marker="s", color="#2ca02c", linewidth=2.0, label="Global L1 CosSim")
plt.plot(sparsities, cossim_data["Structured Ln (Encoder-Only)"], marker="^", color="#d62728", linewidth=2.0, linestyle="--", label="Structured L2 (Encoder-Only)")
plt.plot(sparsities, cossim_data["Random (3-seed mean)"], marker="x", color="#9467bd", linewidth=1.8, linestyle=":", label="Random (3-Seed Mean)")
plt.title("Figure 2: Semantic Embedding Drift vs. Baseline Vector Space", fontsize=12, fontweight="bold")
plt.xlabel("Linear Layer Sparsity (%)", fontsize=11)
plt.ylabel("Mean Cosine Similarity to Baseline", fontsize=11)
plt.ylim(0.2, 1.05)
plt.legend(frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig2_embedding_drift.png", dpi=300)
plt.close()

# ── Fig 3: Storage Footprint (FP32 vs. INT8) ──
plt.figure(figsize=(7, 4.2))
sp_labels = ["0%", "15%", "30%", "45%", "60%", "75%"]
fp32_sizes = [255.57, 255.57, 255.57, 255.57, 255.57, 255.57]
int8_sizes = [154.76, 154.77, 154.77, 154.77, 154.77, 154.77]
x = np.arange(len(sp_labels))
width = 0.35
plt.bar(x - width/2, fp32_sizes, width, label="FP32 Dense Container (255.57 MB)", color="#4C72B0")
plt.bar(x + width/2, int8_sizes, width, label="Dynamic INT8 Container (~154.77 MB)", color="#DD8452")
plt.xticks(x, sp_labels)
plt.xlabel("Sparsity Target", fontsize=11)
plt.ylabel("Serialized Checkpoint Size (MB)", fontsize=11)
plt.title("Figure 3: Serialized Storage Footprint (FP32 vs. Dynamic INT8)", fontsize=12, fontweight="bold")
plt.ylim(0, 300)
plt.legend(frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig3_size_comparison.png", dpi=300)
plt.close()

# ── Fig 4: Latency vs. Sparsity (Hardware Disconnect) ──
plt.figure(figsize=(7, 4.5))
spars_sub = [0, 15, 30, 45, 60, 75]
lat_impl_a = [221.35, 136.07, 135.77, 133.59, 133.88, 133.60] # Notebook Colab unconstrained threads
lat_impl_b = [170.33, 170.50, 170.62, 170.87, 170.95, 171.10] # Verified single-threaded CPU
lat_int8 = [95.03, 78.86, 78.06, 76.15, 78.35, 76.48]
plt.plot(spars_sub, lat_impl_b, marker="s", color="#2ca02c", linewidth=2.2, label="Implementation B: Single-Threaded CPU (No Speedup)")
plt.plot(spars_sub, lat_impl_a, marker="o", color="#1f77b4", linewidth=2.0, linestyle="--", label="Implementation A: Colab Host CPU (Artifact)")
plt.axhline(170.33, color="#2ca02c", linestyle=":", alpha=0.7, label="Dense FP32 Baseline (170.33 ms)")
plt.plot(spars_sub, lat_int8, marker="^", color="#ff7f0e", linewidth=2.0, label="Dynamic INT8 CPU Latency (~78-95 ms)")
plt.title("Figure 4: CPU Latency vs. Sparsity (Hardware-Sparsity Disconnect)", fontsize=12, fontweight="bold")
plt.xlabel("Linear Layer Sparsity (%)", fontsize=11)
plt.ylabel("Median Latency (ms)", fontsize=11)
plt.legend(frameon=True, fontsize=8.5)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig4_latency_comparison.png", dpi=300)
plt.close()

# ── Fig 5: Pareto Frontier (Accuracy vs. Latency) ──
plt.figure(figsize=(7.5, 4.8))
# Points: (Latency, Accuracy)
points = [
    ("Dense FP32", 170.33, 75.00, "#1f77b4", "o"),
    ("Dense FP16", 170.33, 75.00, "#aec7e8", "s"),
    ("Dense INT8", 95.03, 50.00, "#ff7f0e", "^"),
    ("Dense INT4", 170.33, 75.00, "#2ca02c", "v"),
    ("Local L1 15%", 170.50, 75.00, "#1f77b4", "o"),
    ("Local L1 30%", 170.62, 62.50, "#1f77b4", "o"),
    ("Global L1 15%", 170.50, 75.00, "#2ca02c", "s"),
    ("Global L1 30%", 170.62, 75.00, "#2ca02c", "s"),
    ("Global L1 45%", 170.87, 12.50, "#2ca02c", "s"),
    ("Global L1 45% + KD", 170.87, 37.50, "#d62728", "*"),
    ("Global L1 15% + INT8", 78.86, 75.00, "#ff7f0e", "^"),
    ("Global L1 30% + INT8", 78.06, 37.50, "#ff7f0e", "^"),
    ("Global L1 30% + INT4", 170.62, 37.50, "#2ca02c", "v"),
]
for label, lat, acc, col, mk in points:
    plt.scatter(lat, acc, color=col, marker=mk, s=110, alpha=0.9, edgecolors="black", linewidths=0.5)

plt.annotate("Optimal: Global L1 15% + INT8\n(78.86ms, 75.0% Top-5)", xy=(78.86, 75.0), xytext=(85, 82),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5), fontsize=8.5, fontweight="bold")
plt.annotate("KD Recovery (+25 pp)", xy=(170.87, 37.5), xytext=(125, 42),
             arrowprops=dict(facecolor='red', shrink=0.08, width=1, headwidth=5), fontsize=8.5, color="red")
plt.title("Figure 5: Pareto Frontier — Contextual Accuracy vs. CPU Latency", fontsize=12, fontweight="bold")
plt.xlabel("Median CPU Latency (ms) [Lower is Better]", fontsize=11)
plt.ylabel("Top-5 Contextual Accuracy (%) [Higher is Better]", fontsize=11)
plt.ylim(0, 105)
plt.xlim(65, 200)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig5_pareto_frontier.png", dpi=300)
plt.close()

# ── Fig 6: Precision Spectrum (FP32 -> INT8 -> INT4 Footprint) ──
plt.figure(figsize=(7, 4.2))
prec_labels = ["FP32 Full", "FP16 Half", "Dynamic INT8", "Simulated INT4*"]
prec_sizes = [255.57, 127.80, 154.76, 31.94]
bars = plt.bar(prec_labels, prec_sizes, color=["#4C72B0", "#5B9BD5", "#DD8452", "#55A868"], edgecolor="black", linewidth=0.5)
plt.ylabel("Model Checkpoint Size (MB / MiB)", fontsize=11)
plt.title("Figure 6: Storage Footprint Across Quantization Precisions (RQ3)", fontsize=12, fontweight="bold")
plt.ylim(0, 290)
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + 5, f"{y:.1f} MB", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig6_quantization_spectrum.png", dpi=300)
plt.close()

# ── Fig 7: Layer-wise Component Sparsity Breakdown Heatmap (RQ2) ──
plt.figure(figsize=(8, 4.8))
heatmap_data = np.array([
    [52.42, 52.96, 52.45, 62.40, 51.28, 60.84],
    [49.76, 52.88, 43.17, 61.50, 42.31, 59.29],
    [49.17, 52.59, 46.42, 56.20, 46.40, 51.78],
    [50.91, 54.93, 44.69, 51.61, 45.11, 48.27],
    [51.50, 53.23, 44.89, 49.06, 44.66, 46.85],
    [53.74, 56.79, 44.95, 48.06, 44.65, 45.91],
])
layers = [f"Layer {i}" for i in range(6)]
components = ["FFN Lin1", "FFN Lin2", "Key Proj", "Out Proj", "Query Proj", "Val Proj"]
sns.heatmap(heatmap_data, annot=True, fmt=".1f", xticklabels=components, yticklabels=layers, cmap="YlGnBu", cbar_kws={'label': 'Sparsity %'})
plt.title("Figure 7: Non-Uniform Sparsity Distribution under Global L1 Pruning (RQ2)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig7_layer_sensitivity_heatmap.png", dpi=300)
plt.close()

# ── Fig 8: Distillation Accuracy Recovery Bar Chart (RQ4) ──
plt.figure(figsize=(6.5, 4.2))
distill_labels = ["Dense Baseline", "Pruned 45% (Pre-KD)", "Recovered 45% (Post-KD)"]
top1_vals = [37.50, 12.50, 18.75]
top5_vals = [75.00, 12.50, 37.50]
x = np.arange(len(distill_labels))
width = 0.32
b1 = plt.bar(x - width/2, top1_vals, width, label="Top-1 Context Acc (%)", color="#4C72B0", edgecolor="black", linewidth=0.5)
b2 = plt.bar(x + width/2, top5_vals, width, label="Top-5 Context Acc (%)", color="#55A868", edgecolor="black", linewidth=0.5)
plt.xticks(x, distill_labels, fontsize=10)
plt.ylabel("Accuracy (%)", fontsize=11)
plt.title("Figure 8: Accuracy Recovery via Lightweight Distillation (RQ4)", fontsize=12, fontweight="bold")
plt.ylim(0, 100)
for bar in b1:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + 1.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=8.5)
for bar in b2:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + 1.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
plt.legend(frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig8_distillation_recovery.png", dpi=300)
plt.close()

print("Successfully generated all 8 publication figures in plots/ directory!")
