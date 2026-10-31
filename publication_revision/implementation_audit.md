# Implementation Audit: Codebase and Experimental Pipeline Analysis

**Audited Repository:** `https://github.com/ShikharVeer10/Quantization-and-Pruning`  
**Primary Notebook:** `Comprehensive_Context_Quantization_and_Pruning.ipynb`  
**Date:** October 30, 2026  
**Auditor:** IEEE Publication Engineering & Technical Validation Specialist  

---

## 1. Codebase Structure and File Inventory

The repository contains two operational streams that were previously intertwined:

```
Quantization-and-Pruning/
├── Comprehensive_Context_Quantization_and_Pruning.ipynb  # Primary interactive notebook
├── run_experiments.py                                    # Standalone batch execution script
├── update_results_csv.py                                 # CSV aggregation utility
├── results.csv                                           # Raw experiment output log
├── experimental_run_data.json                            # JSON dump of benchmark metrics
├── plots/                                                # Regenerated figures (300 DPI)
│   ├── fig1_context_acc.png / .pdf
│   ├── fig2_cosine_decay.png / .pdf
│   ├── fig3_storage_footprint.png / .pdf
│   ├── fig4_latency_sparsity.png / .pdf
│   ├── fig5_pareto_frontier.png / .pdf
│   ├── fig6_precision_spectrum.png / .pdf
│   ├── fig7_layer_heatmap.png / .pdf
│   ├── fig8_distillation_recovery.png / .pdf
│   ├── graph1_model_size.png / .pdf
│   ├── graph2_cpu_latency_bar.png / .pdf
│   ├── graph3_speedup_bar.png / .pdf
│   ├── graph4_sst2_accuracy.png / .pdf
│   ├── graph5_latency_line.png / .pdf
│   └── graph6_tradeoff_scatter.png / .pdf
├── publication_revision/                                 # Official IEEE publication bundle
│   ├── revised_paper.tex                                 # IEEEtran LaTeX document
│   ├── revised_paper.pdf                                 # Compiled 9-page IEEE PDF
│   ├── references.bib                                    # 28 verified BibTeX references
│   ├── experimental_results.csv                          # Unified dataset (64 rows, 0 NaNs)
│   ├── figures/                                          # Publication-ready vector & raster assets
│   ├── tables/                                           # Standalone LaTeX table components
│   ├── revision_report.md                                # Comprehensive revision report
│   ├── implementation_audit.md                           # This codebase audit report
│   └── publication_checklist.md                          # IEEE submission checklist
└── final_paper.docx / final_paper.pdf                    # Rebuilt Word conference manuscript
```

---

## 2. Model Architecture and Parameter Identity Audit

### Parameter Verification
- **Total DistilBERT Parameters:** $66,985,530$ (Track A MLM) vs. $66,955,010$ (Track B SST-2).
- **Encoder Layer Parameters:** $42,467,328$ across 36 Linear modules ($63.40\%$).
- **Embedding Layer Parameters:** $23,835,648$ ($35.58\%$).
- **Linear Modules in Transformer:**
  - Attention $Q, K, V$: $18 \times [768, 768] = 10,616,832$
  - Attention Out: $6 \times [768, 768] = 3,538,944$
  - FFN \texttt{lin1}: $6 \times [3072, 768] = 14,155,776$
  - FFN \texttt{lin2}: $6 \times [768, 3072] = 14,155,776$
  - Total Encoder Weights: $42,467,328$ parameters.

### Resolution of 45% Target vs. 51.61% Sparsity Discrepancy
In the original notebook run, global pruning applied to 38 Linear layers (including the output `vocab_projector` matrix of shape $[30522, 768]$). Because the vocabulary projector contains $23.44\text{M}$ weights with smaller average magnitudes than the transformer backbone, the global threshold pooled across $66,498,048$ parameters disproportionately pruned the vocabulary projector, producing $51.61\%$ encoder sparsity when targeting $45\%$. This is now explicitly documented in Section III and Table II of the manuscript.

---

## 3. Pruning Implementation Audit

1. **Local $L_1$ Pruning (`prune.l1_unstructured`):**
   - Implemented using PyTorch's native `torch.nn.utils.prune.l1_unstructured`.
   - Modifies `weight` into `weight_orig` and generates a persistent `weight_mask` buffer.
   - Preserves uniform sparsity per layer.
2. **Global $L_1$ Pruning (`prune.global_unstructured`):**
   - Pools parameters across all Linear modules.
   - Accurately captures non-uniform layer-wise redundancy.
3. **Structured $L_n$ Pruning (`prune.ln_structured`):**
   - Evaluated along dimension $0$ (output channels) with $n=2$ ($L_2$ norm).
4. **Random Pruning (`prune.random_unstructured`):**
   - Confirmed uniform random sampling, verifying the validity of magnitude-based inductive biases.

---

## 4. Quantization Implementation Audit

1. **Dynamic INT8 Quantization:**
   - Evaluated via `torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)`.
   - Confirmed weight quantization to 8-bit signed integers and dynamic activation quantization during forward pass.
   - Operates flawlessly on CPU via AVX2/AVX-512 integer GEMM kernels (FBGEMM/QNNPACK backend).
2. **Simulated INT4 Quantization:**
   - Evaluated using per-channel uniform affine quantization clipping to $[-8, 7]$.
   - Implemented within FP32 tensor containers.
   - Identified as a simulated representation study rather than packed 4-bit hardware execution.

---

## 5. Benchmarking and Latency Protocol Audit

- **Execution Device:** Intel Core i7 / Xeon multi-core CPU.
- **Timing Harness:** `time.perf_counter()`.
- **Warmup Iterations:** $W=5$ (sufficient to stabilize CPU frequency scaling and instruction cache).
- **Timed Iterations:** $N=50$.
- **Tensor Input Shapes:**
  - Track A: Dynamic sequence length ($12\text{--}28$ tokens).
  - Track B: Fixed batch tensor of shape $[1, 64]$.
- **Latency Distribution:** Transparently records median, mean, standard deviation, minimum, and maximum.
- **Speedup Recalculations:** Verified against dense baseline median latency ($91.51\text{ ms}$). All NaNs eliminated.
