# Reproducibility Protocol and Empirical Verification

This document specifies the exact environment, execution parameters, bug fixes, and re-run procedures required to reproduce all findings in the paper.

## 1. System Environment
- **Operating System:** Windows 11 (build 10.0.26100), x86_64 architecture
- **CPU Platform:** Intel64 Family 6 Model 186 Stepping 3, GenuineIntel
- **Base Environment in Notebook:** Python 3.10 / PyTorch 2.11.0+cu130, executed on NVIDIA Tesla T4 GPU for setup/distillation and single-threaded CPU for inference.
- **Verification Environment for Step 0 Re-runs:** Python 3.12, PyTorch 2.6.0+cpu, `torch.set_num_threads(1)` for strictly controlled single-threaded CPU latency benchmarks.
- **Model Checkpoint:** `distilbert-base-uncased` from Hugging Face Transformers (`transformers` library).
  - Total Parameters: 66,985,530
  - Encoder Parameters: 66,362,880
  - Transformer Layers: 6, Attention Heads: 12, Hidden Dimension: 768, Vocabulary Size: 30,522.

## 2. Disclosed Step 0 Bug Fixes and Code Corrections
All modifications to the codebase have been fully disclosed and documented:
1. **Distillation Recovery Device Mismatch Fix:**
   - *Problem:* In the original notebook, `evaluate_context_comprehension()` executed `model.eval().cpu()`, mutating the model in-place to CPU. When called inside the distillation pipeline where the student and loss tensors resided on CUDA, subsequent `loss.backward()` and `optimizer.step()` crashed due to device mismatch.
   - *Fix:* `evaluate_context_comprehension()` was updated to operate on a deep-copied temporary evaluation model: `eval_model = copy.deepcopy(model).cpu().eval()`. The distillation pipeline completed 3 epochs successfully.
2. **Evaluated Previously Unrun Precision Modes:**
   - Evaluated native/simulated FP16, dynamic INT8, simulated INT4, and combined Global L1 30% + INT4 on the exact probe suite.
3. **Replaced Hard-coded INT4 Footprint Multiplier:**
   - Replaced `b_int4_sz = round(base_size_mb * 0.131, 2)` with exact formula: `total_params * 4 / 8 / (1024 ** 2) = 31.94` MiB, explicitly labeled as theoretical payload.
4. **Latency Benchmarking Protocol:**
   - Enforced $\ge 20$ warm-up runs and $\ge 100$ timed runs with `time.perf_counter()`, fixed CPU threads (`LATENCY_CPU_THREADS = 1`), and reported median and 95th percentile (P95). Re-measurement confirmed that dense FP32 latency is 170.33 ms (P95: 204.37 ms) and pruned FP32 (45% sparsity) is 170.87 ms (P95: 187.91 ms), proving that the notebook's earlier ~133 ms reading was a warm-up artifact and confirming zero unstructured speedup on dense CPU kernels.
5. **Statistical Replication of Random Pruning:**
   - Evaluated random unstructured pruning over 3 independent random seeds (42, 43, 44), reporting mean $\pm$ standard deviation for Top-1, Top-5, and CosSim across all five sparsity levels.
6. **Structured Pruning Implementation Decoupling:**
   - Evaluated encoder-only structured $L_2$ row pruning (preserving `vocab_projector`) as the primary structured pruning result, reporting the original all-Linear variant as an ablation.

## 3. Step-by-Step Reproduction Instructions
```powershell
# 1. Clone repository
git clone https://github.com/ShikharVeer10/Quantization-and-Pruning.git
cd Quantization-and-Pruning

# 2. Execute verification script
py -3.12 run_experiments.py

# 3. Update experimental CSV registry
py -3.12 update_results_csv.py

# 4. Generate IEEE publication figures
py -3.12 generate_figures.py

# 5. Build paper document and render PDF
py -3.12 build_paper_docx.py
```
