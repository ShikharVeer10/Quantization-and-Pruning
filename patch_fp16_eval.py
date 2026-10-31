"""Add the defined FP16 baseline to the notebook's evaluation registry."""
from pathlib import Path
import json

p = Path(__file__).with_name("Comprehensive_Context_Quantization_and_Pruning.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
cell = nb["cells"][8]
src = "".join(cell["source"])
marker = "# Dense INT8\nbase_int8 = quantize_dynamic_int8(base_mlm_model)"
addition = '''# Dense FP16
base_fp16 = quantize_fp16(base_mlm_model).cpu()
b_fp16_top1, b_fp16_top5 = evaluate_context_comprehension(base_fp16, CONTEXT_TEST_SUITE, tokenizer)
b_fp16_cos_sim = evaluate_embedding_drift(base_mlm_model.distilbert, base_fp16.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
b_fp16_lat, b_fp16_p95 = benchmark_latency(base_fp16, tokenizer)
b_fp16_sz = compute_model_size_mb(base_fp16, "saved_artifacts/baseline_fp16.pth")

experimental_registry.append({
    "Regime": "Dense FP16 (0% Pruned)",
    "Pruning Method": "None",
    "Sparsity %": 0.0,
    "Quantization": "FP16",
    "Size (MB)": b_fp16_sz,
    "Size Red. %": round((1 - b_fp16_sz / base_size_mb) * 100, 1),
    "Top-1 Acc %": b_fp16_top1,
    "Top-5 Acc %": b_fp16_top5,
    "Embedding CosSim": b_fp16_cos_sim,
    "Median Latency (ms)": b_fp16_lat,
    "P95 Latency (ms)": b_fp16_p95,
    "Speedup": round(base_median_ms / b_fp16_lat, 2)
})
live_model_pool[("Dense Baseline", 0, "FP16")] = base_fp16

# Dense INT8
base_int8 = quantize_dynamic_int8(base_mlm_model)'''
if marker not in src:
    if '"Regime": "Dense FP16 (0% Pruned)"' in src:
        print("FP16 baseline already present")
    else:
        raise RuntimeError("Could not locate dense INT8 insertion point")
else:
    src = src.replace(marker, addition, 1)
    cell["source"] = src.splitlines(keepends=True)
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Added dense FP16 evaluation. Notebook not run.")
