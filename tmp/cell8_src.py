# ============================================================
# Step 8: Multi-Dimensional Systematic Experimentation
# ============================================================

os.makedirs("saved_artifacts", exist_ok=True)
experimental_registry = [baseline_record]

# Configuration spectrum
SPARSITY_LEVELS = [0.15, 0.30, 0.45, 0.60, 0.75]
PRUNING_METHODS = [
    ("Local L1", prune_local_l1),
    ("Global L1", prune_global_l1),
    ("Structured Ln", prune_structured_ln),
    ("Random (Ablation)", prune_random)
]

print("Starting Multi-Dimensional Compression & Representation Grid Search...")
print(f"Total experiment configurations to evaluate: {len(PRUNING_METHODS) * len(SPARSITY_LEVELS) * 2 + 3}\n")

live_model_pool = {
    ("Dense Baseline", 0, "FP32"): base_mlm_model
}

for method_name, prune_fn in PRUNING_METHODS:
    print(f"\n{'='*30} Evaluating Pruning Method: {method_name} {'='*30}")
    for amount in SPARSITY_LEVELS:
        pct = int(amount * 100)
        print(f"\n--- Pruning Level: {pct}% ---")

        # 1. Prune FP32
        pruned_fp32 = prune_fn(base_mlm_model, amount)
        actual_sparsity = measure_sparsity(pruned_fp32)

        top1, top5 = evaluate_context_comprehension(pruned_fp32, CONTEXT_TEST_SUITE, tokenizer)
        cos_sim = evaluate_embedding_drift(base_mlm_model.distilbert, pruned_fp32.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
        median_lat, _ = benchmark_latency(pruned_fp32, tokenizer)
        sz_mb = compute_model_size_mb(pruned_fp32, f"saved_artifacts/{method_name.replace(' ', '_')}_{pct}_fp32.pth")

        rec_fp32 = {
            "Regime": f"{method_name} {pct}%",
            "Pruning Method": method_name,
            "Sparsity %": actual_sparsity,
            "Quantization": "FP32",
            "Size (MB)": sz_mb,
            "Size Red. %": round((1 - sz_mb / base_size_mb) * 100, 1),
            "Top-1 Acc %": top1,
            "Top-5 Acc %": top5,
            "Embedding CosSim": cos_sim,
            "Median Latency (ms)": median_lat,
            "Speedup": round(base_median_ms / median_lat, 2)
        }
        experimental_registry.append(rec_fp32)
        print(f"  [FP32] Sparsity: {actual_sparsity}% | Top-1: {top1}% | Top-5: {top5}% | CosSim: {cos_sim:.4f} | Latency: {median_lat} ms")

        if pct in [30, 45, 60] and method_name in ["Local L1", "Global L1"]:
            live_model_pool[(method_name, pct, "FP32")] = pruned_fp32

        # 2. Dynamic INT8 on top of Pruned Model
        pruned_int8 = quantize_dynamic_int8(pruned_fp32)
        int8_top1, int8_top5 = evaluate_context_comprehension(pruned_int8, CONTEXT_TEST_SUITE, tokenizer)
        int8_cos_sim = evaluate_embedding_drift(base_mlm_model.distilbert, pruned_int8.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
        int8_lat, _ = benchmark_latency(pruned_int8, tokenizer)
        int8_sz = compute_model_size_mb(pruned_int8, f"saved_artifacts/{method_name.replace(' ', '_')}_{pct}_int8.pth")

        rec_int8 = {
            "Regime": f"{method_name} {pct}% + INT8",
            "Pruning Method": method_name,
            "Sparsity %": actual_sparsity,
            "Quantization": "Dynamic INT8",
            "Size (MB)": int8_sz,
            "Size Red. %": round((1 - int8_sz / base_size_mb) * 100, 1),
            "Top-1 Acc %": int8_top1,
            "Top-5 Acc %": int8_top5,
            "Embedding CosSim": int8_cos_sim,
            "Median Latency (ms)": int8_lat,
            "Speedup": round(base_median_ms / int8_lat, 2)
        }
        experimental_registry.append(rec_int8)
        print(f"  [INT8] Sparsity: {actual_sparsity}% | Top-1: {int8_top1}% | Top-5: {int8_top5}% | CosSim: {int8_cos_sim:.4f} | Latency: {int8_lat} ms | Size: {int8_sz} MB")

        if pct in [30, 45, 60] and method_name in ["Local L1", "Global L1"]:
            live_model_pool[(method_name, pct, "Dynamic INT8")] = pruned_int8

# Quantization-only baselines (INT8 & INT4)
print("\n--- Evaluating Unpruned Multi-Bit Quantization Baselines ---")
# Dense FP16
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
base_int8 = quantize_dynamic_int8(base_mlm_model)
b_int8_top1, b_int8_top5 = evaluate_context_comprehension(base_int8, CONTEXT_TEST_SUITE, tokenizer)
b_int8_cos_sim = evaluate_embedding_drift(base_mlm_model.distilbert, base_int8.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
b_int8_lat, _ = benchmark_latency(base_int8, tokenizer)
b_int8_sz = compute_model_size_mb(base_int8, "saved_artifacts/baseline_int8.pth")

experimental_registry.append({
    "Regime": "Dense INT8 (0% Pruned)",
    "Pruning Method": "None",
    "Sparsity %": 0.0,
    "Quantization": "Dynamic INT8",
    "Size (MB)": b_int8_sz,
    "Size Red. %": round((1 - b_int8_sz / base_size_mb) * 100, 1),
    "Top-1 Acc %": b_int8_top1,
    "Top-5 Acc %": b_int8_top5,
    "Embedding CosSim": b_int8_cos_sim,
    "Median Latency (ms)": b_int8_lat,
    "Speedup": round(base_median_ms / b_int8_lat, 2)
})
live_model_pool[("Dense Baseline", 0, "Dynamic INT8")] = base_int8

# Dense INT4
base_int4 = quantize_simulated_int4(base_mlm_model)
b_int4_top1, b_int4_top5 = evaluate_context_comprehension(base_int4, CONTEXT_TEST_SUITE, tokenizer)
b_int4_cos_sim = evaluate_embedding_drift(base_mlm_model.distilbert, base_int4.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
b_int4_lat, _ = benchmark_latency(base_int4, tokenizer)
# INT4 theoretical linear layer footprint ~ 33.5 MB
total_params = sum(p.numel() for p in base_mlm_model.parameters())
b_int4_sz = round(total_params * 4 / 8 / (1024 ** 2), 2)
print(f"INT4 theoretical payload: {b_int4_sz:.2f} MiB (not a serialized file)")

experimental_registry.append({
    "Regime": "Dense INT4 (0% Pruned)",
    "Pruning Method": "None",
    "Sparsity %": 0.0,
    "Quantization": "Simulated INT4",
    "Size (MB)": b_int4_sz,
    "Size Red. %": round((1 - b_int4_sz / base_size_mb) * 100, 1),
    "Top-1 Acc %": b_int4_top1,
    "Top-5 Acc %": b_int4_top5,
    "Embedding CosSim": b_int4_cos_sim,
    "Median Latency (ms)": b_int4_lat,
    "Speedup": round(base_median_ms / b_int4_lat, 2)
})
live_model_pool[("Dense Baseline", 0, "Simulated INT4")] = base_int4

# Combined: Global L1 30% + INT4
pruned_int4 = quantize_simulated_int4(prune_global_l1(base_mlm_model, 0.30))
p_int4_top1, p_int4_top5 = evaluate_context_comprehension(pruned_int4, CONTEXT_TEST_SUITE, tokenizer)
p_int4_cos_sim = evaluate_embedding_drift(base_mlm_model.distilbert, pruned_int4.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
p_int4_lat, _ = benchmark_latency(pruned_int4, tokenizer)

experimental_registry.append({
    "Regime": "Global L1 30% + INT4",
    "Pruning Method": "Global L1",
    "Sparsity %": 30.0,
    "Quantization": "Simulated INT4",
    "Size (MB)": b_int4_sz,
    "Size Red. %": round((1 - b_int4_sz / base_size_mb) * 100, 1),
    "Top-1 Acc %": p_int4_top1,
    "Top-5 Acc %": p_int4_top5,
    "Embedding CosSim": p_int4_cos_sim,
    "Median Latency (ms)": p_int4_lat,
    "Speedup": round(base_median_ms / p_int4_lat, 2)
})
live_model_pool[("Global L1", 30, "Simulated INT4")] = pruned_int4

print("\n✓ Multi-Dimensional Grid Search Execution Completed!")

