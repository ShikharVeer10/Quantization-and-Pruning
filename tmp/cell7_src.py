# ============================================================
# Step 7: Profile Baseline (Uncompressed FP32 DistilBERT)
# ============================================================

print("Evaluating Baseline DistilBERT (Dense FP32)...\n")

base_top1, base_top5 = evaluate_context_comprehension(base_mlm_model, CONTEXT_TEST_SUITE, tokenizer)
base_cos_sim = evaluate_embedding_drift(base_mlm_model.distilbert, base_mlm_model.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
base_median_ms, base_p95_ms = benchmark_latency(base_mlm_model, tokenizer)
base_size_mb = compute_model_size_mb(base_mlm_model)

baseline_record = {
    "Regime": "Dense Baseline",
    "Pruning Method": "None",
    "Sparsity %": 0.0,
    "Quantization": "FP32",
    "Size (MB)": base_size_mb,
    "Size Red. %": 0.0,
    "Top-1 Acc %": base_top1,
    "Top-5 Acc %": base_top5,
    "Embedding CosSim": base_cos_sim,
    "Median Latency (ms)": base_median_ms,
    "Speedup": 1.00
}

df_baseline = pd.DataFrame([baseline_record])
display(df_baseline)

