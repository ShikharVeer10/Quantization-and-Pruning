# ============================================================
# Step 10 (NEW): Sparse Representation Recovery via Knowledge Distillation (RQ4)
# ============================================================
# Investigates whether post-pruning distillation can recover context accuracy
# without increasing model size, validating the core thesis of Zafrir et al. (2021).

print("Setting up Sparse Knowledge Distillation Recovery Pipeline...")

TARGET_SPARSITY = 0.45
print(f"Target Baseline Pruned Model: Global L1 ({int(TARGET_SPARSITY*100)}% Sparsity)")

student_model = prune_global_l1(base_mlm_model, TARGET_SPARSITY)
student_model.to(device)
teacher_model = copy.deepcopy(base_mlm_model).to(device).eval()

# Pre-distillation baseline measurement
pre_top1, pre_top5 = evaluate_context_comprehension(student_model, CONTEXT_TEST_SUITE, tokenizer)
print(f"Pre-Distillation Performance  -> Top-1 Acc: {pre_top1}% | Top-5 Acc: {pre_top5}%")

# Create binary masks to prevent pruned zeros from regrowing during gradient descent
sparsity_masks = {}
for name, param in student_model.named_parameters():
    if "weight" in name and param.requires_grad:
        sparsity_masks[name] = (param.data != 0).float().to(device)

# Curate a lightweight context distillation corpus
distill_corpus = [
    "Artificial intelligence models process natural language through contextual token representations.",
    "The solar system consists of eight planets orbiting around the central star known as the sun.",
    "Computer software relies on efficient data structures and optimized algorithms to execute quickly.",
    "Modern medicine uses biological knowledge and clinical trials to discover new treatments for disease.",
    "Deep learning architectures adjust internal weights through gradient descent and backpropagation.",
    "The global economy relies on international trade, monetary policy, and digital communication networks.",
    "Renewable energy generation includes photovoltaic solar panels and massive offshore wind turbines.",
    "The human brain contains billions of interconnected neurons communicating via electrical synapses."
]

optimizer = torch.optim.AdamW(student_model.parameters(), lr=3e-5)
kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
mse_loss_fn = nn.MSELoss()

student_model.train()
EPOCHS = 3
TEMPERATURE = 2.0

print(f"\nExecuting Knowledge Distillation ({EPOCHS} Epochs on Representation Corpus)...")
for epoch in range(EPOCHS):
    total_loss = 0.0
    for text in distill_corpus:
        inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding="max_length").to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
            teacher_hidden = teacher_outputs.hidden_states[-1] if hasattr(teacher_outputs, "hidden_states") and teacher_outputs.hidden_states else None

        student_outputs = student_model(**inputs)
        student_logits = student_outputs.logits

        # Soft label matching via KL Divergence on vocabulary logits
        loss_logits = kl_loss_fn(
            nn.functional.log_softmax(student_logits / TEMPERATURE, dim=-1),
            nn.functional.softmax(teacher_logits / TEMPERATURE, dim=-1)
        ) * (TEMPERATURE ** 2)

        loss = loss_logits
        loss.backward()

        # Enforce exact sparsity mask: zero out gradients on pruned connections
        with torch.no_grad():
            for name, param in student_model.named_parameters():
                if name in sparsity_masks:
                    param.grad *= sparsity_masks[name]

        optimizer.step()
        total_loss += loss.item()

    print(f"  Epoch {epoch+1}/{EPOCHS} Complete - Loss: {total_loss / len(distill_corpus):.4f}")

student_model.eval()

# Post-distillation performance measurement
post_top1, post_top5 = evaluate_context_comprehension(student_model, CONTEXT_TEST_SUITE, tokenizer)
post_cos_sim = evaluate_embedding_drift(base_mlm_model.distilbert, student_model.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
post_sparsity = measure_sparsity(student_model)

print(f"\n=== Post-Distillation Verification ===")
print(f"  • Sparsity Maintained   : {post_sparsity:.2f}% (Exact zero-weights preserved)")
print(f"  • Top-1 Context Acc     : {pre_top1}% -> {post_top1}% ({post_top1 - pre_top1:+.2f} pp)")
print(f"  • Top-5 Context Acc     : {pre_top5}% -> {post_top5}% ({post_top5 - pre_top5:+.2f} pp)")
print(f"  • Embedding Cosine Sim  : {post_cos_sim:.4f}")

# Register recovered model in experimental records
experimental_registry.append({
    "Regime": f"Global L1 {int(TARGET_SPARSITY*100)}% + Distillation",
    "Pruning Method": "Global L1 (Distilled)",
    "Sparsity %": post_sparsity,
    "Quantization": "FP32",
    "Size (MB)": compute_model_size_mb(student_model),
    "Size Red. %": 0.0,
    "Top-1 Acc %": post_top1,
    "Top-5 Acc %": post_top5,
    "Embedding CosSim": post_cos_sim,
    "Median Latency (ms)": benchmark_latency(student_model, tokenizer)[0],
    "Speedup": round(base_median_ms / benchmark_latency(student_model, tokenizer)[0], 2)
})
live_model_pool[("Global L1 (Distilled)", int(TARGET_SPARSITY*100), "FP32")] = student_model.cpu()

