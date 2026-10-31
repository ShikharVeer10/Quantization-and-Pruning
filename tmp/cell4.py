# ============================================================
# Step 4: Core Profilers (Context Acc, Embedding Drift, Latency, Size)
# ============================================================

def evaluate_context_comprehension(model, test_suite, tok):
    """Measures Top-1 and Top-5 accuracy on predicting the masked token across contexts."""
    model.eval().cpu()
    top1_correct = 0
    top5_correct = 0
    total = len(test_suite)

    with torch.no_grad():
        for item in test_suite:
            text = item["text"]
            target = item["target"].lower().strip()

            inputs = tok(text, return_tensors="pt")
            mask_token_index = torch.where(inputs["input_ids"] == tok.mask_token_id)[1]

            if len(mask_token_index) == 0:
                continue

            outputs = model(**inputs)
            mask_logits = outputs.logits[0, mask_token_index[0], :]

            # Top-5 predicted token ids
            top5_tokens = torch.topk(mask_logits, 5, dim=-1).indices.tolist()
            top5_words = [tok.decode([tid]).strip().lower() for tid in top5_tokens]

            if len(top5_words) > 0 and top5_words[0] == target:
                top1_correct += 1
            if target in top5_words:
                top5_correct += 1

    top1_acc = round((top1_correct / total) * 100, 2)
    top5_acc = round((top5_correct / total) * 100, 2)
    return top1_acc, top5_acc

def extract_mean_pooled_embedding(encoder, text, tok):
    """Extracts the 768-dim mean-pooled contextual sentence embedding."""
    encoder.eval().cpu()
    with torch.no_grad():
        inputs = tok(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
        outputs = encoder(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        return mean_pooled.squeeze(0)

def evaluate_embedding_drift(base_encoder, compressed_encoder, sentence_pairs, tok):
    """
    Calculates Cosine Similarity between baseline embedding space
    and compressed model embedding space across diverse sentences.
    Higher cosine similarity (closer to 1.0) = less semantic representation drift.
    """
    similarities = []
    cos_sim = nn.CosineSimilarity(dim=0)

    for sent1, sent2 in sentence_pairs:
        for sent in [sent1, sent2]:
            emb_base = extract_mean_pooled_embedding(base_encoder, sent, tok)
            emb_comp = extract_mean_pooled_embedding(compressed_encoder, sent, tok)
            sim = cos_sim(emb_base, emb_comp).item()
            similarities.append(sim)

    return round(float(np.mean(similarities)), 4)

def measure_sparsity(model):
    """Calculates exact fraction of zero weights across all Linear layers."""
    total, zeros = 0, 0
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            w = mod.weight.detach().cpu()
            total += w.numel()
            zeros += (w == 0).sum().item()
    return round((zeros / total * 100), 2) if total > 0 else 0.0

def benchmark_latency(model, tok, n_runs=50, warmup=5):
    """Measures median and P95 CPU inference latency using time.perf_counter()."""
    model.eval().cpu()
    sample_text = "The quick brown fox jumps over the lazy dog to verify execution speed."
    inputs = tok(sample_text, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
    inputs = {k: v.cpu() for k, v in inputs.items()}

    with torch.no_grad():
        for _ in range(warmup):
            model(**inputs)

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(**inputs)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    median_ms = round(float(np.median(latencies)), 2)
    p95_ms = round(float(np.percentile(latencies, 95)), 2)
    return median_ms, p95_ms

def compute_model_size_mb(model, temp_path="temp_model.pth"):
    """Saves state_dict to disk and measures exact size in Megabytes."""
    torch.save(model.state_dict(), temp_path)
    sz_mb = round(os.path.getsize(temp_path) / (1024 ** 2), 2)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return sz_mb

print("All quantitative profiling functions compiled successfully.")

