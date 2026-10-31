import copy
import json
import os
import platform
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Set seeds and threads
torch.manual_seed(42)
np.random.seed(42)
LATENCY_CPU_THREADS = 1
torch.set_num_threads(LATENCY_CPU_THREADS)

cpu_model = platform.processor() or platform.uname().processor
print(f"CPU Model: {cpu_model}")
print(f"PyTorch version: {torch.__version__}")
print(f"Threads: {torch.get_num_threads()}")

device = torch.device("cpu")

# Load model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)
base_mlm_model.eval()

# Prompts and pairs
CONTEXT_TEST_SUITE = [
    {"text": "The students are studying hard because [MASK] have an exam tomorrow.", "target": "they", "category": "Syntax/Pronoun"},
    {"text": "She walked into the library and borrowed a [MASK] to read.", "target": "book", "category": "Syntax/Noun"},
    {"text": "Neither the teacher nor the students [MASK] aware of the schedule change.", "target": "were", "category": "Syntax/Agreement"},
    {"text": "He drove the car safely down the [MASK] during the storm.", "target": "road", "category": "Syntax/Context"},
    {"text": "Paris is the capital and largest city of [MASK].", "target": "france", "category": "Knowledge/Geography"},
    {"text": "The sun rises in the east and sets in the [MASK].", "target": "west", "category": "Knowledge/Astronomy"},
    {"text": "A doctor works in a [MASK] to treat patients.", "target": "hospital", "category": "Knowledge/Domain"},
    {"text": "Water boils at one hundred degrees [MASK].", "target": "celsius", "category": "Knowledge/Science"},
    {"text": "Neural networks learn complex patterns by adjusting their [MASK] during training.", "target": "weights", "category": "Tech/AI"},
    {"text": "Computers process digital information using binary [MASK] of zeros and ones.", "target": "code", "category": "Tech/CS"},
    {"text": "The primary language used for web page structure is [MASK].", "target": "html", "category": "Tech/Web"},
    {"text": "Quantum computers leverage the principle of [MASK] to perform parallel calculations.", "target": "superposition", "category": "Tech/Physics"},
    {"text": "To protect their privacy, users should always use a strong [MASK] for their accounts.", "target": "password", "category": "Security"},
    {"text": "The detective examined the crime scene looking for any hidden [MASK].", "target": "clues", "category": "Reasoning"},
    {"text": "Renewable energy sources include solar, wind, and [MASK] power.", "target": "hydro", "category": "Energy"},
    {"text": "The heart pumps oxygenated [MASK] throughout the human body.", "target": "blood", "category": "Biology"}
]

SEMANTIC_SENTENCE_PAIRS = [
    ("Artificial intelligence is transforming modern industries.", "Machine learning algorithms are revolutionizing commercial workflows."),
    ("The chef prepared a delicious gourmet meal with fresh ingredients.", "The cook made an exquisite dinner using fresh organic produce."),
    ("The stock market experienced significant volatility today.", "Financial equities fluctuated wildly during today's trading session."),
    ("Planetary scientists discovered water ice beneath the Martian surface.", "Astronomers detected subterranean frozen water on the red planet."),
    ("A peaceful forest with towering redwood trees.", "A dense woodland filled with ancient giant trees.")
]

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

def evaluate_context_comprehension(model, test_suite, tok):
    eval_model = copy.deepcopy(model).cpu().eval()
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
            outputs = eval_model(**inputs)
            mask_logits = outputs.logits[0, mask_token_index[0], :]
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
    total, zeros = 0, 0
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            w = mod.weight.detach().cpu()
            total += w.numel()
            zeros += (w == 0).sum().item()
    return round((zeros / total * 100), 2) if total > 0 else 0.0

def benchmark_latency(model, tok, n_runs=100, warmup=20):
    torch.set_num_threads(LATENCY_CPU_THREADS)
    eval_model = copy.deepcopy(model).cpu().eval()
    sample_text = "The quick brown fox jumps over the lazy dog to verify execution speed."
    inputs = tok(sample_text, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
    with torch.no_grad():
        for _ in range(warmup):
            eval_model(**inputs)
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            eval_model(**inputs)
            latencies.append((time.perf_counter() - t0) * 1000.0)
    median_ms = round(float(np.median(latencies)), 2)
    p95_ms = round(float(np.percentile(latencies, 95)), 2)
    return median_ms, p95_ms

def compute_model_size_mb(model, temp_path="tmp_model.pth"):
    torch.save(model.state_dict(), temp_path)
    sz = os.path.getsize(temp_path) / (1024 * 1024)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return round(sz, 2)

# Pruning definitions
def prune_local_l1(model_orig, amount):
    m = copy.deepcopy(model_orig).cpu()
    if amount > 0:
        for _, module in m.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=amount)
        for _, module in m.named_modules():
            if isinstance(module, nn.Linear):
                prune.remove(module, "weight")
    m.eval()
    return m

def prune_global_l1(model_orig, amount):
    m = copy.deepcopy(model_orig).cpu()
    if amount > 0:
        parameters_to_prune = []
        for _, module in m.named_modules():
            if isinstance(module, nn.Linear):
                parameters_to_prune.append((module, "weight"))
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
        for module, name in parameters_to_prune:
            prune.remove(module, name)
    m.eval()
    return m

def prune_structured_ln_all(model_orig, amount):
    """Original implementation that prunes all Linear including vocab_projector"""
    m = copy.deepcopy(model_orig).cpu()
    if amount > 0:
        for _, module in m.named_modules():
            if isinstance(module, nn.Linear):
                prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
        for _, module in m.named_modules():
            if isinstance(module, nn.Linear):
                prune.remove(module, "weight")
    m.eval()
    return m

def prune_structured_ln_encoder(model_orig, amount):
    """Encoder-only structured pruning (preserves vocab_projector)"""
    m = copy.deepcopy(model_orig).cpu()
    if amount > 0:
        for name, module in m.named_modules():
            if isinstance(module, nn.Linear) and "vocab_" not in name:
                prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
        for name, module in m.named_modules():
            if isinstance(module, nn.Linear) and "vocab_" not in name:
                prune.remove(module, "weight")
    m.eval()
    return m

def prune_random(model_orig, amount, seed=42):
    torch.manual_seed(seed)
    m = copy.deepcopy(model_orig).cpu()
    if amount > 0:
        for _, module in m.named_modules():
            if isinstance(module, nn.Linear):
                prune.random_unstructured(module, name="weight", amount=amount)
        for _, module in m.named_modules():
            if isinstance(module, nn.Linear):
                prune.remove(module, "weight")
    m.eval()
    return m

# Quantization functions
def quantize_dynamic_int8(model_orig):
    m = copy.deepcopy(model_orig).cpu()
    q_model = torch.quantization.quantize_dynamic(
        m,
        {nn.Linear},
        dtype=torch.qint8
    )
    q_model.eval()
    return q_model

def quantize_fp16(model_orig):
    m = copy.deepcopy(model_orig).cpu()
    m = m.half()
    m.eval()
    return m

def quantize_simulated_int4(model_orig):
    m = copy.deepcopy(model_orig).cpu()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                w = mod.weight.data
                scale = torch.max(torch.abs(w), dim=-1, keepdim=True)[0] / 7.0
                scale = torch.clamp(scale, min=1e-8)
                w_int4 = torch.clamp(torch.round(w / scale), -8, 7)
                mod.weight.data = w_int4 * scale
    m.eval()
    return m

results = {}

# 1. Distillation recovery
print("\n--- Running Distillation Recovery (Step 1) ---")
target_sparsity = 0.45
student_model = prune_global_l1(base_mlm_model, target_sparsity)
teacher_model = copy.deepcopy(base_mlm_model).eval()

pre_top1, pre_top5 = evaluate_context_comprehension(student_model, CONTEXT_TEST_SUITE, tokenizer)
pre_cos_sim = evaluate_embedding_drift(base_mlm_model.distilbert, student_model.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
pre_sparsity = measure_sparsity(student_model)
print(f"Pre-distillation: Top-1={pre_top1}%, Top-5={pre_top5}%, CosSim={pre_cos_sim:.4f}, Sparsity={pre_sparsity}%")

sparsity_masks = {}
for name, param in student_model.named_parameters():
    if "weight" in name and param.requires_grad:
        sparsity_masks[name] = (param.data != 0).float()

optimizer = torch.optim.AdamW(student_model.parameters(), lr=3e-5)
kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
student_model.train()
EPOCHS = 3
TEMPERATURE = 2.0

for epoch in range(EPOCHS):
    total_loss = 0.0
    for text in distill_corpus:
        inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher_model(**inputs).logits
        student_logits = student_model(**inputs).logits
        loss = kl_loss_fn(
            nn.functional.log_softmax(student_logits / TEMPERATURE, dim=-1),
            nn.functional.softmax(teacher_logits / TEMPERATURE, dim=-1)
        ) * (TEMPERATURE ** 2)
        loss.backward()
        with torch.no_grad():
            for name, param in student_model.named_parameters():
                if name in sparsity_masks:
                    param.grad *= sparsity_masks[name]
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/{EPOCHS} Loss: {total_loss / len(distill_corpus):.4f}")

student_model.eval()
post_top1, post_top5 = evaluate_context_comprehension(student_model, CONTEXT_TEST_SUITE, tokenizer)
post_cos_sim = evaluate_embedding_drift(base_mlm_model.distilbert, student_model.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
post_sparsity = measure_sparsity(student_model)
print(f"Post-distillation: Top-1={post_top1}%, Top-5={post_top5}%, CosSim={post_cos_sim:.4f}, Sparsity={post_sparsity}%")

results["distillation"] = {
    "pre_top1": pre_top1, "pre_top5": pre_top5, "pre_cos_sim": pre_cos_sim, "pre_sparsity": pre_sparsity,
    "post_top1": post_top1, "post_top5": post_top5, "post_cos_sim": post_cos_sim, "post_sparsity": post_sparsity
}

# 2. Evaluate FP16, Dense INT8, Dense INT4, Global L1 30% + INT4
print("\n--- Evaluating Precision Configurations (Step 2 & 3) ---")
# Dense FP16
print("Evaluating Dense FP16...")
# PyTorch CPU doesn't support float16 for all operations, test carefully
try:
    base_fp16 = quantize_fp16(base_mlm_model)
    fp16_top1, fp16_top5 = evaluate_context_comprehension(base_fp16, CONTEXT_TEST_SUITE, tokenizer)
    fp16_cos_sim = evaluate_embedding_drift(base_mlm_model.distilbert, base_fp16.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
    fp16_sz = compute_model_size_mb(base_fp16)
except Exception as e:
    print(f"FP16 CPU inference note: {e}")
    # On CPU, float16 can be evaluated by casting back to float32 for operations or using native if supported
    base_fp16 = copy.deepcopy(base_mlm_model).half()
    # evaluate by running forward
    fp16_sz = compute_model_size_mb(base_fp16)
    # CPU forward for half can fail on older CPU kernels; let's check
    try:
        inputs = tokenizer("Sample text", return_tensors="pt")
        base_fp16(**inputs)
        fp16_top1, fp16_top5 = evaluate_context_comprehension(base_fp16, CONTEXT_TEST_SUITE, tokenizer)
        fp16_cos_sim = evaluate_embedding_drift(base_mlm_model.distilbert, base_fp16.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
    except Exception as e2:
        print(f"Native CPU float16 forward unsupported: {e2}")
        # Evaluate weights cast to half and back to float32
        base_fp16_sim = copy.deepcopy(base_mlm_model)
        for p in base_fp16_sim.parameters():
            p.data = p.data.half().float()
        fp16_top1, fp16_top5 = evaluate_context_comprehension(base_fp16_sim, CONTEXT_TEST_SUITE, tokenizer)
        fp16_cos_sim = evaluate_embedding_drift(base_mlm_model.distilbert, base_fp16_sim.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)

print(f"Dense FP16: Top-1={fp16_top1}%, Top-5={fp16_top5}%, CosSim={fp16_cos_sim:.4f}, Size={fp16_sz} MB")

# Dense INT8
print("Evaluating Dense INT8...")
base_int8 = quantize_dynamic_int8(base_mlm_model)
int8_top1, int8_top5 = evaluate_context_comprehension(base_int8, CONTEXT_TEST_SUITE, tokenizer)
int8_cos_sim = evaluate_embedding_drift(base_mlm_model.distilbert, base_int8.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
int8_sz = compute_model_size_mb(base_int8)
print(f"Dense INT8: Top-1={int8_top1}%, Top-5={int8_top5}%, CosSim={int8_cos_sim:.4f}, Size={int8_sz} MB")

# Dense INT4
print("Evaluating Dense INT4...")
base_int4 = quantize_simulated_int4(base_mlm_model)
int4_top1, int4_top5 = evaluate_context_comprehension(base_int4, CONTEXT_TEST_SUITE, tokenizer)
int4_cos_sim = evaluate_embedding_drift(base_mlm_model.distilbert, base_int4.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
total_params = sum(p.numel() for p in base_mlm_model.parameters())
int4_theoretical_mib = round(total_params * 4 / 8 / (1024 ** 2), 2)
print(f"Dense INT4: Top-1={int4_top1}%, Top-5={int4_top5}%, CosSim={int4_cos_sim:.4f}, Theoretical Size={int4_theoretical_mib} MiB")

# Global L1 30% + INT4
print("Evaluating Global L1 30% + INT4...")
g30_int4 = quantize_simulated_int4(prune_global_l1(base_mlm_model, 0.30))
g30_int4_top1, g30_int4_top5 = evaluate_context_comprehension(g30_int4, CONTEXT_TEST_SUITE, tokenizer)
g30_int4_cos_sim = evaluate_embedding_drift(base_mlm_model.distilbert, g30_int4.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
print(f"Global L1 30% + INT4: Top-1={g30_int4_top1}%, Top-5={g30_int4_top5}%, CosSim={g30_int4_cos_sim:.4f}")

results["precision"] = {
    "fp16": {"top1": fp16_top1, "top5": fp16_top5, "cos_sim": fp16_cos_sim, "size_mb": fp16_sz},
    "dense_int8": {"top1": int8_top1, "top5": int8_top5, "cos_sim": int8_cos_sim, "size_mb": int8_sz},
    "dense_int4": {"top1": int4_top1, "top5": int4_top5, "cos_sim": int4_cos_sim, "theoretical_size_mib": int4_theoretical_mib},
    "global_l1_30_int4": {"top1": g30_int4_top1, "top5": g30_int4_top5, "cos_sim": g30_int4_cos_sim, "theoretical_size_mib": int4_theoretical_mib}
}

# 4. Latency measurement (100 runs, 20 warm-up)
print("\n--- Running Benchmarked Latency (Step 4) ---")
dense_lat, dense_p95 = benchmark_latency(base_mlm_model, tokenizer, n_runs=100, warmup=20)
print(f"Dense FP32 Latency: Median={dense_lat} ms, P95={dense_p95} ms")

g45_fp32 = prune_global_l1(base_mlm_model, 0.45)
g45_lat, g45_p95 = benchmark_latency(g45_fp32, tokenizer, n_runs=100, warmup=20)
print(f"Global L1 45% FP32 Latency: Median={g45_lat} ms, P95={g45_p95} ms")

dense_int8_lat, dense_int8_p95 = benchmark_latency(base_int8, tokenizer, n_runs=100, warmup=20)
print(f"Dense INT8 Latency: Median={dense_int8_lat} ms, P95={dense_int8_p95} ms")

results["latency"] = {
    "cpu_model": cpu_model,
    "threads": LATENCY_CPU_THREADS,
    "dense_fp32": {"median": dense_lat, "p95": dense_p95},
    "global_l1_45_fp32": {"median": g45_lat, "p95": g45_p95},
    "dense_int8": {"median": dense_int8_lat, "p95": dense_int8_p95}
}

# 5. Statistics: Random control with 3 seeds
print("\n--- Running Random Control Statistics (Step 5) ---")
random_stats = {}
seeds = [42, 43, 44]
for spar in [0.15, 0.30, 0.45, 0.60, 0.75]:
    top1_list, top5_list, cos_list = [], [], []
    for s in seeds:
        m_rand = prune_random(base_mlm_model, spar, seed=s)
        t1, t5 = evaluate_context_comprehension(m_rand, CONTEXT_TEST_SUITE, tokenizer)
        c_sim = evaluate_embedding_drift(base_mlm_model.distilbert, m_rand.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
        top1_list.append(t1)
        top5_list.append(t5)
        cos_list.append(c_sim)
    pct = int(spar * 100)
    random_stats[pct] = {
        "top1_mean": round(float(np.mean(top1_list)), 2),
        "top1_std": round(float(np.std(top1_list)), 2),
        "top5_mean": round(float(np.mean(top5_list)), 2),
        "top5_std": round(float(np.std(top5_list)), 2),
        "cos_mean": round(float(np.mean(cos_list)), 4),
        "cos_std": round(float(np.std(cos_list)), 4),
        "raw_top1": top1_list, "raw_top5": top5_list, "raw_cos": cos_list
    }
    print(f"Random {pct}%: Top-1={random_stats[pct]['top1_mean']} +/- {random_stats[pct]['top1_std']}%, "
          f"Top-5={random_stats[pct]['top5_mean']} +/- {random_stats[pct]['top5_std']}%, "
          f"CosSim={random_stats[pct]['cos_mean']} +/- {random_stats[pct]['cos_std']}")

results["random_seeds"] = random_stats

# 6. Structured Pruning: Encoder-only vs All-Linear ablation
print("\n--- Running Structured Pruning Encoder-Only vs All-Linear (Step 6) ---")
structured_comparison = {}
for spar in [0.15, 0.30, 0.45, 0.60, 0.75]:
    pct = int(spar * 100)
    # Encoder only
    m_enc = prune_structured_ln_encoder(base_mlm_model, spar)
    enc_top1, enc_top5 = evaluate_context_comprehension(m_enc, CONTEXT_TEST_SUITE, tokenizer)
    enc_cos = evaluate_embedding_drift(base_mlm_model.distilbert, m_enc.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
    # All linear (original)
    m_all = prune_structured_ln_all(base_mlm_model, spar)
    all_top1, all_top5 = evaluate_context_comprehension(m_all, CONTEXT_TEST_SUITE, tokenizer)
    all_cos = evaluate_embedding_drift(base_mlm_model.distilbert, m_all.distilbert, SEMANTIC_SENTENCE_PAIRS, tokenizer)
    structured_comparison[pct] = {
        "encoder_only": {"top1": enc_top1, "top5": enc_top5, "cos_sim": enc_cos},
        "all_linear_ablation": {"top1": all_top1, "top5": all_top5, "cos_sim": all_cos}
    }
    print(f"Structured {pct}%: Encoder-only Top-1={enc_top1}%, Top-5={enc_top5}%, CosSim={enc_cos:.4f} | "
          f"All-Linear Top-1={all_top1}%, Top-5={all_top5}%, CosSim={all_cos:.4f}")

results["structured_comparison"] = structured_comparison

with open("experimental_run_data.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nSuccessfully executed all experiments and saved to experimental_run_data.json!")
