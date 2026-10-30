# Quantization and Pruning of DistilBERT

An empirical notebook for measuring how pruning and reduced precision affect a small masked language model's contextual predictions, embeddings, storage, and CPU latency.

The study uses [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) and evaluates compression methods against a dense FP32 baseline.

## What is evaluated

- **Pruning:** local L1 unstructured, global L1 unstructured, structured L2 row/channel pruning, and random unstructured pruning.
- **Precision:** FP32, FP16, dynamic INT8, and simulated per-channel INT4.
- **Sparsity:** 0%, 15%, 30%, 45%, 60%, and 75%.
- **Task quality:** top-1 and top-5 masked-token accuracy on 16 linguistic probes.
- **Representation stability:** cosine similarity between dense and compressed 768-dimensional mean-pooled embeddings across five sentence pairs.
- **System metrics:** exact serialized model size, median CPU latency, and 95th-percentile latency.
- **Additional analysis:** layer/component sensitivity and sparse knowledge-distillation recovery at 45% global L1 sparsity.

The notebook also launches a Gradio dashboard for masked-token prediction and semantic embedding-drift comparisons.

## Repository contents

| File | Description |
| --- | --- |
| `Comprehensive_Context_Quantization_and_Pruning.ipynb` | Complete experiment, analysis, visualizations, and interactive dashboard |
| `plots/` | Generated after execution; contains eight comparison figures |
| `saved_artifacts/` | Generated after execution; contains serialized experiment models |

## Requirements

- Python 3.9+
- PyTorch
- A CPU is sufficient; a CUDA-enabled GPU is recommended for the full experiment
- Internet access on first run to download DistilBERT from Hugging Face

The notebook installs the remaining Python packages automatically:

```text
transformers>=4.40.0
datasets
accelerate
evaluate
gradio
matplotlib
seaborn
pandas
scipy
numpy
```

## Run the notebook

1. Open the notebook in Jupyter or Google Colab.
2. Run the cells from top to bottom.
3. Allow the model and tokenizer to download on the first run.
4. Review the comparison tables and figures.
5. Launch the final Gradio cell to interactively test compressed models.

The experiment fixes Python, NumPy, and PyTorch random seeds to `42`. Full execution evaluates 43 configurations and writes results to `saved_artifacts/` and `plots/`.

## Outputs

The notebook generates:

1. A master comparison table covering accuracy, embedding cosine similarity, sparsity, model size, and latency.
2. Eight figures:
   - accuracy versus sparsity
   - embedding drift
   - model size
   - latency
   - accuracy/size Pareto frontier
   - quantization comparison
   - layer sensitivity
   - distillation recovery
3. A live dashboard with:
   - masked-token top-5 predictions
   - inference latency and measured sparsity
   - sentence-pair similarity and embedding drift

## Important notes

- INT4 is **simulated** using per-channel uniform quantization; it is not hardware-backed 4-bit inference.
- Pruning is applied to linear-layer weights and then made permanent. Unstructured zeros do not automatically produce proportional CPU speedups on dense hardware.
- Reported latency depends on the execution environment, PyTorch build, and available hardware. Treat it as a local benchmark rather than a universal result.

## Reference

The pruning experiments are motivated by:

> Zafrir et al., “Prune Once for All: Sparse Pre-Trained Language Models,” 2021.

