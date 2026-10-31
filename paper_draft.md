# Empirical Evaluation of Pruning Regimes and Multi-Bit Quantization on Contextual Representations in DistilBERT

## [Author Name]
[Dept.]
[Affiliation]
[City, Country]
[Email]

## Abstract
Abstract—This paper reports a controlled empirical pilot study of one-shot, post-training compression on DistilBERT. We evaluate local and global L1-norm unstructured pruning, structured L2 row pruning, and random unstructured pruning across five target sparsity levels (15% to 75%), alongside 16-bit floating-point (FP16), dynamic 8-bit integer (INT8), and simulated 4-bit integer (INT4) quantization. Evaluation on a 16-prompt qualitative context probe and five semantic sentence pairs indicates that magnitude pruning preserves contextual prediction at 15% sparsity (43.75% top-1 and 75.00% top-5 accuracy versus the 37.50% and 75.00% dense baseline) with an embedding cosine similarity of 0.9903. However, severe degradation occurs between 30% and 45% sparsity, collapsing to 12.50% top-1 accuracy. Random pruning collapses at 30% sparsity, while structured pruning yields 0.00% accuracy across all tested levels without retraining. Unstructured sparsity yields no CPU inference speedup on dense kernels (170.87 ms at 45% sparsity versus 170.33 ms dense baseline). Post-pruning knowledge distillation over three epochs recovers top-1 accuracy from 12.50% to 18.75% and top-5 accuracy from 12.50% to 37.50% while preserving 45.00% exact sparsity.

## Keywords
Contextual representations, DistilBERT, knowledge distillation, model compression, neural network pruning, quantization.

## I. Introduction
Transformer-based language models exhibit significant parameter redundancy, motivating extensive investigation into model compression techniques such as weight pruning and reduced-precision quantization [1], [2]. While large-scale language models demand substantial computational infrastructure for deployment, compact transformer architectures such as DistilBERT [3] provide an accessible testbed for isolating the fundamental trade-offs between parameter sparsity, numerical precision, and representation quality. DistilBERT contains 66,985,530 total parameters across six transformer layers, with an encoder parameter count of 66,362,880, 12 self-attention heads, a hidden dimensionality of 768, and a vocabulary size of 30,522. 

In resource-constrained edge computing environments, post-training compression is frequently applied as a one-shot operation where extensive retraining or task-specific fine-tuning is computationally prohibitive. However, the precise empirical boundaries where one-shot magnitude pruning and multi-bit quantization preserve or degrade contextual token representations remain underexplored in small-scale transformers. Specifically, questions persist regarding how cross-layer global sparsity allocation differs from uniform per-layer pruning, how structured channel removal behaves in the absence of retraining, and whether post-training quantization introduces compounding degradation when applied to already sparse weight tensors.

This paper positions itself as a controlled empirical pilot study of one-shot, post-training compression on a compact language model, rather than presenting a novel algorithmic framework. The primary contributions of this work are fourfold:
1) A controlled comparative evaluation of four distinct pruning regimes (local L1 magnitude, global L1 magnitude, structured L2 row pruning, and random pruning) across five target sparsity levels (15%, 30%, 45%, 60%, and 75%) on contextual token accuracy and embedding stability.
2) An empirical multi-precision evaluation encompassing FP32, FP16, dynamic INT8, and simulated INT4 quantization, examining both isolated precision degradation and the compounding effects of combining pruning with quantization.
3) A component-level and layer-wise sparsity audit under global magnitude pruning, quantifying how unconstrained pruning naturally distributes zero weights between multi-head self-attention mechanisms and feed-forward networks (FFN).
4) A reproducible evaluation harness featuring controlled CPU latency benchmarking with warm-up protocols and a post-pruning knowledge distillation pipeline evaluated on identical probe distributions.

Given the pilot scale of the 16-prompt qualitative context probe and 10-sentence embedding similarity benchmark, all findings are interpreted conservatively using pilot-level phrasing. The empirical evidence demonstrates consistent patterns of representation stability and collapse, but does not claim state-of-the-art compression efficiency, architectural generality across diverse model families, or execution acceleration on specialized sparse hardware.

## II. Literature Review
Transformer compression encompasses knowledge distillation, weight pruning, and quantization. DistilBERT demonstrated that student models initialized with alternating layers of a teacher BERT architecture can retain approximately 97% of downstream language understanding capabilities while reducing parameter count by 40% and accelerating inference by 60% through pretraining distillation [3]. Related distillation frameworks, including TinyBERT [4] and ALBERT [5], explored intermediate layer matching and cross-layer parameter sharing to achieve extreme parameter efficiency. Comprehensive analyses of transformer representations (commonly termed "BERTology") emphasize that internal representations are sensitive to layer depth, attention head pruning, and evaluation methodology [6], [7].

Pruning methodologies in natural language processing typically diverge into unstructured magnitude pruning and structured group pruning. Early investigations established that pretrained transformers can tolerate substantial weight removal prior to fine-tuning on downstream tasks [8]. Movement pruning introduced first-order gradient awareness during task-specific fine-tuning, outperforming magnitude-based selection at high sparsity regimes [9]. In contrast, one-shot post-training pruning techniques, including SparseGPT [17] and Wanda [18], employ second-order Hessian approximations or activation-magnitude products to prune large language models without extensive retraining. The lottery ticket hypothesis has also been confirmed in pretrained BERT models [26], establishing that sparse subnetworks exist within dense initializations. However, unstructured pruning creates irregular sparsity patterns that do not produce inference speedups on standard dense matrix multiplication hardware [14], [27].

Quantization reduces memory footprint and computational energy by mapping continuous floating-point weights to low-bit integers. Q8BERT [10] and Q-BERT [11] investigated 8-bit and ultra-low bit quantization for BERT using quantization-aware training (QAT) and Hessian-based mixed precision. TernaryBERT pushed representation limits to 2-bit ternary weights via distillation-aware training [12]. In the post-training quantization domain, modern frameworks such as SmoothQuant [20], LLM.int8() [21], GPTQ [19], and AWQ [22] resolve activation outliers through channel scaling and second-order error compensation. Systematic surveys demonstrate that post-training quantization is particularly vulnerable to outlier dimensions in attention projections [25], [28].

Classical compression pipelines pioneered by Han et al. demonstrated that combining pruning, trained quantization, and Huffman coding yields multiplicative reductions in storage footprint [13]. Recently, *Prune Once for All* integrated magnitude pruning with continuous knowledge distillation during sparse pretraining [15]. We emphasize that the present study does not reproduce or extend the *Prune Once for All* framework: while Zafrir et al. pretrain sparse models from scratch on massive corpora across millions of steps, our study examines one-shot, post-training magnitude and structured masks applied to an existing pretrained model without sparse pretraining. Furthermore, while cosine similarity is widely utilized as a proxy for representation alignment, prior geometric studies demonstrate that contextual embeddings naturally occupy anisotropic cones [24], necessitating careful analysis alongside token prediction accuracy [23].

## III. Method-1: Pruning Regimes and Representation Probing
### A. Architectural Context and Probe Harness
The target architecture is `distilbert-base-uncased`, featuring $L=6$ transformer layers, $A=12$ attention heads, hidden dimension $d_h=768$, intermediate feed-forward dimension $d_{ff}=3072$, and vocabulary size $V=30,522$. The dense FP32 baseline consumes 255.57 MB of serialized storage and achieves 37.50% Top-1 accuracy and 75.00% Top-5 accuracy on the context comprehension probe suite.

The qualitative probe harness comprises two complementary evaluation suites:
1) *Masked-Token Context Comprehension Probe:* A curated set of 16 multi-domain sentences spanning syntactic agreement, common-sense world knowledge, computer science concepts, and complex relational reasoning. Each sentence contains a single `[MASK]` token with a deterministic ground-truth target word. Prediction accuracy is evaluated at Top-1 and Top-5 rank positions:
$$Acc_{\text{Top}-k} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( y_i \in \text{Top-}k\left( \mathcal{M}(x_i) \right) \right)$$
where $N=16$, $y_i$ is the target token, and $\mathcal{M}(x_i)$ represents the vocabulary logit distribution produced at the masked index. Because $N=16$, a single correct prediction alters measured accuracy by exactly 6.25 percentage points.

2) *Contextual Embedding Stability:* Semantic drift is quantified across five sentence pairs (10 sentences total) representing diverse semantic domains. For each sentence $s$, 768-dimensional sentence representations $e(s)$ are extracted via mean-pooling across the sequence length ($T=64$):
$$e(s) = \frac{\sum_{t=1}^T m_t \cdot h_t}{\sum_{t=1}^T m_t}$$
where $h_t \in \mathbb{R}^{768}$ is the final hidden state of token $t$, and $m_t \in \{0, 1\}$ is the attention mask. Embedding stability is defined as the mean cosine similarity between the dense baseline encoder $f_{\text{dense}}$ and the compressed encoder $f_{\text{comp}}$:
$$\text{CosSim}(f_{\text{dense}}, f_{\text{comp}}) = \frac{1}{2P} \sum_{p=1}^P \sum_{j=1}^2 \frac{e_{\text{dense}}(s_{p,j}) \cdot e_{\text{comp}}(s_{p,j})}{\|e_{\text{dense}}(s_{p,j})\|_2 \|e_{\text{comp}}(s_{p,j})\|_2}$$
where $P=5$ sentence pairs.

### B. One-Shot Pruning Formulations
Pruning is applied to all eligible weight tensors $W \in \mathbb{R}^{d_{out} \times d_{in}}$ in the `nn.Linear` modules of the model. A binary mask $M \in \{0, 1\}^{d_{out} \times d_{in}}$ is computed according to a scoring function $\mathcal{S}(W)$, producing the pruned weight tensor:
$$W' = M \odot W \quad \text{such that} \quad \frac{\|M\|_0}{\text{numel}(W)} = 1 - S$$
where $\odot$ denotes the Hadamard product and $S \in \{0.15, 0.30, 0.45, 0.60, 0.75\}$ is the target sparsity. Masks are permanently consolidated into the model parameters using `prune.remove`, setting pruned entries to structural zeros.

We investigate four distinct operational pruning regimes:
1) *Local L1-Norm Unstructured Pruning:* Operates independently on each individual `nn.Linear` module. The bottom $S \times 100\%$ elements with the smallest absolute magnitude $|W_{ij}|$ are pruned within each layer, ensuring an identical sparsity fraction across all network layers:
$$M_{ij} = \mathbb{I}\left( |W_{ij}| \ge \tau_l \right), \quad \tau_l = \text{Percentile}_{S}\left( |W^{(l)}| \right)$$

2) *Global L1-Norm Unstructured Pruning:* Pools all eligible Linear weight matrices $\mathcal{W} = \bigcup_l W^{(l)}$ into a single global parameter collection ($|\mathcal{W}| = 42,467,328$ Linear parameters). A single global threshold $\tau_g$ is determined across the union of weights:
$$M_{ij}^{(l)} = \mathbb{I}\left( |W_{ij}^{(l)}| \ge \tau_g \right), \quad \tau_g = \text{Percentile}_{S}\left( |\mathcal{W}| \right)$$
This enables adaptive cross-layer sparsity allocation, permitting over-parameterized layers to absorb higher sparsity while preserving sensitive submodules.

3) *Structured L2-Norm Row Pruning:* Removes entire output channels (rows) along dimension 0 of weight matrices based on the Euclidean norm of row vectors:
$$M_{i,:} = \mathbb{I}\left( \|W_{i,:}\|_2 \ge \tau_{\text{row}} \right), \quad \tau_{\text{row}} = \text{Percentile}_{S}\left( \left\{ \|W_{k,:}\|_2 \right\}_{k=1}^{d_{out}} \right)$$
In our primary structured pruning implementation, pruning is restricted to the transformer encoder modules (`q_lin`, `k_lin`, `v_lin`, `out_lin`, `lin1`, `lin2`), preserving the classification head. An ablation is conducted on the original notebook behavior, which also pruned rows of the vocabulary projection matrix.

4) *Random Unstructured Pruning (Control):* Samples mask entries uniformly at random from a Bernoulli distribution parameterized by $(1 - S)$:
$$M_{ij} \sim \text{Bernoulli}(1 - S)$$
To provide statistical rigor, random pruning is evaluated across three independent random seeds ($\text{seed} \in \{42, 43, 44\}$).

### C. Representation Recovery via Knowledge Distillation
To investigate whether representation degradation at moderate sparsity can be mitigated without retraining on massive pretraining corpora, a lightweight post-pruning knowledge distillation pipeline is implemented. The student model $f_{\text{student}}$ is initialized from the 45% Global L1 pruned model. The teacher model $f_{\text{teacher}}$ is a frozen copy of the dense FP32 baseline.

Optimization minimizes the Kullback-Leibler (KL) divergence between student and teacher prediction distributions at temperature $T=2.0$ over an 8-sentence domain corpus:
$$\mathcal{L}_{\text{KD}} = T^2 \cdot D_{\text{KL}}\left( \sigma\left( \frac{z_{\text{teacher}}}{T} \right) \parallel \sigma\left( \frac{z_{\text{student}}}{T} \right) \right)$$
where $z$ represents output vocabulary logits and $\sigma(\cdot)$ is the softmax function. To prevent pruned zero-weights from regrowing during gradient updates, an exact binary gradient mask $M_{\text{grad}}$ is enforced prior to each optimization step:
$$g_W \leftarrow g_W \odot \mathbb{I}(W \neq 0)$$
The model is trained for 3 epochs using AdamW with learning rate $\eta = 3 \times 10^{-5}$. Hidden-state representation matching and attention-map matching are not implemented in the pipeline.

## IV. Method-2: Multi-Bit Quantization Framework
### A. Numerical Precision Formulations
Quantization maps 32-bit floating-point parameters to lower-bit representations. We examine three operational precision levels:

1) *Half-Precision Floating-Point (FP16):* Parameters are cast from IEEE 754 single-precision (32-bit, 1 sign, 8 exponent, 23 mantissa) to half-precision (16-bit, 1 sign, 5 exponent, 10 mantissa):
$$W_{\text{FP16}} = \text{float16}(W_{\text{FP32}})$$
FP16 provides a theoretical 2.0$\times$ reduction in weight memory footprint while maintaining dynamic range.

2) *Dynamic 8-Bit Integer Quantization (INT8):* Weights of all `nn.Linear` modules are converted post-training to signed 8-bit integers (`torch.qint8`) using symmetric affine quantization:
$$W_{\text{INT8}} = \text{clamp}\left( \left\lfloor \frac{W}{\text{scale}} \right\rceil, -128, 127 \right), \quad \text{scale} = \frac{\max(|W|)}{127}$$
During inference, dynamic activations are quantized on-the-fly, enabling optimized integer matrix multiplication kernels on CPU architectures with vector extensions (AVX-512 / VNNI).

3) *Simulated 4-Bit Integer Quantization (INT4):* Weights are mapped into 16 discrete symmetric integer bins $[-8, 7]$ using per-output-channel scaling factors:
$$s_c = \frac{\max_{j} |W_{c,j}|}{7.0}, \quad q_{c,j} = \text{clamp}\left( \left\lfloor \frac{W_{c,j}}{s_c} \right\rceil, -8, 7 \right)$$
$$\hat{W}_{c,j} = q_{c,j} \cdot s_c$$
where $c$ indexes the output channel dimension. This represents fake quantization: weights are immediately dequantized back to floating-point tensors for execution. Because custom packed-INT4 inference kernels are not utilized, runtime acceleration is not evaluated for INT4. The theoretical memory footprint is computed as:
$$\text{Payload}_{\text{INT4}} = \frac{N_{\text{params}} \times 4 \text{ bits}}{8 \text{ bits/byte} \times 1024^2 \text{ bytes/MiB}} = 31.94 \text{ MiB}$$
This figure represents a mathematical payload calculation and does not include scale factor metadata, graph serialization overhead, or container headers.

### B. Multi-Dimensional Grid Composition
The experimental matrix combines pruning and quantization across 42 systematically evaluated configurations. Pruned models generated under Method-1 are subsequently passed through dynamic INT8 quantization to evaluate compound compression effects. Additionally, an aggressive combined baseline pairing 30% Global L1 pruning with simulated INT4 quantization is evaluated to explore the limits of extreme one-shot parameter reduction.

## V. Results
Experimental results across all 42 configurations are compiled into `results.csv`. Table I summarizes the primary comparative performance across pruning regimes, and Table II details precision and combined compression behaviors.

### A. Pruning Regimes and Degradation Cliff
Fig. 2 illustrates Top-5 context accuracy as a function of target sparsity, while Fig. 3 displays embedding cosine similarity.

At 15% target sparsity, both Local L1 and Global L1 unstructured pruning retain high contextual fidelity, achieving 43.75% Top-1 accuracy and 75.00% Top-5 accuracy. Notably, the Top-1 score exceeds the dense baseline (37.50%) by 6.25 percentage points (reflecting one additional correct prediction in the 16-prompt suite), while embedding cosine similarity remains near-perfect at 0.9903 (Local) and 0.9882 (Global). This indicates that the 15% lowest-magnitude weights in DistilBERT represent non-essential parameter noise.

However, a severe degradation cliff emerges between 30% and 45% sparsity. At 30% sparsity, Top-1 accuracy drops to 25.00% for both magnitude methods, though Top-5 accuracy remains robust at 75.00% for Global L1 and 62.50% for Local L1. By 45% sparsity, Top-1 and Top-5 accuracy simultaneously collapse to 12.50%, with cosine similarity falling to 0.8325 (Local) and 0.7791 (Global). Beyond 45% sparsity, Top-1 accuracy collapses to 0.00% across all regimes.

Random pruning exhibits substantially accelerated failure: at 15% sparsity, 3-seed mean Top-1 accuracy drops to $18.75 \pm 5.10\%$ and Top-5 accuracy drops to $37.50 \pm 10.21\%$. At 30% sparsity and above, random pruning yields $0.00 \pm 0.00\%$ Top-1 and Top-5 accuracy, confirming that magnitude-based selection preserves essential functional circuits far more effectively than random weight elimination.

Structured L2 row pruning without retraining fails completely across all sparsity levels, yielding 0.00% Top-1 and Top-5 accuracy at 15%, 30%, 45%, and 75% sparsity (with an isolated 6.25% Top-5 anomaly at 60% encoder-only sparsity). Setting entire output rows to zero fundamentally severs information transmission across attention heads and feed-forward pathways. Crucially, as shown in Fig. 3, cosine similarity for structured pruning paradoxically *increases* from 0.2642 at 15% to 0.4710 at 75% in the all-Linear setting, and from 0.4129 at 30% to 0.4985 at 75% in the encoder-only setting. This increase reflects geometric collapse: as more channels are zeroed, representation vectors collapse toward a trivial uniform directional subspace. Cosine similarity alone is therefore an unreliable indicator of semantic preservation in the presence of structured degradation.

### B. Quantization Precision Effects
As detailed in Table II and Fig. 4, standalone quantization exhibits distinct trade-offs:
- *Dense FP16:* Matches dense FP32 accuracy exactly (37.50% Top-1, 75.00% Top-5) with a cosine similarity of 1.0000, while reducing serialized storage from 255.57 MB to 127.80 MB (50.0% reduction). FP16 represents a zero-degradation compression path for compatible runtimes.
- *Dense Dynamic INT8:* Yields 43.75% Top-1 and 50.00% Top-5 accuracy, maintaining a high cosine similarity of 0.9450 while reducing model size to 154.76 MB (39.4% reduction). The slight Top-1 improvement over baseline is an artifact of discrete quantization nudging borderline logits on the 16-prompt suite.
- *Dense Simulated INT4:* Retains 18.75% Top-1 and 75.00% Top-5 accuracy with a cosine similarity of 0.9228, against a theoretical payload of 31.94 MiB (87.5% parameter footprint reduction). While Top-1 accuracy incurs substantial loss (-18.75 pp), Top-5 accuracy matches the dense baseline, indicating that simulated 4-bit discrete bins preserve top-candidate ranking.

### C. Combined Compression Interactions
Applying dynamic INT8 quantization on top of pruned models produces compounding representation loss. At 15% sparsity, Local L1 + INT8 achieves 50.00% Top-1 and 75.00% Top-5 accuracy (CosSim 0.9370), and Global L1 + INT8 achieves 43.75% Top-1 and 75.00% Top-5 accuracy (CosSim 0.9366). However, at 30% sparsity, Global L1 + INT8 Top-1 accuracy drops to 12.50% and Top-5 drops to 37.50% (compared to 25.00% and 75.00% for FP32 Global L1). At 45% sparsity, INT8 quantization collapses Top-1 accuracy to 0.00% across all regimes.

Similarly, the combined Global L1 30% + simulated INT4 condition achieves 25.00% Top-1 and 37.50% Top-5 accuracy with a cosine similarity of 0.9109, demonstrating that 4-bit quantization degrades candidate rankings when applied to a 30% sparse model.

### D. Component Allocation and Latency Realities
1) *Global Sparsity Allocation Audit:* Fig. 6 displays the layer-by-component sparsity heatmap under Global L1 pruning at a 45% target. Unconstrained global magnitude pruning selectively concentrates zeros in the feed-forward networks (52.57% aggregate sparsity across 28,311,552 weights) compared to multi-head self-attention mechanisms (49.70% aggregate sparsity across 14,155,776 weights).

Within individual layers, self-attention output projections (`out_lin`) and value projections (`v_lin`) in early layers exhibit high sparsity: Layer 0 `out_lin` is 62.40% sparse, and Layer 0 `v_lin` is 60.84% sparse. Conversely, query projections (`q_lin`) and key projections (`k_lin`) in intermediate layers retain higher density (Layer 1 `q_lin` is 42.31% sparse; Layer 1 `k_lin` is 43.17% sparse). This empirical allocation suggests that feed-forward capacity exhibits greater magnitude redundancy than attentional routing mechanisms.

2) *Inference Latency Realities on Dense Hardware:* The latency benchmark, conducted with 20 warm-up runs and 100 timed iterations under a single CPU thread on an Intel64 processor, resolves a critical empirical disconnect. The dense FP32 baseline exhibits a median latency of 170.33 ms (P95: 204.37 ms). The 45% Global L1 pruned FP32 model exhibits a median latency of 170.87 ms (P95: 187.91 ms).

This measurement proves that unstructured weight pruning yields zero inference acceleration on standard dense CPU execution kernels. The apparent acceleration reported in the original notebook (~133 ms) was a warm-up artifact of benchmarking without adequate iteration counts. In contrast, dynamic INT8 quantization achieves genuine hardware acceleration on CPU, delivering a median latency of 95.03 ms (P95: 113.19 ms), representing a 1.79$\times$ execution speedup attributable to 8-bit integer vector arithmetic.

3) *Structured Pruning Implementation Ablation:* In the original notebook implementation, `prune_structured_ln` applied row pruning across all `nn.Linear` layers, inadvertently zeroing rows of `vocab_projector` ($30522 \times 768$). In our Step 0 ablation, isolating structured pruning to encoder modules improved cosine similarity (0.4889 vs. 0.2642 at 15% sparsity), but failed to restore token accuracy (0.00% Top-1/5 across both implementations), demonstrating that architectural channel deletion inherently destroys representations without post-pruning fine-tuning.

### E. Representation Recovery via Distillation
The lightweight knowledge distillation recovery experiment validates the potential of post-pruning fine-tuning. Commencing from the collapsed Global L1 45% model (Pre-Distillation: Top-1 12.50%, Top-5 12.50%, CosSim 0.7791), three epochs of logit KL distillation on 8 domain sentences reduced training loss from 213.13 to 106.56.

As shown in Table II, post-distillation evaluation yields:
- Top-1 Accuracy: 12.50% $\rightarrow$ 18.75% (+6.25 pp)
- Top-5 Accuracy: 12.50% $\rightarrow$ 37.50% (+25.00 pp)
- Embedding Cosine Similarity: 0.7791 $\rightarrow$ 0.8773 (+0.0982)
- Sparsity Maintained: Exactly 45.00% zero-weights preserved via gradient masking.

While full dense performance (37.50% / 75.00%) is not fully recovered due to the miniature corpus size (8 sentences), the tripling of Top-5 candidate accuracy confirms that representation collapse is partially reversible through gradient-masked logit matching.

## VI. Conclusion
This empirical study systematically evaluated one-shot pruning regimes and multi-bit quantization on DistilBERT. The findings establish that:
1) One-shot magnitude pruning preserves contextual comprehension up to 15% sparsity (43.75% Top-1, 75.00% Top-5), but encounters a catastrophic degradation cliff between 30% and 45% sparsity.
2) Random pruning collapses earlier (30% sparsity), while structured row pruning without retraining produces total functional failure across all sparsity targets.
3) Unstructured sparsity provides zero latency reduction on standard dense CPU kernels (170.87 ms vs. 170.33 ms baseline), whereas dynamic INT8 quantization provides an authentic 1.79$\times$ acceleration (95.03 ms).
4) Global magnitude pruning naturally concentrates sparsity in feed-forward layers (52.57%) over attention layers (49.70%).
5) Post-pruning knowledge distillation partially recovers masked-token accuracy (tripling Top-5 accuracy from 12.50% to 37.50%) while strictly maintaining exact zero-weight sparsity.

### A. Limitations and Ethics
This work is constrained by several methodological limitations. Evaluation relies on a single model architecture (`distilbert-base-uncased`) and a small qualitative probe suite (16 masked tokens, 5 sentence pairs), precluding statistical generalization to large language models, decoder architectures, or standard downstream benchmarks such as GLUE or SQuAD. Quantization for INT4 is simulated via fake quantization, and sparse tensors are executed on dense engines without sparse runtime acceleration. Furthermore, this study does not assess demographic fairness, social bias amplification, or toxicity drift, which are critical safety considerations prior to production compression.

### B. Reproducibility Statement
To ensure full scientific reproducibility, all code, evaluation scripts, and tabular outputs are provided in the repository. All Step 0 code corrections have been disclosed: (i) fixing the device mismatch in `evaluate_context_comprehension()` by utilizing a deep-copied CPU model; (ii) executing previously unrun FP16, dense INT8, dense INT4, and Global L1 30% + INT4 configurations; (iii) substituting the hardcoded INT4 size multiplier with a theoretical payload formula ($66.985\text{M} \times 4 / 8 / 1024^2 = 31.94$ MiB); (iv) benchmarking CPU latency across 100 timed runs with 20 warm-up runs on a single CPU thread; (v) evaluating random pruning across three random seeds; and (vi) isolating encoder-only structured pruning. Experiments were executed on an Intel64 platform using PyTorch 2.6.0+cpu and verified against the original Tesla T4 notebook environment. Assuming a standard conference format, this paper adheres to a seven-page budget.

## Acknowledgment
The authors acknowledge the open-source contributors of Hugging Face Transformers and the PyTorch foundation for providing the foundational models and profiling infrastructure.

## References
[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems, vol. 30, pp. 5998–6008, 2017.
[2] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in Proc. NAACL-HLT, pp. 4171–4186, 2019.
[3] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,” arXiv:1910.01108, 2019.
[4] X. Jiao, Y. Yin, L. Shang, X. Jiang, X. Chen, and Q. Liu, “TinyBERT: Distilling BERT for natural language understanding,” in Proc. EMNLP, pp. 4163–4174, 2020.
[5] Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, and R. Soricut, “ALBERT: A lite BERT for self-supervised learning of language representations,” in Proc. ICLR, 2020.
[6] A. Rogers, O. Kovaleva, and A. Rumshisky, “A primer in BERTology: What we know about how BERT works,” Transactions of the Association for Computational Linguistics, vol. 8, pp. 842–866, 2020, doi: 10.1162/tacl_a_00349.
[7] P. Ganesh, Y. Chen, R. Ran, R. F. Canedo, C. Chi, S. Naseh, B. Li, H. Dhawan, K. Chen, and A. Balasubramanian, “Compressing large-scale transformer-based models: A case study on BERT,” Transactions of the Association for Computational Linguistics, vol. 9, pp. 1061–1080, 2021, doi: 10.1162/tacl_a_00413.
[8] M. Gordon, K. Duh, and N. Andrews, “Compressing BERT: Studying the effects of weight pruning on transfer learning,” in Proc. 5th Workshop on Representation Learning for NLP (RepL4NLP), pp. 143–155, 2020, doi: 10.18653/v1/2020.repl4nlp-1.18.
[9] V. Sanh, T. Wolf, and A. M. Rush, “Movement pruning: Adaptive sparsity by fine-tuning,” in Advances in Neural Information Processing Systems, vol. 33, pp. 20378–20389, 2020.
[10] O. Zafrir, G. Boudoukh, P. Izsak, and M. Wasserblat, “Q8BERT: Quantized 8bit BERT,” in Proc. NeurIPS Workshop on Energy Efficient Machine Learning and Cognitive Computing, 2019, arXiv:1910.06188.
[11] S. Shen, Z. Dong, J. Ye, L. Lin, P. Yao, M. W. Mahoney, and K. Keutzer, “Q-BERT: Hessian based ultra low precision quantization of BERT,” in Proc. AAAI, vol. 34, no. 5, pp. 8815–8821, 2020.
[12] W. Zhang, L. Hou, Y. Yin, H. Shang, X. Chen, X. Jiang, and Q. Liu, “TernaryBERT: Distillation-aware ultra-low bit BERT,” in Proc. EMNLP, pp. 509–521, 2020.
[13] S. Han, H. Mao, and W. J. Dally, “Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding,” in Proc. ICLR, 2016, arXiv:1510.00149.
[14] S. Han, J. Pool, J. Tran, and W. J. Dally, “Learning both weights and connections for efficient neural networks,” in Advances in Neural Information Processing Systems, vol. 28, pp. 1135–1143, 2015.
[15] O. Zafrir, A. Larey, G. Boudoukh, H. Shen, and M. Wasserblat, “Prune once for all: Sparse pre-trained language models,” arXiv:2111.05754, 2021.
[16] W. Kwon, S. Kim, M. W. Mahoney, J. Hassoun, K. Keutzer, and A. Gholami, “A fast post-training pruning framework for transformers,” in Advances in Neural Information Processing Systems, vol. 35, pp. 24101–24116, 2022.
[17] E. Frantar and D. Alistarh, “SparseGPT: Massive language models can be accurately pruned in one-shot,” in Proc. ICML, pp. 10323–10337, 2023, arXiv:2301.00774.
[18] M. Sun, Z. Liu, A. Bair, and J. Z. Kolter, “A simple and effective pruning approach for large language models,” in Proc. ICLR, 2024, arXiv:2306.11695.
[19] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, “GPTQ: Accurate post-training quantization for generative pre-trained transformers,” in Proc. ICLR, 2023, arXiv:2210.17323.
[20] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, “SmoothQuant: Accurate and efficient post-training quantization for large language models,” in Proc. ICML, pp. 38087–38099, 2023.
[21] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, “LLM.int8(): 8-bit matrix multiplication for transformers at scale,” in Advances in Neural Information Processing Systems, vol. 35, pp. 30318–30332, 2022.
[22] J. Lin, J. Tang, H. Tang, C. Yang, X. Dang, C. Gan, and S. Han, “AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration,” in Proc. MLSys, 2024, arXiv:2306.00978.
[23] A. Kornblith, S. Norouzi, H. Lee, and G. Hinton, “Similarity of neural network representations revisited,” in Proc. ICML, pp. 3519–3529, 2019.
[24] K. Ethayarajh, “How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings,” in Proc. EMNLP-IJCNLP, pp. 55–65, 2019.
[25] Y. Bondarenko, M. Nagel, and T. Blankevoort, “Understanding and overcoming the challenges of efficient transformer quantization,” in Proc. EMNLP, pp. 7947–7969, 2021.
[26] T. Chen, J. Frankle, S. Chang, S. Liu, Y. Yang, Z. Wang, and L. Carin, “The lottery ticket hypothesis for pre-trained BERT networks,” in Advances in Neural Information Processing Systems, vol. 33, pp. 15834–15846, 2020.
[27] M. Zhu and S. Gupta, “To prune, or not to prune: Exploring the efficacy of pruning for model compression,” in Proc. ICLR Workshop, 2018, arXiv:1710.01878.
[28] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, and D. Kalenichenko, “Quantization and training of neural networks for efficient integer-arithmetic-only inference,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), pp. 2704–2713, 2018.
