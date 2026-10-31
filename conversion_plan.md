# Conversion plan by venue family

Venue names and scope were checked against official 2026 pages on October 30, 2026. Both 2026 submission deadlines have passed; these are verified format/scope examples, and a future edition's call must be checked again before conversion.

## IEEE conference paper as drafted

- Format: retain the supplied IEEE conference Word template, two-column body, named styles, numeric references, and seven-page maximum assumption (six content pages plus references).
- Evidence expected: complete the missing precision and recovery runs, controlled CPU latency protocol, three-seed random control, and an encoder-only structured-pruning ablation. Keep the small prompt set explicitly qualitative; add a held-out pseudo-perplexity test if feasible.
- Claims: describe a single-model pilot; avoid generalization, state-of-the-art, and hardware-acceleration claims. Keep Prune Once for All as related work only.

## ACL-family workshop: ACL 2026 Student Research Workshop

- Scope: the ACL 2026 Student Research Workshop is a forum for students working across computational linguistics, NLP, and machine learning. This empirical NLP pilot fits the broad subject scope if the author meets student eligibility. The direct-submission and ARR deadlines were March 18 and April 15, 2026; the conference ran July 5-7, 2026, so this is a future-edition plan, not a live 2026 submission option. Official call: https://2026.aclweb.org/calls/student_research_workshop/.
- Format: use the current *ACL template, anonymization, and that edition's page and limitation requirements. Replace the IEEE title/author block, Roman section labels, and numeric citation style with the workshop's required ACL conventions; check whether the venue uses ARR review.
- Likely reviewer requests: more than one encoder (e.g., BERT-base and RoBERTa), standard downstream evaluation such as GLUE or SQuAD, stronger held-out MLM evaluation, repeated seeds, a repair/recovery ablation, and comparison with established pruning and quantization baselines.
- Claims: state that results concern DistilBERT and a small probe suite; avoid model-wide or linguistic-generalization claims. Clarify that fake INT4 is a numerical simulation and that sparse masks were not evaluated with sparse kernels.

## Efficient-ML workshop: AXIOM at NeurIPS 2026

- Scope: AXIOM (Foundations of Efficient Deep Learning) explicitly lists pruning, quantization, sparsity, compression, representation learning for efficiency, and preliminary empirical results. The 2026 short-paper limit is four pages excluding references, with the NeurIPS workshop template and double-blind review. The submission deadline was August 29, 2026; only camera-ready for accepted papers is still due November 15, 2026. Official CFP: https://axiom-neurips2026.github.io/.
- Format: convert from IEEE Word to the NeurIPS 2026 workshop format, anonymize for review, compress to four pages plus references, and follow the workshop's specific checklist. This draft is not a camera-ready submission and was not submitted.
- Likely reviewer requests: multiple models including BERT-base, RoBERTa, and a small decoder LM; GLUE/SQuAD or held-out language-modeling results; real INT4 implementation such as a supported bitsandbytes or GPTQ pipeline; fine-tuning after pruning; sparse-runtime latency on compatible hardware; and full repeated-seed uncertainty reporting.
- Claims: distinguish theoretical size, serialized size, and runtime performance. Remove any speed claim unless verified on a named sparse runtime and hardware. Present this paper as a pilot unless the expanded experiments justify broader conclusions.
