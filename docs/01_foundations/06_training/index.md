# 06 - Training Foundations & Mental Models

## Executive Overview

The **Training Foundations** module establishes the mathematical, structural, and algorithmic principles required to understand how Transformer models learn, optimize parameters, and generate sequences. It decouples theoretical concepts from runtime pipeline scripts, providing detailed breakdowns of attention mechanics, tokenization dynamics, loss optimization, and precision budgeting.

---

## Core Training Principles

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 1. The Operational Scaling Law                                         │
│    Data Volume >> Vocab Size ∝ Model Dimension (d_model) ∝ Depth (N)   │
└────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────────┐
│ 2. The Mixed-Precision Static State Budget (16 Bytes / Parameter)      │
│    M_static = Weights (2B) + Gradients (2B) + AdamW FP32 States (12B)  │
└────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Target-Only Supervised Fine-Tuning Loss                             │
│    L_SFT(θ) = -(1 / |Y|) * Σ log P_θ(y_t | x, y_<t), Prompt Mask = -100│
└────────────────────────────────────────────────────────────────────────┘

```

---

## Module Roadmap

| Module                                                                                                                                              | Core Subject              | Primary Focus & Deliverables                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[01 Foundational Primer](https://www.google.com/search?q=01_foundational_primer/01_permutation_invariance_and_positions.md)**                     | Transformer Mechanics     | Permutation invariance, QKV matrix retrieval, Attention vs. FFN anatomy, residual highway gradients, and VRAM memory arithmetic.                               |
| **[02 Data Regimes & Tokenization](https://www.google.com/search?q=02_data_regimes_and_tokenization/01_ingestion_topologies_pattern_a_b_cache.md)** | Data Dynamics & Ingestion | Ingestion patterns (A, B, and Ephemeral Gold Cache), the capacity paradox on small corpora, Char vs. BPE vs. BBPE, vocab compression, and sequence packing.    |
| **[03 Architectural Topologies](https://www.google.com/search?q=03_architectural_topologies/01_autoregressive_decoder_minigpt.md)**                 | Model Architectures       | Decoder-only (MiniGPT), Masked Encoder (MiniBERT), Seq2Seq (MiniT5), attention variants (MHA, GQA, MQA, SDPA), RoPE, SwiGLU, and symmetric Weight Tying.       |
| **[04 Optimization & Loss Dynamics](https://www.google.com/search?q=04_optimization_and_loss_dynamics/01_safety_perimeter_gates_1_to_4.md)**        | Training Dynamics         | Gates 1 to 4 safety checks, target-only loss masking ($-100$), AdamW momentum states, linear warmup + cosine decay, gradient accumulation, and early stopping. |
| **[05 Inference Decoding & Calibration](https://www.google.com/search?q=05_inference_decoding_and_calibration/01_kv_caching_mechanics.md)**         | Generation & Inference    | $\mathcal{O}(T)$ KV-caching acceleration, greedy vs. multinomial sampling, logit temperature scaling, Top-$k$/Top-$p$ nucleus sampling, and offline testing.   |
| **[06 Infrastructure & Resilience](https://www.google.com/search?q=06_infrastructure_and_resilience/01_zero_framework_modular_pytorch.md)**         | System Engineering        | Zero-framework modular PyTorch, immutable Git SHA logging, `latest` vs. `best` checkpoint policies, and OOM fault tolerance.                                   |

---

## Detailed Sub-Chapter Breakdown

### 01 Foundational Primer

- **[01 Permutation Invariance & Positions](https://www.google.com/search?q=01_foundational_primer/01_permutation_invariance_and_positions.md):** Why attention treats inputs as unordered sets and how position embeddings inject spatial coordinates.
- **[02 QKV Retrieval Analogy](https://www.google.com/search?q=01_foundational_primer/02_qkv_retrieval_analogy.md):** Projecting tokens into search queries, key tags, and value content via scaled dot-product attention.
- **[03 Attention vs FFN Anatomy](https://www.google.com/search?q=01_foundational_primer/03_attention_vs_ffn_anatomy.md):** Social communication across tokens (Attention) versus pointwise memory recall and non-linear reflection (FFN).
- **[04 Residual Highways & Gradients](https://www.google.com/search?q=01_foundational_primer/04_residual_highways_and_gradients.md):** Skip connections ($x + \text{SubLayer}(x)$) as identity conduits preventing vanishing gradients.
- **[05 VRAM & Precision Budgeting](https://www.google.com/search?q=01_foundational_primer/05_vram_and_precision_budgeting.md):** Memory formulas for FP32/BF16/FP16 weights, activations ($\mathcal{O}(L^2)$), and 16-byte/param AdamW allocations.

### 02 Data Regimes & Tokenization

- **[01 Ingestion Topologies](https://www.google.com/search?q=02_data_regimes_and_tokenization/01_ingestion_topologies_pattern_a_b_cache.md):** Contrasting Pattern A (static pre-tokenized storage) with Pattern B (JIT compilation) and Ephemeral Gold caching.
- **[02 Capacity Paradox in Small Regimes](https://www.google.com/search?q=02_data_regimes_and_tokenization/02_capacity_paradox_in_small_regimes.md):** Overcoming memorization in data-constrained regimes through structural regularization.
- **[03 Target Benchmarks & Corpora](https://www.google.com/search?q=02_data_regimes_and_tokenization/03_target_benchmarks_and_corpora.md):** Baseline environments (TinyShakespeare, TinyStories, WikiText-103) for rapid validation.
- **[04 Char vs BPE vs Byte-Level BBPE](https://www.google.com/search?q=02_data_regimes_and_tokenization/04_char_vs_bpe_vs_byte_level_bbpe.md):** Progression from character encoding to sub-word BPE, and internalizing syntax/whitespace delimiters with BBPE.
- **[05 Vocab Compression & Stuttering](https://www.google.com/search?q=02_data_regimes_and_tokenization/05_vocab_compression_and_stuttering.md):** Limiting vocabulary size to prevent embedding table hypertrophy and token stuttering during inference.
- **[06 Special Tokens & Delimiters](https://www.google.com/search?q=02_data_regimes_and_tokenization/06_special_tokens_and_delimiters.md):** Injecting structural delimiters (`<|im_start|>`, `<|im_end|>`, `[MASK]`, `<SOS>`).
- **[07 Conversation Splitting & Isolation](https://www.google.com/search?q=02_data_regimes_and_tokenization/07_conversation_splitting_and_isolation.md):** Partitioning datasets at document boundaries to eliminate intra-dialogue leakage.
- **[08 Sequence Packing Mechanics](https://www.google.com/search?q=02_data_regimes_and_tokenization/08_sequence_packing_mechanics.md):** Concatenating dialogue chunks into fixed $B \times L$ matrices in `/dev/shm` to eliminate padding waste.

### 03 Architectural Topologies

- **[01 Autoregressive Decoder (MiniGPT)](https://www.google.com/search?q=03_architectural_topologies/01_autoregressive_decoder_minigpt.md):** Next-token causal generation constrained by lower-triangular attention masks (`tril`).
- **[02 Masked Encoder (MiniBERT)](https://www.google.com/search?q=03_architectural_topologies/02_masked_bidirectional_minibert.md):** Unmasked bidirectional representation learning optimized for masked token reconstruction.
- **[03 Seq2Seq Model (MiniT5)](https://www.google.com/search?q=03_architectural_topologies/03_seq2seq_encoder_decoder_minit5.md):** Coupling bidirectional encoding with autoregressive generation via Cross-Attention and Teacher Forcing.
- **[04 Attention Variants (MHA, GQA, MQA, SDPA)](https://www.google.com/search?q=03_architectural_topologies/04_attention_variants_mha_gqa_mqa_sdpa.md):** Multi-Head vs. Multi-Query vs. Grouped-Query trade-offs and FlashAttention / SDPA kernel acceleration.
- **[05 Positional Encodings (RoPE, ALiBi)](https://www.google.com/search?q=03_architectural_topologies/05_positional_encodings_rope_alibi.md):** Absolute lookup tables versus Rotary Position Embeddings and linear attention biases.
- **[06 Activations & Normalization](https://www.google.com/search?q=03_architectural_topologies/06_activations_and_normalization.md):** Mitigating dead neurons with GELU/SwiGLU and stabilizing deep networks with Pre-LayerNorm / RMSNorm.
- **[07 Symmetric Weight Tying](https://www.google.com/search?q=03_architectural_topologies/07_symmetric_weight_tying.md):** Sharing tensors between the Embedding Matrix and LM Head to cut parameter counts and stabilize embeddings.

### 04 Optimization & Loss Dynamics

- **[01 Safety Perimeter (Gates 1-4)](https://www.google.com/search?q=04_optimization_and_loss_dynamics/01_safety_perimeter_gates_1_to_4.md):** Structural verification points enforcing file integrity, schema consistency, split isolation, and tensor shapes.
- **[02 SFT Cross-Entropy & Loss Masking](https://www.google.com/search?q=04_optimization_and_loss_dynamics/02_sft_cross_entropy_and_loss_masking.md):** Minimizing cross-entropy loss strictly across assistant turns by setting prompt token labels to $-100$.
- **[03 Gradient Mechanics & Norm Clipping](https://www.google.com/search?q=04_optimization_and_loss_dynamics/03_gradient_mechanics_and_norm_clipping.md):** Parameter gradient updates, $\mathcal{O}(1)$ `zero_grad(set_to_none=True)` memory optimization, and $\|\nabla \theta\|_2 \le 1.0$ clipping.
- **[04 AdamW & Selective Weight Decay](https://www.google.com/search?q=04_optimization_and_loss_dynamics/04_adamw_and_selective_weight_decay.md):** First and second moment tracking with selective weight decay applied only to 2D matrices.
- **[05 LR Warmup & Cosine Annealing](https://www.google.com/search?q=04_optimization_and_loss_dynamics/05_lr_warmup_and_cosine_annealing.md):** Linear warmup ramps followed by smooth cosine decay down to $0.1 \times \eta_{\text{max}}$.
- **[06 Micro-Batching, Grad Accum & AMP](https://www.google.com/search?q=04_optimization_and_loss_dynamics/06_micro_batching_grad_accum_amp.md):** Large effective batch sizes ($B_{\text{eff}} = B_{\text{micro}} \times K_{\text{accum}}$) on memory-constrained GPUs via `torch.cuda.amp`.
- **[07 Early Stopping & Snapshotting](https://www.google.com/search?q=04_optimization_and_loss_dynamics/07_early_stopping_and_snapshotting.md):** Tracking validation loss plateaus with patience counters to persist optimal checkpoints.

### 05 Inference Decoding & Calibration

- **[01 KV-Caching Mechanics](https://www.google.com/search?q=05_inference_decoding_and_calibration/01_kv_caching_mechanics.md):** Caching Key and Value projections to reduce autoregressive inference complexity from $\mathcal{O}(T^2)$ to $\mathcal{O}(T)$.
- **[02 Deterministic vs Stochastic Sampling](https://www.google.com/search?q=05_inference_decoding_and_calibration/02_deterministic_vs_stochastic_sampling.md):** Comparing greedy argmax selection with multinomial probability sampling.
- **[03 Temperature, Top-k, Top-p](https://www.google.com/search?q=05_inference_decoding_and_calibration/03_temperature_top_k_top_p.md):** Modulating Softmax distribution entropy and filtering long-tail tokens.
- **[04 Offline Inference Verification](https://www.google.com/search?q=05_inference_decoding_and_calibration/04_offline_inference_verification.md):** Verifying checkpoint coherence and prompt adherence prior to pipeline evaluation.

### 06 Infrastructure & Resilience

- **[01 Zero-Framework Modular PyTorch](https://www.google.com/search?q=06_infrastructure_and_resilience/01_zero_framework_modular_pytorch.md):** Building transparent model architectures without third-party wrapper bloat.
- **[02 Cryptographic Provenance & Git SHA](https://www.google.com/search?q=06_infrastructure_and_resilience/02_cryptographic_provenance_and_git_sha.md):** Binding runs to immutable Git commit hashes and dataset lineage hashes.
- **[03 Checkpoint Lineage (Latest vs Best)](https://www.google.com/search?q=06_infrastructure_and_resilience/03_checkpoint_lineage_latest_vs_best.md):** Managing recovery checkpoints (`latest`) versus validated minimal-loss production checkpoints (`best`).
- **[04 Fault-Tolerant Compute Lifecycle](https://www.google.com/search?q=06_infrastructure_and_resilience/04_fault_tolerant_compute_lifecycle.md):** Handling Out-Of-Memory (OOM) exceptions and orchestrating automated GPU teardowns.
