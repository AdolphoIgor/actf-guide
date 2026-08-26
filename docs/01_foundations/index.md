# 01 - Foundations

## Executive Overview

The **Foundations** section establishes the theoretical mental models and architectural principles required for production LLMOps. It isolates theoretical foundations from operational pipeline execution, covering distributed data topologies, Transformer attention mechanics, tokenization dynamics, optimization calculus, inference calibration, and post-training governance frameworks.

---

## Foundation Modules

### Core Operational Principles

- **[01 Enterprise LLMOps](01_the_enterprise_llmops.md):** Full-stack architecture connecting data warehouses, distributed ETL, training engines, and model registries.
- **[02 Continuous Training Role](02_the_continuous_training_role.md):** The scope of continuous training specialists, automated lifecycle triggers, and production risk boundaries.
- **[03 Medallion Architecture](03_the_medallion_architecture.md):** Storage lifecycle management across Bronze (raw landing), Silver (universal cleaned text), and Gold (tokenized feature stores).
- **[04 Decoupled Workflow](04_decoupled_workflow.md):** Ingestion triggers (scheduled batch, volumetric CDC, drift alerts) and the 3-tier configuration pattern.
- **[05 Closed-Loop Guardrails](05_closed_loop_guardrails.md):** The 7-gate safety blueprint, automated verification perimeters, and pipeline circuit breakers.

---

### [06 Training: Theoretical Mental Models](06_training/index.md)

- **01 Foundational Primer:** Permutation invariance, spatial positional need, QKV retrieval analogy, attention vs. FFN layer anatomy, residual highways, and precision memory budgeting (FP32, BF16, FP16, AdamW states).
- **02 Data Regimes & Tokenization:** Storage topologies (Pattern A vs. B vs. Ephemeral Gold Cache), the capacity paradox on small corpora, Char vs. BPE vs. Byte-Level BBPE, vocabulary compression, deterministic document-level splitting, and sequence packing.
- **03 Architectural Topologies:** Autoregressive Decoders (MiniGPT), Masked Encoders (MiniBERT), Seq2Seq (MiniT5), attention topologies (MHA, GQA, MQA, SDPA), positional schemes (RoPE, ALiBi), activation/norm dynamics (SwiGLU, RMSNorm), and symmetric weight tying.
- **04 Optimization & Loss Dynamics:** Safety perimeter rationale (Gates 1-4), target-only SFT loss masking ($-100$), gradient clipping, AdamW momentum mechanics, cosine warmup schedules, micro-batching, AMP, and early stopping.
- **05 Inference Decoding & Calibration:** $\mathcal{O}(T)$ KV-caching, greedy vs. multinomial sampling, temperature calibration, Top-$k$/Top-$p$ nucleus truncation, and offline validation.
- **06 Infrastructure & Resilience:** Modular PyTorch engines, immutable Git SHA provenance, `latest` vs. `best` checkpoint policies, and OOM recovery hooks.

---

### [07 Evaluating: Governance & Evaluation Mental Models](07_evaluating/index.md)

- **01 Evaluation Frameworks & Metrics:** Continuous metrics (Cross-Entropy, Perplexity $\text{PPL} = e^{\mathcal{L}}$), fixed qualitative probing sets, downstream benchmarks (GSM8K, FinQA, HumanEval), and LLM-as-a-Judge calibration matrices.
- **02 Controlled Ablation Framework:** Standardized single-variable experimental matrices evaluating Positional Encodings (Exp 1), Attention Head Topologies (Exp 2), and Activation/Norm Pairings (Exp 3).
- **03 Automated Governance & Registry:** Telemetry streaming, Gate 5 assertion math ($\Delta_{\text{metric}} \ge \epsilon$), staging quarantine lifecycles, and MLflow Model Registry promotion rules.
