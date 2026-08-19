# The Training Pipeline

## Executive Overview
The **Training Pipeline** executes model-specific supervised fine-tuning (SFT). Operating downstream of the model-agnostic Data Pipeline, it ingests universal Silver text, performs **Just-In-Time (JIT)** ChatML templating (Step 11) and multi-threaded token sequence packing (Step 12) inside shared memory (`/dev/shm`), verifies tensor integrity at Gates 3 and 4, executes parameter updates with assistant-only loss masking, and stages candidate checkpoints for Gatekeeper evaluation.

---

## Architecture Flow

```text
       [ Universal Silver Text (s3://.../silver/) ]
                            │
                            ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Phase 1: JIT Data Compilation (In-Memory in /dev/shm)    │
 │ • Step 11: Dynamic ChatML / Jinja Template Formatting    │
 │ • Step 12: Rust BPE Tokenization, Packing & Loss Masking │
 │ • Gate 3: Split Isolation (Zero Leakage, 95/5 Ratio)     │
 │ • Gate 4: Pre-Flight Tensor Validation (B x L, Vocab)    │
 └──────────────────────────┬───────────────────────────────┘
                            │
                            ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Phase 2: Distributed Parameter Optimization              │
 │ • Supervised Fine-Tuning with Target-Only Loss Masking   │
 │ • Hardware Engine: CPU SIMD (AVX-512) OR GPU (FSDP2)     │
 │ • Out-of-Band Telemetry & In-Flight Circuit Breakers     │
 └──────────────────────────┬───────────────────────────────┘
                            │
                            ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Phase 3: Staging Handshake                               │
 │ • Export candidate checkpoint to ephemeral staging bucket│
 │ • Signal Gatekeeper Pipeline for Gate 5 Benchmark Eval   │
 └──────────────────────────────────────────────────────────┘

```

---

## Chapter Roadmap

* **[Step 11: Pre-Tokenization Audit & Schema Alignment](https://www.google.com/search?q=step_11_pre_tokenization_audit_and_schema_alignment.md):** Dynamic Jinja2 chat template extraction and formatting at runtime.
* **[Step 12: Tokenization & Sequence Packing](https://www.google.com/search?q=step_12_tokenization_and_sequence_packing.md):** High-speed Rust sub-word tokenization, sequence packing ($B \times L$), and assistant-only label masking ($-100$).
* **Training Execution & Memory Dynamics:** Static vs. activation memory math, AdamW state sharding, and AVX-512 CPU acceleration.
* **Governance Handshake:** Staging candidate checkpoints prior to MLflow Model Registry promotion.