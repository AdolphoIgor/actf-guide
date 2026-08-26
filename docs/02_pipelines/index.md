# 02 - Pipelines

## Executive Overview

The **Pipelines** section details the three execution loops comprising the automated continuous training lifecycle: data ingestion and curation, distributed training with JIT compilation, and post-flight gatekeeper evaluation.

## Pipeline Architecture

```text
[ The Data Pipeline ] ──► [ The Training Pipeline ] ──► [ The Gatekeeper Pipeline ]
  (Steps 1–10 / CPU)       (JIT Steps 11–12 / SFT)       (Benchmark Eval / MLflow)

```

---

## Module Navigation

- **[01 The Data Pipeline](https://www.google.com/search?q=01_the_data_pipeline/index.md):** Ingestion mechanics, closed-loop contracts, exact RocksDB deduplication, AST syntax parsing, PII redaction, and Gate 1 / Gate 2 safety perimeter checks.
- **[02 The Training Pipeline](https://www.google.com/search?q=02_the_training_pipeline/index.md):** Just-In-Time (JIT) chat templating (Step 11), multi-threaded sequence packing (Step 12), Gate 3 / Gate 4 validation, and SFT loss masking.
- **[03 The Gatekeeper Pipeline](https://www.google.com/search?q=03_the_gatekeeper_pipeline/index.md):** Post-training benchmark evaluation (Gate 5), LLM-as-a-Judge scoring, regression detection, and MLflow Model Registry promotion.

```

```
