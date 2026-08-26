# ACTF Guide: Enterprise LLMOps & Automated Continuous Training Architecture

---

## Executive Overview

The **Automated Continuous Training Flow (ACTF)** architecture guide is an end-to-end technical reference designed for engineering teams deploying autonomous, production-grade LLMOps pipelines.

Modern production deployments face severe data drift, silent quality degradation, and astronomical compute costs. This repository details how to decouple data engineering from parameter optimization, enforce multi-tiered mathematical and syntactic guardrails, execute distributed fine-tuning, and gate deployment using objective, post-flight evaluation suites.

```text
[ Data Warehouse ] ───► [ Distributed ETL ] ───► [ Data Versioning ]
(Snowflake/BigQuery)       (Ray / Spark)             (DVC / S3)
                                                         │
                                                         ▼
[ Orchestrator ] ─────► [ Training Compute ] ───► [ Model Governance ]
(Prefect / Airflow)       (Axolotl / TRL)          (MLflow / W&B)
                                                         │
                                                         ▼
[ Gateway / Router ] ◄── [ Inference Serving ] ◄── [ Automated Eval ]
(Portkey / Kong)          (vLLM / Triton)         (Braintrust/Ragas)
                                                         │
                                                         ▼
[ Live Observability ] ──► (If metrics drift, alerts trigger the Orchestrator to restart)
(Langfuse / Arize)

```

---

## Architectural Core Pillars

```text
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Foundations & Storage Topologies (Medallion Lakehouse & Decoupled Micro-Pipelines) │
└──────────────────────────────────────────┬────────────────────────────────────────────┘
                                           │
                                           ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ 2. High-Throughput Data Pipeline (Zero-Copy Arrow Memory & Branching Curation DAG)    │
└──────────────────────────────────────────┬────────────────────────────────────────────┘
                                           │
                                           ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ 3. Distributed Training & SFT Optimization (Loss Masking, Quantization & CPU SIMD)    │
└──────────────────────────────────────────┬────────────────────────────────────────────┘
                                           │
                                           ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ 4. The 7 Engineering Gates & Gatekeeper Governance (Automated Promotion & Rollback)   │
└───────────────────────────────────────────────────────────────────────────────────────┘

```

### 1. Foundations & Storage Topologies

- **Medallion Data Lake Layout:** Implements Hive-style partitioning across **Bronze** (raw, un-mutated landing data), **Silver** (cleaned, schema-standardized text), and **Gold** (packed integer tensors and gold-standard evaluation benchmarks).
- **Decoupled Workflow Model:** High-frequency, CPU-bound ETL workflows run independently from protected, high-cost GPU training compute across three core ingestion patterns:

1. _Scheduled Batch (Time-Based):_ Routine delta extraction over fixed cron intervals.
2. _Event-Driven Volumetric (Data-Size Based):_ Change Data Capture (CDC) accumulators triggering processing only upon reaching critical mass ($N_{\text{unprocessed}} \ge N_{\text{threshold}}$).
3. _On-Demand Observability (Drift-Based):_ Real-time production telemetry webhooks catching semantic drift ($\text{PSI} > \tau_{\text{drift}}$) or task accuracy regressions ($\text{Accuracy}_{\text{live}} < 0.85$).

### 2. High-Throughput Data Pipeline & Memory Architecture

- **Zero-Copy Ingestion:** Leverages Apache Arrow's 3-buffer contiguous memory layout (Validity Bitmap, Offsets Buffer, and Value Buffer) mapped directly into the Linux virtual memory segment (`/dev/shm`).
- **Branching Curation DAG:**
- _Track A (Prose):_ Macro-linguistic heuristic filters, 128-permutation MinHash LSH fuzzy deduplication, and Classifier-Based Quality Filtering [CQF](cite: 1).
- _Track B (Code & Technical):_ Regex snippet disambiguation, license-stripping line-level LSH, Pygments lexer verification, and Tree-Sitter AST syntax compilers[cite: 1].

- **Embedded Key-Value Deduplication:** Embedded RocksDB Key-Value stores utilizing Lock-Free Skiplists bound to `/dev/shm` to bypass the Python GIL and spill memory gracefully to NVMe during spikes[cite: 1].

### 3. Distributed Training & Optimization Profile

- **Target Model Profile:** Standardized around lightweight Transformer architectures (`Qwen/Qwen2.5-0.5B-Instruct`) featuring Grouped-Query Attention (GQA), Rotary Position Embeddings (RoPE), and SwiGLU activations.
- **Target-Only SFT Loss Masking:** Full ChatML format alignment with assistant-only cross-entropy loss calculation, masking user prompt tokens to $-100$ in PyTorch:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{\vert{}Y\vert{}} \sum_{t \in Y} \log P_\theta(y_t \mid x, y_{<t})$$

- **Compute Flexibility:** Multi-node GPU sharding strategies (PyTorch FSDP2 / DeepSpeed ZeRO-3) paired with vectorized CPU SIMD (AVX-512 / AMX) acceleration pathways.

### 4. Gatekeeper Governance & Closed-Loop Telemetry

- **Automated Gatekeeper:** Post-training evaluation framework running deterministic unit checks, domain gold-standard benchmarks (GSM8K, HumanEval, FinQA), and calibrated LLM-as-a-Judge evaluations before model registry promotion.
- **The 7 Pipeline Gates:** Sequential verification checkpoints across Ingestion (Bronze), Pre-Tokenization (Silver), Data Leakage, Pre-Flight Tensors, Model Promotion, Live Inference Serving, and Continuous Drift Monitoring.
- **Global Circuit Breakers:** Immediate, automated teardown of cloud compute instances upon assertion failures or loss divergence ($\text{NaN} / \infty$).

---

## Medallion Storage Lifecycle & Tiering Strategy

To prevent uncontrolled cloud storage billing while ensuring absolute data lineage reproducibility, datasets transition across automated cloud lifecycle policies:

| Data Layer             | Physical File Format                   | Processing State                                                                | Cloud Lifecycle Action Rule                                                | Business & Architectural Purpose                                                                                            |
| ---------------------- | -------------------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Bronze (Raw)**       | Raw Parquet / Compressed JSONL         | Un-mutated source extracts with appended ingestion metadata.                    | Transition to Cold / Deep Archive after 30 days. Purge after 365 days.     | Serves as immutable audit backup. Kept in cold tier to allow complete pipeline replay without hitting production databases. |
| **Silver (Cleaned)**   | Columnar Snappy Parquet                | Pristine, deduplicated, schema-enforced instruction-tuning schemas.             | Transition to Infrequent Access (Cool) after 90 days. Retain Indefinitely. | Corporate System of Record. Preserved to enable re-tokenization, feature re-extraction, or historical model retraining.     |
| **Gold (Model-Ready)** | Tokenized Binary Shards / Arrow Arrays | Context-padded, packed integer tensors ($B \times L$) and gold evaluation sets. | Retain Indefinitely in Standard (Hot) Object Storage.                      | High-value, refined assets queried actively by multi-node training clusters and evaluation gatekeepers.                     |

```text
s3://company-ai-datalake/
│
├── bronze/                                    # RAW LANDING LAYER (Archive after 30d)
│   ├── oracle_crm/customer_chats/             # Source & Entity descriptors
│   │   └── year=2026/month=06/day=09/         # Hive-style date partitions
│   │       ├── chunk_091214_raw.jsonl
│   │       └── chunk_091530_raw.jsonl
│   └── box_storage/technical_pdfs/
│       └── year=2026/month=06/day=01/
│           └── extracted_text_01.jsonl
│
├── silver/                                    # CLEANED & STANDARDIZED LAYER (Cool tier)
│   └── support_tickets/standardized_train_pairs/
│       └── year=2026/month=06/
│           ├── data_v1_001.snappy.parquet     # Formatted schemas, PII-masked
│           └── data_v1_002.snappy.parquet
│
└── gold/                                      # MODEL-READY FEATURE STORE (Hot tier)
    └── model_qwen_1_7b/evaluation_golden_set/
        └── eval_version_2_4.parquet           # Tokenized tensors & benchmarks

```

---

## Detailed Data Curation DAG & Infrastructure Mapping

The data pipeline runs as an asynchronous, branching Directed Acyclic Graph (DAG) that isolates natural language prose from code/technical text before re-converging for safety sanitization and sequence packing[cite: 1]:

```text
                        [ SHARED TRUNK: INGESTION ]
                                     │
                  [ NODE 1: intake_and_provenance_router ]
                         (Memory/IO CPU Pod: 8 vCPU, 64GB)
                                     │
             ┌───────────────────────┴───────────────────────┐
             ▼                                               ▼
[ BRANCH A: NATURAL LANGUAGE PROSE ]        [ BRANCH B: CODE & TECHNICAL DOMAINS ]
             │                                               │
 [ NODE 2A: nl_heuristics_and_dedup ]          [ NODE 2B: code_syntax_and_dedup ]
     (Memory-Optimized CPU Pod)                      (Compute-Dense CPU Pod)
             │                                               │
   [ NODE 3A: nl_quality_and_lang ]              [ NODE 3B: code_validation_and_ast ]
     (GPU-Dense Inference Pod)                      (CPU/GPU Hybrid Pod)
             │                                               │
             └───────────────────────┬───────────────────────┘
                                     │
                        [ SHARED TRUNK: CONVERGENCE ]
                                     │
                     [ NODE 4: safety_and_decontamination ]
                           (GPU-Dense / Compute Pod)
                                     │
                     [ NODE 5: tokenization_and_packing ]
                      (Compute-Optimized CPU Pod: 16 vCPU)

```

### Node Execution Specifications

1. **Node 1 (`intake_and_provenance_router`):**[cite: 1]

- _Profile:_ Memory/IO-Optimized CPU [AWS `r6i.2xlarge`: 8 vCPUs, 64 GB RAM, 16 GB `/dev/shm`](cite: 1).
- _Tasks:_ Unicode NFKC normalization, boilerplate DOM stripping, 64-bit MurmurHash3 exact deduplication via embedded RocksDB, and Apache Arrow schema metadata routing[cite: 1].

1. **Node 2A (`nl_heuristics_and_dedup`):**[cite: 1]

- _Profile:_ Memory-Optimized CPU Pod[cite: 1].
- _Tasks:_ Vectorized punctuation, symbol, stop-word, and n-gram repetition filters, followed by 128-permutation MinHash LSH fuzzy deduplication [$\ge 0.85$ Jaccard threshold](cite: 1).

1. **Node 3A (`nl_quality_and_lang`):**[cite: 1]

- _Profile:_ GPU-Dense / High-Throughput Inference [NVIDIA L4 / T4](cite: 1).
- _Tasks:_ Classifier-Based Quality Filtering (CQF score threshold $\ge 0.65$) and paragraph-level fastText language identification [`lid.176.bin`](cite: 1).

1. **Node 2B (`code_syntax_and_dedup`):**[cite: 1]

- _Profile:_ Compute-Dense CPU Pod[cite: 1].
- _Tasks:_ Disables prose penalties; executes regex syntax extraction, license header stripping, line-level MinHash LSH, and AST fingerprinting[cite: 1].

1. **Node 3B (`code_validation_and_ast`):**[cite: 1]

- _Profile:_ CPU-Dense Parsing Pod[cite: 1].
- _Tasks:_ Pygments lexer verification, bracket/indentation consistency checks, and Tree-Sitter AST syntax tree validation[cite: 1].

1. **Node 4 (`safety_and_decontamination`):**[cite: 1]

- _Profile:_ GPU-Dense Inference Pod[cite: 1].
- _Tasks:_ In-cluster toxicity classification, NER-based PII redaction matrices, and 13-gram benchmark decontamination against evaluation suites [HumanEval, GSM8K, FinQA](cite: 1).

1. **Node 5 (`tokenization_and_packing`):**[cite: 1]

- _Profile:_ Compute-Optimized CPU [AWS `c6i.4xlarge`: 16 vCPUs](cite: 1).
- _Tasks:_ Pre-tokenization schema audits, multi-threaded Rust BPE tokenization, variable-length sequence concatenation, and fixed-size sequence packing [$L = 2048 / 4096$](cite: 1).

---

## The 7 Engineering & Safety Gates

To eliminate silent failures and contain compute expenditure, the architecture enforces a strict multi-gated verification perimeter:

```text
                                [ Production Live Telemetry ]
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
          [ Gate 7: Drift Monitor ]                     [ Gate 6: Serving Gate ]
          (Triggers Airflow CT DAG)                     (Schema & Skew Defense)
                      │
                      ▼
         [ Gate 1: Ingestion Gate ] ────► (Bronze File & Row Integrity)
                      │
                      ▼
       [ Gate 2: Pre-Tokenization Gate ] ─► (Clean UTF-8 & Context Bounds)
                      │
                      ▼
       [ Gate 3: Data Leakage Gate ] ────► (Zero Overlap & 95/5 Balance)
                      │
                      ▼
      [ Gate 4: Pre-Flight Tensor Gate ] ─► (B x L Matrix & Vocab Audits)
                      │
                      ▼
       [ Gate 5: Automated Gatekeeper ] ──► (Benchmark vs. Production Delta)

```

| Gate Identifier                       | Operating Domain    | Verification Target    | Assertion & Boundary Criteria                                                                   | Automated Circuit Breaker Action                                  |
| ------------------------------------- | ------------------- | ---------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Gate 7: Continuous Drift Monitor**  | Production Feedback | Statistical Data Drift | $\text{PSI}(P_{\text{live}}, P_{\text{base}}) > \tau_{\text{drift}}$ or error rate spike.       | **Trigger:** Dispatches API webhook to wake up Airflow CT DAG.    |
| **Gate 6: Inference Serving Gate**    | Live Microservice   | Training-Serving Skew  | Request feature types, payload bounds, and normalization ranges.                                | **Block:** Drops malformed payloads before model execution.       |
| **Gate 1: Ingestion Gate**            | In-DAG Pipeline     | Bronze Data Integrity  | Valid magic bytes, zero file corruption, row count $> N_{\text{min}}$.                          | **Halt:** Aborts DAG; locks raw data partition immutability.      |
| **Gate 2: Pre-Tokenization Gate**     | In-DAG Pipeline     | Silver Data Schema     | Non-null fields, clean UTF-8 encodings, character length bounds.                                | **Halt:** Aborts DAG before distributed tokenization allocation.  |
| **Gate 3: Data Leakage & Split Gate** | In-DAG Pipeline     | Partition Isolation    | Cryptographic zero-overlap between splits; exact 95/5 train/val ratio.                          | **Halt:** Aborts DAG to prevent contaminated evaluation.          |
| **Gate 4: Pre-Flight Tensor Gate**    | In-DAG Pipeline     | Binary Tensors         | Matrix shape $B \times L$, token IDs $0 \le \text{ID} < V$, binary mask $\{0, 1\}$.             | **Halt:** Aborts DAG before allocating expensive GPU clusters.    |
| **Gate 5: Automated Gatekeeper**      | In-DAG Pipeline     | Model Performance      | $\text{Score}_{\text{cand}} \ge \text{Score}_{\text{prod}} + \epsilon$ across domain gold sets. | **Promote/Abort:** Promotes to Registry or halts and alerts team. |

---

## Multi-Container Infrastructure & Docker Topology

Workloads run on dedicated, resource-isolated pods provisioned from four consolidated Docker images to eliminate container bloat[cite: 1]:

```text
                     [ REPOSITORY WORKSPACE ]
                                │
   ┌────────────────┬───────────┴───────────┬────────────────┐
   ▼                ▼                       ▼                ▼
Orchestrator   Spark Ingest             Ray CPU           Ray GPU
(Airflow Slim) (PySpark/JDBC)       (No CUDA/Math)    (CUDA/PyTorch)
   │                │                       │                │
   ▼                ▼                       │                ▼
orchestrator   spark-ingest                 │        classifier_and_
   pod             pod                      │           safety pod
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
      exact_dedup_and_heuristics                             fuzzy_dedup_lsh
               (Node 1)                                         (Node 2)

```

```yaml
services:
  # Node 1: Memory/IO-Optimized CPU Pod
  exact-dedup-heuristics:
    image: ray-cpu-image
    container_name: actf-node-1
    command: python /home/ray/workspace/2-data-prep/node_1_heuristics.py
    deploy:
      resources:
        limits:
          cpus: "4"
          memory: 8G

  # Node 2: Memory-Bound Distributed Node
  fuzzy-dedup-lsh:
    image: ray-cpu-image
    container_name: actf-node-2
    command: python /home/ray/workspace/2-data-prep/node_2_fuzzy_lsh.py
    deploy:
      resources:
        limits:
          cpus: "8"
          memory: 32G # High RAM for MinHash LSH matrix permutations
```

---

## Repository Structure

```text
actf-guide/
├── .github/
│   └── workflows/
│       ├── lint-docs.yml              # Markdown linting & strict MkDocs validation
│       └── deploy-docs.yml            # Automated GitHub Pages deployment pipeline
├── docs/
│   ├── index.md                       # Site homepage & executive landing page
│   ├── 01_foundations/
│   │   ├── index.md                   # Foundations overview & reading roadmap
│   │   ├── 01_the_enterprise_llmops.md
│   │   ├── 02_the_continuous_training_role.md
│   │   ├── 03_decoupled_workflow.md
│   │   └── 04_the_medallion_architecture.md
│   ├── 02_pipelines/
│   │   ├── index.md                   # End-to-end pipeline orchestration map
│   │   ├── 01_the_data_pipeline/
│   │   │   ├── index.md
│   │   │   ├── 01_foundations/
│   │   │   │   ├── index.md
│   │   │   │   └── 01_closed_loop_guardrails.md
│   │   │   ├── 02_data_writing/
│   │   │   │   ├── index.md
│   │   │   │   └── 01_data_writing.md
│   │   │   ├── 03_filtering_and_tokenization/
│   │   │   │   ├── index.md
│   │   │   │   ├── phase_01_shared_ingestion/
│   │   │   │   │   ├── step_01_normalization.md
│   │   │   │   │   ├── step_02_boilerplate_stripping.md
│   │   │   │   │   ├── step_03_exact_deduplication.md
│   │   │   │   │   └── step_04_metadata_inspection_and_routing.md
│   │   │   │   ├── phase_02_domain_specific_processing/
│   │   │   │   │   ├── track_a_standard_natural_language_branch/
│   │   │   │   │   └── track_b_specialized_domain_branch/
│   │   │   │   └── phase_03_reconvergence_and_tokenization/
│   │   │   │       ├── step_09_safety_and_pii_redaction.md
│   │   │   │       ├── step_10_cross_dataset_decontamination.md
│   │   │   │       ├── step_11_pre_tokenization_audit_and_schema_alignment.md
│   │   │   │       └── step_12_tokenization_and_sequence_packing.md
│   │   │   └── 04_raw_data_ingestion/
│   │   │       ├── index.md
│   │   │       └── 01_raw_data_ingestion.md
│   │   ├── 02_the_training_pipeline/  # Distributed compute & SFT optimization
│   │   └── 03_the_gatekeeper_pipeline/# Evals, registry promotion & rollback
│   └── 03_infrastructure/
│       ├── index.md
│       └── 01_the_infrastructure.md   # Node sizing, KubeRay, Docker & compose.yaml
├── .markdownlint.json                 # Linter formatting & style constraints
├── mkdocs.yml                         # MkDocs Material theme & navigation manifest
├── LICENSE.md                         # Dual-licensing legal specifications
└── README.md                          # Repository documentation overview

```

---

## Contributing & Governance Standards

- **Strict Markdown Linting:** All additions must comply with the style constraints configured in `.markdownlint.json`.
- **Navigation Integrity:** Every sub-folder must contain an `index.md` landing page to ensure seamless tree parsing in MkDocs Material (`navigation.indexes`).
- **CI/CD Enforcement:** All pull requests automatically trigger the GitHub Actions test suite (`lint-docs.yml`) to validate internal markdown links and compile the site under `--strict` mode before merging into `main`.

---

## License

This project operates under a **Dual-Licensing Model**:

- **Documentation & Chapters (`/docs`):** Licensed under the **[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/)**. You are free to share and copy the material with attribution; commercial exploitation and distribution of modified adaptations are strictly prohibited.
- **Code Snippets, Dockerfiles, Configurations & Scripts:** Licensed under the **[MIT License](LICENSE.md)**. You are free to copy, modify, and incorporate the code into proprietary or open-source software distributions.

Refer to [LICENSE.md](LICENSE.md) for full legal text and licensing terms.
