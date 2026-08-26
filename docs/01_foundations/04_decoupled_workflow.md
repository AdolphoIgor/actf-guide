# Decoupled Data Workflow, Training Execution, and Model Governance

In enterprise LLMOps, the data pipeline, training execution, and evaluation governance are strictly decoupled into independent micro-pipelines. They do not execute as a single monolithic script. Instead, they communicate asynchronously via declarative manifests, event triggers, and state contracts.

This decoupling enforces strict failure domain isolation: high-frequency, CPU-bound data curation never competes with or stalls expensive GPU training infrastructure, while model architectures and hyperparameters remain completely pluggable without requiring changes to orchestration DAGs.

---

## 1. Ingestion & Data Workflow Trigger Patterns

The Data Pipeline operates upstream, processing raw ingestion data into universal, model-agnostic **Silver-layer** datasets. Depending on data velocity and domain dynamics, enterprises execute this workflow via three core trigger patterns:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Pattern 1: Scheduled Batch (Time-Based)                                │
│ Cron Schedule ──► Extract Deltas ──► Clean & Curate ──► Silver S3      │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ Pattern 2: Event-Driven Volumetric (Data-Size Based)                   │
│ CDC Accumulator (N >= N_threshold) ──► Trigger Orchestrator ──► Silver │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ Pattern 3: On-Demand Observability (Drift / Telemetry Webhooks)        │
│ Telemetry Alert (PSI > Threshold OR Acc < 85%) ──► Trigger CT Loop     │
└────────────────────────────────────────────────────────────────────────┘

```

### A. The Scheduled Batch Pattern (Time-Based)

- **Execution Mechanics:** An orchestrator triggers a scheduled cron job (e.g., weekly at `00:00 UTC`). Distributed CPU workers extract incremental raw data logs, execute data normalization, deduplication, heuristic filtering, quality scoring, sensitive information masking, and benchmark decontamination, committing the clean data to the Silver storage layer.
- **Downstream Training Trigger:** Does **not** automatically trigger GPU training. Generating a fresh Silver data partition simply updates storage cold tiers, awaiting a dedicated volumetric or performance threshold before provisioning compute.

### B. The Event-Driven Volumetric Pattern (Data-Size Based)

- **Execution Mechanics:** Used when data arrival is non-deterministic. A Change Data Capture (CDC) stream or database listener maintains a watermark counter tracking unprocessed records ($N_{\text{unprocessed}}$).
- **Assertion Condition:** When $N_{\text{unprocessed}} \ge N_{\text{threshold}}$, a trigger event fires the data curation pipeline. The batch is processed into the Silver layer, and the ingestion watermark is committed.

### C. The On-Demand Observability Pattern (Drift-Based)

- **Execution Mechanics:** Live production telemetry monitors real-world model outputs via observability platforms.
- **Assertion Condition:** When statistical data drift exceeds baseline tolerances ($\text{PSI} > \tau_{\text{drift}}$) or live task accuracy regresses ($\text{Accuracy}_{\text{live}} < 0.85$), an automated webhook fires. The orchestrator intercepts the alert, extracts the underperforming production slices, formats a targeted correction partition, and triggers the continuous training loop.

---

## 2. The Architectural Boundary: Silver vs. Gold Layers

To ensure that data curation remains $100\%$ model-agnostic and reusable across any model architecture (e.g., Qwen, Llama, Mistral), the boundary between the Data Pipeline and Training Pipeline is strictly enforced at the **Silver Layer**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ DATA CURATION PIPELINE (CPU Infrastructure / Model-Agnostic)           │
│ Operations: Unicode Normalization, Exact & Fuzzy Dedup, Heuristics,    │
│             Quality Scoring, PII Masking, Evaluation Decontamination   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Outputs
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ SILVER STORAGE LAYER (Pristine, Universal Text Format)                 │
│ Schema: [{"role": "user", "content": "..."}, {"role": "assistant"}]   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Ingested at Runtime
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TRAINING PIPELINE (Compute Infrastructure / Model-Specific)            │
│ JIT Template Formatting: Injects model-specific chat control tokens    │
│ JIT Tokenization: Multi-threaded BPE encoding & sequence packing       │
│ Forward/Backward Passes: Target-only loss masking (-100 on prompts)    │
└────────────────────────────────────────────────────────────────────────┘

```

- **Data Pipeline Scope (Raw Ingestion to Silver Layer):** Focuses exclusively on universal text hygiene, quality scoring, and safety compliance. It has zero knowledge of tokenizers, vocabularies, or context window lengths.
- **Training Pipeline Scope (Silver Layer to Parameter Optimization):** Performs **Just-In-Time (JIT) Compilation** in shared virtual memory at worker boot time. It loads the specific model's chat template, applies Byte-Pair Encoding tokenization, masks prompt tokens to $-100$ in the cross-entropy loss tensor, and feeds packed $B \times L$ sequence matrices directly to the model parameters.

---

## 3. Declarative Decoupling: The 3-Tier Configuration Pattern

Embedding model architectures or hyperparameters directly inside an orchestration workflow is an anti-pattern that violates immutable infrastructure standards. Instead, the architecture isolates parameters into three decoupled tiers:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Tier 1: Git-Versioned Model Recipe Manifest (.yaml)                    │
│ • base_model: "Qwen/Qwen2.5-0.5B-Instruct" | chat_template: "chatml"   │
│ • learning_rate: 2e-5 | seq_len: 2048 | lora_r: 16                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Pointers passed via
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Tier 2: Orchestration Trigger Payload (Run Config)                     │
│ {"config_uri": "s3://ml-configs/recipes/qwen2.5_0.5b_sft_v2.yaml"}     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Injected into Container at Startup
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Tier 3: Universal Execution Container (Model-Agnostic Engine)          │
│ Generic PyTorch container reading config_uri at runtime                │
└────────────────────────────────────────────────────────────────────────┘

```

1. **Model Recipe Manifest (`.yaml`):** Version-controlled file defining model-specific attributes (base weights, tokenizer configs, context lengths, optimizer hyperparameters).
2. **Orchestration Plane (Trigger Payload):** The orchestrator acts as a generic scheduler, receiving a URI pointer to the configuration file without hardcoding training parameters.
3. **Execution Plane (Universal Container):** The training container spins up generically, pulls the declarative manifest from object storage, executes on-the-fly tokenization, and binds the run's Git commit hash to tracking metadata. Swapping models requires changing only the configuration URI.

---

## 4. Governance Handshake: Automated Gatekeeper & Model Registry

The responsibility of the Continuous Training specialist concludes once an approved checkpoint is delivered to the governance registry.

To prevent broken or regressing models from contaminating production, candidate models **never enter the Model Registry directly from training**. An **Automated Gatekeeper Pipeline** functions as the mandatory perimeter guard:

```text
[ Training Pipeline Completes ]
               │
               ▼
[ Ephemeral Artifact Store / S3 ] ──► (Raw un-evaluated weights & run metrics)
               │
               ▼
[ Automated Gatekeeper Pipeline ] ──► (Evaluates Benchmarks, Safety & Accuracy)
               │
      ┌────────┴──────────────────────────────────┐
      │                                           │
[ PASS: Score >= Baseline ]              [ FAIL: Regression Detected ]
      │                                           │
      ▼                                           ▼
[ MLflow Model Registry ]                [ Automated Circuit Breaker ]
• Officially registered & versioned      • Candidate dropped / quarantined
• Tagged: Approved-For-Staging           • Compute torn down; alert dispatched
• Handed to Deployment Layer

```

- **Staging Artifact Storage:** The training job exports raw checkpoints to an isolated, ephemeral staging path and logs step-level loss curves to experiment tracking platforms.
- **Gatekeeper Verification & Evaluation Benchmarks:** The evaluation pipeline loads the candidate weights, runs deterministic syntax/code checks, benchmarks against domain-specific gold-standard test suites, and calculates evaluation deltas against the active production baseline:

$$\Delta_{\text{metric}} = \text{Score}_{\text{candidate}} - \text{Score}_{\text{production}}$$

- **Registry Promotion:** If and only if $\Delta_{\text{metric}} \ge \epsilon$, the Gatekeeper commits the model to the **Model Registry** and assigns the promotion tag:

$$\texttt{status} = \texttt{"Approved-For-Staging"}$$

- **Circuit Breaker Action:** If the candidate regresses on any core benchmark, the pipeline halts immediately, drops the staging checkpoint, and dispatches a diagnostic traceback alert to the engineering team.

---

## 5. End-to-End Pipeline Split & Governance Matrix

| Pipeline Segment        | Primary Execution Engine                 | Input Layer                          | Output Layer                 | Governance & Safety Responsibilities                                                                                                                |
| ----------------------- | ---------------------------------------- | ------------------------------------ | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data Pipeline**       | Distributed CPU Pods (Ray / Spark)       | Raw Bronze Files                     | Clean Silver Parquet Files   | **Ingestion & Schema Integrity:** Enforces file validity, non-corruption, data sanitization, and structured schema standardization.                 |
| **Training Pipeline**   | PyTorch / Distributed Cluster            | Clean Silver Files + Config Manifest | Staging Checkpoint Artifacts | **Partition & Tensor Integrity:** Enforces strict zero-leakage dataset splits, valid matrix boundaries ($B \times L$), and loss convergence.        |
| **Gatekeeper Pipeline** | Automated Evaluation Suites / LLM Judges | Staging Candidate Checkpoint         | Registered Model Asset       | **Pre-Registry Benchmark Gate:** Quantifies candidate accuracy and loss deltas against production baselines before applying `Approved-For-Staging`. |
