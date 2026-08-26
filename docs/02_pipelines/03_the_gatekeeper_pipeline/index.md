# The Gatekeeper Pipeline: Autonomous Certification, Safety Arbitration, and Release Governance

## 1. System Architecture and Pipeline Topology

The Gatekeeper Pipeline operates as an autonomous release arbiter between the training compute cluster and the production serving fleet. It evaluates candidate checkpoints ($\theta_{\text{cand}}$) against the active production baseline ($\theta_{\text{base}}$) using deterministic invariant assertions, paired statistical hypothesis testing, symmetric LLM-as-a-Judge tournaments, and hardware serving verification.

```text
The End-to-End Gatekeeper Certification Architecture:

┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: INGESTION & ZERO-TOLERANCE INVARIANT SCREENING (TIER 1)                         │
│                                                                                          │
│  Staged Artifact ──► [ Checksum Integrity ] ──► [ AST Syntax 100% ] ──► [ EOS Stop >=99% ]│
│                                                                              │           │
│                                                                              ▼           │
│  Quarantine ◄── [ PII Leak Zero-Tolerance ] ◄── [ KV-Cache Parity Delta < 1e-3 ]         │
└────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │ All Hard Invariants Pass
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: STATISTICAL NON-INFERIORITY & SIGNIFICANCE BATTERY (TIER 2)                     │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ DUAL-PARADIGM BENCHMARK ARBITRATION                                                │  │
│  │  • Multiple-Choice: Length-Normalized Log-Likelihood (MMLU, ARC, HellaSwag)        │  │
│  │  • Generative CoT: Regex Exact Match (GSM8K, MATH)                                 │  │
│  │  • Functional Code: Sandboxed Execution Pass@1 (HumanEval, MBPP)                   │  │
│  └─────────────────────────────────────────┬──────────────────────────────────────────┘  │
│                                            │                                             │
│                                            ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ STATISTICAL REGRESSION TESTS                                                       │  │
│  │  • Paired McNemar Chi-Square: Assert p >= 0.05 or Improvements >= Regressions      │  │
│  │  • Empirical Bootstrap (B=10k): Assert Lower 95% CI >= -0.5% Non-Inferiority Margin│  │
│  │  • Wilcoxon Signed-Rank Test on Token Cross-Entropy Loss Distributions             │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │ Statistical Parity Verified
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: SUBJECTIVE ARENA & OPERATIONAL SLA GOVERNANCE (TIERS 3 & 4)                     │
│                                                                                          │
│  [ Symmetric LLM Judge ] ──► Bidirectional Pairing ──► Wilson 95% CI Lower Bound >= 0.50 │
│                                            │                                             │
│                                            ▼                                             │
│  [ Operational Health ] ──► Expected Calibration Error (ECE <= 0.06) & Serving SLA Checks│
│                                            │                                             │
│                                            ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ MASTER PROMOTION ARBITER                                                           │  │
│  │  • Emit Cryptographically Signed Audit Receipt (`gatekeeper_receipt.json`)         │  │
│  │  • Execute Atomic MLflow Registry Swap: Set Version to `@champion`                 │  │
│  │  • Demote Previous Baseline to `@archived` | On Failure: Lock in `@quarantined`    │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────────┘

```

---

## 2. Pipeline Document Blueprint

```text
docs/02_pipelines/03_the_gatekeeper_pipeline/
├── step_01_candidate_checkpoint_ingestion_and_hashing.md
├── step_02_tier1_zero_tolerance_invariant_assertions.md
├── step_03_tier2_downstream_task_benchmark_execution.md
├── step_04_tier2_mcnemar_and_bootstrap_significance_testing.md
├── step_05_tier3_symmetric_llm_judge_tournament.md
├── step_06_tier4_calibration_and_serving_sla_verification.md
├── step_07_gatekeeper_audit_receipt_generation.md
└── step_08_mlflow_registry_atomic_promotion_and_rollback.md

```

---

## 3. Step Lifecycle and Operational Execution Index

The Gatekeeper runs as an automated eight-step pipeline. Any failure in earlier tiers halts execution and routes the checkpoint to quarantine:

### Phase I: Ingestion and Hard Deterministic Verification

- **[Step 01: Candidate Checkpoint Ingestion and Hashing](https://www.google.com/search?q=step_01_candidate_checkpoint_ingestion_and_hashing.md)**
  Streams weights from ephemeral staging, verifies SHA-256 digests against training manifests, and provisions isolated sandboxes.
- **[Step 02: Tier 1 Zero-Tolerance Invariant Assertions](https://www.google.com/search?q=step_02_tier1_zero_tolerance_invariant_assertions.md)**
  Validates 100% Python AST syntax compliance on code outputs, verifies $\ge 99\%$ EOS stop tokens, audits for zero PII leaks, and asserts KV-cache logit parity $\Vert{}Z_{\text{naive}} - Z_{\text{cached}}\Vert{}_\infty < 10^{-3}$.

### Phase II: Statistical Non-Inferiority and Significance

- **[Step 03: Tier 2 Downstream Task Benchmark Execution](https://www.google.com/search?q=step_03_tier2_downstream_task_benchmark_execution.md)**
  Runs evaluation batteries across MMLU, GSM8K, ARC, and HumanEval using length-normalized log-likelihood scoring and sandboxed execution.
- **[Step 04: Tier 2 McNemar and Bootstrap Significance Testing](https://www.google.com/search?q=step_04_tier2_mcnemar_and_bootstrap_significance_testing.md)**
  Executes paired McNemar Chi-Square tests to flag discrete capability regressions and computes $10,000$-sample Empirical Bootstrap confidence intervals to confirm non-inferiority margins ($\Delta \ge -0.5\%$).

### Phase III: Subjective Quality, Calibration, and Release Governance

- **[Step 05: Tier 3 Symmetric LLM Judge Tournament](https://www.google.com/search?q=step_05_tier3_symmetric_llm_judge_tournament.md)**
  Runs symmetric order-swapped pairwise evaluations ($J(A, B)$ and $J(B, A)$) to cancel position bias, computing Wilson Score $95\%$ confidence intervals.
- **[Step 06: Tier 4 Calibration and Serving SLA Verification](https://www.google.com/search?q=step_06_tier4_calibration_and_serving_sla_verification.md)**
  Measures Expected Calibration Error across 15 probability bins ($\text{ECE} \le 0.06$) and profiles Inter-Token Latency (ITL) and memory footprints under serving concurrency.
- **[Step 07: Gatekeeper Audit Receipt Generation](https://www.google.com/search?q=step_07_gatekeeper_audit_receipt_generation.md)**
  Compiles test scorecards, diagnostic traces, and metadata into a signed `gatekeeper_receipt.json` sidecar.
- **[Step 08: MLflow Registry Atomic Promotion and Rollback](https://www.google.com/search?q=step_08_mlflow_registry_atomic_promotion_and_rollback.md)**
  Mutates the `@champion` model alias in the MLflow registry, archives the prior version, dispatches fleet deployment webhooks, and manages emergency rollbacks.

---

## 4. Gatekeeper Four-Tier Decision Matrix

| Evaluation Tier           | Metric / Target            | Mathematical Threshold                                                | Evaluation Method          | Action on Failure                         |
| ------------------------- | -------------------------- | --------------------------------------------------------------------- | -------------------------- | ----------------------------------------- |
| **Tier 1: Safety & Spec** | AST Code Parse Rate        | $\text{Rate} == 1.0$ ($100\%$)                                        | `ast.parse` syntax scan    | **Hard Quarantine**; abort run            |
| **Tier 1: Safety & Spec** | EOS Delimiter Emission     | $\text{Rate} \ge 0.99$ ($99\%$)                                       | Stop sequence matcher      | **Hard Quarantine**; runaway gen          |
| **Tier 1: Safety & Spec** | KV-Cache Parity            | $\Vert{}Z_{\text{naive}} - Z_{\text{cached}}\Vert{}_\infty < 10^{-3}$ | Max logit delta check      | **Hard Quarantine**; cache drift          |
| **Tier 1: Safety & Spec** | PII Violation Count        | $\text{Count} == 0$                                                   | Regex / NER entity scan    | **Hard Quarantine**; safety leak          |
| **Tier 2: Benchmarks**    | Paired Item Delta          | $p \ge 0.05 \lor n_{\text{imp}} \ge n_{\text{reg}}$                   | McNemar Chi-Square test    | **Statistical Quarantine**; regression    |
| **Tier 2: Benchmarks**    | Capability Non-Inferiority | $L_{0.025}(\Delta) \ge -0.005$                                        | Bootstrap $95\%$ Lower CI  | **Statistical Quarantine**; margin breach |
| **Tier 3: LLM Judge**     | Tournament Win Rate        | $\hat{p} \ge 0.52$ ($52\%$)                                           | Symmetric Pairwise Arena   | **Judge Quarantine**; preference loss     |
| **Tier 3: LLM Judge**     | Wilson Lower Bound         | $p_{\text{lower}} \ge 0.50$                                           | Wilson Score $95\%$ CI     | **Judge Quarantine**; parity unproven     |
| **Tier 4: Operations**    | Probability Calibration    | $\text{ECE} \le 0.06$                                                 | 15-Bin Reliability Diagram | **Calibration Quarantine**; uncalibrated  |
| **Tier 4: Operations**    | Inter-Token Latency        | $\text{ITL} \le \text{SLA Limit}$                                     | Hardware serving benchmark | **Operational Quarantine**; slow decode   |

---

## 5. Core Governance Invariants

The Gatekeeper Pipeline enforces five release invariants across all evaluation runs:

1. **Deterministic Short-Circuiting:** Any single Tier 1 failure (AST parse, EOS missing, KV-cache mismatch, PII leak) immediately halts execution and prevents downstream benchmarking.
2. **Paired Statistical Testing:** Never evaluate candidate improvements on aggregate averages alone; discrete benchmark deltas must be evaluated per-item via paired McNemar tests.
3. **Symmetric Position Invariance:** LLM-as-a-Judge evaluations must execute both forward and reverse configurations ($A/B$ and $B/A$) to cancel positional bias.
4. **Signed Provenance Sidecar:** Every promoted or quarantined artifact must carry an immutable, signed `gatekeeper_receipt.json` detailing exact p-values, test metrics, and Git commit lineage.
5. **Zero-Downtime Rollback Readiness:** When promoting a new `@champion`, the pipeline verifies that the preceding production version is tagged as `@archived` and capable of serving traffic within 5 seconds of an anomaly alert.
