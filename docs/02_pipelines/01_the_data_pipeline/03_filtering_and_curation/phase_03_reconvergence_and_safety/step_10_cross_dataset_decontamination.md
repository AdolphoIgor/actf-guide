# Step 10: Cross-Dataset Decontamination

## 1. Core Objective

Executing as the second processing stage within **Phase 3 - Reconvergence & Tokenization**, Step 10 identifies and purges benchmark text leaks, evaluation question overlaps, and verbatim test-set contamination across the unified, re-converged data stream.

Sequenced directly after **Step 9 (Safety & PII Redaction)** and prior to **Step 11 (Pre-Tokenization Audit & Schema Alignment)**, Step 10 targets a critical data-science vulnerability: evaluation set contamination. By comparing candidate pre-training records against an automated registry of public benchmarks (e.g., FinQA, MedQA, GSM8k, HumanEval) and internal gold-standard evaluation suites, Step 10 prevents benchmark memorization. This ensures that downstream post-training evaluation metrics and automated model deployment gatekeepers remain statistically valid and un-corrupted.

---

## 2. Theoretical & Architectural Justification

When training models across massive corpora, uncurated web dumps and domain repositories frequently contain test questions, prompt structures, or explicit answer keys from industry-standard benchmarks or proprietary evaluation sets.

Allowing benchmark overlap to leak into the pre-training dataset introduces three critical failure modes:

### A. Metric Inflation & Gatekeeper Invalidation

If a language model pre-trains on the exact sequences used to evaluate its reasoning performance, its test scores on downstream evaluation benchmarks artificially skyrocket. This phenomenon—data contamination—invalidates automated evaluation gatekeepers, rendering model deployment decision metrics meaningless because the system cannot distinguish between true semantic generalization and rote memorization.

### B. Overfitting to Evaluation Syntax

Pre-training on test set questions biases the model's token probability distributions toward specific test prompt structures. During inference, the model fails to generalize to minor prompt re-phrasings, demonstrating fragile, superficial performance rather than robust domain reasoning.

### C. Computational Intractability of Monolithic String Searching

Comparing billions of incoming training tokens against thousands of multi-page evaluation sets using standard string-matching algorithms introduces an $O(N \cdot M)$ computational bottleneck. Step 10 resolves this by implementing a **Two-Tier Down-Selected Search Pipeline**, utilizing fast $N$-gram hash pre-filtering ($O(1)$ set lookups) followed by targeted Longest Common Subsequence (LCS) sequence alignment for flagged candidates.

---

## 3. Theoretical Execution Mechanics

Step 10 processes re-converged data streams through a three-stage decontamination pipeline:

```text
                     [ Re-Converged Stream Post-Step 9 ]
                                     │
                                     ▼
         ┌───────────────────────────────────────────────────────┐
         │ Stage 1: Automated Evaluation Registry Sync &         │
         │          N-Gram Hash Indexing                         │
         └───────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
         ┌───────────────────────────────────────────────────────┐
         │ Stage 2: Tier 1 - Vectorized N-Gram Hash Pre-Filter   │
         └───────────────────────────┬───────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼ (Zero Hash Collisions)          ▼ (Hash Collision Detected)
         ┌────────────────────────┐      ┌────────────────────────┐
         │ Clear: Advance to      │      │ Tier 2: LCS & Sequence │
         │ Step 11                │      │ Alignment Verification │
         └────────────────────────┘      └────────────┬───────────┘
                                                      │
                                             ┌────────┴────────┐
                                             ▼                 ▼
                                    (LCS Ratio > 0.20)  (LCS Ratio ≤ 0.20)
                                    ┌────────────────┐  ┌────────────────┐
                                    │ Evict / Redact │  │ Clear: Advance │
                                    │ Record         │  │ to Step 11     │
                                    └────────────────┘  └────────────────┘

```

### Stage 1: Automated Evaluation Registry Sync & N-Gram Indexing

1. **Dynamic Evaluation Suite Sync:** At pipeline execution startup, the decontamination module automatically ingests all active benchmark texts directly from the enterprise evaluation catalog registry. This guarantees that whenever new internal gold-standard evaluation sets or public benchmarks are added, the decontamination pipeline absorbs those sequences automatically.
2. **N-Gram Hash Matrix Construction:** Benchmark text sets are split into sliding $N$-gram sequences (typically $N \in [5, 13]$ depending on sentence specificity). Each benchmark $N$-gram is hashed into a fixed 64-bit integer signature using a non-cryptographic hash function (MurmurHash3 / XXHash) and loaded into a high-speed, memory-mapped Finite State Transducer (FST) or unified hash set matrix.

### Stage 2: Tier 1 - Fast Vectorized N-Gram Hash Pre-Filtering

1. **Sliding Window Tokenization:** Every incoming training document is sliced into identical sliding $N$-gram windows (e.g., $13$-grams).
2. **$O(1)$ Hash Match Verification:** The pipeline checks candidate document $N$-gram signatures against the evaluation hash matrix in memory:

- **Zero Hash Matches:** The document is cleared immediately as non-contaminated and advances to Step 11.
- **Hash Collision Detected:** The document is flagged as a potential leak candidate and routed to Tier 2 for sequence alignment validation.

### Stage 3: Tier 2 - Longest Common Subsequence (LCS) & Sequence Alignment Verification

For candidate records flagged in Tier 1, the engine executes an explicit string alignment check to differentiate coincidental $N$-gram phrase overlaps from true structural data leaks:

1. **LCS Ratio Calculation:** The length of the Longest Common Subsequence ($\text{LCS}$) between the candidate training record $D_{\text{train}}$ and the matched benchmark document $D_{\text{bench}}$ is calculated:

$$\text{LCS Ratio} = \frac{\text{Length}\left( \text{LCS}(D_{\text{train}}, D_{\text{bench}}) \right)}{\text{Length}(D_{\text{bench}})}$$

1. **Decontamination Decision Boundaries:**

- **Exact Sequence Match:** If the candidate record shares $\ge 20$ consecutive identical tokens with a benchmark evaluation item, it represents a direct structural leak.
- **Substantial Subsequence Overlap:** If $\text{LCS Ratio} > 0.20$, the matching benchmark sections are surgically redacted, or the entire record is evicted from the training stream via a zero-copy Boolean mask array.

---

## 4. Decontamination Strategy Matrix

| Document Contamination Profile      | Structural / Match Signature                                                            | Theoretical Engine                    | Pipeline Action                                       | Downstream Impact in Phase 3                                           |
| ----------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------- |
| **Un-Contaminated Training Record** | Zero $13$-gram hash collisions with evaluation index.                                   | Vectorized Sliding Window Hash Filter | **Retained:** Advances to Step 11.                    | Re-converged data advances to pre-tokenization audit.                  |
| **Coincidental Phrase Overlap**     | $13$-gram match present, but $\text{LCS Ratio} \le 0.20$ and $< 20$ consecutive tokens. | LCS Alignment & Transducer Engine     | **Retained:** Flagged as false positive match.        | Preserves valid domain prose without false-positive dropping.          |
| **Partial Evaluation Leakage**      | $\text{LCS Ratio} > 0.20$ relative to benchmark item length.                            | String Transducer Alignment Engine    | **Surgically Redacted:** Leaked sub-sequence removed. | Eliminates benchmark prompt overlaps while retaining surrounding text. |
| **Direct Benchmark Match**          | $\ge 20$ consecutive token match or exact question-answer match.                        | FST String Transducer Engine          | **Pruned:** Document fully evicted.                   | Guarantees absolute integrity of automated model registry gatekeepers. |

---

## 5. Algorithmic Principles & Theoretical Tooling

- **Finite State Transducer (FST) Matrices:** Memory-efficient, high-throughput string transducer structures capable of holding millions of 64-bit integer benchmark hashes in memory.
- **Sliding Window N-Gram Hash Generators:** Vectorized hash computation utilities engineered to slice text streams into overlapping $N$-grams and generate signatures at microsecond speeds.
- **Longest Common Subsequence (LCS) Engines:** Sequence alignment utilities designed to calculate sub-sequence overlap ratios between candidate records and target benchmark sets.
- **Automated Registry Sync Hooks:** Workflow orchestrator synchronization hooks that dynamically pull the latest benchmark suites from the evaluation catalog before pipeline execution.
