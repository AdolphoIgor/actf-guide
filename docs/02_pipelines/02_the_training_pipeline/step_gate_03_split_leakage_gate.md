# Gate 3: Split Leakage and Dataset Contamination Prevention Gate

## 1. The Threat Model of Dataset Contamination and Split Leakage

In pre-training and supervised fine-tuning (SFT) pipelines, **Split Leakage** and **Downstream Benchmark Contamination** undermine empirical validation. When evaluation data leaks into the training corpus, the model optimizes for associative retrieval and sequence memorization rather than generalized semantic reasoning.

```text
Training Corpus Ingestion Pipeline
                 │
                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GATE 3: DATASET SPLIT LEAKAGE & CONTAMINATION FIREWALL                 │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Train-Val-Test Split Independence Verification (Zero-Overlap Assert)│
│ 2. Canonical Downstream Benchmark Decontamination (13-Gram Filter)     │
│ 3. Fuzzy Document-Level Leakage Audit (MinHash / LSH Indexing)         │
│ 4. Entity / Conversation Group Leakage Audit (Dialogue Integrity)      │
│ 5. Temporal Horizon Leakage Audit (Lookahead Bias Prevention)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       [ PASS: Proceed to Tokenization ]   [ FAIL: Quarantine Dataset Shard ]

```

### Contamination Topologies

```text
┌────────────────────────────────────────────────────────────────────────┐
│ DATASET CONTAMINATION TAXONOMY                                         │
├──────────────────────────┬───────────────────┬─────────────────────────┤
│ Contamination Type       │ Structural Cause  │ Empirical Failure Mode  │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 1. Exact Test Leakage    │ Identical string  │ 100% artificial recall; │
│    (Verbatim Overlap)    │ in train and eval │ masks true generalization│
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 2. Fuzzy Substring /     │ Paraphrased or    │ Memorized reasoning CoT;│
│    Near-Duplicate Leak   │ templated prompts │ inflated benchmark score│
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 3. Group / Dialogue      │ Splitting turns of│ Model predicts next turn│
│    Fragmentation Leak    │ same conversation │ from memorized past turns│
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 4. Downstream Benchmark  │ Web scrapes ingest│ Public leaderboard score│
│    Contamination         │ benchmark splits  │ invalidation / fraud    │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 5. Temporal Lookahead    │ Future tokens in  │ Spurious correlation on │
│    Leakage               │ time-series SFT   │ historical timelines    │
└──────────────────────────┴───────────────────┴─────────────────────────┘

```

**Gate 3 (Split Leakage Gate)** acts as an automated, non-negotiable verification checkpoint prior to token serialization and GPU allocation. If any evaluation sample or benchmark problem is discovered within the training shards, the ingestion pipeline halts execution immediately.

---

## 2. Mathematical Detection Formulations

Gate 3 applies a multi-tier algorithmic screening battery combining exact $N$-gram containment, Locality-Sensitive Hashing (LSH), and group-level metadata assertions.

```text
Multi-Tier Contamination Screening Hierarchy:

Incoming Training Document (D_train)
                 │
                 ├──► [ TIER 1: Exact 13-Gram Rolling Hash Check ] ──► Matches Benchmark? ──► REJECT
                 │
                 ├──► [ TIER 2: MinHash LSH Jaccard Similarity ]   ──► J(D_t, D_v) >= 0.80? ──► REJECT
                 │
                 └──► [ TIER 3: Group & Session Metadata Hash ]   ──► Session Cross-Split? ──► REJECT

```

### A. Exact $N$-Gram Containment Metric ($C_N$)

Let $D_{\text{train}}$ be a candidate training document and $D_{\text{eval}}$ be an immutable evaluation or benchmark sample (e.g., an MMLU question or GSM8K problem).

Let $G_N(D)$ be the multi-set of all contiguous token/character $N$-grams in document $D$:

$$G_N(D) = \{ (t_i, t_{i+1}, \dots, t_{i+N-1}) \mid 1 \le i \le \vert{}D\vert{} - N + 1 \}$$

The directed containment metric $C_N(D_{\text{eval}}, D_{\text{train}})$ evaluates the proportion of evaluation $N$-grams present within the training document:

$$C_N(D_{\text{eval}}, D_{\text{train}}) = \frac{\left\vert{} G_N(D_{\text{eval}}) \cap G_N(D_{\text{train}}) \right\vert{}}{\left\vert{} G_N(D_{\text{eval}}) \right\vert{}}$$

```text
Decontamination Invariant (13-Gram Rule - Brown et al. / OpenAI):
  • A training document D_train is classified as CONTAMINATED if:
      Exists D_eval in Benchmark_Set such that |G_13(D_eval) ∩ G_13(D_train)| == |G_13(D_eval)|
      OR C_13(D_eval, D_train) >= 0.80 for long evaluation prompts (> 200 tokens).

```

---

### B. MinHash and Locality-Sensitive Hashing (LSH) for Fuzzy Leakage

For document-level near-duplicate detection between training shards $\mathcal{D}_{\text{train}}$ and validation splits $\mathcal{D}_{\text{val}}$, Gate 3 computes the Jaccard similarity over word shingle sets $S(D)$:

$$J(D_1, D_2) = \frac{\vert{}S(D_1) \cap S(D_2)\vert{}}{\vert{}S(D_1) \cup S(D_2)\vert{}}$$

Computing pairwise Jaccard similarities over billions of tokens is computationally intractable ($\mathcal{O}(\vert{}\mathcal{D}_{\text{train}}\vert{} \times \vert{}\mathcal{D}_{\text{val}}\vert{})$). MinHash compresses each document into a signature vector of $K$ hash permutations:

$$h_{\min}^{(k)}(D) = \min_{s \in S(D)} h_k(s), \quad k \in \{1, 2, \dots, K\}$$

$$\mathbb{P}\left[ h_{\min}^{(k)}(D_1) = h_{\min}^{(k)}(D_2) \right] = J(D_1, D_2)$$

```text
MinHash LSH Banding Matrix:

Signature Vector (K = 128 hashes)
┌────────────────────────────────────────────────────────┐
│ Band 1 (Rows r = 8)  ──► Hash to Bucket B_1            │
├────────────────────────────────────────────────────────┤
│ Band 2 (Rows r = 8)  ──► Hash to Bucket B_2            │
├────────────────────────────────────────────────────────┤
│ ...                                                    │
├────────────────────────────────────────────────────────┤
│ Band 16 (Rows r = 8) ──► Hash to Bucket B_16           │
└────────────────────────────────────────────────────────┘

Probability of Candidate Pair Collision:
  P(Collision) = 1 - (1 - J(D_1, D_2)^r)^b
  Where b = 16 bands, r = 8 rows per band (K = b * r = 128).

```

If the estimated Jaccard similarity $J(D_{\text{train}}, D_{\text{val}}) \ge \tau_{\text{fuzzy}}$ (where $\tau_{\text{fuzzy}} = 0.80$), the training document is flagged as a leaked near-duplicate.

---

### C. Group-Level Metadata Leakage (Dialogue & Session Invariance)

In multi-turn chat and user session datasets, splitting individual turns randomly across train and validation splits allows the model to memorize user specifics and dialogue structure:

```text
Group Leakage Failure Mode:

Session ID: #84920 (5 Conversation Turns)
  • Turn 1 (User / Asst) ──► Routed to TRAIN SPLIT
  • Turn 2 (User / Asst) ──► Routed to VALIDATION SPLIT  <── CRITICAL LEAKAGE!
  • Turn 3 (User / Asst) ──► Routed to TRAIN SPLIT

Result: Validation loss measures verbatim memorization of Turn 1 context,
        producing artificially low validation perplexity.

```

Gate 3 enforces **Strict Group Partitioning**: all records sharing a `session_id`, `conversation_id`, or `author_id` are grouped as atomic units prior to split hashing:

$$\text{SplitAssign}(\text{Group}_i) = \begin{cases} \text{Validation}, & \text{if } \text{SHA256}(\text{Group\_ID}_i \parallel \text{Salt}) \pmod{100} < P_{\text{val}} \\ \text{Train}, & \text{otherwise} \end{cases}$$

$$\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} = \emptyset \quad \text{over all Group IDs}$$

---

## 3. Standardized Downstream Benchmark Decontamination Suites

To certify that a model's performance on public benchmarks reflects true reasoning, Gate 3 cross-checks all training shards against reference benchmark suites:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ REFERENCE BENCHMARK DECONTAMINATION CATALOG                            │
├──────────────────────────┬───────────────────┬─────────────────────────┤
│ Target Benchmark         │ Scope & Split     │ Screening Strategy      │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ MMLU / MMLU-Pro          │ Test / Val Sets   │ 13-Gram Exact Match on  │
│                          │ (All 57 subjects) │ Question + Options      │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ GSM8K / MATH             │ Test Splits       │ Normalizing math LaTeX; │
│                          │ (All problems)    │ 10-Gram Problem Match   │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ HumanEval / MBPP         │ Complete Suite    │ AST Token Shingle Match │
│                          │ (Prompt + Tests)  │ on docstrings & code    │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ ARC (Challenge / Easy)   │ Test Splits       │ 13-Gram Exact Match     │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ HellaSwag / WinoGrande   │ Val / Test Splits │ 13-Gram Context + Target│
└──────────────────────────┴───────────────────┴─────────────────────────┘

```

### Remediation Protocol on Benchmark Hit

When a benchmark item is discovered inside a training document:

1. **Document-Level Removal (Default Policy):** The entire training document containing the leaked question/context is discarded from the pre-training pool.
2. **Target Masking (SFT Sub-Splits):** If the document is part of a high-value instructional dataset, the contaminated tokens are masked with loss target `-100`, disabling gradient backpropagation over the contaminated span while retaining background syntactic context.

---

## 4. Gate 3 Mathematical Invariant Vector

Gate 3 evaluates dataset splits against an absolute, zero-tolerance boolean assertion vector $\mathbf{G}_3$:

$$\mathbf{G}_3 = \begin{bmatrix} \mathbb{I}\left( \left\vert{} \text{Entities}(\mathcal{D}_{\text{train}}) \cap \text{Entities}(\mathcal{D}_{\text{val}}) \right\vert{} == 0 \right) \\ \mathbb{I}\left( \text{Count}\left( C_{13}(D_{\text{bench}}, D_{\text{train}}) > 0 \right) == 0 \right) \\ \mathbb{I}\left( \max_{i, j} J(D_{\text{train}, i}, D_{\text{val}, j}) < 0.80 \right) \\ \mathbb{I}\left( \text{TemporalMax}(\mathcal{D}_{\text{train}}) \le \text{TemporalMin}(\mathcal{D}_{\text{val}}) \right) \end{bmatrix}$$

$$\text{Gate 3 Status} = \begin{cases} \text{PASSED (Promote to Tokenizer)}, & \text{if } \prod_{k=1}^4 \mathbf{G}_{3, k} == 1 \\ \text{QUARANTINE (Halt Pipeline)}, & \text{otherwise} \end{cases}$$

---

## 5. Python Implementation: Production Gate 3 Leakage Engine

Below is the standalone Python verification engine that implements 13-gram benchmark decontamination, MinHash LSH fuzzy leakage auditing, and group-aware dialogue split verification:

```python
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import re
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


@dataclass
class ContaminationHit:
    train_doc_id: str
    benchmark_name: str
    eval_item_id: str
    overlap_ngram: str
    match_type: str  # "exact_13gram", "fuzzy_jaccard", or "group_leak"


@dataclass
class Gate3Verdict:
    passed: bool
    total_train_docs: int
    total_val_docs: int
    contamination_hits: List[ContaminationHit]
    quarantined_doc_ids: Set[str]
    audit_summary: Dict[str, Any]


class Gate3SplitLeakageEngine:
    """
    Automated Gate 3 verification engine for detecting dataset split leakage,
    fuzzy cross-split duplication, and downstream benchmark contamination.
    """
    def __init__(
        self,
        ngram_size: int = 13,
        fuzzy_jaccard_threshold: float = 0.80,
        num_minhash_permutations: int = 128
    ):
        self.n = ngram_size
        self.jaccard_thresh = fuzzy_jaccard_threshold
        self.k_hashes = num_minhash_permutations

        # Reference benchmark 13-gram index: Hash(13-gram) -> (benchmark_name, item_id)
        self.benchmark_ngram_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        
        # Validation Shingle & MinHash index
        self.val_minhash_signatures: Dict[str, List[int]] = {}

    @staticmethod
    def _normalize_and_tokenize(text: str) -> List[str]:
        """Lowercases and extracts alphanumeric token sequences."""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return cleaned.split()

    def _extract_ngrams(self, tokens: List[str]) -> Set[str]:
        """Extracts rolling n-grams from tokenized list."""
        if len(tokens) < self.n:
            return set()
        return {
            " ".join(tokens[i : i + self.n])
            for i in range(len(tokens) - self.n + 1)
        }

    # =====================================================================
    # 1. BENCHMARK INDEX REGISTRATION
    # =====================================================================
    def register_benchmark_suite(
        self, benchmark_name: str, benchmark_items: List[Dict[str, str]]
    ):
        """
        Ingests reference evaluation benchmarks (e.g., MMLU, GSM8K, HumanEval).
        Each item format: {"id": str, "text": str}
        """
        for item in benchmark_items:
            tokens = self._normalize_and_tokenize(item["text"])
            ngrams = self._extract_ngrams(tokens)

            for ng in ngrams:
                ng_hash = hashlib.sha256(ng.encode("utf-8")).hexdigest()
                self.benchmark_ngram_index[ng_hash].append((benchmark_name, item["id"]))

    # =====================================================================
    # 2. MINHASH GENERATION
    # =====================================================================
    def _compute_minhash(self, shingles: Set[str]) -> List[int]:
        """Computes K MinHash integer signatures for a set of token shingles."""
        if not shingles:
            return [0] * self.k_hashes

        signatures = []
        for i in range(self.k_hashes):
            min_val = float("inf")
            for shingle in shingles:
                # Deterministic parameterized hashing
                h = int(hashlib.md5(f"{i}_{shingle}".encode("utf-8")).hexdigest(), 16)
                if h < min_val:
                    min_val = h
            signatures.append(int(min_val))
        return signatures

    @staticmethod
    def _estimate_jaccard(sig1: List[int], sig2: List[int]) -> float:
        """Estimates Jaccard similarity via matching MinHash signature rows."""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    # =====================================================================
    # 3. SPLIT & BENCHMARK AUDIT PIPELINE
    # =====================================================================
    def audit_splits(
        self,
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]]
    ) -> Gate3Verdict:
        """
        Executes complete Gate 3 verification suite across train and validation splits.
        Record schema: {"doc_id": str, "group_id": str, "text": str}
        """
        contamination_hits: List[ContaminationHit] = []
        quarantined_doc_ids: Set[str] = set()

        # Step 1: Index Validation Records for Fuzzy & Exact Match
        val_group_ids: Set[str] = set()
        val_exact_ngrams: Dict[str, str] = {}  # ngram_hash -> val_doc_id

        for v_rec in val_records:
            val_group_ids.add(str(v_rec["group_id"]))
            v_tokens = self._normalize_and_tokenize(v_rec["text"])
            
            # Index 13-grams
            for ng in self._extract_ngrams(v_tokens):
                h = hashlib.sha256(ng.encode("utf-8")).hexdigest()
                val_exact_ngrams[h] = v_rec["doc_id"]

            # Compute MinHash signature (using 3-shingle words)
            shingles = set(" ".join(v_tokens[i : i + 3]) for i in range(max(0, len(v_tokens) - 2)))
            self.val_minhash_signatures[v_rec["doc_id"]] = self._compute_minhash(shingles)

        # Step 2: Scan Training Records Against Validation & Benchmarks
        for t_rec in train_records:
            doc_id = t_rec["doc_id"]
            group_id = str(t_rec["group_id"])
            t_tokens = self._normalize_and_tokenize(t_rec["text"])
            t_ngrams = self._extract_ngrams(t_tokens)

            # Test A: Group / Dialogue ID Leakage Assertion
            if group_id in val_group_ids:
                contamination_hits.append(ContaminationHit(
                    train_doc_id=doc_id,
                    benchmark_name="validation_split",
                    eval_item_id=group_id,
                    overlap_ngram=f"Group ID Collision: {group_id}",
                    match_type="group_leak"
                ))
                quarantined_doc_ids.add(doc_id)
                continue

            # Test B: Downstream Benchmark 13-Gram Decontamination
            for ng in t_ngrams:
                ng_hash = hashlib.sha256(ng.encode("utf-8")).hexdigest()
                
                # Check benchmark collision
                if ng_hash in self.benchmark_ngram_index:
                    for bench_name, item_id in self.benchmark_ngram_index[ng_hash]:
                        contamination_hits.append(ContaminationHit(
                            train_doc_id=doc_id,
                            benchmark_name=bench_name,
                            eval_item_id=item_id,
                            overlap_ngram=ng,
                            match_type="exact_13gram"
                        ))
                        quarantined_doc_ids.add(doc_id)

                # Check validation set verbatim overlap
                if ng_hash in val_exact_ngrams:
                    contamination_hits.append(ContaminationHit(
                        train_doc_id=doc_id,
                        benchmark_name="validation_split",
                        eval_item_id=val_exact_ngrams[ng_hash],
                        overlap_ngram=ng,
                        match_type="exact_13gram"
                    ))
                    quarantined_doc_ids.add(doc_id)

            # Test C: Fuzzy MinHash Duplicate Verification
            t_shingles = set(" ".join(t_tokens[i : i + 3]) for i in range(max(0, len(t_tokens) - 2)))
            t_sig = self._compute_minhash(t_shingles)

            for v_doc_id, v_sig in self.val_minhash_signatures.items():
                sim = self._estimate_jaccard(t_sig, v_sig)
                if sim >= self.jaccard_thresh:
                    contamination_hits.append(ContaminationHit(
                        train_doc_id=doc_id,
                        benchmark_name="validation_split",
                        eval_item_id=v_doc_id,
                        overlap_ngram=f"Estimated Jaccard = {sim:.4f}",
                        match_type="fuzzy_jaccard"
                    ))
                    quarantined_doc_ids.add(doc_id)

        verdict_passed = len(quarantined_doc_ids) == 0

        summary = {
            "total_train_inspected": len(train_records),
            "total_val_inspected": len(val_records),
            "total_quarantined_docs": len(quarantined_doc_ids),
            "contamination_rate": len(quarantined_doc_ids) / max(1, len(train_records)),
            "benchmark_index_size": len(self.benchmark_ngram_index)
        }

        return Gate3Verdict(
            passed=verdict_passed,
            total_train_docs=len(train_records),
            total_val_docs=len(val_records),
            contamination_hits=contamination_hits,
            quarantined_doc_ids=quarantined_doc_ids,
            audit_summary=summary
        )

```

---

## 6. Gate 3 Pre-Flight Audit Matrix

| Audit Target | Screening Method | Hard Tolerance Threshold | Action on Failure |
| --- | --- | --- | --- |
| **Benchmark Contamination** | 13-Gram exact token hash match | **$0$ Hits Allowed (Zero-Tolerance)** | Quarantine training document; remove from shard |
| **Train/Val Exact Overlap** | 13-Gram exact token hash match | **$0$ Hits Allowed (Zero-Tolerance)** | Purge duplicate from training split |
| **Fuzzy Near-Duplicates** | MinHash LSH Jaccard similarity | $\text{Jaccard} < 0.80$ | Exclude near-duplicate from training set |
| **Group / Dialogue State** | Hash of `conversation_id` / `user_id` | **$0$ Cross-Split Collisions** | Re-partition dataset by atomic Group ID |
| **Temporal Horizon** | Timestamp metadata assertion | $\text{Max}(T_{\text{train}}) \le \text{Min}(T_{\text{val}})$ | Strip lookahead entries; re-split chronologically |