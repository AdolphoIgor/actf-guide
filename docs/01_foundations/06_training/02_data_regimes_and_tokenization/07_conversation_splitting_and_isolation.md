# Deterministic Conversation-Level Data Splitting and Partition Isolation

## 1. The Intra-Dialogue Leakage Vulnerability

In production language model training, data splitting cannot be treated as a simple tensor-slicing operation. A fundamental failure mode in continuous training pipelines is **intra-dialogue data leakage**, which occurs when a dataset is partitioned *after* text concatenation or sequence packing, rather than at strict document and conversation boundaries.

```text
INCORRECT: Slicing After Sequence Packing (Severe Data Leakage)
Packed Stream: [ ... Turn 1 (Doc A) | Turn 2 (Doc A) | Turn 3 (Doc A) | Turn 1 (Doc B) ... ]
                                     ▲
                            SPLIT BOUNDARY (90%)
Train Split receives: [ Turn 1 (Doc A), Turn 2 (Doc A) ]
Val Split receives:   [ Turn 3 (Doc A), Turn 1 (Doc B) ]
Result: The model memorizes Doc A's context in training, causing artificially low validation loss.

CORRECT: Slicing at Document Boundaries Prior to Packing
Silver Records:   [ Conversation A ]  [ Conversation B ]  [ Conversation C ]
                          │                   │                   │
                  Deterministic Hash  Deterministic Hash  Deterministic Hash
                          ▼                   ▼                   ▼
                      TRAIN SET            VAL SET            TRAIN SET

```

If the first two turns of a multi-turn troubleshooting session land in the training set and the resolution turn lands in the validation set, the validation metric no longer measures generalization. The model has already memorized the entity names, technical state, and syntax of that specific interaction during training, leading to **artificially deflated validation loss** and false positives in pre-flight evaluation.

---

## 2. Deterministic Hash Partitioning Mechanics

Distributed continuous training environments execute across multiple compute nodes and ephemeral worker processes. Relying on pseudo-random runtime shuffling (`random.shuffle()` with arbitrary seeds) introduces non-deterministic partition drift when workers restart, scale, or process shards out of order.

To guarantee reproducibility across distributed training runs, data partitioning must use **deterministic cryptographic or non-cryptographic hashing** over immutable document keys (`doc_id`, `conversation_id`, or normalized canonical text hash).

```text
                                [ Silver Record ]
                         {"conversation_id": "conv_8f9a2c", ...}
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │   Cryptographic Hash Digest   │
                       │   SHA-256("conv_8f9a2c")      │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │     Uniform Integer Space     │
                       │     Big-Endian Hex -> Uint64  │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │     Bucket Normalization      │
                       │     Bucket = Uint64 % 100     │
                       └───────────────┬───────────────┘
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 │                     │                     │
                 ▼                     ▼                     ▼
         Bucket < 90           90 <= Bucket < 95        Bucket >= 95
      ┌──────────────┐          ┌─────────────┐        ┌─────────────┐
      │  TRAIN POOL  │          │  VAL POOL   │        │  TEST POOL  │
      │  (0 to 89)   │          │  (90 to 94) │        │ (95 to 99)  │
      └──────────────┘          └─────────────┘        └─────────────┘

```

### Mathematical Formulation

Let $\mathcal{D} = \{c_1, c_2, \dots, c_N\}$ be the set of all curated Silver conversations. For each conversation $c_i$, we extract its immutable unique identifier $\text{ID}(c_i)$.

The partition assignment $P(c_i)$ is computed as:

$$h(c_i) = \text{SHA256}(\text{ID}(c_i)) \pmod{100}$$

$$P(c_i) = \begin{cases} \text{TRAIN}, & \text{if } 0 \le h(c_i) < R_{\text{train}} \\ \text{VAL}, & \text{if } R_{\text{train}} \le h(c_i) < R_{\text{train}} + R_{\text{val}} \\ \text{TEST}, & \text{if } R_{\text{train}} + R_{\text{val}} \le h(c_i) < 100 \end{cases}$$

Where $R_{\text{train}} = 90$, $R_{\text{val}} = 5$, and $R_{\text{test}} = 5$.

Because the output of SHA-256 is uniformly distributed across the hash space, this function guarantees exact ratio allocations without requiring inter-node communication or state synchronization.

---

## 3. Partition Allocations and Operational Roles

Every partitioned dataset in the ACTF framework fulfills a distinct role across the training and evaluation lifecycle:

| Dataset Partition | Target Split Ratio | Ingestion Stage | Primary Operational Role | Gradient Updates |
| --- | --- | --- | --- | --- |
| **Train Set** | **$90\%\text{--}95\%$** | JIT Step 12 Packing | Parameter optimization via forward/backward backpropagation loops. | **Active** ($\nabla \theta \ne 0$) |
| **Validation Set** | **$5\%$** | JIT Step 12 Packing | Periodic loss tracking (`estimate_loss`), early stopping triggers, and LR decay monitoring. | **Disabled** (`@torch.no_grad()`) |
| **Test Set (Holdout)** | **$5\%$** | Isolated Gold Storage | Post-training Gate 5 Gatekeeper benchmark; regression testing against production baseline. | **Disabled** (`eval` mode) |

### The Holdout Isolation Rule

The **Test Partition** must remain completely untouched throughout the training loop. It is never exposed to the training container's `estimate_loss` function. It is consumed exclusively by downstream evaluation runners in **`03_the_gatekeeper_pipeline`** to ensure an unbiased audit before model registry promotion.

---

## 4. Cryptographic Validation: The Gate 3 Leakage Perimeter

Before any training job allocates GPU VRAM or executes sequence packing, **Gate 3 (Split Leakage Gate)** runs an automated verification check against the partitioned pools.

```text
┌────────────────────────────────────────────────────────────────────────┐
│ GATE 3 VERIFICATION CRITERIA                                           │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Exact Set Disjointness:                                             │
│    Assert: Set(Train_IDs) ∩ Set(Val_IDs) == ∅                          │
│    Assert: Set(Train_IDs) ∩ Set(Test_IDs) == ∅                         │
│    Assert: Set(Val_IDs)   ∩ Set(Test_IDs) == ∅                         │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Volumetric Tolerance:                                               │
│    Actual_Train_Ratio ∈ [ Target_Train_Ratio ± 1.5% ]                  │
│    Actual_Val_Ratio   ∈ [ Target_Val_Ratio   ± 1.0% ]                  │
├────────────────────────────────────────────────────────────────────────┤
│ 3. MinHash 13-Gram Cross-Split Decontamination:                        │
│    Jaccard_Similarity(Doc_Train, Doc_Val) < 0.80                       │
└────────────────────────────────────────────────────────────────────────┘

```

If even a single conversation ID appears in both the training and validation partitions, or if an exact duplicate text snippet bypasses deduplication, Gate 3 triggers an immediate assertion failure and terminates the DAG before compute resources are provisioned.

---

## 5. Python Implementation: Deterministic Partitioning Engine

Below is the standalone, production-grade partitioner implementing deterministic conversation-level hashing, ratio enforcement, and split disjointness validation:

```python
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


class DeterministicDataSplitter:
    """
    Partitions multi-turn conversations into Train, Validation, and Test
    pools using deterministic SHA-256 hashing at document boundaries.
    """
    def __init__(
        self,
        train_ratio: float = 0.90,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05
    ):
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
        self.train_threshold = int(train_ratio * 100)
        self.val_threshold = int((train_ratio + val_ratio) * 100)

    def _hash_conversation(self, doc_id: str) -> int:
        """Maps an arbitrary document ID to a uniform integer in [0, 99]."""
        digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
        # Convert first 8 bytes of hex digest to unsigned 64-bit int
        int_value = int(digest[:16], 16)
        return int_value % 100

    def split_records(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Splits in-memory conversation records into Train, Val, and Test lists."""
        train_pool = []
        val_pool = []
        test_pool = []

        for record in records:
            # Fallback to content hash if conversation_id is missing
            doc_id = record.get("conversation_id")
            if not doc_id:
                doc_id = hashlib.sha256(json.dumps(record["messages"]).encode("utf-8")).hexdigest()

            bucket = self._hash_conversation(doc_id)

            if bucket < self.train_threshold:
                train_pool.append(record)
            elif bucket < self.val_threshold:
                val_pool.append(record)
            else:
                test_pool.append(record)

        return train_pool, val_pool, test_pool

    def verify_split_integrity(
        self,
        train: List[Dict[str, Any]],
        val: List[Dict[str, Any]],
        test: List[Dict[str, Any]]
    ) -> bool:
        """Gate 3 Pre-Flight Assertion: Validates 100% disjointness between splits."""
        def extract_ids(dataset):
            return {
                r.get("conversation_id") or hashlib.sha256(json.dumps(r["messages"]).encode()).hexdigest()
                for r in dataset
            }

        train_ids = extract_ids(train)
        val_ids = extract_ids(val)
        test_ids = extract_ids(test)

        # Assert zero intersection across all combinations
        train_val_overlap = train_ids.intersection(val_ids)
        train_test_overlap = train_ids.intersection(test_ids)
        val_test_overlap = val_ids.intersection(test_ids)

        if train_val_overlap or train_test_overlap or val_test_overlap:
            raise ValueError(
                f"🚨 GATE 3 FAILED: Split leakage detected!\n"
                f"Train ∩ Val Overlap: {len(train_val_overlap)}\n"
                f"Train ∩ Test Overlap: {len(train_test_overlap)}\n"
                f"Val ∩ Test Overlap: {len(val_test_overlap)}"
            )

        print("✨ GATE 3 PASSED: Zero partition leakage verified.")
        return True

```
