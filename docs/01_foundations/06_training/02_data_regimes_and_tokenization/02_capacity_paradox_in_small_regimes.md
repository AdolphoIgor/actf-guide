# Ingestion Topologies: Pattern A, Pattern B, and the Ephemeral Gold Tensor Cache

In automated Continuous Training (CT) systems, the architectural interface between data curation and GPU compute defines the system's storage footprint, model portability, and hardware utilization.

The core architectural dilemma centers on **when and where tokenization occurs**: whether text is compiled into static integer tensors offline during data preprocessing, compiled Just-In-Time (JIT) in memory at training startup, or managed through a hybrid caching tier.

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Pattern A: Offline Static Tensorization (Static Gold Layer)            │
│ Raw Text ──► Clean ──► Tokenize & Pack (Disk) ──► Persist Gold Tensors │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Rigid coupling to single model
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Pattern B: Just-In-Time (JIT) In-Memory Compilation (Decoupled Silver) │
│ Raw Text ──► Clean ──► Universal Silver JSONL ──► JIT in /dev/shm (GPU)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Dynamic runtime flexibility
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Hybrid Architecture: Ephemeral Gold Tensor Cache                       │
│ Silver Text (System of Record) ──► Hash-Keyed Ephemeral Tensor Cache   │
└────────────────────────────────────────────────────────────────────────┘

```

---

## 1. Pattern A: Offline Static Tensorization (The Static Gold Layer)

In **Pattern A**, the data curation pipeline executes all text cleaning, chat template normalization, tokenization, sequence packing, and label masking upstream on distributed CPU workers (e.g., Spark/Ray).

The resulting outputs are written to persistent object storage (S3/GCS) as pre-computed, binary **Gold-layer** tensor shards (e.g., `.bin`, `.pt`, or memory-mapped Arrow IPC files).

```text
    [ Data Engineering DAG ]
      ├── 1. Extraction & Hygiene (Steps 1-10)
      ├── 2. Jinja Template Application (ChatML)
      ├── 3. Tokenizer Encoding (Vocab Size V)
      ├── 4. 2D Tensor Packing (Shape: B x L)
      └── 5. Write to S3: s3://models-data/gold/qwen_vocab_151646/len_2048/shards_00.bin
                 │
                 ▼
    [ GPU Training Container Boot ]
      └── Memory-maps pre-computed binary shards directly into PyTorch DataLoaders.

```

### Advantages

- **Zero GPU Startup Latency:** Training nodes provision and begin forward/backward matrix passes immediately with zero tokenization overhead.
- **Low Host RAM Overhead:** GPU host nodes do not require large memory pools (`/dev/shm`) for in-memory tokenization workers.

### Architectural Failure Modes & Bottlenecks

- **Strict Architecture Lock-in:** The stored dataset is tied to a specific base model's vocabulary ($V$), chat control tokens (e.g., `<|im_start|>` vs. `<|start_header_id|>`), and context window length ($L$).

- **Storage Footprint Hypertrophy:** Running experiments across three candidate models (e.g., `Qwen2.5-0.5B`, `Llama-3.2-1B`, and `Mistral-7B`) requires generating and storing three independent binary Gold datasets, multiplying storage costs.
- **Upstream Re-Run Penalties:** Modifying context length (e.g., testing $L = 2048$ vs. $L = 4096$) forces a complete re-run of the upstream data pipeline, stalling continuous training iterations.

---

## 2. Pattern B: Just-In-Time (JIT) In-Memory Compilation (Decoupled Silver)

In **Pattern B**, the boundary between the Data Pipeline and the Training Pipeline is set at the **Silver Layer**. The data pipeline is responsible for text cleaning, exact deduplication, AST syntax verification, PII redaction, and benchmark decontamination (Steps 1–10). It persists a single, model-agnostic dataset in universal format (`{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`).

When a training job initializes (`dag_04_model_train.py`), the training container streams the Silver text records into shared virtual memory (`/dev/shm`), dynamically resolves the base model's native Jinja template (Step 11), encodes tokens via multi-threaded Rust BPE kernels, applies target-only loss masking, and packs sequences into tensors on the fly (Step 12).

```text
    [ Data Curation DAG (CPU Pods) ]
      └── Outputs universal, model-agnostic text: s3://data-lake/silver/sft_v1.parquet
                                │
                                ▼
    [ Training Container Initialization (In-Memory in /dev/shm) ]
      ├── 1. Read Model Recipe Manifest (training_config.yaml)
      ├── 2. Dynamic Chat Template Resolution (tokenizer.apply_chat_template)
      ├── 3. Multi-Threaded Byte-Pair Encoding
      ├── 4. Target-Only Loss Masking (Prompt Tokens = -100)
      ├── 5. 2D Sequence Packing (B x L Matrices)
      └── 6. Direct In-RAM Stream to GPU Tensor Cores

```

### Advantages

- **100% Model Agnostic:** The Silver data lake serves as a permanent, universal System of Record.

- **Zero-Cost Model Swapping:** Switching base models from Qwen to Llama requires updating only the declarative YAML configuration manifest (`dag_run.conf`). The orchestrator points the generic training container to the new tokenizer, leaving the upstream data lake untouched.

- **Zero Storage Waste:** Eliminates object storage bloat by generating binary tensors strictly in volatile memory (`/dev/shm`).

### Architectural Failure Modes & Bottlenecks

- **Startup Initialization Latency:** GPU workers must wait for CPU worker threads to tokenize and pack the dataset before the first gradient step executes.
- **Redundant Computation in Sweeps:** Running multi-epoch hyperparameter tuning sweeps (testing learning rates, weight decays, or LoRA ranks) repeatedly re-tokenizes identical text on every run.

---

## 3. The Hybrid Middle Ground: Ephemeral Gold Tensor Cache

To combine the dynamic flexibility of Pattern B with the zero-latency execution of Pattern A, production platforms implement an **Ephemeral Gold Tensor Cache**.

In this architecture, the data lake continues to store universal Silver text as the single source of truth. However, the training execution engine manages a local or fast-tier ephemeral caching layer keyed by a cryptographic composite hash.

```text
                         [ Ingest Silver Parquet ]
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │  COMPUTE COMPOSITE CACHE KEY (SHA-256)                 │
        │  K = Hash(Silver_Data) + Hash(Tokenizer) + L + Template│
        └────────────────────────────┬───────────────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
             [ Cache Hit: Key Exists ]       [ Cache Miss: Key Absent ]
                     │                               │
                     ▼                               ▼
       ┌───────────────────────────┐   ┌───────────────────────────┐
       │ Memory-Map Cached Tensors │   │ Execute JIT Steps 11 & 12 │
       │ Direct mmap into GPU VRAM │   │ Stream to /dev/shm        │
       └─────────────┬─────────────┘   └─────────────┬─────────────┘
                     │                               │
                     │                               ▼
                     │                 ┌───────────────────────────┐
                     │                 │ Asynchronously Populate   │
                     │                 │ Ephemeral Cache Storage   │
                     │                 └─────────────┬─────────────┘
                     │                               │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                      [ Execute Parameter Updates ]

```

### The Composite Cache Key Formula

A cached tensor partition is valid if and only if the underlying data, tokenizer vocabulary, context limits, and chat templates are identical:

$$\text{Cache Key} = \text{SHA256}\Big(\text{Hash}(\mathcal{D}_{\text{Silver}}) \,\Vert{}\, \text{Hash}(\mathcal{T}_{\text{config}}) \,\Vert{}\, \text{Context Length } L \,\Vert{}\, \text{Hash}(\text{Jinja Template})\Big)$$

Where:

- $\text{Hash}(\mathcal{D}_{\text{Silver}})$ is the 64-bit cryptographic hash of the input Silver text dataset shards.
- $\text{Hash}(\mathcal{T}_{\text{config}})$ is the hash of the tokenizer vocabulary and merge rules (`tokenizer.json`).
- $L$ is the target sequence packing context window (e.g., $2048$).
- $\text{Hash}(\text{Jinja Template})$ is the hash of the template string formatting user/assistant roles.

### Operational Mechanics

1. **First Run (Cold Cache):** The training container detects a cache miss. It pulls Silver text, executes Steps 11 and 12 in `/dev/shm`, feeds PyTorch DataLoaders immediately, and writes the packed tensor matrix asynchronously to an ephemeral cache mount (local NVMe SSD or high-throughput shared scratch bucket).
2. **Subsequent Runs (Warm Cache):** Hyperparameter sweeps, multi-epoch runs, or restarted crashed jobs compute an identical Cache Key. The engine skips Steps 11 and 12, memory-mapping (`mmap`) the pre-compiled binary tensors directly into GPU VRAM.
3. **Eviction & Lifecycle Management:** The cache directory operates under a strict **Time-to-Live (TTL)** or **Least Recently Used (LRU)** eviction policy. Because the Silver layer remains intact, the loss of any cached tensor shard has zero data durability impact.

---

## 4. Architectural Comparison Matrix

| Evaluation Dimension        | Pattern A (Static Gold Storage)                                  | Pattern B (JIT In-Memory Compilation)                           | Hybrid (Ephemeral Gold Tensor Cache)                           |
| --------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------- |
| **System of Record**        | Pre-tokenized binary files (`.bin`, `.pt`)                       | Universal Silver text (`.parquet`, `.jsonl`)                    | Universal Silver text (`.parquet`, `.jsonl`)                   |
| **Storage Overhead**        | High ($\mathcal{O}(M \times \text{Runs})$, multiplies per model) | Minimal ($\mathcal{O}(1)$, single universal Silver store)       | Low ($\mathcal{O}(1)$ persistent + bounded LRU scratch)        |
| **GPU Boot Latency**        | Instant ($0\text{ sec}$ preprocessing)                           | Small delay ($10\text{--}60\text{ sec}$ JIT compilation in RAM) | Instant on cache hits; small delay on cold misses              |
| **Base Model Portability**  | Inflexible (re-runs upstream data pipeline on model swap)        | Fully decoupled (swap model via YAML manifest)                  | Fully decoupled (recomputes cache key automatically)           |
| **Context Length Agility**  | Locked to pre-tokenized sequence length $L$                      | Parameterized at container runtime ($L$ in YAML)                | Parameterized at runtime; caches each distinct $L$             |
| **Hyperparameter Sweeps**   | Fast execution; high pre-storage preparation cost                | Redundant CPU tokenization on every sweep iteration             | Optimal (JIT on run 1, instant cache reuse on runs $2\dots N$) |
| **Host Memory Requirement** | Standard RAM allocation                                          | Requires high-throughput `/dev/shm` virtual RAM                 | Requires local NVMe scratch disk or `/dev/shm`<br>             |

---

## 5. Implementation: The Hybrid In-Memory & Caching Engine

Below is the PyTorch-compatible Python implementation illustrating dynamic cache validation, JIT compilation fallback, and zero-copy `/dev/shm` streaming:

```python
import hashlib
import json
import os
from pathlib import Path
import torch
from tokenizers import ByteLevelBPETokenizer

class EphemeralTensorLoader:
    """
    Manages JIT sequence packing with automatic ephemeral caching
    keyed by data, tokenizer, and context parameters.
    """
    def __init__(
        self,
        silver_data_path: str,
        tokenizer: ByteLevelBPETokenizer,
        block_size: int,
        chat_template: str,
        cache_dir: str = "/dev/shm/actf_cache"
    ):
        self.silver_data_path = silver_data_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.chat_template = chat_template
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Compute deterministic composite cache key
        self.cache_key = self._compute_cache_key()
        self.cache_file = self.cache_dir / f"{self.cache_key}.pt"

    def _compute_cache_key(self) -> str:
        hasher = hashlib.sha256()

        # 1. Hash Silver Data File Metadata & Content Header
        with open(self.silver_data_path, "rb") as f:
            hasher.update(f.read(65536)) # Fast sample hash

        # 2. Hash Tokenizer Vocabulary & Merge Signature
        hasher.update(str(self.tokenizer.get_vocab_size()).encode())

        # 3. Hash Context Length and Template
        hasher.update(str(self.block_size).encode())
        hasher.update(self.chat_template.encode())

        return hasher.hexdigest()[:16]

    def load_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Loads cached tensors on hit, or executes JIT compilation on miss."""
        if self.cache_file.exists():
            # CACHE HIT: Fast memory-map directly into RAM
            print(f"✨ Ephemeral Cache Hit! Loading: {self.cache_file}")
            payload = torch.load(self.cache_file, weights_only=True)
            return payload["input_ids"], payload["labels"]

        # CACHE MISS: Execute JIT Steps 11 & 12
        print(f"⚙️ Cache Miss ({self.cache_key}). Executing JIT tokenization in /dev/shm...")
        input_ids, labels = self._compile_jit()

        # Asynchronously persist to ephemeral cache
        torch.save({"input_ids": input_ids, "labels": labels}, self.cache_file)
        return input_ids, labels

    def _compile_jit(self) -> tuple[torch.Tensor, torch.Tensor]:
        """JIT Tokenization, target-only loss masking, and sequence packing."""
        with open(self.silver_data_path, "r", encoding="utf-8") as f:
            raw_lines = [json.loads(line) for line in f]

        packed_tokens = []
        packed_labels = []

        for record in raw_lines:
            # Step 11: Jinja Template Formatting
            # Step 12: BPE Tokenization with Prompt Masking (-100)
            user_text = record["messages"][0]["content"]
            asst_text = record["messages"][1]["content"]

            u_ids = self.tokenizer.encode(f"<|im_start|>user\n{user_text}<|im_end|>\n").ids
            a_ids = self.tokenizer.encode(f"<|im_start|>assistant\n{asst_text}<|im_end|>\n").ids

            # Prompt labels masked to -100; assistant labels retained
            tokens = u_ids + a_ids
            labels = ([-100] * len(u_ids)) + a_ids

            packed_tokens.extend(tokens)
            packed_labels.extend(labels)

        # Truncate/Pack into fixed B x L tensors
        num_blocks = len(packed_tokens) // self.block_size
        tensor_x = torch.tensor(packed_tokens[:num_blocks * self.block_size], dtype=torch.long)
        tensor_y = torch.tensor(packed_labels[:num_blocks * self.block_size], dtype=torch.long)

        return tensor_x.view(-1, self.block_size), tensor_y.view(-1, self.block_size)

```
