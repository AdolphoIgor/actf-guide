# The Enterprise Infrastructure & Compute Topology

## 1. Architectural Paradigm & Lifecycle Separation

Modern data curation architectures for Large Language Models (LLMs) require a strict separation between **model-agnostic curation** and **model-dependent ingestion**.

Treating tokenization and sequence packing as an immutable upstream data step creates significant technical debt: any change in model target, vocabulary size, tokenizer scheme (e.g., BPE vs. WordPiece), or context window length requires re-running the entire multi-stage cleaning pipeline from scratch.

To solve this, the Automated Continuous Training Framework (ACTF) splits data processing across two isolated lifecycles and maps them to specialized compute topologies:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA CURATION LIFECYCLE                         │
│                    (Model-Agnostic / Storage DAG)                      │
│                                                                        │
│  Ingestion ──> Domain Branching ──> Curation & Dedup ──> Safety/PII   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                         [ Clean Text Storage ]
                           (Silver Data Layer)
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                       MODEL TRAINING LIFECYCLE                         │
│                     (Model-Dependent / GPU DAG)                        │
│                                                                        │
│   Schema Policy ──> Dynamic Tokenization ──> GPU Sequence Packing      │
└────────────────────────────────────────────────────────────────────────┘

```

1. **The Data Curation Lifecycle (Model-Agnostic):** Ingests raw multi-modal corpora, enforces structural and linguistic heuristics, eliminates exact and near-duplicate logic clones, redacts sensitive identifiers, and mitigates benchmark decontamination. Outputs clean, high-signal canonical text.
2. **The Model Training Lifecycle (Model-Dependent):** Ingests clean canonical text on-demand, injects architecture-specific tokenizer policies (such as code-indentation flags or special token framing), and packs sequences directly within GPU memory buffers for distributed training loops.

---

## 2. Infrastructure DAG Topology

The compute architecture enforces physical execution boundaries. Every transformation node is aligned with a dedicated hardware profile (I/O-bound, memory-dense, compute-dense, or GPU-accelerated) to optimize throughput, prevent cross-domain heuristic contamination, and maximize compute Return on Investment (ROI).

```text
                        [ SHARED INGESTION TRUNK ]
                                    │
                  [ NODE 1: Intake & Provenance Router ]
                       (I/O & Memory-Optimized CPU)
                                    │
             ┌──────────────────────┴──────────────────────┐
             ▼                                             ▼
[ BRANCH A: NATURAL LANGUAGE PROSE ]       [ BRANCH B: CODE & TECHNICAL DOMAINS ]
             │                                             │
 [ NODE 2A: Prose Heuristics & Dedup ]     [ NODE 2B: Code Syntax & Dedup      ]
      (Memory-Optimized CPU)                    (Compute-Dense CPU)
             │                                             │
 [ NODE 3A: Semantic Quality & LID   ]     [ NODE 3B: AST Compilation & Syntax ]
      (GPU-Accelerated Inference)               (Compute-Dense CPU)
             │                                             │
             └──────────────────────┬──────────────────────┘
                                    │
                        [ SHARED CONVERGENCE TRUNK ]
                                    │
                     [ NODE 4: Safety & Decontamination ]
                         (GPU-Accelerated Inference)
                                    │
                     ═════════════════════════════════
                       [ Storage Handoff Boundary ]
                     ═════════════════════════════════
                                    │
                     [ NODE 5: Tokenization & Packing ]
                         (GPU-Accelerated Training)
                                    │
                     [ Distributed Model Retraining ]

```

---

## 3. Node Architecture & Compute Allocation

### Phase 1: Shared Ingestion Trunk (Curation Lifecycle)

#### Node 1: Intake & Provenance Router

- **Workload Role:** Ingests uncurated records from raw landing zones, standardizes character representations, removes document boilerplate, eliminates exact duplicates, and directs records toward domain-specific branches.
- **Hardware Profile:** I/O & Memory-Optimized CPU Pool with high-speed shared memory IPC buffers and local NVMe caching.
- **Architectural Responsibilities:**
- _Unicode Standardization:_ Enforces universal Unicode normalization across heterogeneous data formats to eliminate encoding anomalies and reconstructs broken hyphenations caused by OCR/PDF extractions.
- _Boilerplate Elimination:_ Strips structural DOM artifacts, navigation trees, licensing banners, and website headers before memory allocation.
- _Zero-Serialization Exact Deduplication:_ Hashes normalized document signatures into an in-memory key-value state store. Exact duplicates are pruned immediately without incurring compute overhead downstream.
- _Metadata-Driven Early Branching:_ Inspects table headers against a strict provenance whitelist. Explicit technical data routes directly to Branch B; untagged web documents undergo zero-copy symbol density profiling and micro-pass classification to determine whether they belong in Branch A or Branch B.

---

### Phase 2: Domain-Isolated Processing Tracks (Curation Lifecycle)

#### Branch A: Natural Language Prose Track

#### Node 2A: Prose Heuristics & Fuzzy Deduplication

- **Workload Role:** Filters noisy web prose and performs near-duplicate document clustering.
- **Hardware Profile:** Memory-Optimized CPU Pool (High RAM capacity for large-scale Locality-Sensitive Hashing indices).
- **Architectural Responsibilities:**
- _Vectorized Macro-Heuristics:_ Uses low-level columnar compute kernels to filter documents violating natural language statistical properties (e.g., extreme punctuation-to-word ratios, high symbol density, low stop-word frequency, or repetitive character loops).
- _Document-Level MinHash LSH:_ Generates permutation signatures over word-level n-grams and executes banded Locality-Sensitive Hashing (LSH) to identify, cluster, and drop documents exceeding target Jaccard similarity thresholds.

#### Node 3A: Semantic Quality & Language Identification

- **Workload Role:** Evaluates deep semantic quality and verifies linguistic consistency.
- **Hardware Profile:** GPU-Accelerated High-Throughput Inference Pool.
- **Architectural Responsibilities:**
- _Classifier-Based Quality Filtering (CQF):_ Executes embedding models to score semantic depth, logical coherence, and informational density against a curated reference distribution. Low-scoring records are dropped.
- _Paragraph-Level Language Identification:_ Scans documents to detect accidental code-switching, broken machine translations, or out-of-distribution languages.

---

#### Branch B: Code & Technical Domain Track

#### Node 2B: Code Syntax & Layout Disambiguation

- **Workload Role:** Evaluates structured source code, markup, and technical documentation without applying natural language heuristic penalties.
- **Hardware Profile:** Compute-Dense CPU Pool (High core count for multi-threaded regex and syntax parsing).
- **Architectural Responsibilities:**
- _Syntax Isolation:_ Extracts fenced code blocks from mixed documentation using compiled regex state machines.
- _Layout Profiling:_ Detects minified scripts, compressed assets, and unformatted data dumps by evaluating line-length distributions.
- _Boilerplate-Stripped Line Deduplication:_ Prunes common license headers prior to computing line-level MinHash signatures, identifying copy-pasted utility blocks across distinct repositories.

#### Node 3B: AST Compilation & Grammar Validation

- **Workload Role:** Validates structural code integrity and logical uniqueness via grammar parsing.
- **Hardware Profile:** Compute-Dense CPU Pool.
- **Architectural Responsibilities:**
- _AST Parsing & Normalization:_ Compiles source files into concrete Abstract Syntax Trees (ASTs). Replaces user-defined identifiers with canonical placeholders to identify and prune logic clones that differ only by variable renaming.
- _Syntax Error Verification:_ Calculates the density of syntax error nodes within the compiled AST. Files containing critical compilation breaks are discarded.
- _Multi-Dialect Fallback:_ Routes non-compiling structured queries through fallback dialect engines to recover valid domain-specific statements.

---

### Phase 3: Shared Convergence Trunk (Curation Lifecycle)

#### Node 4: Safety & Decontamination

- **Workload Role:** Enforces universal safety policies and benchmark integrity across all re-converged data branches prior to storage persistence.
- **Hardware Profile:** GPU-Accelerated Compute Pool.
- **Architectural Responsibilities:**
- _Automated PII Redaction:_ Uses Named Entity Recognition (NER) token classifiers combined with pattern extraction matrices to mask sensitive identifiers (e.g., credentials, personal addresses, personal IDs) with deterministic category tokens.
- _Toxicity & Risk Mitigation:_ Runs sequence classification models over all candidate text to detect and drop dangerous content or policy-violating text.
- _Cross-Benchmark Decontamination:_ Indexes evaluation benchmark prompts across standardized n-gram registries. Any training segment exhibiting verbatim overlap with evaluation sets is scrubbed to prevent synthetic metric inflation.
- _Storage Handoff:_ Persists model-agnostic, curated data into the Silver Data Lakehouse partition.

---

### Phase 4: Model Ingestion Trunk (Training Lifecycle)

#### Node 5: Model-Dependent Tokenization & Sequence Packing

- **Workload Role:** Ingests curated Silver datasets on-demand, applies target model configurations, and prepares fixed-length training tensors.
- **Hardware Profile:** GPU-Accelerated Training Node Pool (utilizing high-bandwidth Host-to-Device memory and pinned VRAM buffers).
- **Architectural Responsibilities:**
- _Pre-Tokenization Audit & Policy Injection:_ Validates byte-level UTF-8 integrity and applies model-specific tokenizer flags (e.g., whitespace retention rules for code tracks vs. chat/system template formatting for conversational prose).
- _In-Memory Tokenization:_ Encodes text streams into integer token sequences using the active model's vocabulary.
- _Direct Sequence Packing:_ Concatenates variable-length token arrays and packs them into fixed context windows (e.g., 2048, 4096, 8192 tokens) delimited by model-specific sequence boundary tokens. Attention masks and position arrays are constructed directly inside GPU memory, feeding training workers without disk-serialization bottlenecks.

---

## 4. Architectural Design Principles

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        CORE DESIGN PRINCIPLES                          │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Upstream Compute Filtering                                          │
│    Cheap, zero-copy CPU operations execute first; expensive GPU        │
│    inference runs exclusively on pre-filtered, surviving records.      │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Domain Isolation                                                    │
│    Structured code and technical documentation are shielded from       │
│    prose heuristic penalties via early metadata routing.               │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Universal Compliance Enforcement                                    │
│    Safety guardrails, PII redaction, and benchmark decontamination     │
│    converge globally across 100% of all data streams.                  │
├────────────────────────────────────────────────────────────────────────┤
│ 4. Decoupled Tokenization Lifecycle                                    │
│    Tokenization and sequence packing run on-demand on training GPUs,   │
│    enabling the curated corpus to support multiple model targets.      │
└────────────────────────────────────────────────────────────────────────┘

```

---

## 5. Compute Profile & Topology Matrix

| Node Identifier                  | Conceptual Workload Domain | Hardware Architecture      | Primary Computational Engines                              | Lifecycle / Pipeline Phase |
| -------------------------------- | -------------------------- | -------------------------- | ---------------------------------------------------------- | -------------------------- |
| **Node 1** (`Intake & Router`)   | Shared Ingestion           | I/O & Memory-Optimized CPU | Vectorized C++ Kernels, Embedded KV-Store, Fast Classifier | Data Curation DAG          |
| **Node 2A** (`Prose Heuristics`) | Branch A: Natural Language | Memory-Optimized CPU       | Columnar Compute Kernels, MinHash LSH Graph Index          | Data Curation DAG          |
| **Node 3A** (`Semantic Quality`) | Branch A: Natural Language | GPU-Accelerated Inference  | Neural Embeddings, Quality Scoring, Language ID            | Data Curation DAG          |
| **Node 2B** (`Code Syntax`)      | Branch B: Code & Technical | Compute-Dense CPU          | Compiled Regex Engines, Line-Level MinHash                 | Data Curation DAG          |
| **Node 3B** (`AST Validation`)   | Branch B: Code & Technical | Compute-Dense CPU          | Concrete Syntax Tree Parsers, Dialect Transpilers          | Data Curation DAG          |
| **Node 4** (`Safety & Decon`)    | Shared Convergence         | GPU-Accelerated Inference  | NER Classifiers, Guardrail Encoders, N-Gram Registry       | Data Curation DAG          |
| **Node 5** (`Tokenize & Pack`)   | Model-Dependent Ingestion  | GPU Training Node Pool     | Parallel Byte-Pair Encoder, Memory Sequence Packer         | Model Training DAG         |
