# 01 - The Enterprise Infrastructure

## 1. Core Objective & Architectural Overview

To achieve production-grade data curation for large language models, enterprise data pipelines must map every linguistic transformation directly to specialized physical hardware pools. By aligning specific task nodes with targeted hardware profiles, this infrastructure eliminates resource bottlenecks, optimizes compute Return on Investment (ROI), and ensures strict execution boundaries across multi-modal data streams.

---

## 2. The Curation-to-Infrastructure Mapping DAG

The pipeline processes raw data through a branching Directed Acyclic Graph (DAG) that isolates natural language prose from code and technical domains, re-converging for global safety checks and sequence packing.

```text
                        [ SHARED TRUNK: INGESTION ]
                                    │
                  [ NODE 1: intake_and_provenance_router ]
                         (Memory/IO CPU Pod: 8 vCPU, 64GB)
                                    │
             ┌──────────────────────┴──────────────────────┐
             ▼                                             ▼
[ BRANCH A: NATURAL LANGUAGE PROSE ]       [ BRANCH B: CODE & TECHNICAL DOMAINS ]
             │                                             │
 [ NODE 2A: nl_heuristics_and_dedup ]       [ NODE 2B: code_syntax_and_dedup   ]
     (Memory-Optimized CPU Pod)                     (Compute-Dense CPU Pod)
             │                                             │
 [ NODE 3A: nl_quality_and_lang     ]       [ NODE 3B: code_validation_and_ast ]
     (GPU-Dense Inference Pod)                     (CPU/GPU Hybrid Pod)
             │                                             │
             └──────────────────────┬──────────────────────┘
                                    │
                        [ SHARED TRUNK: CONVERGENCE ]
                                    │
                     [ NODE 4: safety_and_decontamination ]
                           (GPU-Dense / Compute Pod)
                                    │
                     [ NODE 5: tokenization_and_packing ]
                      (Compute-Optimized CPU Pod: 16 vCPU)

```

---

## 3. Phase-by-Phase Node & Hardware Specifications

### Phase 1: Shared Ingestion Trunk

#### `NODE 1: intake_and_provenance_router`

* **Target Data:** Raw ingested records from Bronze landing zones.
* **Hardware Profile:** Memory/IO-Optimized CPU Pod (e.g., AWS `r6i.2xlarge`: 8 vCPUs, 64 GB RAM).
* **Storage Allocation:** High-speed `/dev/shm` shared memory mount ($16\text{ GB}$) + Local NVMe SSD storage.
* **Step 1: Normalization & Reassembly**
* **Engine:** PyArrow C++ String Kernels & `unicodedata`.
* **Execution:** Converts heterogeneous text strings into normalized Unicode NFKC format. Stitches broken hyphenated line breaks resulting from OCR or PDF extraction outputs (e.g., `comp-\nuting` $\rightarrow$ `computing`).
* **Memory Footprint:** Executes zero-copy in-place mutations on Apache Arrow RecordBatches inside `/dev/shm`.


* **Step 2: Boilerplate Stripping**
* **Engine:** Custom Trafilatura / C++ HTML parser wrappers.
* **Execution:** Purges recurring website navigation headers, footers, cookie banners, legal disclaimers, and tracking scripts.
* **Memory Footprint:** Processes raw text streams directly, dropping non-content DOM nodes prior to schema allocation.


* **Step 3: Exact Deduplication**
* **Engine:** Embedded RocksDB (C++ Key-Value Store) backed by 64-bit MurmurHash3 / XXHash signatures.
* **Execution:** Computes a 64-bit cryptographic hash of the normalized text and queries the in-memory RocksDB index sitting in `/dev/shm`. If the hash exists, the record is dropped instantly with zero CPU serialization tax; if unique, the hash is committed to RocksDB.
* **Failure State:** If RAM allocation spikes, RocksDB seamlessly spills excess hash tables down to the local NVMe drive to prevent Out-Of-Memory (OOM) crashes.


* **Step 4: Metadata Inspection & Provenance Routing**
* **Engine:** `pyarrow.KeyValueMetadata` API & fastText Micro-Classifier.
* **Execution:** Inspects the schema header of each Apache Arrow table:
* *Rule 1 (Metadata Match):* If `source_type` carries whitelisted tags (`github_repo`, `financial_pdf`, `latex_math`), the record batch is assigned a `branch_id = 1` and routed directly to **Branch B**.
* *Rule 2 (Un-tagged Web Scrapes):* If `source_type == "web_crawl"`, a vectorized C++ string counter (`pyarrow.compute.ascii_string_count`) checks symbol densities (`{`, `}`, `;`, `=`). If symbol density exceeds the threshold, a fastText micro-pass verifies code validity. Valid code routes to **Branch B**; general prose routes to **Branch A**.





---

### Phase 2: Branching Domain Processing

#### Branch A: Natural Language Prose Track

#### `NODE 2A: nl_heuristics_and_dedup`

* **Target Data:** Common Crawl scrapes, news articles, digital books, and general web prose.
* **Hardware Profile:** Memory-Optimized CPU Pod (High RAM allocation for MinHash LSH tables).
* **Step 5a: Macro-Linguistic Heuristic Filters**
* **Engine:** Apache Arrow Vectorized Compute Filters.
* **Execution:** Evaluates macro-linguistic prose constraints across every document:
* *Punctuation Ratio:* Drops documents with a punctuation-to-word ratio $> 0.3$ (log dumps/tables) or equal to $0.0$ (word salad).
* *Symbol-to-Word Ratio:* Drops documents where operational symbols (`#`, `$`, `%`, `@`) exceed $10\%$ of total word count.
* *Stop-Word Density:* Drops documents with stop-word density $< 5\%$ (signals SKU lists or error dumps).
* *N-Gram Repetition:* Calculates 2-gram, 3-gram, and 4-gram repetition loops to prune web-scraping artifacts.




* **Step 6a: Document-Level MinHash LSH (Fuzzy Deduplication)**
* **Engine:** MinHash LSH ($128$ permutation functions) + Connected Components Graph Clustering.
* **Execution:** Calculates $128$ integer signatures per document, slices signatures into LSH bands, and queries an on-disk memory-mapped Faiss/RocksDB index to eliminate documents exhibiting $\ge 0.85$ Jaccard similarity.



#### `NODE 3A: nl_quality_and_lang`

* **Target Data:** Surviving prose documents from Node 2A.
* **Hardware Profile:** GPU-Dense / High-Throughput Inference Pod (e.g., NVIDIA L4 / T4 instance).
* **Step 7a: Classifier-Based Quality Filtering (CQF)**
* **Engine:** ONNX Runtime / TensorRT executing a fine-tuned Quality Model (e.g., fastText / BGE Quality Classifier).
* **Execution:** Evaluates semantic prose quality against a gold-standard baseline (Wikipedia/textbooks). Generates a probability score $S \in [0.0, 1.0]$. Documents scoring $< 0.65$ are dropped.


* **Step 8a: Language Inconsistency & Code-Switching**
* **Engine:** Paragraph-level fastText Language Identification (`lid.176.bin`).
* **Execution:** Scans text at the paragraph level to detect unindexed language shifts or messy code-switching (e.g., an English document degrading into un-translated machine text). Drops non-target language blocks.



---

#### Branch B: Code & Technical Domain Track

#### `NODE 2B: code_syntax_and_dedup`

* **Target Data:** GitHub repositories, ArXiv LaTeX papers, financial balance sheets, and technical manuals.
* **Hardware Profile:** Compute-Dense CPU Pod (High CPU core count for parallel syntax parsing).
* **Step 5b: Syntax & Snippet Disambiguation**
* **Engine:** PyArrow C++ String Counters, fastText Code Engine, and Rust Regex Crates.
* **Execution:** Disables prose heuristic penalties (punctuation and symbol thresholds).
* *Code Extraction:* Isolates code blocks (`<pre><code>`, markdown fences) from mixed technical text.
* *Syntax Filtering:* Uses C++/Rust regex engines to verify programming keyword distributions (`import`, `def`, `public void`, `SELECT * FROM`). Prunes OCR garbage and malformed database dumps.




* **Step 6b: Code-Aware & Line-Level Deduplication**
* **Engine:** Line-Level MinHash LSH & Abstract Syntax Tree (AST) Fingerprinting.
* **Execution:** Strips standard boilerplate license headers (e.g., Apache 2.0, MIT) prior to hashing, then executes Line-Level LSH and AST fingerprinting to remove duplicate utility functions and copy-pasted code snippets across repositories.



#### `NODE 3B: code_validation_and_ast`

* **Target Data:** Surviving code and technical documents from Node 2B.
* **Hardware Profile:** CPU-Dense Parsing Pod.
* **Step 7b: Domain Quality & Lexer Validation**
* **Engine:** Pygments Lexer Engine & Custom Domain Parsers.
* **Execution:** Evaluates syntactic validity:
* *Lexer Tokenization:* Passes code through language-specific Pygments lexers (Python, C++, Java, SQL) to verify syntax integrity.
* *Indent & Brace Matching:* Validates structural brace matching (`{}` balance) and indentation consistency (`\t` vs. space tabs).




* **Step 8b: Multi-Language Code Filtering**
* **Engine:** Tree-Sitter AST Parsers.
* **Execution:** Parses source code into concrete Abstract Syntax Trees. If a file contains catastrophic syntax errors that prevent AST compilation, it is purged.



---

### Phase 3: Shared Convergence Trunk

#### `NODE 4: safety_and_decontamination`

* **Target Data:** Surviving datasets from Branch A and Branch B (Re-converged stream).
* **Hardware Profile:** GPU-Dense / Compute Inference Pod.
* **Step 9: Safety Guardrails & PII Redaction**
* **Engine:** In-cluster Toxicity Classifiers + Vectorized Named Entity Recognition (NER) Masking Matrices.
* **Execution:** Runs across all re-converged data:
* *PII Masking:* Replaces Social Security Numbers, private API keys, emails, IP addresses, and corporate secrets with deterministic token masks (e.g., `<PII_EMAIL>`).
* *Toxicity Filtering:* Flags and purges severe hate speech, dangerous content, or explicit non-consensual material using lightweight guardrail models.




* **Step 10: Decontamination**
* **Engine:** N-Gram Overlap Indexing against Evaluation Sets (FinQA, MedQA, HumanEval, GSM8K).
* **Execution:** Cross-references incoming data blocks against benchmark evaluation suites. If an exact 13-gram or longer match is detected between a training document and a test prompt, the overlapping segment is scrubbed to prevent benchmark contamination and metric inflation.



#### `NODE 5: tokenization_and_packing`

* **Target Data:** Fully cleaned, safe, decontaminated text stream.
* **Hardware Profile:** Compute-Optimized CPU Pod (e.g., AWS `c6i.4xlarge`: 16 vCPUs).
* **Step 11: Pre-Tokenization Audit & Schema Alignment**
* **Engine:** Apache Arrow Schema Validator.
* **Execution:** Performs pre-flight verification:
* Confirms all string encodings are clean UTF-8.
* Verifies no null records or zero-length string arrays exist.
* Maps domain-specific tokenization flags (e.g., instructing the tokenizer to preserve explicit `\t` indentation flags for Code Branch shards while stripping them for Prose shards).




* **Step 12: Tokenization & Sequence Packing**
* **Engine:** Multi-threaded Rust Tokenizer Backend (`tokenizers` crate).
* **Execution:** Encodes text strings into integer Token IDs via Byte-Pair Encoding (BPE). Concatenates variable-length token arrays and packs them into fixed-size sequence matrices (e.g., $2048$ or $4096$ tokens) separated by EOS (`End-of-Sequence`) tokens. Exports final binary shards directly to storage for GPU ingestion.



---

## 4. Key Architectural Takeaways

1. **Compute ROI Optimization:** Expensive GPU operations (Quality Classifiers, PII NER models) execute exclusively on data that has survived cheap, early CPU-based filtering gates.
2. **Zero Loss of Technical Assets:** Code, math, and financial data are explicitly shielded from natural language heuristic filters at Step 4, moving into an isolated processing branch optimized for technical structures.
3. **Global Compliance & Integrity:** All processing streams re-converge at Node 4, guaranteeing that safety guardrails, PII redaction, and benchmark decontamination are enforced globally across $100\%$ of the corpus.

---

## 5. Containerization Strategy: Images vs. Pods

In enterprise container management, an **Image** is a static blueprint (defining installed binaries, OS packages, and Python environments), whereas a **Pod/Node** is an active running process mapped to specific physical hardware (CPU, RAM, or GPU).

Creating a unique Dockerfile for every pipeline node causes massive image bloat, slow build times, and duplicate dependency installations. Instead, the architecture consolidates workload requirements into four standardized Docker images launched across hardware-isolated pods.

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

### The Consolidated 4-Image Mapping

* **Image 1: `orchestrator-image**` $\rightarrow$ Spins up the orchestration control plane (`orchestrator pod`: Airflow Scheduler / Webserver).
* **Image 2: `spark-ingest-image**` $\rightarrow$ Spins up `spark-ingest pod` (executes Step 0 - Spark Ingestion).
* **Image 3: `ray-cpu-image**` $\rightarrow$ Spins up `exact_dedup_and_heuristics` (Node 1), `fuzzy_dedup_lsh` (Node 2), and `tokenization_and_packing` (Node 5). These tasks require Python text processing, math libraries, and Rust-compiled binaries without CUDA overhead.
* **Image 4: `ray-gpu-image**` $\rightarrow$ Spins up `classifier_and_safety` (Node 3A / Node 4) to support heavy CUDA drivers and deep learning inference model weights.

---

## 6. Monorepo Build Context & Production Dockerfile

To copy shared dependencies cleanly across subdirectories in a monorepo, Docker builds must execute from the **Root of the Monorepo** as the Build Context, referencing specific Dockerfiles via the `-f` flag.

### Production Dockerfile: `./2-data-prep/Dockerfile` (`ray-cpu-image`)

```dockerfile
# Use the official Ray CPU image as our base
FROM rayproject/ray:2.40.0-py310
USER root

# Install light OS dependencies for text processing and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally for lightning-fast dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin"

USER ray
WORKDIR /home/ray/workspace

# 1. COPY ONLY dependency definitions first (Optimizes Docker layer caching)
COPY ./pyproject.toml ./uv.lock ./
COPY ./2-data-prep/pyproject.toml ./2-data-prep/

# 2. Sync dependencies using uv
RUN uv pip install --no-cache -r 2-data-prep/pyproject.toml

# 3. COPY application execution scripts into the image
COPY ./2-data-prep/node_1_heuristics.py ./2-data-prep/
COPY ./2-data-prep/node_2_fuzzy_lsh.py ./2-data-prep/
COPY ./3-model-training/node_4_tokenize_pack.py ./3-model-training/

```

### CLI Build Command (Executed from Monorepo Root)

```bash
docker build -t ray-cpu-image -f ./2-data-prep/Dockerfile .

```

*(Note: The trailing `.` specifies the monorepo root as the build context, allowing Docker to resolve references across all subfolders).*

---

## 7. Multi-Container Orchestration (`compose.yaml`)

Using `compose.yaml`, services deploy from the unified `ray-cpu-image` while overriding commands, environment variables, and physical resource limits per workload node:

```yaml
services:
  # Node 1: Memory/IO CPU Pod
  exact-dedup-heuristics:
    image: ray-cpu-image
    container_name: actf-node-1
    command: python /home/ray/workspace/2-data-prep/node_1_heuristics.py
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G

  # Node 2: Memory-Bound Distributed Cluster Node
  fuzzy-dedup-lsh:
    image: ray-cpu-image
    container_name: actf-node-2
    command: python /home/ray/workspace/2-data-prep/node_2_fuzzy_lsh.py
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 32G  # Allocates expanded RAM for LSH shingle matrices

```

---

## 8. Infrastructure Mapping Matrix

| Node Identifier | Target Workload Domain | Hardware Profile & Allocation | Primary Software Engines | Assigned Pipeline Steps |
| --- | --- | --- | --- | --- |
| **Node 1** (`intake_and_provenance_router`) | Shared Ingestion | Memory/IO-Optimized CPU (`r6i.2xlarge`: 8 vCPU, 64GB RAM, 16GB `/dev/shm`) | PyArrow C++, Trafilatura, RocksDB, fastText | Steps 1, 2, 3, 4 |
| **Node 2A** (`nl_heuristics_and_dedup`) | Branch A: Prose | Memory-Optimized CPU (Expanded RAM for LSH tables) | PyArrow Compute, MinHash LSH, Faiss / RocksDB | Steps 5a, 6a |
| **Node 3A** (`nl_quality_and_lang`) | Branch A: Prose | GPU-Dense / High-Throughput Inference (NVIDIA L4 / T4) | ONNX Runtime, TensorRT, fastText (`lid.176.bin`) | Steps 7a, 8a |
| **Node 2B** (`code_syntax_and_dedup`) | Branch B: Technical | Compute-Dense CPU (High vCPU core count) | PyArrow C++, Rust Regex, Line-Level MinHash LSH | Steps 5b, 6b |
| **Node 3B** (`code_validation_and_ast`) | Branch B: Technical | CPU-Dense Parsing Pod | Pygments Lexer, Tree-Sitter AST Compiler | Steps 7b, 8b |
| **Node 4** (`safety_and_decontamination`) | Shared Convergence | GPU-Dense / Compute Inference Pod | Toxicity Guardrails, NER Masking, N-Gram Indexer | Steps 9, 10 |
| **Node 5** (`tokenization_and_packing`) | Shared Convergence | Compute-Optimized CPU (`c6i.4xlarge`: 16 vCPUs) | PyArrow Schema Engine, Rust `tokenizers` BPE | Steps 11, 12 |