# Closed-Loop Data Contract & Training Guardrails

## Why Enforcing These Gates Is Indispensable

In traditional software engineering, compilation and unit tests prove that an application is syntactically correct and structurally stable. In Machine Learning Operations (MLOps)—specifically within automated continuous training (CT) and domain adaptation—traditional unit tests are completely blind. An automated training script can execute flawlessly from line 1 to line 1000 without throwing a single code exception, yet still output a completely broken, hallucinating, or financially destructive model.

This introduces the concept of **Silent Failures**, the most hazardous risk in AI infrastructure. Implementing this multi-gated blueprint is indispensable for three core operational reasons:

1. **Cost Containment and "Fail-Fast" Infrastructure:** Training and fine-tuning modern Transformer/LLM architectures require spinning up multi-node GPU clusters that cost massive amounts of money per hour. Without these gates, a pipeline might pull data with mutated text encodings (failing Gate 2) or pass unexpected token index dimensions to the model's embedding matrix (failing Gate 3). If this happens, the infrastructure will compute garbage for hours or experience a catastrophic segmentation fault deep into the training run. By validating boundaries linearly, you enforce a fail-fast philosophy: the pipeline trips its circuit breakers at the lightweight CPU level before a single GPU is ever provisioned, saving tens of thousands of dollars in wasted compute.
2. **Eliminating the "Garbage In, Garbage Out" Paradox:** Continuous training assumes that the incoming data stream is healthy and representative of the target domain. However, data sources are non-deterministic; upstream data engineers might change a schema, an export database might drop text fields, or real-world consumer behavior might suddenly drift overnight. If your pipeline blindly accepts this data, the model will experience catastrophic forgetting or severe degradation. These gates act as mathematical filters, guaranteeing that only high-fidelity, structurally aligned, and statistically viable datasets are permitted to shape the model's internal weights.
3. **Achieving True, Hands-Off Automation:** A pipeline without gates isn't automated; it is dangerous. True automation requires the system to act as its own Autonomous Gatekeeper. By coupling the In-DAG Training Loop (which guarantees mathematical and structural correctness) with the Production Feedback Loop (which uses telemetry to monitor live real-world drift), you close the MLOps circle. The architecture becomes self-healing—automatically detecting when the model is decaying, initiating an airtight, risk-free retraining cycle, and deploying the upgrade safely without human intervention.

This framework transforms your Apache Airflow environment from a basic task scheduler into a rigorous, mathematical security perimeter for your enterprise AI assets.

---

## The Gold-Standard Blueprint Architecture

The architecture operates as a DAG-based (Directed Acyclic Graph) conditional sequence within Apache Airflow, divided into two distinct operating zones:

- **Part 1: The Core Airflow Training Loop (In-DAG Gates 1 to 5):** These gates live directly inside your continuous training DAG. They execute sequentially; if one fails, the pipeline immediately halts to protect downstream compute and budget.
- **Part 2: The Production Feedback Loop (Inference Boundaries - Gates 6 & 7):** These gates run outside your training DAG within the live inference system, providing the real-time telemetry that alerts the Airflow pipeline when to execute.

---

## Detailed Execution Mechanics & Step-by-Step Flow

### Phase 0: Production Telemetry & Trigger Loop (Part 2: Production Feedback Loop)

- **Gate 7: Continuous Drift Monitor (The Trigger)**
- **Location:** An asynchronous, background monitor checking production logs daily or weekly outside the main training DAG.
- **What it Checks:** Statistical Data Drift. Continuously monitors live production feature streams via statistical distance algorithms (e.g., Population Stability Index [PSI], Kolmogorov-Smirnov test, or Wasserstein Distance) to see if real-world data has shifted away from the training baseline.
- **Relevance & Action:** Mandatory Trigger. This gate is the automated starter engine for the pipeline. When this monitor detects heavy drift exceeding baseline bounds, it fires an API call that wakes up the main Continuous Training (CT) Airflow DAG to execute Gates 1 through 5.

- **Gate 6: Inference Serving Gate (The Microservice Gatekeeper)**
- **Location:** Runs asynchronously inside the live API microservice, immediately before incoming user feature payloads hit the `model.predict()` endpoint.
- **What it Checks:** Training-Serving Skew. Ensures live request feature schemas and value normalization ranges match the baseline characteristics the model was trained on.
- **Relevance & Action:** Contextual. It stops production systems from passing malformed requests that would cause the model to output hallucinations or junk data.

---

### Phase 1: Ingestion, Branching DAG & Heavy Data Pruning

#### Shared Ingestion Trunk

- **Raw Data Intake:** Raw data is extracted from staging storage/warehouses into the processing workspace.
- **Gate 1: The Ingestion Gate (Bronze Check)**
- **Location:** Immediately after raw data is extracted from external storage/warehouses into the pipeline workspace.
- **What it Checks:** Data Integrity & Volumetrics. Ensures file exports succeeded, files are uncorrupted, schemas are intact, and row counts/file sizes fall within expected baseline bounds.
- **Relevance & Action:** Mandatory. Enforces data immutability—once validated, this raw partition is signed off forever.

- **Step 1: Normalization & Unicode Reassembly:** Fixes broken Unicode, maps characters to standard NFKC format, and stitches hyphenated line breaks from OCR/PDF extraction.
- **Step 2: Boilerplate Stripping:** Purges recurring website navigation headers, footers, disclaimers, and cookie policies.
- **Step 3: Exact Deduplication:** Generates 64-bit cryptographic signatures (MurmurHash3) to drop exact character duplicates via an embedded RocksDB in-memory index.
- **Step 4: Metadata Inspector & Router:** Inspects Apache Arrow schema headers (`pyarrow.KeyValueMetadata`). Documents with whitelisted technical provenance (`source_type: github_repo`, `doc_type: financial_pdf`) are assigned to Branch B. Un-tagged web pages undergo vectorized C++ symbol density checks (`pyarrow.compute.ascii_string_count`) and fastText code detection to route prose to Branch A and valid code snippets to Branch B.

#### Branch A: Natural Language Prose Track

- **Step 5a: Macro-Linguistic Heuristic Filters:** Evaluates macro-linguistic prose constraints across every document:
- **Punctuation Ratio:** Drops text with punctuation-to-word ratios $> 0.3$ or equal to $0.0$.
- **Symbol-to-Word Threshold:** Drops text where operational symbols (`#`, `$`, `%`, `@`) exceed $10\%$ of total word count.
- **Stop-Word Density:** Drops text with stop-word densities $< 5\%$ (signals SKU lists or automated error dumps).
- **N-Gram Repetition Filter:** Prunes repeating 2-gram, 3-gram, and 4-gram phrase loops from bad web scraping.

- **Step 6a: Document-Level MinHash LSH (Fuzzy Deduplication):** Computes 128 MinHash signatures per document and executes Connected Components graph clustering to eliminate documents with $\ge 0.85$ Jaccard similarity.
- **Step 7a: Classifier-Based Quality Filtering (CQF):** Scores prose semantic quality against a gold-standard baseline using linear embeddings/quality classifiers. Drops text scoring below $0.65$.
- **Step 8a: Language Inconsistency & Code-Switching:** Paragraph-level fastText processing (`lid.176.bin`) to drop messy code-switching or unindexed language shifts.

#### Branch B: Code & Technical Domain Track

- **Step 5b: Syntax & Snippet Disambiguation:** Bypasses prose heuristic penalties. Uses PyArrow C++ character counters and fastText code models to validate code structures, isolate embedded code fences (`<pre><code>`), and prune minified assets, compiler stack traces, or OCR garbage.
- **Step 6b: Code-Aware & Line-Level Deduplication:** Strips boilerplate license headers and executes Line-Level MinHash LSH and Abstract Syntax Tree (AST) fingerprinting to eliminate duplicate utility functions across repositories.
- **Step 7b: Domain Quality & Lexer Validation:** Passes code through language-specific Pygments lexers (Python, C++, Java, SQL) to verify syntax integrity, structural brace matching (`{}` balance), and indentation consistency.
- **Step 8b: AST Syntax Validation:** Uses Tree-Sitter AST parsers to verify full syntactic validity. Drops files with catastrophic syntax errors that prevent AST compilation.

#### Shared Convergence Trunk

- **Step 9: Safety Guardrails & PII Redaction:** Re-converges surviving data from Branch A and Branch B. Executes in-cluster toxicity scoring and NER-matrix masking (redacting phone numbers, emails, IP addresses, and private API keys).
- **Step 10: Cross-Dataset Decontamination:** Cross-references data blocks against downstream evaluation sets (FinQA, MedQA, HumanEval, GSM8K) to eliminate exact 13-gram overlapping and prevent metric inflation.

---

### Phase 2: Serialization, Tokenization & Partitioning

- **Gate 2: The Pre-Tokenization Gate (Silver Check)**
- **Location:** Post-cleaning/filtering, right before data hits the tokenization and sequence-packing cluster.
- **What it Checks:** Schema & Character Constraints. Verifies text fields are non-null, string encodings (like UTF-8) are clean and flawless, and text lengths align with target context boundaries.
- **Relevance & Action:** Mandatory. Prevents string encoding errors or malformed text from crashing downstream distributed preprocessing workers.

- **Step 11: Pre-Tokenization Audit & Schema Alignment:** Maps domain-specific tokenization configurations (e.g., instructing the tokenizer to preserve explicit `\t` indentation flags for Code Branch shards while stripping them for Prose shards).
- **Step 12: Tokenization & Sequence Packing:** Converts clean text into integer Token IDs via a multi-threaded Rust tokenization backend. Concatenates variable-length arrays and packs them into fixed-size sequence matrices (e.g., 2048 or 4096 tokens) wrapped with proper attention masks.
- **Data Splitting:** Splits the packed sequence matrix into Train, Validation, and Test sets.
- **Gate 3: The Data Leakage & Split Gate**
- **Location:** Immediately after generating Train/Validation/Test data partitions.
- **What it Checks:**
- **Partition Isolation:** Uses hash/ID matching to ensure absolute zero overlap between training and evaluation datasets.
- **Partition Split Ratio:** Enforces strict split ratios of the dataset shards (e.g., ensuring a target split of exactly 95% training and 5% validation) and verifies that the validation set contains sufficient token volume to yield a statistically significant perplexity calculation.

- **Relevance & Action:** Mandatory. Guarantees that evaluation metrics (like loss and perplexity) are scientifically honest before the model progresses.

- **Gate 4: The Pre-Flight Tensor Gate (ML Engineering Check)**
- **Location:** Across the tokenized Parquet/binary array shards, right before provisioning GPU compute.
- **What it Checks:** Mathematical Constraints.
- Verifies matrix shapes are strictly $B \times L$ (Batch Size $\times$ Sequence Length).
- Runs vectorized audits ensuring all token IDs fall within the vocabulary range ($0 \le \text{Token ID} < V$).
- Confirms the `attention_mask` consists strictly of binary bits ($\{0, 1\}$).

- **Relevance & Action:** Mandatory. Passing an out-of-bounds token ID to an embedding layer throws an asynchronous CUDA segmentation fault, stalling expensive clusters. This gate guarantees training run success.

---

### Phase 3: Model Compute & Deployment Gatekeeping

- **Compute Engine Execution:** Deep learning distributed training execution block (e.g., Axolotl / PyTorch FSDP).
- **Gate 5: The Automated Gatekeeper**
- **Location:** The final conditional block of the training pipeline, post-evaluation.
- **What it Checks:** Performance Deltas. Programmatically compares the fine-tuned model’s metrics on a Domain Gold Standard benchmark against the active production model.
- **Relevance & Action:** Mandatory. Functions as the ultimate deployment circuit breaker. If the new model beats the production baseline, it passes to the Registry as a deployment candidate; if it degrades, it alerts the team and aborts.

---

## The Global Circuit Breaker Specification

If any gate (1 through 5) throws an assertion failure or meets a breach condition, an automated circuit breaker hook intercepts the pipeline process. It immediately halts the compute job, prevents active cluster autoscalers from keeping costly GPU nodes online, dispatches an emergency alert to the engineering team with complete log tracebacks, and teardowns all ephemeral infrastructure resources instantly.

---

## Summary Matrix: The 7 Engineering & Safety Gates

| Gate Identifier                                  | Operating Loop Zone         | Execution Phase / Location       | Primary Inspection Target                                      | Circuit Breaker & Pipeline Action                                    |
| ------------------------------------------------ | --------------------------- | -------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Gate 7: Continuous Drift Monitor**             | Part 2: Production Feedback | Asynchronous background monitor  | Statistical data drift (PSI, Wasserstein distance)             | **Trigger:** Fires API call to wake up main Airflow CT DAG.          |
| **Gate 6: Inference Serving Gate**               | Part 2: Production Feedback | Live API Microservice            | Training-Serving Skew & schema validation                      | **Block:** Rejects malformed requests at microservice boundary.      |
| **Gate 1: Ingestion Gate (Bronze Check)**        | Part 1: In-DAG Training     | Post-Raw Extraction              | File integrity, non-corruption, baseline row counts            | **Halt:** Aborts DAG; enforces data immutability.                    |
| **Gate 2: Pre-Tokenization Gate (Silver Check)** | Part 1: In-DAG Training     | Post-Cleaning / Pre-Tokenization | Null fields, clean UTF-8 encodings, context length bounds      | **Halt:** Aborts DAG before tokenization cluster allocation.         |
| **Gate 3: Data Leakage & Split Gate**            | Part 1: In-DAG Training     | Post-Dataset Partitioning        | Partition isolation (zero overlap) & 95/5 split ratios         | **Halt:** Aborts DAG to prevent contaminated evaluation.             |
| **Gate 4: Pre-Flight Tensor Gate**               | Part 1: In-DAG Training     | Post-Sequence Packing / Pre-GPU  | $B \times L$ shapes, $0 \le \text{Token ID} < V$, binary masks | **Halt:** Aborts DAG before provisioning expensive GPUs.             |
| **Gate 5: The Automated Gatekeeper**             | Part 1: In-DAG Training     | Post-Training Evaluation         | Metric deltas against active production baseline               | **Promote/Abort:** Registers winning candidates; blocks regressions. |
