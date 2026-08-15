# Step 7a: Natural Language CQF

## 1. Core Objective

Executing within **Track A (Natural Language Prose)** of **Phase 2 - Domain Specific Processing**, Step 7a evaluates incoming prose documents using probabilistic semantic classification.

Sequenced directly after **Step 6a (Document-Level MinHash LSH)** and prior to **Step 8a (Language Inconsistency & Code-Switching)**, Step 7a transitions the pipeline from macro-linguistic pattern matching to semantic quality grading. By measuring text against a high-quality gold-standard reference pool, Step 7a purges low-value, semantically hollow prose—such as corporate spam blogs, marketing fluff, and SEO-optimized web copy—before data enters language consistency checks, safety guardrails in **Phase 3 - Reconvergence & Tokenization**, or final model training.

---

## 2. Theoretical & Architectural Justification

Macro-linguistic heuristic filters (**Step 5a**) excel at scrubbing structural garbage, system logs, and broken OCR outputs. However, heuristics cannot distinguish between a beautifully formatted corporate spam blog and a dense, highly informative technical manual. Both documents feature clean syntax, standard punctuation distributions, and valid stop-word densities.

Allowing semantically low-quality prose to pass into downstream pre-training stages introduces three critical failure modes:

### A. Semantic Density Degradation

Language models trained on low-density prose (e.g., repetitive marketing text or filler articles) require significantly more token compute to learn complex reasoning capabilities. High-quality semantic filtering concentrates informational density per token, improving model performance per FLOP during pre-training.

### B. Probability Distribution Calibration Loss

Including low-grade web prose shifts the pre-training target distribution toward generic, low-information text generation. This degrades the model's ability to generate concise, accurate, and authoritative responses during downstream fine-tuning.

### C. Re-Compute Waste via Ephemeral Scoring

A common architectural anti-pattern is executing quality classification models and immediately dropping low-scoring records without persisting the scalar quality scores. If downstream training strategies require adjusting the quality retention bar, the entire pipeline must re-run expensive neural classification models. Step 7a resolves this by permanently appending scalar quality scores as metadata columns to the dataset manifest, enabling dynamic, offline threshold adjustments during future continuous training cycles.

---

## 3. Theoretical Execution Mechanics

Step 7a converts the abstract concept of "text quality" into a calibrated binary classification problem, evaluating records routed to **Track A** through a three-stage execution pipeline:

```text
                     [ Track A Stream Post-Step 6a ]
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │ Stage 1: Dual-Corpus Calibration & Model Training    │
         └──────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │ Stage 2: Multi-Tier Inference & Score Generation     │
         └──────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │ Stage 3: Threshold Filtering & Metadata Schema Append│
         └──────────────────────────┴───────────────────────────┘

```

### Stage 1: Reference Set Calibration & Model Training

1. **Reference Pool Construction:** A curated High-Quality (HQ) reference pool $D_{\text{HQ}}$ is established using domain-verified corpora (e.g., audited product documentation, peer-reviewed textbooks, certified knowledge bases). A matching sample of uncurated web text $D_{\text{LQ}}$ represents the background noise distribution.
2. **Probabilistic Classifier Objective:** A classification model $f_{\theta}(x)$ is trained to output a scalar quality score representing the probability $P(\text{HQ} \mid x)$ that a document $x$ shares the style, density, and vocabulary distribution of the HQ reference pool:

$$f_{\theta}(x) = P(\text{HQ} \mid x) \in [0.0, 1.0]$$



### Stage 2: Multi-Tier Execution Architecture

Depending on throughput constraints and compute allocation, Step 7a executes via two distinct architectural profiles:

* **Type A: High-Speed Lexical Profiling (FastText):** Trains an optimized linear classifier over dense token $n$-gram character spaces. Operates on CPU architectures at high MB/s throughput per core, making it ideal for high-volume initial pruning.
* **Type B: Dense Semantic Embedding Profiling (Transformer + Classifier):** Uses a high-efficiency sentence embedding transformer model (e.g., BGE, Arctic) to map text blocks into a dense latent vector space $\mathbf{z} = E(x)$, running a classification layer $g(\mathbf{z})$ on top:

$$P(\text{HQ} \mid x) = \sigma\left( \mathbf{w}^T \mathbf{z} + b \right)$$



This captures deep semantic nuance and conceptual density but requires GPU acceleration.

### Stage 3: Threshold Filtering & Metadata Schema Persistence

1. **Scalar Score Generation:** For every incoming document $x_i$, the classifier computes a scalar quality score $S_i = f_{\theta}(x_i)$.
2. **Threshold Gatekeeping:** Documents meeting or exceeding the hard retention threshold pass to Step 8a; documents below the threshold are evicted:

$$\text{Retention Condition}: S_i \ge 0.65$$


3. **Metadata Schema Extension:** Rather than discarding the computed score, the pipeline appends $S_i$ directly to the data batch schema as a 64-bit float metadata array column (`cqf_quality_score`). This metadata is persisted alongside dataset manifests, allowing engineering teams to audit quality distribution drift or dynamically modify filtering cutoffs in future continuous training iterations without re-executing inference.

---

## 4. Classifier Quality Filtering Execution Matrix

| Classification Profile | Architectural Model | Compute Target | Primary Advantage | Operational Pipeline Role |
| --- | --- | --- | --- | --- |
| **Type A: Lexical CQF** | Linear Model over Word/Character $n$-grams (FastText) | Multi-Core CPU Pods | Extremely high throughput; zero GPU cost; fast execution over massive web dumps. | Primary high-volume semantic filter for general web crawls. |
| **Type B: Dense Semantic CQF** | Sentence Transformer + Linear Classification Layer | GPU Inference Clusters | Captures conceptual nuance, stylistic tone, and dense domain authority. | Precision filtering for high-value domain datasets and fine-tuning corpora. |
| **Metadata Persistence** | Schema Extension (`cqf_quality_score`) | Persistent Storage Manifest | Enables offline threshold tuning without re-running classification models. | Immutable score logging alongside dataset partitions. |

---

## 5. Algorithmic Principles & Theoretical Tooling

* **Supervised Linear Classification Frameworks:** $n$-gram classification models optimized for rapid feature evaluation over raw character arrays.
* **Dense Latent Embedding Transformers:** Lightweight sentence-transformer models engineered to map variable-length prose into dense vector representations.
* **Logistic Regression / Classification Layers:** Calibrated output heads capable of transforming latent vectors into smooth probability distributions $P(\text{HQ} \mid x) \in [0.0, 1.0]$.
* **Schema Extension Engines:** Columnar array manipulation utilities that append persistent metadata arrays directly to binary table headers.