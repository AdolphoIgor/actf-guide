# Step 6a: MinHash Fuzzy Dedup

## 1. Core Objective

Executing within **Track A (Natural Language Prose)** of **Phase 2 - Domain Specific Processing**, Step 6a identifies and purges "near-duplicate" documents—text records that share 80%–95% structural and syntactic similarity while differing in localized details (e.g., syndicated news articles, mirrored blog posts, or boilerplate emails with altered recipient names).

Sequenced directly after **Step 5a (Macro-Linguistic Heuristic Filters)** and prior to **Step 7a (Classifier-Based Quality Filtering)**, Step 6a targets syntactic redundancy that bypassed exact cryptographic hashing in **Step 3 (Exact Deduplication)**. By compressing text into probabilistic signature vectors and clustering candidate matches, Step 6a eliminates corpus over-representation before data enters dense neural quality scoring or final sequence packing in **Phase 3 - Reconvergence & Tokenization**.

---

## 2. Theoretical & Architectural Justification

While exact cryptographic deduplication (**Step 3**) operates in $O(N)$ linear time by hashing identical character strings, identifying near-duplicate documents presents a massive computational challenge.

Allowing near-duplicate prose to pass into downstream processing creates three critical failure modes:

### A. The $O(N^2)$ Pairwise Computational Intractability

Evaluating document similarity across a dataset of $N$ documents using standard set intersection or string edit distance requires pairwise comparison across all document combinations:

$$\text{Total Pairwise Comparisons} = \frac{N(N - 1)}{2} \approx O(N^2)$$

For terabyte- or petabyte-scale corpora containing hundreds of millions of documents, $O(N^2)$ pairwise comparison is computationally impossible. MinHash and Locality-Sensitive Hashing (LSH) solve this by mapping variable-length text sets into fixed-size integer signature vectors and partitioning them into hash buckets, reducing candidate search complexity from quadratic $O(N^2)$ to near-linear $O(N)$.

### B. Over-Representation & Memory Overfitting

Near-duplicate documents—such as press releases republished across hundreds of news domains—expose the language model to identical core prose structures with trivial variations. Training on un-clustered near-duplicates distorts empirical risk minimization, forcing the model to memorize specific sentence templates and increasing the likelihood of verbatim text regurgitation during inference.

### C. Incremental Continuous Training Expansion (The Daily Delta Problem)

In Continuous Training (CT) pipelines, comparing newly ingested batch payloads ("Daily Delta") against raw historical text corpora causes severe compute inflation over time. Maintaining a pre-calculated, disk-backed index of historical signature vectors allows incoming batches to be cross-referenced against historical state in sub-linear time, purging incremental near-duplicates before updating the persistent signature manifest.

---

## 3. Theoretical Execution Mechanics

Step 6a processes records routed to **Track A** through a four-stage near-deduplication and clustering pipeline:

```text
                     [ Track A Stream Post-Step 5a ]
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │ Stage 1: K-Shingle Extraction & MinHash Vectoring    │
         └──────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │ Stage 2: LSH Banding & Candidate Bucket Collision    │
         └──────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │ Stage 3: Distributed Graph Cluster Resolution        │
         └──────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │ Stage 4: Historical Signature Delta Cross-Referencing│
         └──────────────────────────┴───────────────────────────┘

```

### Stage 1: K-Shingle Extraction & MinHash Signature Generation

1. **Shingle Extraction:** Each document text $D$ is parsed into an ordered set $S(D)$ of overlapping $k$-word shingles (typically $k = 5$ word n-grams).
2. **MinHash Signature Permutation:** The shingle set $S(D)$ is passed through $m$ independent, non-cryptographic hash permutations $h_1, h_2, \dots, h_m$ (where $m \in \{128, 240\}$). The minimum hash integer output by each function forms a fixed-length signature vector $V(D)$:

$$V(D) = \left[ \min_{s \in S(D)} h_1(s), \; \min_{s \in S(D)} h_2(s), \; \dots, \; \min_{s \in S(D)} h_m(s) \right]$$


3. **Jaccard Equivalence Property:** The probability that two documents produce an identical minimum hash value for permutation $h_i$ equals their true set Jaccard similarity:

$$P\left( \min_{s \in S(D_1)} h_i(s) = \min_{s \in S(D_2)} h_i(s) \right) = \text{Jaccard}(D_1, D_2) = \frac{\vert{}S(D_1) \cap S(D_2)\vert{}}{\vert{}S(D_1) \cup S(D_2)\vert{}}$$


$$\text{Target Deduplication Threshold}: \text{Jaccard Similarity} \ge 0.85$$



### Stage 2: LSH Banding & Candidate Bucket Collision

To locate candidate duplicate pairs without pairwise evaluation, the signature vector $V(D)$ of length $m$ is divided into $b$ bands, each containing $r$ rows ($m = b \times r$).

* For each band, the $r$ integer values are hashed together into a single band key.
* If two distinct documents share an identical band key in at least **one** of the $b$ bands, they collide in the same hash bucket and are flagged as a candidate near-duplicate pair.

The probability of two documents colliding in an LSH bucket follows a characteristic S-curve:

$$P(\text{LSH Collision}) = 1 - \left( 1 - s^r \right)^b$$

where $s = \text{Jaccard}(D_1, D_2)$. By tuning $b$ and $r$, the pipeline creates a sharp probability step at $s = 0.85$, ensuring near-duplicate pairs are detected with high probability while distant document pairs are ignored.

### Stage 3: Distributed Graph Cluster Resolution

Candidate pairs identified across LSH buckets form an undirected similarity graph $G = (V, E)$, where vertices $V$ represent documents and edges $E$ represent candidate collisions above the similarity threshold.

1. **Connected Components Extraction:** A distributed graph resolution algorithm (e.g., Union-Find / Connected Components) groups connected subgraphs into duplicate families. If Document $X \sim Y$ and $Y \sim Z$, all three are grouped into a single cluster $\{X, Y, Z\}$.
2. **Canonical Selection:** For each duplicate cluster, the pipeline evaluates document structural metrics (retaining the single longest or highest-quality document) and purges all other nodes in the component.

### Stage 4: Historical Signature Delta Cross-Referencing

To prevent incoming streaming records from duplicating text processed in previous execution runs:

1. Incoming record signature bands are cross-referenced against a persistent, disk-backed Key-Value band index or a dense vector index representing historical state.
2. If an incoming record's bands collide with historical index entries above the threshold, the record is flagged as a historical duplicate and purged locally.
3. Surviving unique signatures are committed to the historical index to maintain state continuity across continuous training cycles.

---

## 4. Near-Deduplication Strategy Matrix

| Deduplication Dimension | Algorithmic Mechanism | Target Similarity Domain | Storage & Index Architecture | Operational Action |
| --- | --- | --- | --- | --- |
| **Syntactic Near-Deduplication** | MinHash ($m=128$) + LSH Banding ($b=16, r=8$) | Lexical & structural text overlap ($\text{Jaccard} \ge 0.85$). | Embedded Key-Value Band Index (LSM-Tree mapped in virtual memory) | **Cluster & Purge:** Retain single longest canonical document per cluster. |
| **Semantic Near-Deduplication** | Dense Vector Embeddings + Approximate Nearest Neighbor (ANN) | Conceptual & semantic equivalence (different wording, same meaning). | Embedded C++ Vector Engine (In-Process HNSW / IVF-PQ index mapped to NVMe) | **Purge:** Drop semantic duplicates exceeding cosine similarity threshold. |
| **Historical State Cross-Referencing** | Incremental Band Key Lookup (Daily Delta) | Cross-run duplicate prevention against historical training state. | Persistent Signature Manifest Index | **Filter:** Evict incoming records matching historical band keys. |

---

## 5. Algorithmic Principles & Theoretical Tooling

* **MinHash & LSH Signature Engines:** High-performance vectorization utilities configured to generate permutation hash vectors and partition signatures into LSH bands.
* **Distributed Graph Resolvers:** Graph algorithms (Union-Find and Connected Components) capable of resolving large-scale adjacency lists into isolated document clusters.
* **Embedded Key-Value Signature Indices:** Low-latency, LSM-Tree-backed Key-Value engines mapped directly to virtual memory for zero-overhead band key lookups.
* **Embedded Dense Vector Search Engines:** In-process C++ vector indexing libraries utilizing Hierarchical Navigable Small World (HNSW) graphs or Inverted File Product Quantization (IVF-PQ) for direct memory-mapped semantic similarity queries.