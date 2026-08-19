# Step 6b: Code-Specific MinHash / AST Dedup

## 1. Core Objective

Executing within **Track B (Code & Technical Domains)** of **Phase 2 - Domain Specific Processing**, Step 6b identifies and purges duplicate source code files, repository forks, copy-pasted functions, boilerplate header duplication, and syntactically identical code (semantic clones) across technical datasets.

Sequenced directly after **Step 5b (Code & Syntax Disambiguation)** and prior to **Step 7b (Domain Quality & Lexer Validation)**, Step 6b addresses the unique structural redundancy of software code bases. By decoupling code logic from surface-level formatting and naming conventions, Step 6b combines **Line-Level MinHash LSH** (to eliminate textual near-duplicates) with **Abstract Syntax Tree (AST) Canonicalization** (to eliminate functional logic clones). This prevents token over-representation and model memorization before data enters AST syntax validation (**Step 7b**) or final sequence packing in **Phase 3 - Reconvergence & Tokenization**.

---

## 2. Theoretical & Architectural Justification

Standard document-level deduplication algorithms—such as character or word $N$-gram MinHash (**Step 6a**)—fail catastrophically when applied directly to software code bases due to two fundamental failure modes:

### A. The License Header False Positive

Millions of distinct open-source files share identical 50-line license headers (e.g., Apache 2.0, MIT, BSD) or standard import headers (`import os`, `import sys`). Standard word- or character-level MinHash algorithms evaluate these shared headers as part of the document signature, causing completely unrelated files to exceed the Jaccard similarity threshold ($\text{Jaccard} \ge 0.85$). As a result, unique functional source code is falsely flagged as near-duplicate and accidentally discarded.

### B. The Variable Rename False Negative

Developers routinely copy-paste utility functions and algorithms across repositories, modifying only variable names (e.g., `def calculate_sum(a, b)` versus `def calculate_total(x, y)`), inline comments, or indentation styles. Standard textual and token-level MinHash algorithms evaluate these modified functions as completely distinct files, leaving massive volumes of redundant training signals in the dataset.

Step 6b resolves this by isolating underlying logic from surface representation. It strips boilerplate headers prior to signature generation and combines line-level tokenization with AST-based structural canonicalization to evaluate true code equivalence.

---

## 3. Theoretical Execution Mechanics

Step 6b evaluates records routed to **Track B** through a three-stage deduplication pipeline:

```text
                     [ Track B Stream Post-Step 5b ]
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │ Stage 1: Preprocessing & Boilerplate/Comment Stripping│
         └──────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │ Stage 2: Line-Level MinHash LSH (Textual Duplicates) │
         └──────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │ Stage 3: AST Canonicalization & Structural Hashing   │
         │          (Semantic Clones)                           │
         └──────────────────────────┴───────────────────────────┘

```

### Stage 1: Preprocessing & Boilerplate/Comment Stripping

Before computing hash signatures, input code files undergo structural cleanup to isolate executable logic:

1. **License Header Stripping:** Leading lines are matched against a compiled pattern database of standard open-source licenses (Apache, MIT, BSD, AGPL) and stripped.
2. **Docstring & Comment Removal:** Inline comments (`//`, `#`) and multi-line documentation blocks (`"""..."""`, `/*...*/`) are removed.
3. **Whitespace Normalization:** Trailing spaces are stripped and empty newline sequences are collapsed to prevent trivial formatting variations from altering signature vectors.

### Stage 2: Line-Level MinHash LSH (Textual Near-Duplicates)

Standard text deduplication splits documents into character or word $N$-grams. Code deduplication treats each normalized, non-empty line as an atomic token in a line set $S$:

1. **MinHash Signature Vector Generation:** $m = 128$ independent permutation hash functions generate a signature vector over the line set $S$:

$$h_{\text{min}}(S) = \min_{s \in S} h(s)$$


2. **LSH Banding & Jaccard Evaluation:** Signature vectors are divided into $b = 16$ bands of $r = 8$ rows. Band keys are queried against a local signature index to identify candidate pairs matching across at least one band. Candidate pairs are evaluated for line-level Jaccard similarity:

$$J(S_1, S_2) = \frac{\vert{}S_1 \cap S_2\vert{}}{\vert{}S_1 \cup S_2\vert{}} \ge 0.80$$


3. **Graph Cluster Resolution:** Candidate pairs exceeding the $0.80$ similarity threshold are connected in an adjacency graph. A Connected Components graph resolution algorithm groups duplicate families, retaining the canonical primary repository file and purging duplicates.

### Stage 3: AST Canonicalization & Structural Hashing (Semantic Clones)

Surviving textual files are evaluated for structural logic clones that differ only in naming or formatting:

1. **Abstract Syntax Tree (AST) Generation:** Source code files pass through multi-language AST parsers (supporting languages such as Python, C++, Java, Rust, Go, JavaScript, and SQL) to generate concrete syntax trees.
2. **Node Canonicalization:** All user-defined identifiers (variable names, function names, parameter identifiers) are mapped to generic, canonical placeholders (`VAR_1`, `VAR_2`, `FUNC_1`), while strictly preserving language control-flow nodes (`if`, `for`, `while`, `return`).
3. **Canonical AST Hashing:** A 64-bit non-cryptographic hash signature (MurmurHash3 or XXHash) is computed over the canonicalized AST representation:

$$\text{AST Hash} = H\left( \text{AST}_{\text{canonicalized}} \right)$$



Files or subtrees yielding identical canonical AST hashes represent functional logic clones and are purged, regardless of original variable renaming or comment variations.

---

## 4. Code-Specific Deduplication Strategy Matrix

| Duplication Category | Structural Signature / Pattern | Theoretical Engine | Pipeline Action | Downstream Impact in Phase 2 |
| --- | --- | --- | --- | --- |
| **Header / License Boilerplate** | Identical open-source license text across distinct functional logic. | Pattern Matcher & Prefix Masking Engine | **Header Stripped:** License removed prior to hashing; underlying code body evaluated independently. | Eliminates false-positive deduplication of unrelated code files. |
| **Exact & Near-Duplicate Files** | Files sharing $\ge 0.80$ line-level set similarity (repository forks, modified scripts). | Line-Level MinHash LSH + Band Index | **Pruned:** Graph clustering identifies duplicate cluster; retains single canonical source. | Purges duplicate files and repository forks across Track B. |
| **Refactored / Renamed Functions** | Identical control-flow logic with altered variable names, comments, or whitespace. | AST Canonicalizer + Structural Hash Kernel | **Pruned:** Matching canonical AST hash identifies structural clone; duplicate subtree removed. | Eliminates semantic code clones that bypass textual deduplication. |
| **Auto-Generated Code** | Repeated auto-generated boilerplate (e.g., `// Code generated by protoc-gen-go`). | AST Depth Profiler + Pattern Filter | **Pruned:** Excessively repetitive AST structures flagged and purged via Boolean mask array. | Prevents model over-fitting on auto-generated interface bindings. |

---

## 5. Algorithmic Principles & Theoretical Tooling

* **Multi-Language AST Parsing Engines:** Concrete syntax tree generators capable of parsing source code into structured Abstract Syntax Trees across multiple programming languages.
* **Line-Level MinHash Vectorizers:** Signature generation utilities optimized for hashing discrete lines as atomic tokens over contiguous memory arrays.
* **Embedded Key-Value Signature Indices:** Low-latency Key-Value storage architectures utilized for high-throughput LSH band key lookup and matching.
* **AST Structural Canonicalizers:** Tree-traversal algorithms that rename user-defined variables and functions to generic placeholders while preserving control-flow hierarchy.
* **Non-Cryptographic Hash Kernels:** High-speed 64-bit hashing algorithms (MurmurHash3 / XXHash) engineered for microsecond signature generation over canonicalized AST representations.