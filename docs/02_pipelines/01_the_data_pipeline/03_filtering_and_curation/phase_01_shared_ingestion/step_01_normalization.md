# Step 1: Normalization & Unicode Reassembly

## 1. Core Objective

Situated at the entry point of **Phase 1 - Shared Ingestion**, Step 1 transforms heterogeneous, structurally fractured raw text streams—extracted from multi-format enterprise sources such as scanned PDFs, raw web crawls, and legacy database dumps—into canonical, standardized UTF-8 text representations.

By enforcing Normalization Form Compatibility Composition (NFKC), scrubbing non-printable byte sequences, and stitching layout-hyphenated words prior to Exact Deduplication (Step 3) and Metadata Routing (Step 4), Step 1 establishes essential textual invariants across the entire dataset. This prevents vocabulary fragmentation, eliminates phantom token allocation, and protects context alignment before data branches into **Phase 2 - Domain Specific Processing** or undergoes final sequence packing in **Phase 3 - Reconvergence & Tokenization**.

---

## 2. Theoretical & Architectural Justification

Data ingested from un-sanitized enterprise repositories arrives with severe encoding noise, byte corruption, and layout artifacts. Passing raw text directly to downstream tokenization stages in **Phase 3** introduces three fundamental failure modes in large language model pre-training:

### A. Vocabulary Fragmentation & Canonical Equivalence

The same visual character can be represented by multiple distinct Unicode code point sequences:

* **Canonical Composition (NFC):** A single pre-composed code point representing both character and diacritic (e.g., $\text{U+00E9}$ for `é`).
* **Canonical Decomposition (NFD):** Two separate code points combining a base character and a combining mark (e.g., $\text{U+0065} + \text{U+0301}$ for `e` + `◌́`).

If a corpus contains a mixture of NFC and NFD encodings, Byte-Pair Encoding (BPE) and WordPiece tokenizers fail to recognize their semantic identity, assigning completely different integer Token IDs to identical words.

NFKC normalization resolves this by converting decomposition variants and compatibility characters—such as typographic ligatures ($f_{\text{NFKC}}(\text{U+FB01}) \rightarrow \text{f} + \text{i}$) and stylistic superscripts ($f_{\text{NFKC}}(\text{U+2072}) \rightarrow 2$)—back to unified, standard alphanumeric forms:

$$f_{\text{NFKC}}\left( \text{U+0065} + \text{U+0301} \right) = \text{U+00E9}$$

### B. Sequence Window Capacity Waste via Phantom Tokens

Scanned documents, OCR outputs, and raw HTML DOM trees routinely contain non-printable control bytes, soft hyphens, and zero-width spaces:

* **Control Bytes:** Non-printable ASCII signals ($\text{U+0000}$ through $\text{U+001F}$, excluding valid structural whitespace $\text{U+000A}$ `\n` and $\text{U+0009}$ `\t`).
* **Phantom Characters:** Zero-width spaces ($\text{U+200B}$), zero-width non-joiners ($\text{U+200C}$), and soft hyphens ($\text{U+00AD}$) inserted for visual layout wrapping.
* **Malformed Byte Arrays:** Corrupted byte sequences rendered as replacement characters ($\text{U+FFFD}$).

Sub-word tokenizers cannot map these non-semantic characters to real linguistic concepts. Instead, they either isolate them into byte-fallback tokens or merge them into arbitrary "phantom tokens." This consumes fixed sequence window capacity, skews attention matrix distances, and induces hallucinated token generations during model inference.

### C. High-Throughput Memory Allocation Dynamics

Executing string manipulations via native object-pointer loops introduces a severe computational bottleneck. Unpacking columnar memory tables into individual object pointers, applying string transformations sequentially, and re-packing them back into structured arrays incurs a massive memory-allocation tax:

$$\text{Latency Overhead Ratio} = \frac{\text{Latency}_{\text{Object Pointer Unpacking}}}{\text{Latency}_{\text{Contiguous Vector Kernel}}} \approx 3.33 \times \text{ to } 5.00 \times$$

To maintain maximum throughput across terabyte-scale datasets, Phase 1 mandates that character normalization and sanitization be performed using vectorized, contiguous memory string kernels that execute directly on binary memory arrays without object instantiation loops.

---

## 3. Theoretical Execution Mechanics

In **Phase 1 - Shared Ingestion**, Step 1 processes incoming text batches through a three-stage sequential pipeline:

```text
                       [ Raw Ingestion Data Stream ]
                                     │
                                     ▼
         ┌───────────────────────────────────────────────────────┐
         │ Stage 1: Encoding Conversion & NFKC Normalization     │
         └───────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
         ┌───────────────────────────────────────────────────────┐
         │ Stage 2: Control Byte & Phantom Character Sanitization│
         └───────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
         ┌───────────────────────────────────────────────────────┐
         │ Stage 3: Dictionary-Assisted Layout Hyphen Reassembly │
         └───────────────────────────┴───────────────────────────┘

```

### Stage 1: Encoding Alignment & Canonical NFKC Normalization

1. **Encoding Conversion:** Heterogeneous input character sets (such as ISO-8859-1 or Windows-1252) are auto-detected and re-encoded into clean UTF-8 byte sequences.
2. **NFKC Composition Pass:** Vectorized string kernels convert decomposed characters, diacritics, and compatibility symbols into unified canonical forms across the entire memory array.

### Stage 2: Control Byte & Phantom Character Sanitization

The normalized byte arrays undergo vectorized pattern matching and character replacement:

1. **Control Character Clearing:** Non-printable ASCII control characters ($\text{U+0000}$–$\text{U+001F}$ excluding `\n` and `\t`) and unassigned Unicode categories ($\p{C}$) are purged.
2. **Phantom Character Removal:** Soft hyphens ($\text{U+00AD}$) and zero-width spaces ($\text{U+200B}$) are stripped completely, while malformed byte sequences trigger automated character-repair operations.
3. **Whitespace Trimming:** Outer string margins and trailing blank tokens are standardized.

### Stage 3: Dictionary-Assisted Hyphen Reassembly & Text Flow Restoration

PDF generation and OCR engines split words across page boundaries by inserting hyphens and newline characters (e.g., `en-\nterprise`). Naively deleting all occurrences of `-\n` corrupts valid compound words (e.g., `high-\nquality` becomes `highquality` instead of `high-quality`).

To restore original text flow without semantic loss, Step 1 applies a conditional dictionary-lookup decision tree backed by Finite State Transducers (FST) or in-memory hash maps:

$$\text{Target Pattern}: \text{Prefix} + \text{"-"} + \text{"\n"} + \text{Suffix}$$

```text
                        Evaluate Pattern: Prefix + "-" + "\n" + Suffix
                                               │
                                               ▼
                         ┌──────────────────────────────────────────┐
                         │ Is (Prefix + Suffix) a valid word        │──► YES ──► Output: "enterprise"
                         │ in the baseline dictionary?              │            (Strip hyphen & newline)
                         └────────────────────┬─────────────────────┘
                                              │ NO
                                              ▼
                         ┌──────────────────────────────────────────┐
                         │ Is (Prefix + "-" + Suffix) a valid       │──► YES ──► Output: "high-quality"
                         │ compound word in the dictionary?         │            (Strip newline, retain hyphen)
                         └────────────────────┬─────────────────────┘
                                              │ NO
                                              ▼
                         Fallback: Retain original layout split
                         (Prevents accidental word corruption)

```

---

## 4. Transformation & Sanitization Matrix

| Input Defect / Layout Artifact | Unicode / Byte Representation | Transformation Rule | Output Format | Downstream LLM Impact |
| --- | --- | --- | --- | --- |
| **Decomposed Diacritics (NFD)** | $\text{U+0065} + \text{U+0301}$ (`e` + `◌́`) | NFKC Normalization | $\text{U+00E9}$ (`é`) | Eliminates duplicate Token IDs for identical semantic words in **Phase 3**. |
| **Typographic Ligatures** | $\text{U+FB01}$ (`ﬁ`) | NFKC Decomposition | `f` + `i` | Standardizes vocabulary tokens across diverse publishing sources. |
| **Soft Hyphens / Zero-Width** | $\text{U+00AD}$, $\text{U+200B}$ | String Stripping | *Removed* | Prevents allocation of empty phantom tokens in BPE matrices. |
| **Control Bytes / Null Signals** | $\text{U+0000}$–$\text{U+001F}$ (excl. `\n`, `\t`) | Pattern Clearing | *Removed* | Prevents sequence context alignment drift during pre-training. |
| **Broken Line Layout Hyphen** | `en-\nterprise` | Dictionary FST Lookup | `enterprise` | Restores original word tokens for accurate semantic loss calculation. |
| **Valid Compound Line Break** | `high-\nquality` | Dictionary FST Lookup | `high-quality` | Preserves compound word semantics without accidental token merging. |

---

## 5. Algorithmic Principles & Theoretical Tooling

* **Vectorized Array Kernels:** Low-level C++ compute utilities designed for zero-copy string manipulation on contiguous memory arrays.
* **Encoding & Character Repair Engines:** Libraries implementing standard Unicode character category tables ($\p{C}$) and automated encoding detection heuristics.
* **Lexical Reassembly Engines:** Finite State Transducers (FST) and SymSpell memory-mapped hash structures optimized for microsecond dictionary lookups.