# Step 7b: Domain Quality Check

## 1. Core Objective

Executing within **Track B (Code & Technical Domains)** of **Phase 2 - Domain Specific Processing**, Step 7b evaluates the domain-specific quality, structural health, and maintainability of source code, mathematical documents, and technical papers.

Sequenced directly after **Step 6b (Code-Specific MinHash / AST Dedup)** and prior to **Step 8b (AST Syntax Validation & Multi-Language Filter)**, Step 7b replaces natural prose quality scoring with Domain-Aware Quality Scoring. By evaluating lexical tokenization validity, maintainability metrics, and specialized neural quality models trained on curated technical datasets, Step 7b purges non-informative code stubs, machine-generated spaghetti code, and obfuscated technical noise before data enters final AST syntax validation (**Step 8b**) or sequence packing in **Phase 3 - Reconvergence & Tokenization**.

---

## 2. Theoretical & Architectural Justification

Applying standard Natural Language Classifier-Based Quality Filtering (**Step 7a**) to source code, financial balance sheets, or mathematical documents causes catastrophic data loss. Prose quality models evaluate text against standard human prose distributions (such as Wikipedia or high-grade journalism). Because technical syntax features non-prose keywords, low natural-language perplexity, structural indentation, and symbol-dense lines, standard prose CQF models evaluate valid technical documents as low-quality prose, scoring them near $0.0$ and incorrectly purging them.

Conversely, completely omitting quality filtering for Track B introduces severe dataset contamination:

### A. Non-Informative Stubs & Empty Test Fixtures

Software repositories are saturated with empty function signatures (e.g., `def function_name(): pass`, `raise NotImplementedError`), auto-generated interface bindings, and empty test mocks. Ingesting these files wastes token compute on non-informative structural templates that offer zero educational or reasoning value.

### B. Extreme Cyclomatic Complexity & Machine-Generated Artifacts

Machine-generated code, un-rolled loop structures, and deeply nested decision trees feature extreme cyclomatic complexity. Ingesting these obfuscated structures confuses model attention mechanisms, distorts loss curves, and degrades the model's ability to learn structured algorithmic reasoning.

### C. Obfuscated & Minified Dumps

Minified scripts, variable-dump files lacking inline documentation, and obfuscated assets contain high variable entropy with zero explanatory context. Step 7b applies multi-dimensional quality evaluation designed explicitly for technical syntax, preventing low-quality technical noise from polluting the corpus without misclassifying valid code as "low-quality prose."

---

## 3. Theoretical Execution Mechanics

Step 7b evaluates records routed to **Track B** through a three-stage domain quality pipeline:

```text
                     [ Track B Stream Post-Step 6b ]
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │ Stage 1: Lexical Tokenization & Error Ratio          │
         └──────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │ Stage 2: Structural Complexity & Maintainability     │
         │          Metrics                                     │
         └──────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │ Stage 3: Domain-Specific Neural Quality Scoring      │
         └──────────────────────────┴───────────────────────────┘

```

### Stage 1: Lexical Tokenization & Error Ratio Calculation

Before analyzing semantic or structural quality, the pipeline verifies whether the code payload can be tokenized into valid lexical constructs corresponding to its programming language tag:

1. **Lexical Parsing:** Code blocks are passed through language-specific lexers or concrete syntax parsers.
2. **Lexer Error Ratio:** The engine measures the relative frequency of unrecognized syntax tokens, malformed string literals, and lexical errors relative to total token count:

$$\text{Error Ratio} = \frac{\text{Count}(\text{Lexical Error Tokens})}{\text{Total Token Count}}$$



If $\text{Error Ratio} > 0.05$, the document is flagged as corrupted or un-parseable syntax and purged immediately.

### Stage 2: Structural Complexity & Maintainability Metrics

Syntactically valid code is evaluated against software maintainability and structural complexity bounds:

1. **Docstring-to-Code Ratio:** Calculates the proportion of lines containing inline documentation, explanatory comments, or docstrings relative to executable logic lines. Files exhibiting zero documentation alongside low readability metrics are penalized.
2. **Cyclomatic Complexity ($M$):** Measures the number of linearly independent control-flow paths through a program's source code based on decision graph nodes:

$$M = E - N + 2P$$



where $E$ represents the number of edges, $N$ represents the number of nodes, and $P$ represents the number of connected components in the control-flow graph.
* **Low Complexity Limit ($M < 2$):** Flags trivial function stubs, empty getter/setter wrappers, or non-informative boilerplate across large files.
* **High Complexity Limit ($M > 50$):** Flags un-maintainable spaghetti code, machine-generated decision trees, or obfuscated control flows. Records outside the valid $2 \le M \le 50$ envelope are purged.



### Stage 3: Domain-Specific Neural Quality Scoring

For complex code bases and LaTeX documents, text payloads pass through a specialized, domain-tailored neural quality model fine-tuned on curated technical repositories (e.g., high-star repositories, peer-reviewed papers, and verified technical Q&A platforms):

1. **Inference Evaluation:** The model evaluates code context and structural style, generating a scalar domain quality score $S_{\text{domain}} \in [0.0, 1.0]$.
2. **Quality Gatekeeping:** Documents yielding a domain quality score below the retention cutoff are evicted:

$$\text{Retention Condition}: S_{\text{domain}} \ge 0.60$$



---

## 4. Domain Quality Strategy Matrix

| Technical Content Profile | Structural & Metric Profile | Theoretical Engine | Pipeline Action | Downstream Impact in Phase 2 |
| --- | --- | --- | --- | --- |
| **High-Quality Production Code** | Valid lexer tokens, $0.05 \le \text{Comment Ratio} \le 0.40$, $2 \le M \le 30$, $S_{\text{domain}} \ge 0.60$. | Language Lexer + Domain Quality Model | **Retained:** Passes quality gate. | Advanced to Step 8b for AST syntax validation. |
| **Empty Function / Stub File** | High stub-to-line ratio (`pass`, `return null`), Cyclomatic Complexity $M < 2$. | Structural AST Metric Profiler | **Pruned:** Flagged as trivial boilerplate and dropped. | Prevents model over-fitting on non-informative stubs. |
| **Machine-Generated Spaghetti Code** | Extreme Cyclomatic Complexity ($M > 50$), deep nesting ($> 8$ scope levels), zero comments. | Control-Flow Complexity Engine | **Pruned:** Dropped to protect model attention mechanisms. | Prevents degradation of LLM reasoning convergence. |
| **Corrupted / Partial Syntax File** | Lexer Error Ratio $> 0.05$, un-closed brackets or malformed string literals. | C-Implemented Lexer Backend | **Pruned:** Fails basic lexical tokenization pass. | Purges broken token sequences from the dataset. |
| **Obfuscated / Variable-Renamed Dump** | High variable entropy, single-letter variable density $> 0.70$, $S_{\text{domain}} < 0.40$. | Quantized Domain Quality Model | **Pruned:** Dropped due to low instructional value. | Preserves corpus quality for code pre-training. |

---

## 5. Algorithmic Principles & Theoretical Tooling

* **Language Lexer Backends:** High-performance lexical analysis engines configured to convert raw text streams into language-specific token streams while flagging lexical syntax errors.
* **Control-Flow Graph Calculators:** Abstract Syntax Tree and complexity analysis utilities capable of building control-flow graphs to compute Cyclomatic Complexity ($M = E - N + 2P$) and nesting depth metrics.
* **Quantized Domain Quality Classifiers:** Specialized, lightweight neural classification models fine-tuned on curated technical corpora to evaluate domain authority and instructional value.