# Qualitative Probing Harness and Behavioral Evaluation

## 1. The Gap Between Statistical Metrics and Behavioral Capabilities

While validation loss and perplexity (PPL) measure the overall information-theoretic compression of a dataset, they act as coarse aggregate metrics. A checkpoint with a lower validation loss can still exhibit severe behavioral regressions:

```text
Aggregate Validation Loss (Macro Metric):
  L_val = 1.62 nats ──► Overall statistical likelihood improved by 0.05.
                        (Hides catastrophic failures on low-frequency, high-value tasks)

Micro-Behavioral Regressions Masked by Low Loss:
  ├── Formatting Collapse:       Fails to emit closing JSON brackets or AST-valid code.
  ├── Instruction Drift:         Ignores negative constraints ("Do NOT include headers").
  ├── Sycophancy & Alignment:    Agrees with factually false user premises.
  └── Catastrophic Forgetting:   Loses arithmetic or multi-step reasoning capabilities.

```

A **Qualitative Probing Harness** executes automated, deterministic, and behavioral challenge suites against model checkpoints during training intervals. It converts qualitative behavioral requirements into deterministic assertions, providing a fine-grained capability profile across optimization checkpoints.

```text
Candidate Model Checkpoint (θ_t)
               │
               ▼
┌────────────────────────────────────────────────────────────────────────┐
│ QUALITATIVE PROBING HARNESS (BEHAVIORAL CHALLENGE BATTERY)             │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Schema & Structural Probes      ──► AST Parser / JSON Validator     │
│ 2. Deterministic Reasoning Probes  ──► Symbolic Math / Logic Verifier  │
│ 3. Negative Constraint Probes      ──► Substring / Banned Term Scanner │
│ 4. Formatting Delimiter Probes     ──► Token Boundary / EOS Validator  │
│ 5. Domain Knowledge Probes         ──► Canonical Exact-Match Evaluator │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Aggregated Capability Scorecard
                                    ▼
       [ Pass/Fail Regression Gatekeeper & Checkpoint Scorecard ]

```

---

## 2. Behavioral Probing Taxonomy

Qualitative probes are categorized into five functional archetypes, each targeting a distinct failure mode in continuous training and fine-tuning:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ BEHAVIORAL PROBING TAXONOMY                                            │
├──────────────────────────┬───────────────────┬─────────────────────────┤
│ Probe Category           │ Evaluated Ability │ Failure Signature       │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 1. Structural / Schema   │ Adherence to JSON,│ Syntactic parse errors, │
│    Integrity             │ XML, Python AST   │ unclosed quote/bracket  │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 2. Deterministic         │ Multi-step logic, │ Hallucinated arithmetic,│
│    Reasoning             │ arithmetic exact  │ early answer commitment │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 3. Negative Constraint   │ Negative rule     │ Inability to suppress   │
│    Adherence             │ compliance        │ high-probability n-grams│
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 4. Turn Delimitation     │ Stop condition    │ Runaway generation,     │
│    & EOS Integrity       │ emission          │ header hallucinations   │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 5. Knowledge Invariance  │ Fact retention,   │ Catastrophic forgetting │
│    & Alignment           │ domain groundings │ under continuous tuning │
└──────────────────────────┴───────────────────┴─────────────────────────┘

```

### 1. Structural & Schema Integrity Probes

Evaluates whether the model's outputs parse cleanly under formal grammar engines without human post-processing:

* **JSON Conformance:** `json.loads(output)` must succeed; key names and types must match expected schema definitions.
* **Code AST Parsing:** Output wrapped in code fences must successfully compile into an Abstract Syntax Tree via `ast.parse(code)` without throwing `SyntaxError`.

### 2. Deterministic Reasoning & Exact Match Probes

Tests multi-step symbolic and arithmetic deduction where only a single valid answer exists:

* Multi-digit integer arithmetic (e.g., $147 \times 23 = 3381$).
* Algorithmic trace execution (e.g., "Given list $[3, 1, 4]$, what is the result of `sorted(list)[1]`?").

### 3. Negative Constraint Adherence Probes

Standard autoregressive generation biases models toward high-frequency token associations. Negative constraints test whether the attention mechanism can suppress default priors:

* *Constraint Prompt:* "Explain photosynthesis in exactly three sentences. Do not use the letter 'e'."
* *Assertion:* Output sentence count must equal 3; total count of character `'e'` or `'E'` must equal 0.

### 4. Turn Delimitation & EOS Integrity Probes

Ensures that instruction-tuned models terminate decoding promptly when completing a turn:

* *Prompt:* Input formatted with standard dialogue markers (e.g., `<|im_start|>user\nWhat is the capital of Japan?<|im_end|>\n<|im_start|>assistant\n`).
* *Assertion:* Generation must terminate on `<|im_end|>` or `</s>` within $\le 10$ tokens without leaking `<|im_start|>user` headers.

---

## 3. Evaluation Assertion Mechanics: Deterministic vs. Heuristic

To eliminate variance, qualitative probing relies strictly on **deterministic assertion engines** rather than non-deterministic "Model-as-a-Judge" heuristics.

```text
Probing Input: Prompt P_k ──► Model Generation G_k ──► Deterministic Verification Pipeline
                                                                   │
                         ┌─────────────────────────────────────────┼─────────────────────────────────────────┐
                         ▼                                         ▼                                         ▼
              [ Schema / AST Engine ]                   [ Exact-Match / Regex ]                   [ Constraint Scanner ]
               • json.loads(G_k)                         • re.search(Pattern, G_k)                 • Banned term detection
               • ast.parse(G_k)                          • Normalized String Match                 • Length / Count bounds

```

### Mathematical Pass Rate Formulation

Let $\mathcal{P} = \{p_1, p_2, \dots, p_K\}$ be a suite of $K$ qualitative probe cases. For each probe $p_k$, a deterministic verification function $v_k: \text{String} \to \{0, 1\}$ returns $1$ if all assertions pass, and $0$ otherwise.

$$\text{PassRate}(\mathcal{P}) = \frac{1}{K} \sum_{k=1}^K v_k\left( \text{Generate}(\theta, p_k) \right)$$

$$\text{Category Pass Rate: } \text{Score}_c = \frac{1}{\vert{}K_c\vert{}} \sum_{i \in K_c} v_i(G_i)$$

A checkpoint is certified if and only if every category score exceeds its predefined regression ceiling:

$$\forall c \in \text{Categories}, \quad \text{Score}_c \ge \tau_c$$

---

## 4. Behavioral Drift and Catastrophic Forgetting

During domain-specific continuous fine-tuning (e.g., adapting a general model to biomedical texts or SQL query generation), general capabilities degrade along predictable trajectories:

```text
Capability Degradation Dynamics across Continuous Fine-Tuning:

Pass Rate (%)
 100 ┌─────────────────────────────────────────────────────────────┐
     │                                     Target Domain (Biomedical)
  80 │                                 _ - ──► Climbs to 92%
     │                             _ -
  60 │     General Python AST  _ -
     │    ───────────────────■ - _
  40 │                            - _      Arithmetic Reasoning Drift
     │                                - _ ──► Drops from 85% to 32%
  20 │
   0 └─────────────────────────────────────────────────────────────┴────► Training Steps
     Step 0 (Base Checkpoint)                                       Step 50,000

```

### Probing as an Early Warning System

* **Pre-Divergence Detection:** Behavioral pass rates frequently collapse 1,000–3,000 steps *before* validation loss indicates an issue.
* **Checkpoint Selection:** If Checkpoint $A$ (Step 40k) has validation loss $1.52$ and Reasoning Pass Rate $88\%$, while Checkpoint $B$ (Step 50k) has validation loss $1.49$ but Reasoning Pass Rate $45\%$, the probing harness prevents promoting the degraded Checkpoint $B$.

---

## 5. Python Implementation: Standalone Qualitative Probing Harness

Below is the production implementation of a modular probing harness executing schema parsing, AST compilation, negative constraint scanning, exact match arithmetic, and generating a structured scorecard:

```python
import ast
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ProbeCategory(str, Enum):
    SCHEMA = "schema_integrity"
    REASONING = "deterministic_reasoning"
    CONSTRAINT = "negative_constraints"
    DELIMITER = "turn_delimitation"
    FACTUAL = "factual_retention"


@dataclass
class ProbeTestCase:
    id: str
    category: ProbeCategory
    prompt: str
    assertion_fn: Callable[[str], bool]
    description: str
    temperature: float = 0.0  # Probes default to deterministic argmax
    max_new_tokens: int = 256


@dataclass
class ProbeResult:
    test_id: str
    category: ProbeCategory
    passed: bool
    generated_text: str
    error_message: Optional[str] = None


class QualitativeProbingHarness:
    """
    Automated qualitative evaluation harness for behavioral model testing.
    """
    def __init__(self):
        self.test_suite: List[ProbeTestCase] = []
        self._register_default_probes()

    def add_test(self, test: ProbeTestCase):
        self.test_suite.append(test)

    def _register_default_probes(self):
        # -------------------------------------------------------------
        # 1. SCHEMA PROBES
        # -------------------------------------------------------------
        self.add_test(ProbeTestCase(
            id="schema_valid_json",
            category=ProbeCategory.SCHEMA,
            prompt="Generate a JSON object with keys 'user_id' (integer), 'roles' (list of strings), and 'active' (boolean). Output JSON only.",
            assertion_fn=self._assert_valid_json_schema,
            description="Validates that output parses via json.loads and matches required keys."
        ))

        self.add_test(ProbeTestCase(
            id="schema_python_ast",
            category=ProbeCategory.SCHEMA,
            prompt="Write a Python function named `compute_factorial(n: int) -> int` that handles n=0 and returns n!. Output code only.",
            assertion_fn=self._assert_valid_python_ast,
            description="Validates that code snippet compiles cleanly via Python AST parser."
        ))

        # -------------------------------------------------------------
        # 2. REASONING PROBES
        # -------------------------------------------------------------
        self.add_test(ProbeTestCase(
            id="reasoning_arithmetic_mult",
            category=ProbeCategory.REASONING,
            prompt="Compute the exact value of 147 * 23. Provide only the numeric result.",
            assertion_fn=lambda out: "3381" in re.findall(r"\b\d+\b", out),
            description="Tests exact multi-digit integer multiplication without hallucination."
        ))

        self.add_test(ProbeTestCase(
            id="reasoning_symbolic_reversal",
            category=ProbeCategory.REASONING,
            prompt="Reverse the following sequence of words separated by spaces: 'alpha beta gamma delta'. Output only the reversed words.",
            assertion_fn=lambda out: "delta gamma beta alpha" in out.strip().lower(),
            description="Validates deterministic token order reversal."
        ))

        # -------------------------------------------------------------
        # 3. NEGATIVE CONSTRAINT PROBES
        # -------------------------------------------------------------
        self.add_test(ProbeTestCase(
            id="constraint_banned_letter",
            category=ProbeCategory.CONSTRAINT,
            prompt="Write a short summary of a cat sleeping in the sun. You must strictly avoid using the letter 'e' or 'E' anywhere in your response.",
            assertion_fn=lambda out: ("e" not in out.lower()) and (len(out.strip()) > 20),
            description="Verifies absolute suppression of high-frequency character 'e'."
        ))

        self.add_test(ProbeTestCase(
            id="constraint_max_sentence_count",
            category=ProbeCategory.CONSTRAINT,
            prompt="Explain what an operating system kernel is in exactly two sentences. Do not write more or less than two sentences.",
            assertion_fn=lambda out: len([s for s in re.split(r"[.!?]+", out.strip()) if s.strip()]) == 2,
            description="Enforces strict sentence count boundary."
        ))

        # -------------------------------------------------------------
        # 4. DELIMITER & TERMINATION PROBES
        # -------------------------------------------------------------
        self.add_test(ProbeTestCase(
            id="delimiter_eos_stop",
            category=ProbeCategory.DELIMITER,
            prompt="<|im_start|>user\nSay 'OK'.<|im_end|>\n<|im_start|>assistant\n",
            assertion_fn=lambda out: (
                out.strip().startswith("OK") and
                "<|im_start|>user" not in out and
                len(out.strip().split()) <= 4
            ),
            description="Ensures model stops generation immediately without runaway conversation loops."
        ))

    # --- Assertion Helpers ---
    @staticmethod
    def _assert_valid_json_schema(output: str) -> bool:
        try:
            # Extract JSON block if wrapped in markdown code fences
            cleaned = output.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            data = json.loads(cleaned)
            return (
                isinstance(data, dict)
                and "user_id" in data
                and "roles" in data
                and "active" in data
                and isinstance(data["user_id"], int)
                and isinstance(data["roles"], list)
                and isinstance(data["active"], bool)
            )
        except Exception:
            return False

    @staticmethod
    def _assert_valid_python_ast(output: str) -> bool:
        try:
            cleaned = output.strip()
            if "```python" in cleaned:
                cleaned = cleaned.split("```python")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            tree = ast.parse(cleaned)
            # Verify function name exists in AST
            func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            return "compute_factorial" in func_names
        except Exception:
            return False

    def run_evaluations(
        self, generate_fn: Callable[[str, float, int], str]
    ) -> Dict[str, Any]:
        """
        Executes all registered probes against the provided model generation function.
        
        Args:
            generate_fn: Callable taking (prompt, temperature, max_new_tokens) -> output_text
        """
        results: List[ProbeResult] = []
        category_stats: Dict[ProbeCategory, Dict[str, int]] = {
            cat: {"passed": 0, "total": 0} for cat in ProbeCategory
        }

        for test in self.test_suite:
            generated_text = ""
            passed = False
            err_msg = None

            try:
                generated_text = generate_fn(test.prompt, test.temperature, test.max_new_tokens)
                passed = test.assertion_fn(generated_text)
            except Exception as e:
                passed = False
                err_msg = str(e)

            result = ProbeResult(
                test_id=test.id,
                category=test.category,
                passed=passed,
                generated_text=generated_text,
                error_message=err_msg
            )
            results.append(result)

            # Accumulate statistics
            category_stats[test.category]["total"] += 1
            if passed:
                category_stats[test.category]["passed"] += 1

        # Compute summary scorecard
        total_tests = len(results)
        total_passed = sum(1 for r in results if r.passed)
        overall_pass_rate = (total_passed / total_tests) if total_tests > 0 else 0.0

        category_scores = {}
        for cat, stats in category_stats.items():
            if stats["total"] > 0:
                category_scores[cat.value] = {
                    "passed": stats["passed"],
                    "total": stats["total"],
                    "pass_rate": stats["passed"] / stats["total"]
                }

        scorecard = {
            "overall_pass_rate": overall_pass_rate,
            "total_passed": total_passed,
            "total_probes": total_tests,
            "categories": category_scores,
            "probe_details": [
                {
                    "id": r.test_id,
                    "category": r.category.value,
                    "passed": r.passed,
                    "output_preview": r.generated_text[:120].replace("\n", "\\n"),
                    "error": r.error_message
                }
                for r in results
            ]
        }

        return scorecard

```

---

## 6. Probe Suite Matrix & Acceptance Gate Standards

Before any continuous training checkpoint is promoted to validation scoring or deployment consideration, it must satisfy minimum category thresholds:

| Category | Primary Test Mechanism | Min Pass Threshold | Action on Failure |
| --- | --- | --- | --- |
| **Schema Integrity** | `json.loads` & `ast.parse` | **$100\%$ Strict** | Reject checkpoint; formatting corrupted |
| **Turn Delimitation** | Substring `< | im_start | >` & EOS |
| **Deterministic Reasoning** | Numeric Regex / Exact Match | **$\ge 85\%$** | Flag warning; check reasoning degradation |
| **Negative Constraints** | Banned character & token search | **$\ge 75\%$** | Flag warning; check instruction drift |
| **Factual Invariance** | Gold QA exact substring match | **$\ge 80\%$** | Check for catastrophic domain forgetting |