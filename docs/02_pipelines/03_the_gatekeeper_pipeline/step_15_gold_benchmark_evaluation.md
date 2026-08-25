# Step 15: Gold Benchmark Evaluation and Automated Regression Auditing

## 1. The Role of Gold Benchmark Evaluation in Continuous Training

While validation cross-entropy loss and token perplexity track macro-level convergence on general text distributions, they act as coarse statistical averages. A training checkpoint that achieves a new minimum validation loss can simultaneously suffer catastrophic regressions in code syntax generation, multi-step arithmetic reasoning, or strict negative constraint following.

```text
Training Checkpoint (step_t.pt)
                 │
                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 15: GOLD BENCHMARK EVALUATION HARNESS                             │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Immutable Gold Capability Battery (Zero-Contamination Split)        │
│    • Syntax & Schema: JSON parser, Python AST compiler                 │
│    • Symbolic Reasoning: Exact-match multi-step arithmetic (GSM8K)     │
│    • Factual & Domain Knowledge: Multi-choice log-likelihood (MMLU)    │
│    • Functional Code Synthesis: Sandboxed unit-test execution (HumanEval)
│    • Instruction Following: Verifiable negative constraint probes      │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Dual-Paradigm Scoring Engine                                        │
│    • Log-Likelihood Evaluator: Normalized multiple-choice ranking      │
│    • Generative Execution Engine: Deterministic greedy rollouts        │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Automated Scorecard & Regression Arbiter                            │
│    • Delta Comparison against Reference Baseline Checkpoint (θ_base)   │
│    • Hard Regression Ceiling: Reject if any core capability drops > 0.5%│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       [ Promote to best_model.pt ]     [ Trigger Early Warning / Quarantine ]

```

**Step 15 (Gold Benchmark Evaluation)** executes an immutable, curated evaluation suite against candidate checkpoints. It provides an objective capability scorecard that governs early stopping, model checkpoint selection (`best_model.pt`), and promotion through Gate 5 deployment gates.

---

## 2. Standardized Gold Capability Dimensions & Benchmark Suites

The Gold Benchmark Battery tests five core capability vectors using deterministic evaluation protocols:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ GOLD CAPABILITY BENCHMARK MATRIX                                       │
├──────────────────────────┬───────────────────┬─────────────────────────┤
│ Capability Dimension     │ Target Benchmark  │ Scoring Methodology     │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 1. Schema & Structural   │ Gold JSON /       │ Deterministic parsing:  │
│    Integrity             │ Python AST Probes │ `json.loads`, `ast.parse`│
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 2. Mathematical &        │ GSM8K Subset /    │ Greedy Generative CoT:  │
│    Symbolic Reasoning    │ Arithmetic Gold   │ Regex Exact Match (EM)  │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 3. Multi-Discipline      │ MMLU Core /       │ Length-Normalized       │
│    World Knowledge       │ ARC-Challenge     │ Log-Likelihood Ranking  │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 4. Functional Code       │ HumanEval Mini    │ Subprocess Sandboxed    │
│    Synthesis             │ (Handcrafted Set) │ Execution: Pass@1       │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 5. Constraint & Format   │ IFEval Strict /   │ Rule-based string scan: │
│    Adherence             │ Negative Probes   │ Word bounds, banned keys│
└──────────────────────────┴───────────────────┴─────────────────────────┘

```

---

## 3. Evaluation Mathematical Protocols

### A. Multiple-Choice Log-Likelihood with Length Normalization

For multiple-choice queries with candidate options $C = \{c_1, c_2, \dots, c_K\}$ given prompt $X$, raw log-probabilities bias predictions toward shorter completions:

$$\log P(c_k \mid X) = \sum_{t=1}^{\vert{}c_k\vert{}} \log P(c_{k, t} \mid X, c_{k, <t})$$

The Gold Harness applies length normalization to rank completions fairly:

$$S(c_k \mid X) = \frac{1}{\vert{}c_k\vert{}^\alpha} \sum_{t=1}^{\vert{}c_k\vert{}} \log P(c_{k, t} \mid X, c_{k, <t})$$

$$\hat{y} = \arg\max_{k \in \{1, \dots, K\}} S(c_k \mid X)$$

Where $\alpha = 1.0$ represents standard length averaging.

---

### B. Sandboxed Code Execution (Pass@1)

For code generation tasks, the model generates code completions under deterministic greedy decoding ($T = 0.0$). The generated function is injected into an isolated execution environment with a strict timeout ($t \le 3.0\text{s}$) to evaluate assertion test suites:

$$\text{Pass@1} = \frac{1}{N_{\text{tasks}}} \sum_{i=1}^{N_{\text{tasks}}} \mathbb{I}\Big( \text{Execute}\left( \text{Code}_i, \, \text{Tests}_i \right) == \text{PASSED} \Big)$$

---

### C. Composite Gold Capability Score ($S_{\text{composite}}$)

Individual benchmark scores $s_i \in [0, 1]$ are aggregated into a weighted composite capability metric:

$$S_{\text{composite}} = \sum_{i=1}^M w_i \cdot s_i, \quad \text{where } \sum_{i=1}^M w_i = 1.0$$

A candidate checkpoint qualifies as a new **Global Best** if and only if:

$$S_{\text{composite}}(\theta_{\text{cand}}) > S_{\text{composite}}(\theta_{\text{best\_historical}}) \quad \land \quad \forall i, \; s_i(\theta_{\text{cand}}) \ge s_i(\theta_{\text{base}}) - \delta_{\text{regress}}$$

Where $\delta_{\text{regress}} = 0.005$ ($0.5\%$ maximum allowable capability degradation per task).

---

## 4. Python Implementation: Production Gold Benchmark Suite

Below is the complete standalone implementation of the `GoldBenchmarkEvaluator` supporting multiple-choice log-likelihood evaluation, Chain-of-Thought math extraction, sandboxed code execution, schema validation, and composite scorecard generation:

```python
import ast
import json
import math
import multiprocessing
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. SANDBOXED EXECUTION WORKER
# =====================================================================
def _code_execution_worker(
    full_code: str, entry_point: str, result_queue: multiprocessing.Queue
):
    """Executes Python code in an isolated subprocess."""
    global_namespace = {}
    try:
        exec(full_code, global_namespace)
        result_queue.put({"status": "PASSED", "error": None})
    except Exception as e:
        result_queue.put({"status": "FAILED", "error": f"{type(e).__name__}: {str(e)}"})


# =====================================================================
# 2. MASTER GOLD EVALUATION ENGINE
# =====================================================================
class GoldBenchmarkEvaluator:
    """
    Production-grade Gold Benchmark Evaluation suite executing multi-task
    capability audits and regression verification on training checkpoints.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        execution_timeout_sec: float = 3.0
    ):
        self.model = model.to(device=device, dtype=dtype)
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.timeout = execution_timeout_sec
        self.model.eval()

    # -----------------------------------------------------------------
    # TASK 1: MULTIPLE-CHOICE LOG-LIKELIHOOD (MMLU / ARC)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def evaluate_multiple_choice(
        self,
        samples: List[Dict[str, Any]],
        length_penalty: float = 1.0
    ) -> float:
        """
        Computes accuracy over multiple-choice questions via log-likelihood ranking.
        Sample schema: {"prompt": str, "choices": List[str], "gold_idx": int}
        """
        correct_count = 0

        for sample in samples:
            prompt_text = sample["prompt"]
            choices = sample["choices"]
            gold_idx = sample["gold_idx"]

            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            scores = []

            for choice in choices:
                choice_ids = self.tokenizer.encode(f" {choice}", add_special_tokens=False)
                input_ids = torch.tensor(
                    [prompt_ids + choice_ids], dtype=torch.long, device=self.device
                )

                with torch.autocast(
                    device_type=self.device if self.device == "cuda" else "cpu",
                    dtype=self.dtype
                ):
                    logits = self.model(input_ids)
                    if isinstance(logits, tuple):
                        logits = logits[0]

                # Causal token shift
                shift_logits = logits[0, :-1, :].contiguous()
                shift_labels = input_ids[0, 1:].contiguous()
                log_probs = F.log_softmax(shift_logits, dim=-1)

                # Extract probabilities strictly across choice completion tokens
                start_pos = len(prompt_ids) - 1
                end_pos = start_pos + len(choice_ids)
                target_indices = torch.arange(start_pos, end_pos, device=self.device)
                target_labels = shift_labels[target_indices]

                token_log_probs = log_probs[target_indices, target_labels]
                total_log_prob = token_log_probs.sum().item()

                # Apply length normalization
                normalized_score = total_log_prob / (len(choice_ids) ** length_penalty)
                scores.append(normalized_score)

            predicted_idx = int(torch.argmax(torch.tensor(scores)).item())
            if predicted_idx == gold_idx:
                correct_count += 1

        return correct_count / max(1, len(samples))

    # -----------------------------------------------------------------
    # TASK 2: GENERATIVE CHAIN-OF-THOUGHT MATH (GSM8K)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def evaluate_generative_math(
        self,
        samples: List[Dict[str, str]],
        max_new_tokens: int = 256
    ) -> float:
        """
        Evaluates exact-match arithmetic from greedy CoT completions.
        Sample schema: {"prompt": str, "gold_answer": str}
        """
        correct_count = 0

        for sample in samples:
            prompt_text = sample["prompt"]
            gold_answer = sample["gold_answer"]

            input_ids = torch.tensor(
                [self.tokenizer.encode(prompt_text)],
                dtype=torch.long,
                device=self.device
            )

            # Greedy generation rollout
            curr_tokens = input_ids
            for _ in range(max_new_tokens):
                with torch.autocast(
                    device_type=self.device if self.device == "cuda" else "cpu",
                    dtype=self.dtype
                ):
                    logits = self.model(curr_tokens)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

                if next_token.item() == getattr(self.tokenizer, "eos_token_id", None):
                    break

            completion = self.tokenizer.decode(
                curr_tokens[0, input_ids.shape[1]:], skip_special_tokens=True
            )

            # Extract numeric tokens
            pred_num = self._extract_final_number(completion)
            gold_num = self._extract_final_number(gold_answer)

            if pred_num is not None and gold_num is not None:
                try:
                    if math.isclose(float(pred_num), float(gold_num), rel_tol=1e-5):
                        correct_count += 1
                except ValueError:
                    if pred_num.strip().lower() == gold_num.strip().lower():
                        correct_count += 1

        return correct_count / max(1, len(samples))

    @staticmethod
    def _extract_final_number(text: str) -> Optional[str]:
        """Extracts the final numerical value from text or GSM8K '####' marker."""
        if "####" in text:
            ans = text.split("####")[-1].replace(",", "").replace("$", "").strip()
            match = re.search(r"^-?\d+(?:\.\d+)?", ans)
            if match:
                return match.group(0)

        cleaned = text.replace(",", "").replace("$", "")
        numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        return numbers[-1] if numbers else None

    # -----------------------------------------------------------------
    # TASK 3: SANDBOXED CODE SYNTHESIS (HumanEval Pass@1)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def evaluate_code_synthesis(
        self,
        samples: List[Dict[str, str]],
        max_new_tokens: int = 384
    ) -> float:
        """
        Evaluates Python code synthesis via isolated subprocess execution.
        Sample schema: {"prompt": str, "test_code": str, "entry_point": str}
        """
        passed_count = 0

        for sample in samples:
            prompt_text = sample["prompt"]
            test_code = sample["test_code"]
            entry_point = sample["entry_point"]

            input_ids = torch.tensor(
                [self.tokenizer.encode(prompt_text)],
                dtype=torch.long,
                device=self.device
            )

            curr_tokens = input_ids
            for _ in range(max_new_tokens):
                with torch.autocast(
                    device_type=self.device if self.device == "cuda" else "cpu",
                    dtype=self.dtype
                ):
                    logits = self.model(curr_tokens)
                    if isinstance(logits, tuple):
                        logits = logits[0]

                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

                if next_token.item() == getattr(self.tokenizer, "eos_token_id", None):
                    break

            completion = self.tokenizer.decode(
                curr_tokens[0, input_ids.shape[1]:], skip_special_tokens=True
            )

            # Strip Markdown code fences if generated
            if "```python" in completion:
                code_body = completion.split("```python")[1].split("```")[0].strip()
            elif "```" in completion:
                code_body = completion.split("```")[1].split("```")[0].strip()
            else:
                code_body = completion.strip()

            full_program = f"{prompt_text}\n{code_body}\n\n{test_code}\n\ncheck({entry_point})"

            # Execute in isolated subprocess
            if self._run_sandboxed_test(full_program, entry_point):
                passed_count += 1

        return passed_count / max(1, len(samples))

    def _run_sandboxed_test(self, full_code: str, entry_point: str) -> bool:
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        process = ctx.Process(
            target=_code_execution_worker,
            args=(full_code, entry_point, queue)
        )
        process.start()
        process.join(timeout=self.timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return False  # Timed out

        if not queue.empty():
            res = queue.get()
            return res.get("status") == "PASSED"
        return False

    # -----------------------------------------------------------------
    # TASK 4: SCHEMA & SYNTAX INTEGRITY (JSON / AST)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def evaluate_schema_integrity(
        self,
        samples: List[Dict[str, str]],
        max_new_tokens: int = 128
    ) -> float:
        """
        Validates that generated JSON outputs parse cleanly via json.loads.
        Sample schema: {"prompt": str, "required_keys": List[str]}
        """
        valid_count = 0

        for sample in samples:
            input_ids = torch.tensor(
                [self.tokenizer.encode(sample["prompt"])],
                dtype=torch.long,
                device=self.device
            )

            curr_tokens = input_ids
            for _ in range(max_new_tokens):
                with torch.autocast(
                    device_type=self.device if self.device == "cuda" else "cpu",
                    dtype=self.dtype
                ):
                    logits = self.model(curr_tokens)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
                if next_token.item() == getattr(self.tokenizer, "eos_token_id", None):
                    break

            completion = self.tokenizer.decode(
                curr_tokens[0, input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            if "```json" in completion:
                completion = completion.split("```json")[1].split("```")[0].strip()
            elif "```" in completion:
                completion = completion.split("```")[1].split("```")[0].strip()

            try:
                data = json.loads(completion)
                if isinstance(data, dict):
                    req_keys = sample.get("required_keys", [])
                    if all(k in data for k in req_keys):
                        valid_count += 1
            except Exception:
                pass

        return valid_count / max(1, len(samples))

    # -----------------------------------------------------------------
    # COMPOSITE SCORECARD ARBITER
    # -----------------------------------------------------------------
    def run_gold_battery(
        self,
        mc_samples: List[Dict[str, Any]],
        math_samples: List[Dict[str, str]],
        code_samples: List[Dict[str, str]],
        schema_samples: List[Dict[str, str]],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Executes full battery and returns weighted composite capability scorecard."""
        w = weights or {"mc": 0.25, "math": 0.25, "code": 0.25, "schema": 0.25}

        t_start = time.perf_counter()

        score_mc = self.evaluate_multiple_choice(mc_samples)
        score_math = self.evaluate_generative_math(math_samples)
        score_code = self.evaluate_code_synthesis(code_samples)
        score_schema = self.evaluate_schema_integrity(schema_samples)

        composite_score = (
            w["mc"] * score_mc +
            w["math"] * score_math +
            w["code"] * score_code +
            w["schema"] * score_schema
        )

        duration = time.perf_counter() - t_start

        return {
            "composite_score": composite_score,
            "multiple_choice_acc": score_mc,
            "math_exact_match": score_math,
            "code_pass1": score_code,
            "schema_validity": score_schema,
            "eval_duration_sec": duration
        }

```

---

## 5. Scorecard Regression & Promotion Standards

```text
┌────────────────────────────────────────────────────────────────────────┐
│ GOLD BENCHMARK CERTIFICATION STANDARDS                                 │
├──────────────────────────┬───────────────────┬─────────────────────────┤
│ Capability Metric        │ Minimum Threshold │ Regression Limit vs Prod│
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Schema Syntax Validity   │ 100.0% Strict     │ Δ == 0.0% (Zero Drop)   │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Code Synthesis Pass@1    │ >= 20.0%          │ Δ >= -0.5%              │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Math CoT Exact Match     │ >= 25.0%          │ Δ >= -0.5%              │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Knowledge Log-Likelihood │ >= 40.0%          │ Δ >= -0.5%              │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Composite Gold Score     │ New All-Time High │ Δ >= +0.1% for Best Ckpt│
└──────────────────────────┴───────────────────┴─────────────────────────┘

```

---

## 6. Diagnostic Failure Matrix

| Failure Symptom | Detection Point | Root Cause | Engineering Remediation |
| --- | --- | --- | --- |
| **Schema Syntax Rate $< 100\%$** | JSON / AST Validation | Loss masking omitted formatting tokens or SFT data polluted | Enforce loss on delimiter tokens; inspect formatting SFT shards |
| **Math Exact Match $\to 0\%$** | GSM8K Generative CoT | Loss of Chain-of-Thought reasoning structure | Verify CoT exemplar prompts; check for learning rate collapse |
| **Code Execution Timeout** | HumanEval Subprocess Runner | Infinite loops in generated code syntax | Terminate process; adjust repetition penalties on while/for loops |
| **MC Log-Likelihood Regression** | Length-Normalized MMLU | Knowledge forgetting during aggressive domain fine-tuning | Add general pre-training replay shards ($5\text{--}10\%$ mixture) |
| **Composite Score Stagnation** | Multi-Task Aggregation | Model capacity saturated under current learning rate | Trigger learning rate decay or scale hidden parameter budget |