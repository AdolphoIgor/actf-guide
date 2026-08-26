# Downstream Task Benchmarks, Evaluation Protocols, and Functional Unit Testing

## 1. The Dual Paradigms of Downstream Evaluation

Evaluating whether a language model has acquired generalizable problem-solving capabilities requires testing across downstream task distributions. Downstream evaluation operates under two distinct computational and algorithmic paradigms:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Log-Likelihood Scoring (Multiple-Choice / Cloze Evaluation)         │
│    • Method: Computes sequence log-probabilities across candidate options│
│    • Execution: Single forward pass over choice completions.            │
│    • Output: Ranked probabilities P(Option | Prompt). Fast & deterministic│
│    • Examples: MMLU, ARC, HellaSwag, PIQA, WinoGrande.                 │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Generative Execution & Unit Testing (Free-Form Output Evaluation)   │
│    • Method: Autoregressively generates text, code, or proofs.         │
│    • Execution: Multi-step sampling rollout + Output parsing / Execution│
│    • Output: Functional pass/fail, Exact Match (EM), or Pass@k.        │
│    • Examples: GSM8K, HumanEval, MBPP, MATH, SWE-bench.                │
└────────────────────────────────────────────────────────────────────────┘

```

```text
Log-Likelihood Multiple-Choice Paradigm:
  Prompt: "What is the capital of France?\n"
  ├── Option A: "London" ──► Log-Likelihood: -8.42
  ├── Option B: "Paris"  ──► Log-Likelihood: -0.14  <── Argmax Selection: B
  ├── Option C: "Berlin" ──► Log-Likelihood: -7.91
  └── Option D: "Rome"   ──► Log-Likelihood: -9.05

Generative Functional Unit-Testing Paradigm:
  Prompt: "Write a function `is_prime(n: int) -> bool`..."
  Rollout: "def is_prime(n):\n  if n < 2: return False..."
     │
     ▼
  Sandboxed Execution Environment ──► Run assert is_prime(7) == True
                                 ──► Run assert is_prime(4) == False
                                 ──► Status: PASS (100% Tests Passed)

```

---

## 2. Downstream Benchmark Taxonomy

Production model evaluation measures distinct capability vectors using standardized academic and industry benchmarks:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ DOWNSTREAM CAPABILITY BENCHMARK TAXONOMY                               │
├──────────────────────────┬───────────────────┬─────────────────────────┤
│ Domain                   │ Benchmark Suites  │ Core Capability Tested  │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Multi-Discipline World   │ MMLU, MMLU-Pro,   │ Factual recall, academic│
│ Knowledge & QA           │ ARC-Challenge     │ knowledge, reading comp │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Mathematical & Symbolic  │ GSM8K, MATH,      │ Multi-step arithmetic,  │
│ Reasoning                │ SVAMP             │ algebraic deduction     │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Code Generation &        │ HumanEval, MBPP,  │ Algorithmic syntax, API │
│ Functional Synthesis     │ EvalPlus          │ logic, test satisfaction│
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Grounded Common Sense &  │ HellaSwag, PIQA,  │ Adversarial completion, │
│ Linguistic Reasoning     │ WinoGrande        │ physical intuition      │
└──────────────────────────┴───────────────────┴─────────────────────────┘

```

### 1. Multi-Discipline Academic Knowledge (MMLU / ARC)

- **MMLU (Massive Multitask Language Understanding):** Encompasses 57 subjects across STEM, humanities, social sciences, and professional fields (law, medicine). Evaluated primarily via 5-shot multiple-choice log-likelihood scoring.
- **ARC (AI2 Reasoning Challenge):** Grade-school science questions partitioned into _Easy_ and _Challenge_ sets, designed to resist simple retrieval and word-co-occurrence heuristics.

### 2. Mathematical Reasoning (GSM8K / MATH)

- **GSM8K (Grade School Math 8K):** Multi-step linguistic math word problems requiring 2 to 8 reasoning steps. Evaluated using 8-shot Chain-of-Thought (CoT) prompting with exact numeric string extraction.
- **MATH:** Advanced high-school competition mathematics (algebra, geometry, number theory, calculus) evaluated via exact symbolic LaTeX string extraction.

### 3. Code Generation (HumanEval / MBPP)

- **HumanEval:** 164 handcrafted Python programming tasks containing docstrings, function signatures, reference implementations, and unit test suites. Evaluated via execution in a sandboxed runtime against unit tests.

---

## 3. Mathematical Mechanics of Evaluation Metrics

### A. Multiple-Choice Log-Likelihood Normalization

When scoring multiple-choice options $C \in \{c_1, c_2, \dots, c_K\}$ given prompt context $X$, short answers naturally produce higher raw probabilities than long answers due to multiplying fewer conditional probabilities:

$$P(C \mid X) = \prod_{t=1}^{\vert{}C\vert{}} P(c_t \mid X, c_{<t})$$

$$\log P(C \mid X) = \sum_{t=1}^{\vert{}C\vert{}} \log P(c_t \mid X, c_{<t})$$

To eliminate length bias, evaluation harnesses apply **length normalization** or **unconditioned probability normalization**:

$$\text{Length-Normalized Score: } S_{\text{norm}}(C \mid X) = \frac{1}{\vert{}C\vert{}^\alpha} \sum_{t=1}^{\vert{}C\vert{}} \log P(c_t \mid X, c_{<t})$$

$$\text{Unconditioned Normalization: } S_{\text{uncond}}(C \mid X) = \log P(C \mid X) - \log P(C \mid \text{""})$$

Where $\alpha \in [0.6, 1.0]$ is the length penalty, and $P(C \mid \text{""})$ is the marginal probability of the completion string given an empty prompt.

---

### B. Unbiased Pass@k Metric for Code & Functional Synthesis

Evaluating code generation by generating a single sample per task ($k=1$) introduces high variance. Generating $n$ candidate completions per problem ($n \ge k$) and computing the proportion of problems where at least one candidate passes all unit tests provides a more reliable metric.

Evaluating $\text{Pass@}k$ directly by sampling subsets of size $k$ introduces combinatorial estimation variance. Chen et al. (2021) derived the **unbiased minimum-variance estimator**:

$$\text{Pass@}k = \underset{\text{Problems}}{\mathbb{E}} \left[ 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}} \right] = \underset{\text{Problems}}{\mathbb{E}} \left[ 1 - \frac{\prod_{i=0}^{k-1} (n - c - i)}{\prod_{i=0}^{k-1} (n - i)} \right]$$

Where:

- $n$ is the total number of generated samples per problem (e.g., $n = 200$).
- $c$ is the number of samples that successfully pass all unit tests ($c \le n$).
- $k$ is the target evaluation threshold (e.g., $k \in \{1, 10, 100\}$).

```text
Combinatorial Pass@k Estimator Behavior:
  • If c = 0 (No samples pass):       Pass@k = 1 - (n/n) = 0.0
  • If c = n (All samples pass):      Pass@k = 1 - (0/n) = 1.0
  • If c > 0 and n - c < k:           Pass@k = 1.0 (Guaranteed hit in subset of size k)

```

---

## 4. Standardized Benchmark Evaluation Matrix

| Benchmark         | Target Task           | Primary Paradigm | Scoring Metric     | Few-Shot Protocol | Standard Baseline Expectation           |
| ----------------- | --------------------- | ---------------- | ------------------ | ----------------- | --------------------------------------- |
| **MMLU**          | General Knowledge     | Log-Likelihood   | Accuracy (%)       | 5-shot            | $25.0\%$ (Random) to $>85.0\%$ (SOTA)   |
| **ARC-Challenge** | Grade-School Science  | Log-Likelihood   | Normalized Acc (%) | 25-shot / 0-shot  | $25.0\%$ (Random) to $>90.0\%$ (SOTA)   |
| **GSM8K**         | Math Word Problems    | Generative CoT   | Exact Match (EM)   | 8-shot CoT        | $<10.0\%$ (Untuned) to $>90.0\%$ (SOTA) |
| **HumanEval**     | Python Synthesis      | Generative Code  | Pass@1 / Pass@10   | 0-shot            | $<15.0\%$ (Untuned) to $>80.0\%$ (SOTA) |
| **HellaSwag**     | Commonsense Reasoning | Log-Likelihood   | Normalized Acc (%) | 10-shot / 0-shot  | $25.0\%$ (Random) to $>90.0\%$ (SOTA)   |

---

## 5. Python Implementation: Modular Downstream Benchmark Engine

Below is the standalone, production-grade downstream evaluation harness supporting log-likelihood multiple-choice evaluation, regex-extracted mathematical exact match, and sandboxed execution of unit tests for Pass@k calculation:

```python
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
import math
import multiprocessing
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. LOG-LIKELIHOOD MULTIPLE CHOICE EVALUATION (MMLU / ARC)
# =====================================================================
class MultipleChoiceEvaluator:
    """
    Evaluates multiple-choice tasks by computing and ranking conditional log-likelihoods.
    """
    def __init__(self, model: nn.Module, tokenizer: Any, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def compute_choice_log_likelihood(
        self, prompt: str, completion: str, length_penalty: float = 1.0
    ) -> float:
        """
        Computes conditional log P(completion | prompt) with length normalization.
        """
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)

        input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=self.device)
        prompt_len = len(prompt_ids)
        completion_len = len(completion_ids)

        logits = self.model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]

        # Shift logits for causal alignment: predict token t+1 from t
        shift_logits = logits[0, :-1, :].contiguous()
        shift_labels = input_ids[0, 1:].contiguous()

        # Compute log-softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Slice loss strictly across completion tokens
        target_indices = torch.arange(prompt_len - 1, prompt_len + completion_len - 1, device=self.device)
        target_labels = shift_labels[target_indices]

        token_log_probs = log_probs[target_indices, target_labels]
        total_log_prob = token_log_probs.sum().item()

        # Apply length normalization
        normalized_score = total_log_prob / (completion_len ** length_penalty)
        return normalized_score

    def evaluate_sample(
        self, prompt: str, choices: List[str], correct_choice_idx: int
    ) -> bool:
        """
        Ranks choices by normalized log-likelihood and checks if argmax matches target.
        """
        scores = [
            self.compute_choice_log_likelihood(prompt, f" {choice}")
            for choice in choices
        ]
        predicted_idx = int(torch.argmax(torch.tensor(scores)).item())
        return predicted_idx == correct_choice_idx


# =====================================================================
# 2. GENERATIVE MATH & EXACT MATCH EXTRACTION (GSM8K)
# =====================================================================
class GenerativeMathEvaluator:
    """
    Extracts numerical answers from Chain-of-Thought generations and computes Exact Match.
    """
    @staticmethod
    def extract_numeric_answer(text: str) -> Optional[str]:
        """
        Extracts final numeric answer from GSM8K format (e.g., '#### 42') or trailing regex.
        """
        # 1. Look for explicit GSM8K answer marker
        if "####" in text:
            ans = text.split("####")[-1].strip()
            # Remove thousand-separator commas and currency symbols
            ans = ans.replace(",", "").replace("$", "").strip()
            # Match integer or float pattern
            match = re.search(r"^-?\d+(?:\.\d+)?", ans)
            if match:
                return match.group(0)

        # 2. Fallback: Extract the last valid numeric token in the generation
        cleaned = text.replace(",", "").replace("$", "")
        numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        if numbers:
            return numbers[-1]

        return None

    def evaluate_response(self, generated_text: str, ground_truth_answer: str) -> bool:
        pred = self.extract_numeric_answer(generated_text)
        gold = self.extract_numeric_answer(ground_truth_answer)
        if pred is None or gold is None:
            return False

        # Float-safe exact match check
        try:
            return math.isclose(float(pred), float(gold), rel_tol=1e-5)
        except ValueError:
            return pred.strip().lower() == gold.strip().lower()


# =====================================================================
# 3. CODE SYNTHESIS & PASS@K SANDBOXED RUNNER (HumanEval)
# =====================================================================
def _sandboxed_execution_worker(
    program_code: str, test_code: str, entry_point: str, result_queue: multiprocessing.Queue
):
    """
    Isolated execution worker executing python script against test cases.
    """
    full_code = f"{program_code}\n\n{test_code}\n\ncheck({entry_point})"
    global_namespace = {}
    try:
        exec(full_code, global_namespace)
        result_queue.put("PASSED")
    except Exception as e:
        result_queue.put(f"FAILED: {type(e).__name__}: {e}")


class SandboxedCodeRunner:
    """
    Executes generated Python programs in sandboxed processes with strict timeouts.
    """
    def __init__(self, timeout_sec: float = 3.0):
        self.timeout = timeout_sec

    def execute_test(self, code_solution: str, test_suite: str, entry_point: str) -> bool:
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        process = ctx.Process(
            target=_sandboxed_execution_worker,
            args=(code_solution, test_suite, entry_point, queue)
        )
        process.start()
        process.join(timeout=self.timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return False  # Timed out (e.g., infinite loop)

        if not queue.empty():
            res = queue.get()
            return res == "PASSED"
        return False

    @staticmethod
    def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
        """
        Computes the unbiased Pass@k estimator derived by Chen et al.
        """
        n, c = num_samples, num_correct
        if n - c < k:
            return 1.0
        if c == 0:
            return 0.0

        # Compute 1 - Product_{i=0}^{k-1} (n - c - i) / (n - i)
        prob_no_hits = 1.0
        for i in range(k):
            prob_no_hits *= (n - c - i) / (n - i)

        return 1.0 - prob_no_hits

```

---

## 6. Evaluation Protocols and Bias Guardrails

```text
┌────────────────────────────────────────────────────────────────────────┐
│ DOWNSTREAM EVALUATION GUARDRAILS                                       │
├──────────────────────────────┬─────────────────────────────────────────┤
│ Choice Order Permutation     │ Multiple-choice models exhibit position │
│ Sensitivity                  │ bias (e.g., favoring choice 'A' or 'C').│
│                              │ Average scores across option rotations. │
├──────────────────────────────┼─────────────────────────────────────────┤
│ Few-Shot Example Invariance  │ Prompt template whitespace or exemplar  │
│                              │ ordering can shift accuracy by 3-8%.    │
│                              │ Pin few-shot seeds and prompt templates.│
├──────────────────────────────┼─────────────────────────────────────────┤
│ Contamination Safeguards     │ Enforce strict 13-gram overlap checks   │
│                              │ between training sets and test splits.  │
├──────────────────────────────┼─────────────────────────────────────────┤
│ Sandboxed Isolation          │ Never execute model-generated code      │
│                              │ on the host machine without subprocess  │
│                              │ isolation, cgroups, and strict timeouts.│
└──────────────────────────────┴─────────────────────────────────────────┘

```

### Multiple Choice Option Order Bias Mitigation

Because language models often demonstrate a preference for specific option keys (e.g., selecting choice "A" under uncertainty), accurate benchmark evaluation computes predictions across **all cyclic permutations** of the choices:

$$\text{Permuted Choices for } 4\text{-option QA: } [A, B, C, D] \implies [B, C, D, A] \implies [C, D, A, B] \implies [D, A, B, C]$$

A problem is marked correct only if the model's highest-probability assignment consistently points to the ground-truth semantic entity across all input permutations.
