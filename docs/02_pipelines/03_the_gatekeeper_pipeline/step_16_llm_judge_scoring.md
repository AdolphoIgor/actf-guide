# Step 16: LLM-as-a-Judge Scoring and Automated Capability Arbitration

## 1. The Role of LLM-as-a-Judge in Continuous Training

While deterministic unit tests (Step 15) evaluate closed-form tasks like code AST parsing and arithmetic exact match, they cannot evaluate open-ended instruction following, complex multi-turn reasoning, conversational coherence, or stylistic tone. Human evaluation provides high-quality signal but introduces latency and cost bottlenecks that prevent continuous checkpoint evaluation.

```text
Training Checkpoint (step_t.pt)
                 │
                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 16: AUTOMATED LLM-AS-A-JUDGE SCORING HARNESS                     │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Evaluation Paradigms                                                │
│    • Pairwise Tournament (Candidate θ_cand vs. Production θ_base)     │
│    • Single-Answer Multi-Criteria Pointwise Rubrics (1 to 5 scale)     │
│    • Reference-Guided Factuality & Constraint Verification             │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Systematic Bias Neutralization                                      │
│    • Bidirectional Position Swapping (A/B and B/A trials)              │
│    • Verbosity Penalty Normalization (Length-bias mitigation)          │
│    • Structured Chain-of-Thought (CoT) Critique Generation             │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Statistical Skill Aggregation                                       │
│    • Bradley-Terry Maximum Likelihood Skill Ratings (Elo scaling)      │
│    • Wilson Score Confidence Intervals for Win Rates                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       [ Win Rate >= 52.0% (p < 0.05) ]   [ Win Rate < 50.0% / Regressed ]
       [ Certified for Gate 5 Staging ]   [ Flagged for Alignment Audit  ]

```

**Step 16 (LLM-as-a-Judge Scoring)** deploys a high-capacity judge model (such as Llama-3-70B-Instruct or an external frontier model) to arbitrate candidate quality against production baselines, computing statistical win rates and Bradley-Terry skill ratings across subjective and multi-turn instruction sets.

---

## 2. Evaluation Methodologies: Pairwise vs. Pointwise

```text
┌────────────────────────────────────────────────────────────────────────┐
│ JUDGING PARADIGM COMPARISON                                            │
├──────────────────────────┬───────────────────┬─────────────────────────┤
│ Dimension                │ Pairwise Arena    │ Pointwise Rubric        │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Input Context            │ Prompt + [Resp A, │ Prompt + Single Resp +  │
│                          │  Resp B]          │ Multi-Tier Rubric       │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Output Space             │ Win(A), Win(B),   │ Absolute Score (1 to 5) │
│                          │ Tie, Inconsistent │ per Sub-Dimension       │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Primary Variance Source  │ Position Bias     │ Score Scale Drift /     │
│                          │ (Order effect)    │ Central Tendency Bias   │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Best Operational Use     │ Direct A/B model  │ Longitudinal quality    │
│                          │ tournament ranking│ tracking over epochs    │
└──────────────────────────┴───────────────────┴─────────────────────────┘

```

### A. Pairwise Comparison with Symmetric Order Invariance

When comparing candidate model $\theta_{\text{cand}}$ against baseline $\theta_{\text{base}}$, the judge is vulnerable to **position bias** (favoring Candidate 1 over Candidate 2 regardless of quality).

To neutralize position bias, every prompt $X$ is evaluated across two symmetric trials:

$$\text{Trial 1 (Forward): } J_1 = \text{Judge}\Big( X, \, R_{\text{cand}}, \, R_{\text{base}} \Big)$$

$$\text{Trial 2 (Reverse): } J_2 = \text{Judge}\Big( X, \, R_{\text{base}}, \, R_{\text{cand}} \Big)$$

```text
Decision Mapping across Symmetric Trials:

Trial 1 (Pos 1 = Cand, Pos 2 = Base) │ Trial 2 (Pos 1 = Base, Pos 2 = Cand) │ Final Resolved Outcome
─────────────────────────────────────┼─────────────────────────────────────┼─────────────────────────
Candidate 1 Selected (Prefers Cand)  │ Candidate 2 Selected (Prefers Cand)  │ Consistent WIN (Candidate)
Candidate 2 Selected (Prefers Base)  │ Candidate 1 Selected (Prefers Base)  │ Consistent WIN (Baseline)
Tie Selected                         │ Tie Selected                         │ Consistent TIE
Candidate 1 Selected (Prefers Pos 1) │ Candidate 1 Selected (Prefers Pos 1) │ INCONSISTENT (Position Bias)

```

If the judge selects the first position in both trials, the result is flagged as positional bias and mapped to a **Tie** or excluded from the ranking calculation.

---

### B. Pointwise Multi-Dimensional Rubric Scoring

In pointwise evaluation, candidate responses are scored independently across four core operational axes on a 1-to-5 Likert scale:

1. **Instruction Following (IF):** Strict adherence to all explicit constraints, formatting demands, and negative rules.
2. **Factual Correctness (FC):** Absence of hallucinations, unsupported claims, or logical fallacies.
3. **Clarity & Structure (CS):** Logical paragraph hierarchy, scannable formatting, and grammatical precision.
4. **Conciseness (CO):** Delivering complete answers without conversational filler, repetitive padding, or artificial verbosity.

$$\text{Pointwise Score: } S_{\text{point}} = 0.35 \cdot \text{IF} + 0.35 \cdot \text{FC} + 0.15 \cdot \text{CS} + 0.15 \cdot \text{CO}$$

---

## 3. Mathematical Statistical Modeling

### A. Wilson Score Confidence Interval for Win Rates

Given $N$ pairwise evaluation trials where the candidate model achieves $W$ wins and $T$ ties, the effective win count is:

$$\hat{w} = W + 0.5 \cdot T, \quad \hat{p} = \frac{\hat{w}}{N}$$

To account for small sample sizes and boundary conditions, the **Wilson Score Interval** computes the $95\%$ confidence interval $[p_{\text{lower}}, p_{\text{upper}}]$:

$$p_{\text{lower}}, p_{\text{upper}} = \frac{\hat{p} + \frac{z^2}{2N} \pm z \sqrt{\frac{\hat{p}(1 - \hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}}$$

Where $z = 1.96$ for a $95\%$ confidence level ($\alpha = 0.05$).

- **Promotion Threshold:** The candidate is certified for deployment only if $p_{\text{lower}} \ge 0.50$ (statistically non-inferior to baseline).

---

### B. Bradley-Terry Skill Rating Formulation

Across continuous training runs involving multiple historical checkpoints $M_1, M_2, \dots, M_K$, latent model capability parameters $\gamma_1, \gamma_2, \dots, \gamma_K$ are estimated via maximum likelihood under the Bradley-Terry model:

$$P(M_i \succ M_j) = \frac{\exp(\gamma_i)}{\exp(\gamma_i) + \exp(\gamma_j)}$$

The log-likelihood of all observed pairwise match outcomes across the dataset is maximized:

$$\ln \mathcal{L}(\boldsymbol{\gamma}) = \sum_{i < j} \left[ W_{ij} \ln\left( \frac{\exp(\gamma_i)}{\exp(\gamma_i) + \exp(\gamma_j)} \right) + W_{ji} \ln\left( \frac{\exp(\gamma_j)}{\exp(\gamma_i) + \exp(\gamma_j)} \right) \right]$$

Where $W_{ij}$ is the total number of times Model $i$ defeated Model $j$. Skill ratings are scaled to standard Elo ratings via:

$$R_i = 1000 + 400 \cdot \gamma_i$$

---

## 4. Python Implementation: Production LLM Judge Engine

Below is the standalone implementation of `ProductionLLMJudgeScorer`. It supports structured JSON schema enforcement, bidirectional order swapping, Wilson confidence interval calculation, and Bradley-Terry Elo tracking:

````python
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class PairwiseTrialResult:
    prompt_id: str
    decision_trial_1: str
    decision_trial_2: str
    resolved_winner: str  # "candidate", "baseline", "tie", or "inconsistent"
    is_position_biased: bool
    critique_1: str
    critique_2: str


@dataclass
class JudgeTournamentScorecard:
    total_evaluations: int
    candidate_wins: int
    baseline_wins: int
    ties: int
    inconsistent_trials: int
    raw_win_rate: float
    effective_win_rate: float
    wilson_ci_lower: float
    wilson_ci_upper: float
    certified_promotion: bool


class ProductionLLMJudgeScorer:
    """
    Automated LLM Judge harness executing pairwise symmetric evaluations,
    bias mitigation, and statistical certification.
    """
    def __init__(
        self,
        judge_generate_fn: Callable[[str], str],
        confidence_level_z: float = 1.96,
        min_win_rate_threshold: float = 0.52
    ):
        self.judge_fn = judge_generate_fn
        self.z = confidence_level_z
        self.min_win_rate = min_win_rate_threshold

    def _build_judge_prompt(
        self, prompt: str, resp_1: str, resp_2: str, reference: Optional[str] = None
    ) -> str:
        ref_section = f"\n[Reference Ground Truth]:\n{reference}\n" if reference else ""
        return f"""You are an expert, impartial AI judge evaluating two candidate responses.
Analyze both responses based on:
1. Instruction Following and Negative Constraint Adherence.
2. Factuality, Logical Correctness, and Absence of Hallucinations.
3. Clarity, Scannability, and Organization.
4. Conciseness (penalize irrelevant padding or conversational filler).

[User Prompt]:
{prompt}
{ref_section}
[Candidate Response 1]:
{resp_1}

[Candidate Response 2]:
{resp_2}

Provide a concise, step-by-step critique analyzing the strengths and flaws of both responses.
Then declare the winner as strictly 'Candidate 1', 'Candidate 2', or 'Tie'.

You must respond in valid JSON matching this schema:
{{
  "critique": "<step-by-step reasoning>",
  "winner": "Candidate 1" | "Candidate 2" | "Tie"
}}
"""

    def _parse_judge_json(self, raw_output: str) -> Tuple[str, str]:
        """Extracts critique and winner from model output."""
        try:
            cleaned = raw_output.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            data = json.loads(cleaned)
            winner = data.get("winner", "").strip()
            critique = data.get("critique", "").strip()

            if winner not in ["Candidate 1", "Candidate 2", "Tie"]:
                if "Candidate 1" in winner:
                    winner = "Candidate 1"
                elif "Candidate 2" in winner:
                    winner = "Candidate 2"
                else:
                    winner = "Tie"

            return winner, critique
        except Exception:
            # Fallback regex search
            if re.search(r"Candidate\s*1", raw_output, re.IGNORECASE):
                return "Candidate 1", "Regex parsed fallback"
            elif re.search(r"Candidate\s*2", raw_output, re.IGNORECASE):
                return "Candidate 2", "Regex parsed fallback"
            return "Tie", "Parse failed fallback"

    def evaluate_pair(
        self, prompt_id: str, prompt: str, cand_resp: str, base_resp: str, reference: Optional[str] = None
    ) -> PairwiseTrialResult:
        """
        Executes symmetric forward and reverse trials to eliminate position bias.
        """
        # Trial 1: Candidate = Pos 1, Baseline = Pos 2
        p1 = self._build_judge_prompt(prompt, cand_resp, base_resp, reference)
        raw_1 = self.judge_fn(p1)
        dec_1, crit_1 = self._parse_judge_json(raw_1)

        # Trial 2: Baseline = Pos 1, Candidate = Pos 2
        p2 = self._build_judge_prompt(prompt, base_resp, cand_resp, reference)
        raw_2 = self.judge_fn(p2)
        dec_2, crit_2 = self._parse_judge_json(raw_2)

        # Resolve winner across order swaps
        is_biased = False
        if dec_1 == "Candidate 1" and dec_2 == "Candidate 2":
            resolved = "candidate"
        elif dec_1 == "Candidate 2" and dec_2 == "Candidate 1":
            resolved = "baseline"
        elif dec_1 == "Tie" and dec_2 == "Tie":
            resolved = "tie"
        elif dec_1 == dec_2 and dec_1 in ["Candidate 1", "Candidate 2"]:
            is_biased = True
            resolved = "inconsistent"
        else:
            resolved = "tie"

        return PairwiseTrialResult(
            prompt_id=prompt_id,
            decision_trial_1=dec_1,
            decision_trial_2=dec_2,
            resolved_winner=resolved,
            is_position_biased=is_biased,
            critique_1=crit_1,
            critique_2=crit_2
        )

    def compute_tournament_scorecard(
        self, trial_results: List[PairwiseTrialResult]
    ) -> JudgeTournamentScorecard:
        """
        Aggregates pairwise results and computes Wilson score confidence intervals.
        """
        n_total = len(trial_results)
        w_cand = sum(1 for r in trial_results if r.resolved_winner == "candidate")
        w_base = sum(1 for r in trial_results if r.resolved_winner == "baseline")
        ties = sum(1 for r in trial_results if r.resolved_winner == "tie")
        inconsistents = sum(1 for r in trial_results if r.resolved_winner == "inconsistent")

        # Inconsistent trials are treated as ties for win-rate denominator
        effective_wins = w_cand + 0.5 * (ties + inconsistents)
        effective_p = effective_wins / max(1, n_total)
        raw_win_rate = w_cand / max(1, n_total)

        # Wilson Score Interval
        z = self.z
        denominator = 1.0 + (z ** 2) / n_total
        centre_adjusted_p = effective_p + (z ** 2) / (2.0 * n_total)
        adjusted_std = math.sqrt(
            (effective_p * (1.0 - effective_p) / n_total) + (z ** 2) / (4.0 * (n_total ** 2))
        )

        ci_lower = (centre_adjusted_p - z * adjusted_std) / denominator
        ci_upper = (centre_adjusted_p + z * adjusted_std) / denominator

        # Certification: Lower bound must establish non-inferiority (>= 0.50)
        is_certified = (ci_lower >= 0.50) and (effective_p >= self.min_win_rate)

        return JudgeTournamentScorecard(
            total_evaluations=n_total,
            candidate_wins=w_cand,
            baseline_wins=w_base,
            ties=ties,
            inconsistent_trials=inconsistents,
            raw_win_rate=raw_win_rate,
            effective_win_rate=effective_p,
            wilson_ci_lower=ci_lower,
            wilson_ci_upper=ci_upper,
            certified_promotion=is_certified
        )

````

---

## 5. Certification Matrix and Gating Thresholds

```text
┌────────────────────────────────────────────────────────────────────────┐
│ LLM JUDGE CERTIFICATION STANDARDS                                      │
├──────────────────────────┬───────────────────┬─────────────────────────┤
│ Evaluation Metric        │ Hard Threshold    │ Operational Action      │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Effective Win Rate       │ >= 52.0% vs Base  │ Promoted to Gate 5      │
│ (Candidate vs Baseline)  │                   │ staging pipeline        │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Wilson 95% CI Lower Bound│ >= 50.0%          │ Non-inferiority         │
│                          │                   │ mathematically verified │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Position Bias Rate       │ <= 15.0% of Total │ If exceeded, discard    │
│ (Inconsistent Decisions) │ Trials            │ judge; re-run with CoT  │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Pointwise Factuality (FC)│ >= 4.5 / 5.0      │ Checkpoint rejected if  │
│                          │                   │ factual drift detected  │
└──────────────────────────┴───────────────────┴─────────────────────────┘

```

---

## 6. Diagnostic Failure Matrix

| Failure Symptom                                       | Detection Point                       | Root Cause                                                        | Engineering Remediation                                                    |
| ----------------------------------------------------- | ------------------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **High Position Inconsistency ($> 15\%$)**            | Symmetric Order Pairing               | Judge exhibits strong position bias toward Candidate 1            | Enforce detailed Chain-of-Thought justification prior to decision token    |
| **Verbosity Inflation**                               | Response Length vs. Score Correlation | Baseline generates overly concise answers; judge favors long text | Add explicit length penalty instructions and target token caps             |
| **Self-Enhancement Skew**                             | Model Family Ablation                 | Judge favors models sharing its tokenizer or pre-training data    | Use multi-judge panels with distinct model families (e.g., Llama + Claude) |
| **Wide Confidence Interval ($L_{\text{CI}} < 0.50$)** | Wilson Score Calculation              | Insufficient evaluation samples ($N < 100$)                       | Scale evaluation dataset to $N \ge 250$ balanced prompt items              |
| **Uncalibrated Score Clustering**                     | Pointwise Scoring Distribution        | Judge assigns score 4 or 5 to all inputs without variance         | Anchor prompts with concrete few-shot rubric scoring examples              |
