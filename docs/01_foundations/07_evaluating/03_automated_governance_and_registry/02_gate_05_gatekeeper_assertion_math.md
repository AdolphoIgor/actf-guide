# Gate 5: Gatekeeper Assertion Mathematics and Release Decision Engines

## 1. The Mathematical Purpose of Gate 5

In continuous training and automated fine-tuning pipelines, model promotion cannot rely on raw point estimates or simple averages. A candidate checkpoint $\theta_{\text{cand}}$ that achieves an average accuracy of $74.2\%$ on a benchmark suite compared to the production baseline's $\theta_{\text{base}}$ of $73.8\%$ might still represent a statistically insignificant improvement or hide critical regressions on high-stakes sub-distributions.

```text
Continuous Training Pipeline Checkpoint Pool
                     │
                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GATE 5: GATEKEEPER AUTOMATED ARBITER                                  │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Zero-Tolerance Invariant Checks (Hard Thresholds)                  │
│    • Syntax AST Conformance == 100%                                    │
│    • EOS / Turn-Delimitation Compliance >= 99.0%                       │
│    • PII Leakage / Safety Violation Rate == 0.0%                       │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Statistical Non-Inferiority & Superiority Testing                   │
│    • Paired McNemar's Test on Categorical Benchmark Items (p < 0.05)   │
│    • Wilcoxon Signed-Rank Test on Sample Token-Loss Distributions     │
│    • Bootstrap Confidence Intervals for Metric Deltas (Δ >= -δ_margin) │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Probabilistic Calibration & Uncertainty Boundaries                  │
│    • Expected Calibration Error (ECE) <= ECE_max                       │
│    • Brier Score Divergence <= BS_max                                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       [ PASS: Promoted to Registry ]  [ FAIL: Checkpoint Quarantined ]

```

**Gate 5 (The Gatekeeper Engine)** is the automated mathematical firewall that evaluates whether a candidate checkpoint statistically dominates the current production model across safety, capability, calibration, and regression metrics before promotion to the production model registry.

---

## 2. Statistical Significance & Non-Inferiority Formulations

### A. McNemar's Test for Paired Categorical Benchmarks

When evaluating discrete correctness across $N$ matched test items between the baseline model $\theta_{\text{base}}$ and candidate model $\theta_{\text{cand}}$ (e.g., in MMLU, GSM8K, ARC-Challenge), samples are not independent. Predictions are paired per item, forming a $2 \times 2$ contingency matrix:

```text
Contingency Matrix on N Matched Test Items:

                            Candidate Model (θ_cand)
                            Correct (1)       Incorrect (0)
                        ┌─────────────────┬─────────────────┐
          Correct (1)   │      n_11       │      n_10       │
Baseline                │ (Both Correct)  │  (Regression)   │
Model (θ_base)          ├─────────────────┼─────────────────┤
          Incorrect (0) │      n_01       │      n_00       │
                        │  (Improvement)  │ (Both Incorrect)│
                        └─────────────────┴─────────────────┘

```

Where:

* $n_{10}$ is the number of **regressions** (baseline correct, candidate incorrect).
* $n_{01}$ is the number of **improvements** (baseline incorrect, candidate correct).

To test the null hypothesis $H_0: P(\text{Regression}) = P(\text{Improvement})$ against $H_1: P(\text{Improvement}) > P(\text{Regression})$, the continuity-corrected **McNemar Chi-Squared Statistic** is evaluated:

$$\chi^2 = \frac{\left( \vert{}n_{01} - n_{10}\vert{} - 1 \right)^2}{n_{01} + n_{10}}$$

For small discordant counts ($n_{01} + n_{10} < 25$), the exact two-tailed Binomial test is computed:

$$p\text{-value} = 2 \sum_{k=n_{01}}^{n_{01} + n_{10}} \binom{n_{01} + n_{10}}{k} \left(\frac{1}{2}\right)^{n_{01} + n_{10}}$$

* **Gatekeeper Rejection Criterion:** If $n_{10} > n_{01}$ and $p < 0.05$, the candidate model introduces a statistically significant capability regression and is rejected.

---

### B. Non-Inferiority Margin Testing ($\delta_{\text{margin}}$)

In multi-task continuous learning, improving domain capabilities (e.g., Code Synthesis) must not degrade general reasoning (e.g., MMLU) beyond an acceptable non-inferiority margin $\delta \ge 0$.

Let $\mu_{\text{cand}}$ and $\mu_{\text{base}}$ be the true population means of the evaluation metric.

$$\text{Hypothesis Formulation: } \begin{cases} H_0: \mu_{\text{cand}} - \mu_{\text{base}} \le -\delta & \text{(Candidate is inferior)} \\ H_1: \mu_{\text{cand}} - \mu_{\text{base}} > -\delta & \text{(Candidate is non-inferior)} \end{cases}$$

```text
Non-Inferiority Confidence Interval Geometry:

  Metric Delta (Δ = μ_cand - μ_base)
  ◄─────────────────────────┬─────────────────────────┬─────────────────────────►
                          -δ_margin                  0.0 (Parity)
                            │
  Case A: REJECT (Inferior) ├──[====■====]───────────┼─────────────────────────►
                            │  (Lower bound crosses -δ)
                            │
  Case B: PASS (Non-Inferior)────────────────────────┼─────[====■====]─────────►
                            │                        │  (Lower bound > -δ)

```

Using the **Empirical Bootstrap**, the system draws $B = 10,000$ resamples of size $N$ with replacement from the paired sample delta vector $D = Y_{\text{cand}} - Y_{\text{base}}$, constructing the two-sided $(1 - \alpha)$ percentile confidence interval $[L_{\alpha/2}, U_{1 - \alpha/2}]$.

$$\text{Non-Inferiority Condition: } L_{0.025} > -\delta_{\text{margin}}$$

Where $\delta_{\text{margin}}$ is set per domain (typically $\delta_{\text{margin}} = 0.005$, or $0.5\%$).

---

### C. Wilcoxon Signed-Rank Test for Continuous Token-Loss Distributions

To test whether candidate token-level cross-entropy loss distributions stochastic dominate baseline loss distributions without assuming normal distributions:

Let $d_i = \ell_{\text{base}}(x_i) - \ell_{\text{cand}}(x_i)$ be the paired loss difference on document $i$.

1. Exclude all pairs where $d_i = 0$.
2. Rank the absolute differences $\vert{}d_i\vert{}$ in ascending order: $\text{Rank}(\vert{}d_i\vert{})$.
3. Compute the signed rank sums:

$$W^+ = \sum_{d_i > 0} \text{Rank}(\vert{}d_i\vert{}), \quad W^- = \sum_{d_i < 0} \text{Rank}(\vert{}d_i\vert{})$$

$$W = \min(W^+, W^-)$$

For large evaluation splits ($N > 50$), the standardized $Z$-score evaluates to:

$$Z = \frac{W - \frac{N(N+1)}{4}}{\sqrt{\frac{N(N+1)(2N+1)}{24}}}$$

$$\text{Condition for Significant Loss Reduction: } Z < -1.96 \quad (p < 0.05)$$

---

## 3. Probabilistic Calibration Assertions

A high-performing model that is uncalibrated (e.g., highly overconfident when incorrect) introduces downstream safety risks. Gate 5 audits the model's posterior probability calibration.

```text
Reliability Calibration Curve (Expected Accuracy vs. Confidence):

Accuracy (acc)
 1.0 ┌─────────────────────────────────────────────────────────────┐
     │                                                     _ - - █ │ (Perfect Calibration)
 0.8 │                                             _ - - █   ▄     │
     │                                     _ - - █   ▄             │
 0.6 │                             _ - - █   ▄                     │ █ = Perfect Diagonal
     │                     _ - - █   ▄                             │ ▄ = Overconfident Model
 0.4 │             _ - - █   ▄                                     │
     │     _ - - █   ▄                                             │
 0.2 │ _ █                                                         │
     └─────────────────────────────────────────────────────────────┴────► Confidence (conf)
     0.0        0.2         0.4         0.6         0.8         1.0

```

### A. Expected Calibration Error (ECE)

The predicted token confidence values $\hat{p} = \max_v P(y=v \mid x)$ are partitioned into $M$ equally spaced empirical bins $B_1, B_2, \dots, B_M \subset (0, 1]$ (typically $M = 15$).

For each bin $B_m$:

$$\text{acc}(B_m) = \frac{1}{\vert{}B_m\vert{}} \sum_{i \in B_m} \mathbb{I}(\hat{y}_i = y_i)$$

$$\text{conf}(B_m) = \frac{1}{\vert{}B_m\vert{}} \sum_{i \in B_m} \hat{p}_i$$

The **Expected Calibration Error (ECE)** is the sample-weighted average difference between confidence and accuracy:

$$\text{ECE} = \sum_{m=1}^M \frac{\vert{}B_m\vert{}}{N} \Big\vert{} \text{acc}(B_m) - \text{conf}(B_m) \Big\vert{}$$

$$\text{Gatekeeper Calibration Condition: } \text{ECE}(\theta_{\text{cand}}) \le \text{ECE}_{\max} \quad (\text{Standard: } \text{ECE}_{\max} = 0.06)$$

---

### B. Multi-Class Brier Score

Measures the mean squared error of predicted probability vectors against true one-hot target vectors:

$$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^V \left( P_\theta(y_i = k \mid x_i) - \mathbf{y}_{i, k} \right)^2$$

Where $\mathbf{y}_{i, k} = \mathbb{I}(y_i = k)$.

---

## 4. Multi-Objective Decision Function: Hard Gates vs. Soft Composites

The Gatekeeper evaluates candidate promotion via a hierarchical decision policy:

```text
Hierarchical Decision Policy:

                    [ Candidate Checkpoint Evaluation ]
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │ STAGE 1: Hard Zero-Tolerance Assertions │
                │  • AST Valid == 100%                    │
                │  • EOS Compliance >= 99%                │
                │  • PII Violations == 0                  │
                └────────────────────┬────────────────────┘
                                     │
                       ┌─────────────┴─────────────┐
                       │ ALL PASS                  │ ANY FAIL
                       ▼                           ▼
        ┌─────────────────────────────┐   [ IMMEDIATE REJECTION ]
        │ STAGE 2: Statistical Delta  │   (Quarantine Checkpoint)
        │ • McNemar Regressions p>=0.05│
        │ • Non-inferiority CI >= -δ  │
        │ • ECE <= 0.06               │
        └──────────────┬──────────────┘
                       │
         ┌─────────────┴─────────────┐
         │ ALL PASS                  │ ANY FAIL
         ▼                           ▼
  [ PROMOTE TO REGISTRY ]   [ REJECT CHECKPOINT ]

```

### The Stage 1 Hard Boolean Assertion Vector ($\mathbf{H}$)

$$\mathbf{H} = \begin{bmatrix} \mathbb{I}(\text{AST\_Syntax\_Rate} == 1.0) \\ \mathbb{I}(\text{EOS\_Stop\_Compliance} \ge 0.99) \\ \mathbb{I}(\text{PII\_Leak\_Count} == 0) \\ \mathbb{I}(\text{KV\_Cache\_Logit\_Parity} < 10^{-3}) \end{bmatrix}$$

$$\text{Stage 1 Condition: } \prod_{k=1}^{\vert{}\mathbf{H}\vert{}} \mathbf{H}_k == 1$$

---

## 5. Python Implementation: Production Gatekeeper Assertion Engine

Below is the standalone implementation of the `GatekeeperAssertionEngine`, executing McNemar significance tests, Bootstrap confidence intervals, ECE calibration auditing, and hard safety gates:

```python
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class GatekeeperVerdict:
    passed: bool
    rejection_reasons: List[str]
    hard_gates_passed: bool
    statistical_gates_passed: bool
    metrics_summary: Dict[str, Any]


class GatekeeperAssertionEngine:
    """
    Automated decision arbiter executing Gate 5 statistical hypothesis tests,
    safety assertions, and calibration checks.
    """
    def __init__(
        self,
        non_inferiority_margin: float = 0.005,  # Max allowable regression: 0.5%
        significance_alpha: float = 0.05,
        max_allowable_ece: float = 0.06,
        bootstrap_resamples: int = 10000
    ):
        self.delta_margin = non_inferiority_margin
        self.alpha = significance_alpha
        self.max_ece = max_allowable_ece
        self.n_boot = bootstrap_resamples

    # =====================================================================
    # 1. STATISTICAL TESTS
    # =====================================================================
    @staticmethod
    def compute_mcnemar_test(
        baseline_correct: np.ndarray,
        candidate_correct: np.ndarray
    ) -> Tuple[float, float, int, int]:
        """
        Computes continuity-corrected McNemar test on paired binary outcomes.
        Returns: (chi2_stat, p_value, n_regressions, n_improvements)
        """
        assert len(baseline_correct) == len(candidate_correct), "Array length mismatch"
        
        # Contingency counts
        n_10 = int(np.sum((baseline_correct == 1) & (candidate_correct == 0))) # Regressions
        n_01 = int(np.sum((baseline_correct == 0) & (candidate_correct == 1))) # Improvements
        
        total_discordant = n_10 + n_01
        if total_discordant == 0:
            return 0.0, 1.0, 0, 0

        # Exact Binomial test for small discordant sample counts
        if total_discordant < 25:
            # Two-tailed binomial test under H0: p = 0.5
            k = min(n_10, n_01)
            p_val = 2.0 * stats.binom.cdf(k, total_discordant, 0.5)
            p_val = min(1.0, p_val)
            return 0.0, p_val, n_10, n_01

        # Continuity-corrected Chi-Square
        chi2_stat = ((abs(n_01 - n_10) - 1.0) ** 2) / total_discordant
        p_val = 1.0 - stats.chi2.cdf(chi2_stat, df=1)
        return float(chi2_stat), float(p_val), n_10, n_01

    def compute_bootstrap_ci(
        self,
        baseline_scores: np.ndarray,
        candidate_scores: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Computes paired bootstrap confidence interval for (candidate - baseline).
        Returns: (mean_delta, lower_ci, upper_ci)
        """
        deltas = candidate_scores - baseline_scores
        mean_delta = float(np.mean(deltas))

        # Bootstrap resampling
        rng = np.random.default_rng(seed=1337)
        indices = rng.integers(0, len(deltas), size=(self.n_boot, len(deltas)))
        bootstrap_means = np.mean(deltas[indices], axis=1)

        lower_ci = float(np.percentile(bootstrap_means, (self.alpha / 2.0) * 100))
        upper_ci = float(np.percentile(bootstrap_means, (1.0 - self.alpha / 2.0) * 100))
        return mean_delta, lower_ci, upper_ci

    # =====================================================================
    # 2. CALIBRATION METRICS
    # =====================================================================
    @staticmethod
    def compute_expected_calibration_error(
        confidences: np.ndarray,
        correctness: np.ndarray,
        num_bins: int = 15
    ) -> float:
        """
        Computes Expected Calibration Error (ECE) across confidence bins.
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        n_samples = len(confidences)

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_size = np.sum(in_bin)

            if bin_size > 0:
                bin_acc = np.mean(correctness[in_bin])
                bin_conf = np.mean(confidences[in_bin])
                ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

        return float(ece)

    # =====================================================================
    # 3. MASTER DECISION PIPELINE
    # =====================================================================
    def evaluate_checkpoint(
        self,
        # Hard Safety Inputs
        ast_syntax_pass_rate: float,
        eos_stop_compliance_rate: float,
        pii_violation_count: int,
        kv_cache_max_delta: float,
        # Benchmark Evaluation Pairs: Dict[benchmark_name, (baseline_bools, cand_bools)]
        benchmark_binary_runs: Dict[str, Tuple[np.ndarray, np.ndarray]],
        # Calibration Inputs
        candidate_confidences: np.ndarray,
        candidate_correctness: np.ndarray
    ) -> GatekeeperVerdict:
        rejections = []

        # -------------------------------------------------------------
        # STAGE 1: HARD SAFETY GATES (Zero-Tolerance)
        # -------------------------------------------------------------
        if ast_syntax_pass_rate < 1.0:
            rejections.append(f"Hard Gate Failed: Code AST Syntax Pass Rate = {ast_syntax_pass_rate * 100:.2f}% (Expected 100%)")

        if eos_stop_compliance_rate < 0.99:
            rejections.append(f"Hard Gate Failed: EOS Stop Compliance = {eos_stop_compliance_rate * 100:.2f}% (Expected >= 99%)")

        if pii_violation_count > 0:
            rejections.append(f"Hard Gate Failed: PII Leaks Detected = {pii_violation_count} (Expected 0)")

        if kv_cache_max_delta >= 1e-3:
            rejections.append(f"Hard Gate Failed: KV Cache Logit Delta = {kv_cache_max_delta:.6e} >= 1e-3 tolerance")

        hard_gates_passed = len(rejections) == 0

        # -------------------------------------------------------------
        # STAGE 2: CALIBRATION GATE
        # -------------------------------------------------------------
        ece_score = self.compute_expected_calibration_error(
            candidate_confidences, candidate_correctness
        )
        if ece_score > self.max_ece:
            rejections.append(f"Calibration Failed: ECE = {ece_score:.4f} > max allowable {self.max_ece:.4f}")

        # -------------------------------------------------------------
        # STAGE 3: STATISTICAL BENCHMARK SIGNIFICANCE & NON-INFERIORITY
        # -------------------------------------------------------------
        benchmark_summary = {}

        for name, (base_arr, cand_arr) in benchmark_binary_runs.items():
            base_acc = float(np.mean(base_arr))
            cand_acc = float(np.mean(cand_arr))
            
            # 1. McNemar Significance Test
            chi2, p_val, n_reg, n_imp = self.compute_mcnemar_test(base_arr, cand_arr)

            # 2. Bootstrap Confidence Interval
            mean_d, low_ci, upp_ci = self.compute_bootstrap_ci(base_arr.astype(float), cand_arr.astype(float))

            benchmark_summary[name] = {
                "base_acc": base_acc,
                "cand_acc": cand_acc,
                "delta": mean_d,
                "ci_lower": low_ci,
                "ci_upper": upp_ci,
                "mcnemar_p_value": p_val,
                "regressions": n_reg,
                "improvements": n_imp
            }

            # Evaluation Assertion 1: Reject if candidate is statistically significantly worse
            if n_reg > n_imp and p_val < self.alpha:
                rejections.append(
                    f"Statistical Regression on {name}: Regressions ({n_reg}) > Improvements ({n_imp}) with p={p_val:.4f} < {self.alpha}"
                )

            # Evaluation Assertion 2: Reject if Lower CI violates Non-Inferiority margin
            if low_ci < -self.delta_margin:
                rejections.append(
                    f"Non-Inferiority Violated on {name}: Lower CI {low_ci:.4f} < -{self.delta_margin:.4f}"
                )

        statistical_gates_passed = (len(rejections) == 0) if hard_gates_passed else False
        verdict_passed = hard_gates_passed and (len(rejections) == 0)

        return GatekeeperVerdict(
            passed=verdict_passed,
            rejection_reasons=rejections,
            hard_gates_passed=hard_gates_passed,
            statistical_gates_passed=statistical_gates_passed,
            metrics_summary={
                "ece": ece_score,
                "benchmarks": benchmark_summary,
                "ast_syntax_pass_rate": ast_syntax_pass_rate,
                "eos_stop_compliance_rate": eos_stop_compliance_rate
            }
        )

```

---

## 6. Gate 5 Production Assertion Matrix

| Assertion Dimension | Target Metric | Mathematical Condition | Threshold | Action on Failure |
| --- | --- | --- | --- | --- |
| **Code Syntax** | AST Parse Success Rate | $\text{Rate}_{\text{AST}} == 1.0$ | $100\%$ | **Hard Reject**; quarantine checkpoint |
| **Turn Termination** | EOS Delimiter Emission | $\text{Rate}_{\text{EOS}} \ge 0.99$ | $\ge 99.0\%$ | **Hard Reject**; runaway generation |
| **Data Safety** | PII Entity Count | $\sum \text{Leaks} == 0$ | $0$ leaks | **Hard Reject**; audit training data |
| **Cache Parity** | Max Absolute Logit Delta | $\Vert{}Z_{\text{naive}} - Z_{\text{cached}}\Vert{}_\infty < \epsilon$ | $< 10^{-3}$ | **Hard Reject**; inspect RoPE offsets |
| **Benchmark Regression** | McNemar Paired Chi-Square | $p \ge 0.05 \lor n_{\text{improvements}} \ge n_{\text{regressions}}$ | $p < 0.05$ | **Statistical Reject**; capability loss |
| **Non-Inferiority** | Bootstrap Lower $95\%$ CI | $L_{0.025}(\Delta) > -\delta_{\text{margin}}$ | $\ge -0.5\%$ | **Statistical Reject**; margin breached |
| **Probability Calibration** | Expected Calibration Error | $\text{ECE} \le \text{ECE}_{\max}$ | $\le 0.06$ | **Calibration Reject**; temperature scaling |