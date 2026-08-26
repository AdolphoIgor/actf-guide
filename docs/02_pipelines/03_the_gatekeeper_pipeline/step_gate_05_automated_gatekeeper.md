# Gate 5: Automated Gatekeeper Decision Engine and Production Promotion Pipeline

## 1. The Gatekeeper as the Final Verification Firewall

Before any candidate model checkpoint ($\theta_{\text{cand}}$) is promoted to the enterprise model registry or deployed to serve live user traffic, it must pass **Gate 5 (The Automated Gatekeeper)**.

Gate 5 synthesizes data from all upstream training stages, empirical benchmark evaluations, qualitative probing suites, and inference parity tests into an automated, statistically grounded release decision.

```text
Staged Model Checkpoint (STAGED Lifecycle State)
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GATE 5: MULTI-STAGE AUTOMATED GATEKEEPER PIPELINE                      │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Zero-Tolerance Invariant Checks (Hard Boolean Constraints)          │
│    • 100% Python AST syntax parsing validity on code generations       │
│    • >= 99.0% EOS delimiter emission (Zero runaway generations)        │
│    • KV-Cache logit parity error ||Z_naive - Z_cached||_inf < 1e-3     │
│    • Zero PII leakage and safety violation instances                   │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Statistical Non-Inferiority & Superiority Testing                   │
│    • Paired McNemar Chi-Square tests on categorical benchmark splits   │
│    • Empirical Bootstrap Confidence Intervals (Δ_score >= -δ_margin)   │
│    • Wilcoxon Signed-Rank Test on validation token loss distributions  │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Automated LLM-as-a-Judge Capability Arbitration                     │
│    • Symmetric pairwise win-rate: Wilson 95% CI lower bound >= 0.50   │
├────────────────────────────────────────────────────────────────────────┤
│ 4. Hardware Serving SLA & Calibration Assertions                       │
│    • Expected Calibration Error (ECE) <= 0.06                          │
│    • Inter-Token Latency (ITL) and Peak VRAM fit serving SLA           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │ ALL ASSERTIONS PASS           │ ANY ASSERTION FAILS
                    ▼                               ▼
       [ CERTIFIED PROMOTION ]             [ QUARANTINE ISOLATION ]
       • Update @champion Alias            • Revoke Serving Permissions
       • Sign Manifest Envelope            • Emit Signed Failure Receipt
       • Route Live Serving Traffic        • Alert Engineering Fleet

```

---

## 2. Multi-Tier Decision Architecture

Gate 5 evaluates candidate viability through a hierarchical four-tier evaluation battery. Failure at any tier results in immediate short-circuiting and artifact quarantine:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ GATEKEEPER TIERED EVALUATION TAXONOMY                                  │
├──────────────────────────┬───────────────────┬─────────────────────────┤
│ Evaluation Tier          │ Target Dimensions │ Evaluation Mechanism    │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Tier 1: Hard Invariants  │ Syntax, EOS,      │ Zero-tolerance boolean  │
│         (Safety & Spec)  │ Cache, PII        │ deterministic assertions│
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Tier 2: Benchmark Delta  │ MMLU, GSM8K,      │ Paired McNemar Tests &  │
│         (Non-Inferiority)│ HumanEval, ARC    │ Bootstrap 95% CIs       │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Tier 3: Subjective Arena │ Open Dialogue,    │ Symmetric LLM Judge     │
│         (LLM-as-a-Judge) │ Complex CoT, Tone │ with Wilson CI Bounds   │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Tier 4: Operational SLA  │ Latency, Memory,  │ ECE binning & hardware  │
│         (Serving Health) │ ECE Calibration   │ profiling monitors      │
└──────────────────────────┴───────────────────┴─────────────────────────┘

```

---

## 3. Mathematical Decision Rules and Statistical Invariants

### A. Stage 1: Hard Boolean Invariant Vector ($\mathbf{H}$)

A candidate checkpoint must satisfy all deterministic invariants:

$$\mathbf{H} = \begin{bmatrix} \mathbb{I}(\text{AST\_Syntax\_Pass\_Rate} == 1.0) \\ \mathbb{I}(\text{EOS\_Stop\_Compliance} \ge 0.99) \\ \mathbb{I}(\Vert{}Z_{\text{naive}} - Z_{\text{cached}}\Vert{}_\infty < 10^{-3}) \\ \mathbb{I}(\text{PII\_Leak\_Count} == 0) \end{bmatrix}, \quad \text{Condition: } \prod_{k=1}^4 \mathbf{H}_k == 1$$

---

### B. Stage 2: Paired Statistical Non-Inferiority

For each categorical benchmark $B$, let $n_{10}$ be regressions (baseline correct, candidate wrong) and $n_{01}$ be improvements (baseline wrong, candidate correct). The continuity-corrected McNemar Chi-Square test determines significance:

$$\chi^2 = \frac{\left( \vert{}n_{01} - n_{10}\vert{} - 1 \right)^2}{n_{01} + n_{10}}$$

$$\text{Rejection Condition: } n_{10} > n_{01} \quad \land \quad P(\chi^2 \mid \text{df}=1) < 0.05$$

Additionally, for continuous capability metrics, the empirical bootstrap lower $95\%$ confidence bound $L_{0.025}(\Delta)$ must not breach the non-inferiority margin $\delta_{\text{margin}} = 0.005$ ($0.5\%$):

$$L_{0.025}\left( \text{Score}_{\text{cand}} - \text{Score}_{\text{base}} \right) \ge -\delta_{\text{margin}}$$

---

### C. Stage 3: LLM Judge Pairwise Tournament Arbitration

Under symmetric position-swapped trials across $N$ instruction prompts, let $\hat{p}$ be the effective win rate:

$$\hat{w} = W_{\text{cand}} + 0.5 \cdot T_{\text{tie}}, \quad \hat{p} = \frac{\hat{w}}{N}$$

The Wilson score $95\%$ confidence interval lower bound $p_{\text{lower}}$ must confirm non-inferiority against the baseline:

$$p_{\text{lower}} = \frac{\hat{p} + \frac{z^2}{2N} - z \sqrt{\frac{\hat{p}(1 - \hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}} \ge 0.50 \quad (z = 1.96)$$

---

### D. Stage 4: Probability Calibration (ECE)

The model's output confidence across $M = 15$ bins must satisfy:

$$\text{ECE} = \sum_{m=1}^M \frac{\vert{}B_m\vert{}}{N} \Big\vert{} \text{acc}(B_m) - \text{conf}(B_m) \Big\vert{} \le 0.06$$

---

## 4. Python Implementation: Master Gatekeeper Decision Engine

Below is the standalone implementation of `AutomatedGatekeeperEngine`. It ingests test outputs, executes the statistical batteries, enforces safety invariants, signs audit receipts, and drives registry promotions:

```python
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class GatekeeperReceipt:
    artifact_id: str
    verdict: str  # "PROMOTED" or "QUARANTINED"
    timestamp_utc: str
    tier1_invariants_passed: bool
    tier2_statistics_passed: bool
    tier3_judge_passed: bool
    tier4_operational_passed: bool
    rejection_reasons: List[str]
    scorecard: Dict[str, Any]


class AutomatedGatekeeperEngine:
    """
    Automated Gate 5 production promotion arbiter executing multi-tier
    invariant checks, statistical non-inferiority audits, and signing release receipts.
    """
    def __init__(
        self,
        non_inferiority_margin: float = 0.005,
        significance_alpha: float = 0.05,
        max_allowable_ece: float = 0.06,
        min_judge_win_rate: float = 0.52,
        bootstrap_resamples: int = 10000
    ):
        self.delta_margin = non_inferiority_margin
        self.alpha = significance_alpha
        self.max_ece = max_allowable_ece
        self.min_judge_win = min_judge_win_rate
        self.n_boot = bootstrap_resamples

    # =====================================================================
    # STATISTICAL EVALUATORS
    # =====================================================================
    @staticmethod
    def _evaluate_mcnemar(
        base_correct: np.ndarray, cand_correct: np.ndarray
    ) -> Tuple[float, float, int, int]:
        n_10 = int(np.sum((base_correct == 1) & (cand_correct == 0)))  # Regressions
        n_01 = int(np.sum((base_correct == 0) & (cand_correct == 1)))  # Improvements
        total = n_10 + n_01

        if total == 0:
            return 0.0, 1.0, 0, 0
        if total < 25:
            p_val = 2.0 * float(stats.binom.cdf(min(n_10, n_01), total, 0.5))
            return 0.0, min(1.0, p_val), n_10, n_01

        chi2 = float(((abs(n_01 - n_10) - 1.0) ** 2) / total)
        p_val = float(1.0 - stats.chi2.cdf(chi2, df=1))
        return chi2, p_val, n_10, n_01

    def _evaluate_bootstrap_ci(
        self, base_arr: np.ndarray, cand_arr: np.ndarray
    ) -> Tuple[float, float, float]:
        deltas = cand_arr.astype(float) - base_arr.astype(float)
        mean_d = float(np.mean(deltas))

        rng = np.random.default_rng(seed=1337)
        idx = rng.integers(0, len(deltas), size=(self.n_boot, len(deltas)))
        boot_means = np.mean(deltas[idx], axis=1)

        low = float(np.percentile(boot_means, (self.alpha / 2.0) * 100))
        upp = float(np.percentile(boot_means, (1.0 - self.alpha / 2.0) * 100))
        return mean_d, low, upp

    @staticmethod
    def _compute_wilson_ci(
        wins: int, ties: int, total: int, z: float = 1.96
    ) -> Tuple[float, float, float]:
        eff_wins = wins + 0.5 * ties
        p = eff_wins / max(1, total)
        denom = 1.0 + (z ** 2) / total
        centre = p + (z ** 2) / (2.0 * total)
        spread = z * math.sqrt((p * (1.0 - p) / total) + (z ** 2) / (4.0 * (total ** 2)))
        return p, (centre - spread) / denom, (centre + spread) / denom

    # =====================================================================
    # MASTER DECISION HARNESS
    # =====================================================================
    def arbitrate_release(
        self,
        artifact_id: str,
        # Tier 1: Invariants
        ast_syntax_rate: float,
        eos_compliance_rate: float,
        kv_cache_delta: float,
        pii_leaks: int,
        # Tier 2: Benchmarks Dict[name, (base_bools, cand_bools)]
        benchmarks: Dict[str, Tuple[np.ndarray, np.ndarray]],
        # Tier 3: Judge Outcomes (cand_wins, base_wins, ties, total)
        judge_results: Tuple[int, int, int, int],
        # Tier 4: Operational Metrics
        ece_score: float,
        itl_ms: float,
        max_itl_sla_ms: float,
        peak_vram_gb: float,
        vram_limit_gb: float
    ) -> GatekeeperReceipt:
        rejections: List[str] = []
        scorecard: Dict[str, Any] = {}

        # -------------------------------------------------------------
        # TIER 1: ZERO-TOLERANCE HARD INVARIANTS
        # -------------------------------------------------------------
        t1_passed = True
        if ast_syntax_rate < 1.0:
            t1_passed = False
            rejections.append(f"Tier 1 Failed: Code AST Syntax Pass Rate = {ast_syntax_rate*100:.2f}% (Expected 100%)")
        if eos_compliance_rate < 0.99:
            t1_passed = False
            rejections.append(f"Tier 1 Failed: EOS Compliance = {eos_compliance_rate*100:.2f}% (Expected >= 99%)")
        if kv_cache_delta >= 1e-3:
            t1_passed = False
            rejections.append(f"Tier 1 Failed: KV-Cache Logit Delta = {kv_cache_delta:.6e} >= 1e-3")
        if pii_leaks > 0:
            t1_passed = False
            rejections.append(f"Tier 1 Failed: PII Leaks Detected = {pii_leaks}")

        # -------------------------------------------------------------
        # TIER 2: BENCHMARK NON-INFERIORITY
        # -------------------------------------------------------------
        t2_passed = True
        bench_summary = {}
        for b_name, (base_arr, cand_arr) in benchmarks.items():
            chi2, p_val, n_reg, n_imp = self._evaluate_mcnemar(base_arr, cand_arr)
            mean_d, low_ci, upp_ci = self._evaluate_bootstrap_ci(base_arr, cand_arr)

            bench_summary[b_name] = {
                "delta": mean_d,
                "ci_lower": low_ci,
                "mcnemar_p": p_val,
                "regressions": n_reg,
                "improvements": n_imp
            }

            if n_reg > n_imp and p_val < self.alpha:
                t2_passed = False
                rejections.append(f"Tier 2 Failed: Significant regression on {b_name} (p={p_val:.4f} < {self.alpha})")
            if low_ci < -self.delta_margin:
                t2_passed = False
                rejections.append(f"Tier 2 Failed: Non-inferiority breached on {b_name} (Lower CI {low_ci:.4f} < -{self.delta_margin})")

        scorecard["benchmarks"] = bench_summary

        # -------------------------------------------------------------
        # TIER 3: LLM-AS-A-JUDGE TOURNAMENT ARBITRATION
        # -------------------------------------------------------------
        t3_passed = True
        c_wins, b_wins, ties, n_judge = judge_results
        eff_wr, wilson_low, wilson_upp = self._compute_wilson_ci(c_wins, ties, n_judge)

        scorecard["llm_judge"] = {
            "effective_win_rate": eff_wr,
            "wilson_ci_lower": wilson_low,
            "wilson_ci_upper": wilson_upp
        }

        if wilson_low < 0.50:
            t3_passed = False
            rejections.append(f"Tier 3 Failed: Judge Wilson CI lower bound {wilson_low:.4f} < 0.50")
        if eff_wr < self.min_judge_win:
            t3_passed = False
            rejections.append(f"Tier 3 Failed: Judge Win Rate {eff_wr*100:.1f}% < {self.min_judge_win*100:.1f}% threshold")

        # -------------------------------------------------------------
        # TIER 4: OPERATIONAL SLA & PROBABILISTIC CALIBRATION
        # -------------------------------------------------------------
        t4_passed = True
        if ece_score > self.max_ece:
            t4_passed = False
            rejections.append(f"Tier 4 Failed: Expected Calibration Error {ece_score:.4f} > {self.max_ece}")
        if itl_ms > max_itl_sla_ms:
            t4_passed = False
            rejections.append(f"Tier 4 Failed: Inter-Token Latency {itl_ms:.2f}ms exceeds SLA limit {max_itl_sla_ms:.2f}ms")
        if peak_vram_gb > vram_limit_gb:
            t4_passed = False
            rejections.append(f"Tier 4 Failed: Peak VRAM {peak_vram_gb:.1f}GB exceeds limit {vram_limit_gb:.1f}GB")

        scorecard["operational"] = {
            "ece": ece_score,
            "itl_ms": itl_ms,
            "peak_vram_gb": peak_vram_gb
        }

        overall_pass = t1_passed and t2_passed and t3_passed and t4_passed
        verdict = "PROMOTED" if overall_pass else "QUARANTINED"

        receipt = GatekeeperReceipt(
            artifact_id=artifact_id,
            verdict=verdict,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            tier1_invariants_passed=t1_passed,
            tier2_statistics_passed=t2_passed,
            tier3_judge_passed=t3_passed,
            tier4_operational_passed=t4_passed,
            rejection_reasons=rejections,
            scorecard=scorecard
        )

        return receipt

    def save_signed_receipt(self, receipt: GatekeeperReceipt, output_path: Path):
        """Persists the signed audit receipt sidecar to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(receipt), f, indent=2)
        print(f"Gatekeeper Receipt saved -> Verdict: {receipt.verdict} at {output_path}")

```

---

## 5. Automated Promotion and Rollback Protocol

When the Gatekeeper issues a verdict, the deployment pipeline executes automated promotion or rollback operations:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ GATEKEEPER AUTOMATED ORCHESTRATION CONTRACT                            │
├────────────────────────────────────────────────────────────────────────┤
│ ACTION ON VERDICT: "PROMOTED"                                          │
│  1. Ingest model into MLflow Model Registry as @champion.              │
│  2. Archive previous production version to @archived.                  │
│  3. Spin up Canary serving deployment (1% -> 10% -> 50% -> 100%).      │
│  4. Emit Webhook Notification to MLOps Slack / Monitoring Sinks.       │
├────────────────────────────────────────────────────────────────────────┤
│ ACTION ON VERDICT: "QUARANTINED"                                       │
│  1. Move physical artifact to `/quarantine/{artifact_id}/`.            │
│  2. Tag MLflow version as @quarantined with rejection reasons.         │
│  3. Lock write access (chmod 444) and write `quarantine_receipt.json`. │
│  4. Cordon training shard responsible for data drift.                  │
│  5. Revert serving fleet traffic 100% to active @champion.             │
└────────────────────────────────────────────────────────────────────────┘

```

---

## 6. Gate 5 Production Decision Matrix

| Evaluation Tier         | Metric / Check       | Mathematical Threshold                                                | Failure Action                              |
| ----------------------- | -------------------- | --------------------------------------------------------------------- | ------------------------------------------- |
| **Tier 1 (Invariants)** | AST Code Parse Rate  | $\text{Rate} == 1.0$ ($100\%$)                                        | **Hard Quarantine**; reject release         |
| **Tier 1 (Invariants)** | EOS Stop Compliance  | $\text{Rate} \ge 0.99$ ($99\%$)                                       | **Hard Quarantine**; runaway generation     |
| **Tier 1 (Invariants)** | KV-Cache Parity      | $\Vert{}Z_{\text{naive}} - Z_{\text{cached}}\Vert{}_\infty < 10^{-3}$ | **Hard Quarantine**; cache divergence       |
| **Tier 1 (Invariants)** | PII Leak Count       | $\text{Count} == 0$                                                   | **Hard Quarantine**; security breach        |
| **Tier 2 (Benchmarks)** | McNemar Significance | $p \ge 0.05 \lor n_{\text{imp}} \ge n_{\text{reg}}$                   | **Statistical Quarantine**; regression      |
| **Tier 2 (Benchmarks)** | Bootstrap Lower CI   | $L_{0.025}(\Delta) \ge -0.005$                                        | **Statistical Quarantine**; non-inferiority |
| **Tier 3 (LLM Judge)**  | Wilson Lower Bound   | $p_{\text{lower}} \ge 0.50$                                           | **Judge Quarantine**; preference loss       |
| **Tier 3 (LLM Judge)**  | Effective Win Rate   | $\hat{p} \ge 0.52$ ($52\%$)                                           | **Judge Quarantine**; insufficient margin   |
| **Tier 4 (Operations)** | Expected Calibration | $\text{ECE} \le 0.06$                                                 | **Calibration Quarantine**; uncalibrated    |
| **Tier 4 (Operations)** | Inter-Token Latency  | $\text{ITL} \le \text{SLA Limit}$                                     | **Operational Quarantine**; latency breach  |
