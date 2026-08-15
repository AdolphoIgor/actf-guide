# The Gatekeeper Pipeline

## Executive Overview
The **Gatekeeper Pipeline** provides automated post-flight evaluation for candidate model checkpoints. It executes benchmark test suites, LLM-as-a-Judge setups, and historical regression checks before promoting approved checkpoints to the MLflow Model Registry.

---

## Module Scope
* **Automated Evaluation:** Deterministic code checks, benchmark evaluation (FinQA, GSM8K, HumanEval), and Ragas/Braintrust scoring.
* **Governance & Promotion:** MLflow Registry asset promotion (`Approved-For-Staging`) and circuit breaker rollback hooks.