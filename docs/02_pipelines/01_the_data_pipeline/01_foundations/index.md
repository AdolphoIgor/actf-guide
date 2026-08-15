# Data Pipeline Foundations

## Executive Overview
This section establishes the mathematical security perimeter and error-handling behavior of the data pipeline. It defines the in-DAG training gates and inference serving boundaries required to eliminate silent failures before GPU compute is provisioned.

---

## Key Contents
* **[Closed-Loop Guardrails](01_closed_loop_guardrails.md):** Step-by-step DAG execution, 7 safety gates (Ingestion, Pre-Tokenization, Data Leakage, Pre-Flight Tensor, Gatekeeper, Serving, Drift Monitor), and the global circuit breaker specification.