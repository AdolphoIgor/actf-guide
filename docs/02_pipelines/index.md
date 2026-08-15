# 02 - Pipelines

## Executive Overview
The **Pipelines** section details the three execution loops comprising the automated continuous training lifecycle: data ingestion and transformation, distributed multi-GPU training execution, and post-flight gatekeeper evaluation.

---

## Pipeline Architecture

```text
[ Data Pipeline ] ---> [ Training Pipeline ] ---> [ Gatekeeper Pipeline ]