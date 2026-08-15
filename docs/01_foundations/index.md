# 01 - Foundations

## Executive Overview
The **Foundations** section establishes the core architectural paradigms required for enterprise-grade LLMOps. It covers the shift from manual fine-tuning to autonomous continuous training, the decoupling of data ingestion from GPU compute, and the storage principles governing the Medallion Data Lake.

---

## Chapter Roadmap

| Chapter | Core Focus |
| :--- | :--- |
| **[01 Enterprise LLMOps](01_the_enterprise_llmops.md)** | Full stack integration across warehouse, distributed ETL, orchestrator, training, and serving layers. |
| **[02 Continuous Training Role](02_the_continuous_training_role.md)** | Defining the automated trigger, training execution, evaluation gatekeeper, and production failure modes. |
| **[03 Decoupled Workflow](03_decoupled_workflow.md)** | Scheduled batch, volumetric, and drift-driven ingestion trigger patterns that protect training compute. |
| **[04 Medallion Architecture](04_the_medallion_architecture.md)** | Storage lifecycle management across Bronze (raw), Silver (cleaned), and Gold (tokenized) layers. |