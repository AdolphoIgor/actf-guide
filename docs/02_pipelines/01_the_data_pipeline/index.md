# The Data Pipeline

## Executive Overview
The **Data Pipeline** operates upstream of model training on distributed CPU infrastructure (Ray Data / Apache Spark). It ingests raw records, validates physical and schema boundaries at Gate 1, executes multi-phase filtering, deduplication, and PII masking, and certifies universal, model-agnostic text at Gate 2.

---

## Data Pipeline Sub-Modules

| Sub-Module | Focus Area | Key Artifacts |
| :--- | :--- | :--- |
| **[01 Data Writing](01_data_writing/index.md)** | Warehouse vs. Object Storage ingestion, initial historical loads, and CDC watermarking. | Raw Bronze files |
| **[02 Raw Data Ingestion](02_raw_data_ingestion/index.md)** | Ray Data memory-throttling, Apache Arrow 3-buffer layout, and **Gate 1 (Ingestion Gate)**. | Verified Bronze partitions |
| **[03 Filtering & Curation](03_filtering_and_curation/index.md)** | 3-phase branching DAG (normalization, exact/fuzzy dedup, AST syntax, PII, decontamination) and **Gate 2 (Pre-Tokenization Gate)**. | Universal Silver Parquet |