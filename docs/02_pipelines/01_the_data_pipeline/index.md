# The Data Pipeline

## Executive Overview
The **Data Pipeline** handles all data operations upstream of model training. Operating primarily on distributed CPU infrastructure (Ray Data / Apache Spark), it ingests raw records, enforces closed-loop guardrails, executes multi-phase filtering and deduplication, and prepares model-ready feature shards.

---

## Section Sub-Modules

| Module | Focus Area |
| :--- | :--- |
| **[01 Data Writing](02_data_writing/index.md)** | Warehouse vs. object storage ingestion patterns, initial historical loads, and CDC watermarking. |
| **[02 Raw Data Ingestion](03_raw_data_ingestion/index.md)** | Ray Data memory-throttling mechanics, Apache Arrow 3-buffer layout, and zero-copy shared memory. |
| **[03 Filtering & Tokenization](04_filtering_and_tokenization/index.md)** | 3-phase branching DAG: shared ingestion, Track A (prose), Track B (code), and sequence packing. |