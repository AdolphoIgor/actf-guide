# Filtering and Curation Engine

## Executive Overview

The **Filtering & Curation** module forms the core data-cleaning engine. It takes raw ingested Bronze records and executes a 3-phase branching DAG to eliminate noise, drop duplicates, mask PII, verify syntax rules, decontaminate evaluation sets, and sign off universal Silver datasets at **Gate 2**.

---

## Architectural Workflow

```text
                       [ Shared Ingestion Trunk ]
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
            [ Track A: Prose ]          [ Track B: Code ]
                     │                           │
                     └─────────────┬─────────────┘
                                   ▼
                  [ Phase 3: Reconvergence & Safety ]
                                   │
                                   ▼
                  [ Gate 2: Pre-Tokenization Gate ]

```

---

## Sub-Chapter Roadmap

### Phase 1: Shared Ingestion Trunk

- **[Step 01: Normalization](https://www.google.com/search?q=phase_01_shared_ingestion/step_01_normalization.md):** Unicode NFKC reassembly and hyphenation repair.
- **[Step 02: Boilerplate Stripping](https://www.google.com/search?q=phase_01_shared_ingestion/step_02_boilerplate_stripping.md):** Purging navigation headers, footers, and cookie policies.
- **[Step 03: Exact Deduplication](https://www.google.com/search?q=phase_01_shared_ingestion/step_03_exact_deduplication.md):** 64-bit MurmurHash3 exact deduplication via in-memory RocksDB.
- **[Step 04: Metadata Inspection & Routing](https://www.google.com/search?q=phase_01_shared_ingestion/step_04_metadata_inspection_and_routing.md):** Arrow header analysis and language routing.

### Phase 2: Domain-Specific Processing

- **Track A (Prose):** [Step 5a Heuristics](https://www.google.com/search?q=phase_02_domain_specific_processing/track_a_standard_natural_language_branch/step_5a_standard_heuristics.md), [Step 6a MinHash LSH](https://www.google.com/search?q=phase_02_domain_specific_processing/track_a_standard_natural_language_branch/step_6a_minhash_fuzzy_deduplication.md), [Step 7a CQF Scoring](https://www.google.com/search?q=phase_02_domain_specific_processing/track_a_standard_natural_language_branch/step_7a_natural_language_cqf.md), [Step 8a FastText Language ID](https://www.google.com/search?q=phase_02_domain_specific_processing/track_a_standard_natural_language_branch/step_8a_fasttext_language_id.md).
- **Track B (Code):** [Step 5b Disambiguation](https://www.google.com/search?q=phase_02_domain_specific_processing/track_b_specialized_domain_branch/step_5b_code_and_syntax_disambiguation.md), [Step 6b Code AST Dedup](https://www.google.com/search?q=phase_02_domain_specific_processing/track_b_specialized_domain_branch/step_6b_code_specific_minhash_ast_deduplication.md), [Step 7b Domain Quality](https://www.google.com/search?q=phase_02_domain_specific_processing/track_b_specialized_domain_branch/step_7b_domain_quality_check.md), [Step 8b Syntax Verification](https://www.google.com/search?q=phase_02_domain_specific_processing/track_b_specialized_domain_branch/step_8b_syntax_verification.md).

### Phase 3: Reconvergence & Safety

- **[Step 09: Safety & PII Redaction](https://www.google.com/search?q=phase_03_reconvergence_and_safety/step_09_safety_and_pii_redaction.md):** In-cluster toxicity filtering and NER masking.
- **[Step 10: Cross-Dataset Decontamination](https://www.google.com/search?q=phase_03_reconvergence_and_safety/step_10_cross_dataset_decontamination.md):** N-gram overlap sanitization against evaluation benchmarks.
- **[Gate 2: Pre-Tokenization Gate](https://www.google.com/search?q=phase_03_reconvergence_and_safety/step_gate_02_pre_tokenization_gate.md):** UTF-8 byte validation, schema completeness, and Silver sign-off.
