# Raw Data Ingestion & Memory Architecture

## Executive Overview

This section explains how Ray Data acts as a memory-throttling firewall during raw data ingestion. It covers lazy loading evaluation, block sizing mechanics, Apache Arrow 3-buffer string memory layouts, and OS shared-memory (`/dev/shm`) zero-copy mechanics.

---

## Key Contents

- **[Raw Data Ingestion](01_raw_data_ingestion.md):** Metadata planning vs. streaming execution, target block sizing, Plasma store zero-copy deserialization, and PyArrow streaming C++ JSONL ingestion.
