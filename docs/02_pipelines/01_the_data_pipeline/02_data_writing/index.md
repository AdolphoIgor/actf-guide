# Data Writing & Ingestion Mechanics

## Executive Overview
This section details how raw data lands in the Bronze layer of the Medallion Data Lake. It compares Cloud-Native Data Warehouses against Object Storage Data Lakes and outlines strategies for massive historical initial loads and incremental CDC watermarking.

---

## Key Contents
* **[Data Writing Architecture](01_data_writing.md):** Pattern A vs. Pattern B storage routes, Apache Spark parallel database extraction, Ray Data document ingestion, and CDC/High-Watermark incrementals.