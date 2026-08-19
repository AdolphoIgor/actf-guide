# The Medallion Architecture

In mature, enterprise-scale organizations, cloud object storage (like AWS S3 or Google Cloud Storage) does not use traditional folders. Because object storage is a flat key-value system, "folders" are just string prefixes in a file's path (e.g., `bucket/prefix/file.parquet`). To scale this to billions of files without grinding query engines to a halt, the industry gold standard relies on Hive-Style Partitioning laid across the Medallion Architecture, strictly optimized for Storage Lifecycle Management.

---

## 1. The Bronze Layer: Raw landing

* **What it is:** The core definition of what "Bronze" means in enterprise data engineering is a state of data fidelity. Either the Cloud-Native Data Warehouse Route (BigQuery / Snowflake) or an Object Storage Data Lake (S3 / MinIO) can hold the Bronze Layer. Usually, the raw text is extracted from RDBMS or from raw files (`.pdf`, `.doc`, etc.) via Spark and then dumped into S3/MinIO as a `.parquet` file.
* **The Rule:** No destructive changes. You only append metadata (e.g., `ingestion_timestamp`, `source_system`). If the text in the datasource contains typos, HTML tags, or duplicated rows, it must land in Bronze with those exact flaws intact.
* **Why?** If you change your mind later about how you want to filter your data, or if your tokenization script has a bug, you need to be able to replay the entire pipeline from scratch without querying your production Postgres database again.

---

## 2. The Silver Layer: Cleaned & Structured Text (Ray Data territory)

* **What it is:** The result of your text cleaning, deduplication, PII removal, and structural alignment.
* **The Transformation:** Your pipeline reads the Bronze files, runs your processing logic (via Ray Data), and writes a new set of Parquet files back to a separate `silver/` directory in (S3 / MinIO).
* **The State:** At this layer, the data is pristine, human-readable text perfectly formatted into your instruction-tuning schemas (e.g., `{"instruction": "...", "output": "..."}`).

---

## 3. The Gold Layer: Tokenized Tensors (Weights-Ready)

* **What it is:** The tokenized integer arrays, attention masks, and packaged binary files (like Hugging Face `.arrow` cache chunks or packed tensors) that Axolotl reads directly.
* **The Transformation:** Ray reads the clean Silver text, applies the model's exact tokenizer, applies context window padding/packing, and writes the output to the `gold/` directory.

---

```text
s3://company-ai-datalake/
|
|-- bronze/                                    # RAW LANDING LAYER
|   |-- oracle_crm/                            # Source system descriptor
|   |   L-- customer_chats/                    # Entity descriptor
|   |       |-- year=2026/month=06/day=09/     # Hive-style date partitions
|   |       |   |-- chunk_091214_raw.jsonl     # Flat text append-only files
|   |       |   L-- chunk_091530_raw.jsonl
|   |
|   L-- box_storage/
|       L-- technical_pdfs/
|           L-- year=2026/month=06/day=01/
|               L-- extracted_text_01.jsonl    # Flat text pulled by Ray
|
|-- silver/                                    # CLEANED & STANDARDIZED LAYER
|   L-- support_tickets/
|       L-- standardized_train_pairs/          # Formatted schemas
|           L-- year=2026/month=06/
|               |-- data_v1_001.snappy.parquet # Columnar, clean, compressed
|               L-- data_v1_002.snappy.parquet
|
L-- gold/                                      # MODEL-READY & EVAL FEATURE STORE
    L-- model_qwen_1_7b/
        L-- evaluation_golden_set/
            L-- eval_version_2_4.parquet       # High-value benchmark data

```

---

# Data Storage Lifecycle Management (The Cost Control)

You cannot store every raw file in standard object storage forever; enterprise data bills would spin out of control. Organizations configure Automated Lifecycle Policies directly within the cloud providers (like AWS S3 Lifecycle Rules or GCS Object Lifecycle Management) to transition data based on age and layer value.

## The Tiering Policy Strategy

| Data Layer | Data State | Lifecycle Action Rule | The Business Reason |
| --- | --- | --- | --- |
| **Bronze (Raw)** | Unoptimized Parquet tables. | Move to Deep Archive after 30 days.<br>

<br>Delete after 365 days. | Once Bronze data is parsed and transformed into Silver, it is just a backup. It is rarely accessed. Storing heavy text files in hot storage wastes money. We push it to the cheapest cold tier. |
| **Silver (Cleaned)** | Cleaned, deduplicated, schema-enforced Parquet tables. | Move to Infrequent Access (Cool) after 90 days.<br>

<br>Retain Indefinitely. | This is your corporate System of Record. It must be preserved forever so you can always rebuild your models or extract new features. We put it in a "Cool" tier because we don't query it daily, but it must be completely intact for massive retraining runs. |
| **Gold (Model-Ready)** | Curated evaluation sets, specific training datasets. | Retain Indefinitely in Standard (Hot) Storage. | Gold data is highly refined, relatively small in file size, and queried constantly by active training loops and evaluation engines. It stays in high-performance hot storage for immediate access. |

By pairing this structural layout with automated cloud lifecycle purging rules, a mature organization ensures its data assets remain completely traceable for the Continuous Training pipeline while keeping operational costs contained.

---

## The Live Production Layout

When you look at Bronze and (Silver/Gold) files inside an enterprise storage system, the architecture is structured precisely like this.

Depending on the company's chosen cloud vendor, this storage landing zone is placed in one of three standard locations:

* **AWS (Amazon Web Services):** An Amazon S3 Bucket (e.g., `s3://enterprise-data-lake/bronze/documents_raw/`).
* **GCP (Google Cloud Platform):** A Google Cloud Storage (GCS) Bucket (e.g., `gs://enterprise-data-lake/bronze/documents_raw/`).
* **Azure:** An Azure Data Lake Storage (ADLS) Gen2 Account.

---

## Why is MinIO a Great Alternative?

* **The S3 API Standard:** MinIO offers full API parity with the AWS S3 API. This means you can often migrate your applications with zero code changes.
* **Multi-Cloud/Hybrid Flexibility:** It can be run on-premises (bare metal, VMs), in private clouds, or within public clouds (AWS, GCP, Azure) to eliminate vendor lock-in and centralize your storage framework.
* **Zero Egress Fees:** By hosting MinIO on your own hardware, you eliminate the data transfer and request costs that mount up when pulling data out of public cloud object stores.
* **High Performance:** MinIO is specifically optimized for throughput-heavy applications, making it a favorite for modern analytics, databases, and AI/ML data lakes.