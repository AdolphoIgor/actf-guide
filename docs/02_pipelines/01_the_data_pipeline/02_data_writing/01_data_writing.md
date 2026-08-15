# Data Writing

## The Two Distinct Enterprise Patterns (Bronze data)

### Pattern A: The Cloud-Native Data Warehouse Route (BigQuery / Snowflake)

* **Context:** If an enterprise uses BigQuery as its primary data platform.
* **How it works:** Raw, uncleaned application logs or database dumps are streamed directly into BigQuery tables. These tables are treated as immutable and append-only.
* **The Reality:** When Ray Data streams from BigQuery in this scenario, it is reading the raw, unpolished "Bronze data" directly from the warehouse's internal storage blocks.

### Pattern B: The Object Storage Data Lake Route (Spark + MinIO Setup)

* **Context:** If an enterprise wants to keep cloud costs low or operate on-premises, they use object storage (like MinIO or AWS S3) as its data lake.
* **How it works:** Because querying a production datasource directly for machine learning training would slow down the app and crash the database, Spark is used to quickly clone the raw data and dump it safely into MinIO as flat files.
* **The Reality:** When Ray Data streams in this scenario, it reads those raw flat files (`.parquet`) directly from your MinIO bucket.

---

## Why Pattern B (MinIO/S3) is the Major Choice for LLMOps

While both patterns exist, Pattern B is heavily preferred by Automated CT/Fine-Tuning Specialists for a massive reason: **Decoupling and Portability**.

* **Storage Costs:** Storing terabytes of raw, unstructured text files in MinIO or standard AWS S3 is significantly cheaper than storing that same raw data inside an active BigQuery or Snowflake data warehouse database engine.
* **Compute Localization:** When it is time to train your model using Axolotl, the training cluster needs fast, direct access to the files. It is vastly faster and more secure to have your pipeline pull files directly from an S3-compatible bucket (like MinIO) right next to the GPU, rather than constantly authenticating and querying a remote data warehouse over an external API connection.

> **Note:** Even if an enterprise uses BigQuery as its primary data platform, it is still possible to adopt Pattern B; The way it’s done is by periodically feeding parquet files with data coming from Data Warehouses, Relational Databases, Document Databases, Files, etc.

---

## Initial Historical Load

This is the scariest part of building a Data Lake because it is vastly different from the micro-incremental updates we do periodically. If you try to run your normal daily incremental pipeline on the entire legacy data at once, your infrastructure will completely melt down. Let's look at the mature enterprise blueprint for tackling a dual-source ingestion: a structured relational database (RDBMS) paired with an unstructured mountain of PDFs and Word files.

### 1. The Strategy: The Medallion "Bronze" Landing Zone

In a mature organization, you follow the Medallion Architecture. The objective of the initial load is to get everything into the Bronze Layer (The Raw Landing Zone) inside your cloud object storage (S3/GCS) exactly as it looks at the source. No cleaning, no tokenization, no processing yet.

### 2. Ingesting the RDBMS (The Structured History)

If your RDBMS has too many rows of historical logs, chats, or tabular text, probably, a single worker cannot extract them.

* **The Tooling Setup:**
* **The Manager:** Apache Airflow
* **The Muscle:** Apache Spark (via Databricks, AWS EMR, or a temporary cloud cluster) or a distributed extraction tool like Airbyte.


* **How Airflow Orchestrates the Initial Load:**
* **Parallel Partitions:** Airflow tells Spark to connect to the RDBMS. Spark looks at the database's primary key index (e.g., Row ID 1 to 500,000,000) and divides the extraction into logical chunks.
* **The Extraction Force:** Spark fires up 50 or 100 concurrent worker threads. Worker 1 pulls rows 1 to 5,000,000; Worker 2 pulls rows 5,000,001 to 10,000,000, and so on.
* **The Bronze Drop:** These workers stream the data out of the RDBMS and write it directly into your S3 (or MinIO) Data Lake as compressed, raw Apache Parquet files, organized by date folders (`/bronze/rdbms_raw/year=2026/`).



### 3. Ingesting the Document Storage (PDFs & Word Files)

This is a totally different beast. You have terabytes of unstructured text trapped inside unstructured files sitting in a network file system or an enterprise storage bucket.

* **The Tooling Setup:**
* **The Manager:** Apache Airflow
* **The Muscle:** Ray Data. *Note: While Spark is the king of tabular databases, Ray Data is the undisputed modern gold standard for unstructured AI data ingestion. It handles files, pixels, and text layouts significantly better than Spark.*


* **How Airflow Orchestrates the Initial Load:**
* **The Manifest Task:** Airflow runs a lightweight script that scans the storage directory and generates a "Manifest File"—a massive text list of every single file path that needs to be processed.
* **Distributed File Ingestion:** Airflow calls Ray Data and passes it the manifest list. Ray spins up a CPU-heavy cluster. In an enterprise environment, this cluster is usually hosted on Kubernetes (K8s) using an open-source tool called KubeRay or managed via platforms like Anyscale (the enterprise platform built by the creators of Ray). This allows scaling the task by adding more CPUs (virtual machines), and when it finishes, all VMs are deleted, so it is an on-demand solution.
* **Chunking the Documents:** Ray Data reads the raw files in parallel. Ray workers run binary text-extraction libraries (like `pypdf`, `pdfplumber`, or `docx`) to read the files, strip the text layout, and convert the unstructured files into a flat text schema.
* **The Unstructured Bronze Drop:** Ray outputs these extracted text blocks into a uniform Parquet format and saves them to the storage landing zone (`/bronze/documents_raw/`), which is a logical folder directory located inside a cloud-native Object Storage Service. It is not a standard hard drive or a shared network folder on a specific server.



### 4. How the "State" is Handover to Incremental Mode

Once the initial load is complete and your S3 bucket contains the complete, raw history of your RDBMS and Documents, Airflow records the Watermark.

* **For the RDBMS:** Airflow checks the maximum primary key or `updated_at` timestamp that was written to the Bronze layer (e.g., Max ID: 500,000,000). It saves this value to its internal database state.
* **For the Documents:** Airflow records the timestamp of the initial load run.

---

## Transitioning to Periodical Loadings

Now, your heavy Spark and Ray initial load clusters shut down completely. The next time Airflow wakes up for its routine incremental check, it queries the RDBMS with an explicit filter (e.g., `WHERE id > 500,000,000`). For the documents, it uses an event trigger (like an AWS S3 Bucket Notification) that fires an execution only when a user uploads a new PDF. To keep up with the last time a snapshot was made so you don't load everything every time (called Incremental Loading), mature organizations choose between two core software designs depending on how fast their data changes.

### Approach A: Change Data Capture (CDC) — Maturity Standard

If your legacy systems are transactional databases (like Oracle or PostgreSQL) handling live business operations, you cannot use timestamp queries because they slow down the database. Instead, you use CDC.

* **The Software:** Debezium or Airbyte.
* **How it works:** These tools don't query your tables. They read the database's internal transaction log (like PostgreSQL's WAL or MySQL's binlog) in real-time. Every time a new customer ticket is inserted or an old BLOB text column is updated, the CDC engine instantly captures that delta log, formats it, and streams it to your Data Lake landing zone.
* **Airflow's Role:** Airflow simply keeps the CDC connectors monitored or triggers micro-batches to catalog the incoming data streams.

### Approach B: High-Watermark Querying (Watermarking) — The Batch Standard

If your legacy systems are updated in predictable daily batches or you are querying fixed endpoints (like an Elasticsearch cluster or static file dumps), you use Watermarking.

* **The Software:** Airbyte, Fivetran, or a lightweight Python worker running DLT (data load tool).
* **How it works:** The database maintains a tracking table containing a state value—the "High-Watermark" (usually a timestamp column like `updated_at` or an incremental tracking ID).
* **The Integration Flow:**
1. Airflow triggers an Ingestion Tool (e.g., `dlt` or Airbyte) and passes it the last recorded timestamp from the previous successful run.
2. The Ingestion Tool executes a query explicitly requesting data where `updated_at > last_successful_timestamp`.
3. The tool streams those specific rows directly into the Data Lake as immutable, raw `.parquet` files.
4. Upon completion, the tool writes the new maximum timestamp back to the state tracker, and Airflow marks the job as complete.