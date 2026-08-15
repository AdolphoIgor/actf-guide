# Decoupled Data Workflow and Model Training Workflow

In a mature enterprise pipeline, the data workflow and the model training workflow are decoupled. They do not run as one single massive script. They run as independent micro-pipelines communicating via events or schedules.

Depending on the company's business model and how fast their data changes, enterprises use one of three core ingestion patterns to trigger the data workflow.

---

## 1. The Scheduled Batch Pattern (Time-Based)

This is the most common pattern for traditional enterprise tasks (e.g., fine-tuning a customer service model on the previous week's support logs).

* **How it works:** Apache Airflow runs a scheduled cron job (e.g., every Sunday at 12:00 AM). Airflow wakes up Ray, extracts the last days of raw log data from the datasource, cleans it, versions it with DVC, and saves it as a clean dataset file in an S3 bucket.
* **Does it trigger training?** Usually, no. Generating a new data chunk doesn't automatically mean spending thousands of dollars on GPUs to train a new model. The pipeline stops here, waiting for a separate training evaluation trigger.

---

## 2. The Event-Driven Volumetric Pattern (Data-Size Based)

This pattern is used when data arrives unpredictably, and you only want to generate a new chunk when you have enough new information to make a difference.

* **How it works:** You set up a sensor or a database trigger in a datasource. The database keeps a counter of new, unprocessed training samples.
* **The Rule:** "When unprocessed records are big enough, this triggers the Airflow data preparation DAG." Airflow fires up Ray to process that specific batch, registers it, and resets the counter.

---

## 3. The On-Demand Observability Pattern (Drift-Based)

This is the holy grail. The data workflow is triggered directly by real-world performance failures.

* **How it works:** Your live production model is running. An observability tool (like Langfuse) detects that the model's accuracy on production data has dropped below 85% or that users are downvoting a specific answer type.
* **The Rule:** Langfuse fires a webhook alert. Airflow catches this webhook and instantly triggers the entire end-to-end loop: It commands Ray to grab the specific failed production logs, formats them into a "correction dataset," appends them to the historical training data, and immediately passes the torch to the Axolotl training engine to fix the model.

---

## Summary of the Workflow Split

To keep things efficient, low-stress, and cost-effective, the gold standard rule is:

* **The Data Pipeline (Ray + Airflow)** runs frequently—either on a schedule or on an event basis—to keep datasets fresh and structured in cold storage.
* **The Training Pipeline (Axolotl + W&B)** is a protected, expensive gate. It is only pulled when the compiled data reaches a critical mass, a new base model drops, or production monitoring reports a performance emergency.
* **The Automated Gatekeeper Pipeline (Promptfoo + MLFlow)** is an objective, post-flight evaluation checkpoint. It automatically benchmarks candidate models against domain-specific gold standards, safety guardrails, and active production baselines—promoting winning checkpoints to the deployment registry while instantly blocking regressions from touching production.