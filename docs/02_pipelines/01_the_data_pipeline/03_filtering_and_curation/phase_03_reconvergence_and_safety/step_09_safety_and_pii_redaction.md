# Step 9: Safety & PII Redaction

## 1. Core Objective

Executing as the entry stage of **Phase 3 - Reconvergence & Tokenization**, Step 9 re-converges surviving data streams from **Track A (Natural Language Prose)** and **Track B (Code & Technical Domains)** into a unified processing trunk.

Sequenced directly after domain-specific processing (**Step 8a** and **Step 8b**) and prior to **Step 10 (Cross-Dataset Decontamination)**, Step 9 enforces enterprise safety and compliance guardrails through a split execution strategy: Named Entity Recognition (NER) combined with regular expression pattern masking for Personal Identifiable Information (PII) redaction, alongside multi-vector probabilistic sequence classification for toxicity detection. Step 9 guarantees that data re-entering the shared convergence pipeline contains zero un-masked sensitive identifiers or toxic content before cross-dataset decontamination (**Step 10**) or sequence packing (**Step 12**).

---

## 2. Theoretical & Architectural Justification

When processing millions of corporate documents, web crawls, or technical repositories, datasets routinely contain sensitive Personal Identifiable Information (PII)—such as Social Security Numbers, credit card numbers, phone numbers, email addresses, and personal names—as well as toxic prose, hate speech, or harassment.

Feeding un-sanitized data into downstream training stages introduces three critical failure modes:

### A. PII Memorization & Privacy Leakage

Deep neural networks act as efficient memorizers of high-entropy sequence outliers (such as credit card numbers, personal addresses, or private API keys). Un-redacted PII in pre-training data causes the model to memorize private customer details, exposing the enterprise to privacy violations and data leakage during inference (e.g., via prefix-prompting attacks).

### B. The Limitations of Pure Regex or Pure NER

Alphanumeric PII strings with fixed grammars (such as credit cards or email addresses) are captured efficiently using regular expressions. However, contextual entities (such as human names, physical addresses, or corporate identities) lack static string patterns and cannot be safely identified via regex alone. Conversely, using dense NER models for deterministic alphanumeric strings introduces unnecessary computational overhead. Step 9 resolves this by combining high-speed regex pattern matching with token-level NER sequence classification.

### C. Safety Contamination & Compliance Auditability

Pre-training on un-filtered toxic content (hate speech, severe profanity, harassment) degrades model safety alignment, causing harmful or biased text generation during inference. Furthermore, enterprise governance mandates that toxic records must not be silently deleted; instead, they must be isolated into a secure compliance quarantine path to maintain an auditable paper trail for risk assessment without contaminating the training pipeline.

---

## 3. Theoretical Execution Mechanics

Step 9 processes re-converged data streams through a two-part safety and redaction pipeline:

```text
       [ Track A Data Stream ]            [ Track B Data Stream ]
                  │                                  │
                  └─────────────────┬────────────────┘
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │ Part A: Hybrid PII Detection & Tag Redaction         │
         │         (Regex Patterns + Token-Level NER)           │
         └──────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │ Part B: Multi-Vector Toxicity Classification         │
         └──────────────────────────┬───────────────────────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   │                                 │
                   ▼ (Score ≤ 0.40)                  ▼ (Score > 0.40)
        ┌──────────────────────┐          ┌──────────────────────┐
        │ Pass: Advance to     │          │ Fail: Isolated       │
        │ Step 10              │          │ Compliance Quarantine│
        └──────────────────────┘          └──────────────────────┘

```

### Part A: Hybrid PII Extraction & Redaction

1. **Vectorized Pattern Matching (Structured PII):** Standardized alphanumeric sequences (credit card numbers, Social Security Numbers, emails, IP addresses) are identified using compiled regular expression pattern matchers across memory arrays.
2. **Token-Level NER Sequence Classification (Contextual PII):** Sentences pass through a localized Named Entity Recognition token classifier that outputs BIO (Beginning-Inside-Outside) entity boundary tags:

$$\text{Entity Tag} \in \{\text{B-PER}, \text{I-PER}, \text{B-LOC}, \text{I-LOC}, \text{B-ORG}, \text{I-ORG}\}$$

1. **Immutable Entity Tag Swap:** Matched text arrays are sliced, replacing identified private identifiers with immutable generic entity tags (e.g., `[REDACTED_EMAIL]`, `[REDACTED_NAME]`, `[REDACTED_ADDRESS]`).

### Part B: Localized Toxicity Filtering & Quarantine Routing

1. **Multi-Vector Toxicity Scoring:** Text blocks pass through a high-throughput sequence classification model that calculates probability scores across multiple toxicity dimensions (hate speech, sexual content, harassment, severe profanity):

$$\mathbf{P}_{\text{toxic}}(x) = \left[ P_{\text{hate}}, P_{\text{harassment}}, P_{\text{sexual}}, P_{\text{profanity}} \right] \in [0.0, 1.0]^4$$

1. **Threshold Gatekeeping:** If the maximum score across any toxicity vector crosses the enterprise threshold, the document fails the safety gate:

$$\text{Safety Boundary}: \max \left( \mathbf{P}_{\text{toxic}}(x) \right) > 0.40$$

1. **Compliance Quarantine Isolation:** Records flagged for extreme toxicity are redirected to a secure, write-only compliance quarantine path. This maintains an immutable paper trail for corporate governance teams to audit data source liabilities, completely isolated from the training infrastructure.

---

## 4. Safety & Redaction Strategy Matrix

| Safety Hazard                                   | Detection Mechanism                | Theoretical Signature / Threshold                     | Pipeline Action                                               | Downstream Impact in Phase 3                                       |
| ----------------------------------------------- | ---------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Structured PII** (Emails, SSNs, Credit Cards) | Compiled Regular Expression Engine | Deterministic alphanumeric grammar match              | **Masked:** Replaced with immutable tag (`[REDACTED_EMAIL]`). | Prevents verbatim memorization of structured identifiers.          |
| **Contextual PII** (Human Names, Addresses)     | Token-Level NER Transformer Model  | Entity boundary match ($\text{B-PER}, \text{I-PER}$)  | **Masked:** Replaced with immutable tag (`[REDACTED_NAME]`).  | Eliminates privacy leakage while preserving grammatical structure. |
| **Benign Prose / Technical Code**               | Sequence Toxicity Classifier       | $\max\left(\mathbf{P}_{\text{toxic}}\right) \le 0.40$ | **Retained:** Advances to Step 10.                            | Re-converged into Phase 3 for decontamination.                     |
| **Toxic Content** (Hate Speech, Harassment)     | Multi-Vector Toxicity Classifier   | $\max\left(\mathbf{P}_{\text{toxic}}\right) > 0.40$   | **Quarantined:** Routed to compliance audit directory.        | Protects model alignment and maintains an auditable risk trail.    |

---

## 5. Algorithmic Principles & Theoretical Tooling

- **Hybrid PII Redaction Frameworks:** Enterprise engines combining compiled regular expression matchers with localized, token-level NER models for comprehensive privacy masking.
- **Token-Level Entity Classifiers:** Transformer models fine-tuned for sequence labeling to output precise token entity boundaries ($\text{B-PER}, \text{I-PER}, \text{B-LOC}$).
- **Multi-Label Toxicity Classifiers:** High-throughput sequence classification models compiled for fast inference to evaluate multi-vector toxicity distributions.
- **Auditable Quarantine Routers:** Workflow orchestration hooks configured to intercept non-compliant payloads and direct them to secure, write-only compliance storage paths.
