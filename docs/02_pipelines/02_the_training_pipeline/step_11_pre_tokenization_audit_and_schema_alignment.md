# Step 11: JIT Chat Template & Schema Alignment

## 1. Core Objective & Operational Placement

Executing as the initial stage of the **Training Pipeline**, Step 11 bridges universal, model-agnostic Silver data with model-specific tokenizer requirements.

Operating in-memory within `/dev/shm`, Step 11 dynamically extracts the base model's native Jinja2 chat template (e.g., ChatML for `Qwen2.5-0.5B-Instruct` or Header ID formatting for `Llama-3.2`) and serializes structured JSON dialogues into properly formatted prompt strings immediately prior to tokenization.

```text
 [ Silver Layer Dialogue JSON ] ──► [ Step 11: JIT Chat Template ] ──► [ Step 12: Tokenizer ]
 [{"role": "user", ...}]            "<|im_start|>user\n..."           [151644, 872, ...]

```

---

## 2. Theoretical & Architectural Justification

Universal Silver storage records store conversations as structured dictionary arrays (`[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`). Because different LLM architectures rely on distinct structural delimiters, hardcoding formatting in the upstream Data Pipeline breaks model portability.

By executing chat templating Just-In-Time (JIT) inside the training container, swapping base models (e.g., from Qwen to Llama) requires changing only the configuration manifest URI—leaving the upstream Data Pipeline completely untouched.

---

## 3. Execution Mechanics

1. **Dynamic Jinja2 Template Extraction:** The training worker queries `tokenizer.chat_template` from the model's configuration metadata.
2. **Structural Control Token Injection:** Wraps user, system, and assistant turns with model-specific delimiters:

- **ChatML (`Qwen2.5`):**

```text
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nCalculate 2+2.<|im_end|>\n<|im_start|>assistant\n2 + 2 = 4.<|im_end|>

```

- **Llama-3 (`Llama-3.2`):**

```text
<|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\nCalculate 2+2.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n2 + 2 = 4.<|eot_id|>

```

1. **In-Memory Buffer Streaming:** Formatted text strings pass directly into Step 12 via zero-copy RAM buffers without disk serialization.
