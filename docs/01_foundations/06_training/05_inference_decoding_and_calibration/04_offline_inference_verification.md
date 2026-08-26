# Offline Inference Verification and Model Audit Harness

## 1. The Role of Offline Inference Verification

Before a newly trained or fine-tuned model checkpoint is promoted to a production model registry or deployed to an online serving fleet (e.g., vLLM, TensorRT-LLM), it must pass **Gate 5 (Offline Inference Verification)**.

While loss metrics and perplexity scores confirm optimization convergence on training and validation splits, they do not guarantee that the serialized checkpoint will execute reliably under production inference constraints.

```text
Training Checkpoint (best_model.pt)
                 │
                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GATE 5: OFFLINE INFERENCE VERIFICATION PIPELINE                        │
├────────────────────────────────────────────────────────────────────────┤
│ 1. KV-Cache Numerical Equivalence Check (Logit Parity vs. Naive)       │
│ 2. Delimiter & Stop-Token Generation Compliance (<|im_end|> / EOS)     │
│ 3. Deterministic Gold Unit-Test Regression Suite (Zero Regressions)    │
│ 4. Precision & Quantization Drift Audits (FP32 vs. BF16 vs. FP8)       │
│ 5. Hardware Profiling: TTFT, Inter-Token Latency, Peak VRAM Footprint  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ ALL TESTS PASSED
                                    ▼
       [ Promote to Model Registry & Deploy to Serving Infrastructure ]

```

Offline verification executes standalone, non-networked assertions against the finalized weights to detect low-level implementation bugs, numerical divergence in inference kernels, formatting regressions, and memory leaks before user traffic is routed to the model.

---

## 2. KV-Cache Numerical Equivalence & Logit Parity

A common failure mode in custom inference engines is **KV-cache divergence**, where the single-token decode path generates subtly different logits than the full-sequence forward pass due to incorrect positional index offsets, missing RoPE rotations on cached keys, or misaligned attention masks.

### Logit Parity Mathematical Invariant

Let $X = (x_1, x_2, \dots, x_T)$ be a sequence of token IDs.

- Let $Z_{\text{naive}} \in \mathbb{R}^{T \times V}$ be the logits produced by a standard full-sequence forward pass without caching:

$$Z_{\text{naive}} = \text{Model}(X)$$

- Let $Z_{\text{cached}} \in \mathbb{R}^{T \times V}$ be the logits produced by sequentially feeding tokens $x_t$ one by one while maintaining an active KV cache:

$$z_{t, \text{cached}} = \text{Model}(x_t, \text{kv\_cache}_{<t})$$

The implementation satisfies **Logit Parity** if and only if the maximum absolute difference across all sequence positions and vocabulary coordinates is bounded by machine precision tolerance $\epsilon$:

$$\Delta_{\text{max}} = \max_{1 \le t \le T} \max_{1 \le v \le V} \left\vert{} Z_{\text{naive}}[t, v] - Z_{\text{cached}}[t, v] \right\vert{} < \epsilon$$

Where:

- $\epsilon = 10^{-5}$ for Float32 precision.
- $\epsilon = 10^{-3}$ for BFloat16 precision.

```text
Logit Parity Test Flow:

1. Full Forward Pass (No Cache):
   Input: [ T1, T2, T3, T4 ] ──► [ Full Model Forward ] ──► Logits Matrix Z_naive (4 x V)

2. Step-by-Step Forward Pass (With KV Cache):
   Step 1: Input [ T1 ]       ──► Update Cache ──► Output Logit z_1 (1 x V)
   Step 2: Input [ T2 ] + KV1 ──► Update Cache ──► Output Logit z_2 (1 x V)
   Step 3: Input [ T3 ] + KV2 ──► Update Cache ──► Output Logit z_3 (1 x V)
   Step 4: Input [ T4 ] + KV3 ──► Update Cache ──► Output Logit z_4 (1 x V)
   Concatenate Step Outputs:  ──► Logits Matrix Z_cached (4 x V)

3. Assertion:
   Assert: max(|Z_naive - Z_cached|) < tolerance

```

If $\Delta_{\text{max}} \ge \epsilon$, the KV cache contains an indexing or tensor rotation defect and must be quarantined immediately.

---

## 3. Stop-Condition Compliance and Delimiter Integrity

In Supervised Fine-Tuning (SFT), models learn to terminate responses by emitting designated delimiter tokens (e.g., `<|im_end|>` in ChatML or `<|eot_id|>` in Llama-3).

If loss masking during training accidentally penalized delimiter tokens, or if tokenizer special token definitions were mapped incorrectly, the model will suffer from **Runaway Generation**: generating high-quality answers but failing to emit the stop token, continuing to output trailing gibberish until reaching `max_new_tokens`.

```text
Turn Termination Verification Test:

Input Prompt:
  "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"

Expected Model Behavior:
  Output: "4<|im_end|>" ──► Decoding halts immediately at step 2.

Runaway Generation Failure Mode:
  Output: "4<|im_end|>\n<|im_start|>user\nHello? What else can you do? I can help with math and..."
  Result: Model failed to register <|im_end|> as a stopping signal; breached turn boundary.

```

### Verification Criteria

1. **Delimiter Emission Rate:** Across a deterministic test suite of 100 closed-form queries, the model must emit the configured EOS token in $\ge 99\%$ of runs.
2. **Maximum Length Truncation Rate:** Less than $1\%$ of evaluation samples should terminate due to reaching `max_new_tokens`.
3. **Turn Framing Integrity:** The model must never spontaneously emit system prompt headers (`<|im_start|>system`) during an assistant turn.

---

## 4. Gold Benchmark Unit-Testing & Regression Thresholds

To detect catastrophic forgetting or behavioral regressions across training iterations, Gate 5 runs the candidate checkpoint against an immutable **Gold Reference Evaluation Suite**.

The Gold Suite consists of curated test cases representing foundational capabilities:

- **Syntax & Code Structure:** Syntactic correctness of generated Python/SQL code (AST validity).
- **Deterministic Reasoning:** Mathematical arithmetic and symbolic logic prompts with exact ground-truth solutions.
- **Instruction Constraint Following:** Strict adherence to negative constraints (e.g., "Answer in exactly 3 bullet points without mentioning the word 'blue'").
- **Schema Conformance:** Generating structured outputs that adhere strictly to predefined JSON schemas.

```text
┌────────────────────────────────────────────────────────────────────────┐
│ GOLD CAPABILITY REGRESSION MATRIX                                      │
├──────────────────────────┬───────────────────┬─────────────────────────┤
│ Capability Domain        │ Evaluation Metric │ Hard Regression Ceiling │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Python Code Synthesis    │ AST Parse Pass %  │ 100% Syntax Validity    │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ JSON Schema Generation   │ json.loads() Pass │ 100% Strict Parse       │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Arithmetic Verification  │ Exact Match (EM)  │ >= 95% on Gold Set      │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Delimiter Stopping       │ EOS Emission Rate │ >= 99% on All Tasks     │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ Production Baseline Δ    │ Benchmark Delta   │ Max 0.5% Drop vs Prior  │
└──────────────────────────┴───────────────────┴─────────────────────────┘

```

---

## 5. Performance and Hardware Profiling Metrics

Offline verification profiles execution latency and VRAM consumption across standardized batch sizes and context lengths to ensure serving SLAs will be met.

```text
Inference Latency Profile:

   User Prompt
        │
        ├── Prefill Phase ──► [ Time To First Token (TTFT) ]
        │
        └── Decode Phase  ──► [ Inter-Token Latency (ITL) ] ──► Total Generation Time

```

### 1. Time to First Token (TTFT)

The latency required to ingest the prompt sequence of length $N$ and compute the first output token:

$$\text{TTFT} = t_{\text{first\_token\_emitted}} - t_{\text{request\_received}}$$

TTFT is dominated by the compute-bound Prefill GEMM over prompt tokens.

### 2. Inter-Token Latency (ITL) / Time Per Output Token (TPOT)

The duration required to compute each subsequent single token during the memory-bandwidth bound Decode phase:

$$\text{ITL} = \frac{1}{M - 1} \sum_{i=2}^M (t_i - t_{i-1})$$

Where $M$ is the number of generated completion tokens.

### 3. Peak VRAM Memory Footprint ($M_{\text{peak}}$)

The maximum GPU memory allocated during the generation rollout, computed via `torch.cuda.max_memory_allocated()`. Must satisfy:

$$M_{\text{peak}} = M_{\text{weights}} + M_{\text{KV\_cache}}(B, L) + M_{\text{workspace}} \le 0.90 \times M_{\text{GPU\_Total}}$$

---

## 6. Python Implementation: Offline Verification Harness

Below is the standalone Python verification engine automating KV-cache logit parity testing, delimiter compliance verification, and performance profiling:

````python
import ast
import json
import time
from typing import Any, Callable, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class OfflineInferenceVerifier:
    """
    Automated Gate 5 verification harness for validating model checkpoints
    prior to registry promotion and production deployment.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        eos_token_id: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ):
        self.model = model.to(device=device, dtype=dtype)
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.device = device
        self.dtype = dtype
        self.model.eval()

    # ------------------------------------------------------------------
    # TEST 1: KV-Cache Logit Parity Assertion
    # ------------------------------------------------------------------
    @torch.no_grad()
    def verify_kv_cache_parity(
        self,
        test_sequence: torch.Tensor,
        tolerance: float = 1e-3
    ) -> bool:
        """
        Asserts that sequential decoding with KV cache produces identical logits
        to a full-sequence forward pass without caching.
        """
        test_sequence = test_sequence.to(self.device)
        B, T = test_sequence.shape
        assert B == 1, "Parity test expects batch size of 1"

        # 1. Full-sequence forward pass (No Cache)
        logits_naive, _ = self.model(test_sequence, use_cache=False)

        # 2. Sequential step-by-step pass (With KV Cache)
        logits_cached_list = []
        cache = None

        for t in range(T):
            token_in = test_sequence[:, t : t + 1]
            out, cache = self.model(token_in, kv_cache=cache, use_cache=True)
            logits_cached_list.append(out)

        logits_cached = torch.cat(logits_cached_list, dim=1)

        # 3. Compute absolute error
        abs_diff = torch.abs(logits_naive - logits_cached)
        max_diff = torch.max(abs_diff).item()

        if max_diff > tolerance:
            raise AssertionError(
                f"GATE 5 FAILED: KV-Cache Logit Parity mismatch! "
                f"Max absolute delta {max_diff:.6f} exceeds tolerance {tolerance:.6f}"
            )

        print(f"PASS: KV-Cache Parity Verified (Max Delta: {max_diff:.6e} < {tolerance})")
        return True

    # ------------------------------------------------------------------
    # TEST 2: Stop Token Compliance & Delimiter Audit
    # ------------------------------------------------------------------
    @torch.no_grad()
    def verify_delimiter_compliance(
        self,
        prompts: List[str],
        max_new_tokens: int = 128
    ) -> Dict[str, Any]:
        """
        Verifies that model emits EOS delimiter without runaway generation.
        """
        total_prompts = len(prompts)
        eos_emitted_count = 0
        runaway_count = 0

        for prompt_text in prompts:
            input_ids = torch.tensor(
                [self.tokenizer.encode(prompt_text)],
                dtype=torch.long,
                device=self.device
            )

            curr_tokens = input_ids
            cache = None
            emitted_eos = False

            for _ in range(max_new_tokens):
                # Single token step
                token_to_forward = curr_tokens if cache is None else curr_tokens[:, -1:]
                logits, cache = self.model(token_to_forward, kv_cache=cache, use_cache=True)

                # Greedy selection
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

                if next_token.item() == self.eos_token_id:
                    emitted_eos = True
                    break

            if emitted_eos:
                eos_emitted_count += 1
            else:
                runaway_count += 1

        eos_rate = eos_emitted_count / max(1, total_prompts)
        if eos_rate < 0.98:
            raise AssertionError(
                f"GATE 5 FAILED: Delimiter emission rate ({eos_rate * 100:.1f}%) "
                f"fell below minimum 98% threshold. Runaway generations detected: {runaway_count}"
            )

        print(f"PASS: Delimiter Compliance Verified ({eos_emitted_count}/{total_prompts} stopped cleanly)")
        return {"eos_emission_rate": eos_rate, "runaways": runaway_count}

    # ------------------------------------------------------------------
    # TEST 3: Gold Benchmark Code Syntax Audit
    # ------------------------------------------------------------------
    @torch.no_grad()
    def verify_code_ast_conformance(
        self,
        coding_prompts: List[str],
        generate_fn: Callable[[str], str]
    ) -> bool:
        """
        Validates that generated Python code snippets pass full AST parsing.
        """
        for i, prompt in enumerate(coding_prompts):
            completion = generate_fn(prompt)

            # Extract code between fences if present
            if "```python" in completion:
                code = completion.split("```python")[1].split("```")[0].strip()
            elif "```" in completion:
                code = completion.split("```")[1].split("```")[0].strip()
            else:
                code = completion.strip()

            try:
                ast.parse(code)
            except SyntaxError as err:
                raise AssertionError(
                    f"GATE 5 FAILED: Generated code failed AST syntax check on prompt {i}!\n"
                    f"Error: {err}\nGenerated Code:\n{code}"
                )

        print(f"PASS: Gold Code AST Conformance Verified across {len(coding_prompts)} cases")
        return True

    # ------------------------------------------------------------------
    # TEST 4: Hardware Latency and Memory Profiling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def profile_hardware_performance(
        self,
        prompt_len: int = 512,
        gen_len: int = 64
    ) -> Dict[str, float]:
        """
        Measures Time to First Token (TTFT), Inter-Token Latency (ITL),
        and Peak VRAM footprint.
        """
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        dummy_prompt = torch.randint(
            0, 1000, (1, prompt_len), dtype=torch.long, device=self.device
        )

        # 1. Measure TTFT (Prefill Phase)
        t_start = time.perf_counter()
        logits, cache = self.model(dummy_prompt, use_cache=True)
        if self.device == "cuda":
            torch.cuda.synchronize()
        t_first_token = time.perf_counter()
        ttft_ms = (t_first_token - t_start) * 1000.0

        # 2. Measure ITL (Decode Phase)
        current_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        decode_latencies = []

        for _ in range(gen_len - 1):
            t_step_start = time.perf_counter()
            logits, cache = self.model(current_token, kv_cache=cache, use_cache=True)
            if self.device == "cuda":
                torch.cuda.synchronize()
            t_step_end = time.perf_counter()

            decode_latencies.append((t_step_end - t_step_start) * 1000.0)
            current_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        avg_itl_ms = sum(decode_latencies) / len(decode_latencies)
        throughput_tps = 1000.0 / avg_itl_ms

        peak_vram_mb = 0.0
        if self.device == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        metrics = {
            "ttft_ms": ttft_ms,
            "itl_ms": avg_itl_ms,
            "throughput_tokens_per_sec": throughput_tps,
            "peak_vram_mb": peak_vram_mb
        }

        print(
            f"PASS: Profiling Complete -> TTFT: {ttft_ms:.2f} ms | "
            f"ITL: {avg_itl_ms:.2f} ms ({throughput_tps:.1f} tok/s) | Peak VRAM: {peak_vram_mb:.1f} MB"
        )
        return metrics

````

---

## 7. Gate 5 Verification Checklist & Promotion Policy

Before moving a checkpoint to the production serving registry, all verification items must evaluate to **PASS**:

| Audit Domain          | Test Target           | Verification Mechanism                 | Acceptance Ceiling                     | Action on Failure                        |
| --------------------- | --------------------- | -------------------------------------- | -------------------------------------- | ---------------------------------------- |
| **Cache Integrity**   | KV-Cache Logit Parity | Sequential vs. Full Forward pass delta | $\Delta_{\text{max}} < 10^{-3}$ (BF16) | Halt deployment; inspect RoPE offsets    |
| **Turn Termination**  | EOS / `<              | im_end                                 | >` Delimiter                           | Output stream scan on 100 Gold Prompts   |
| **Code Syntax**       | AST Conformance       | `ast.parse()` on Python code outputs   | $100\%$ valid AST                      | Reject release; review coding SFT data   |
| **Schema Strictness** | Structured JSON/SQL   | `json.loads()` on JSON tasks           | $100\%$ valid JSON                     | Reject release; tune prompt formatting   |
| **Serving SLA**       | Hardware Latency      | ITL / Throughput profiling             | $\text{ITL} \le \text{SLA Limit}$      | Alert hardware team; check quantization  |
| **VRAM Safety**       | Peak Memory Footprint | `torch.cuda.max_memory_allocated()`    | $\le 90\%$ GPU Memory                  | Block batch size promotion; optimize GQA |
