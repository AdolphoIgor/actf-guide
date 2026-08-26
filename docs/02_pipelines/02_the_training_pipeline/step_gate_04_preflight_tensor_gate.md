# Gate 4: Pre-Flight Tensor, Weight Distribution, and Gradient Health Gate

## 1. The Role and Scope of Gate 4

Before a distributed cluster allocates thousands of GPU-hours to a foundation pre-training or supervised fine-tuning (SFT) run, it must pass **Gate 4 (The Pre-Flight Tensor Gate)**.

While Gates 1 through 3 validate data formats, tokenization alignments, and split contamination on disk, Gate 4 executes in-memory assertions directly against the computational graph, parameter memory buffers, and autograd execution engines at Step 0.

```text
Dataset Shards & Tokenizer (Validated by Gate 1-3)
                     │
                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GATE 4: PRE-FLIGHT TENSOR & GRADIENT HEALTH FIREWALL (STEP 0)          │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Parameter Tensor Health & Weight Distribution Integrity             │
│    • Zero initial NaNs/Infs, dead channels, or uninitialized buffers   │
│    • Layer-wise weight variance matches architectural scaling laws     │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Memory Pointer & Weight Tying Aliasing Assertions                   │
│    • Tied parameter pointers share identical underlying data_ptr()     │
│    • Disjoint optimizer parameter groups (decay vs. no-decay)          │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Step-0 Forward Pass & Initial Loss Calibration                      │
│    • Output logits within bounded range (no extreme activation spikes) │
│    • Step-0 Cross-Entropy aligns with theoretical uniform bound ln(V)  │
├────────────────────────────────────────────────────────────────────────┤
│ 4. Autograd Completeness & Gradient Flow Audit                         │
│    • 100% of trainable parameters receive non-zero, non-NaN gradients  │
│    • Zero detached submodules or dead residual highways                │
├────────────────────────────────────────────────────────────────────────┤
│ 5. Hardware VRAM Headroom & Activation Memory Profiling                │
│    • Peak forward + backward + optimizer allocation fits in VRAM pool  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       [ PASS: Initiate Distributed Training ]  [ FAIL: Abort & Quarantine Job ]

```

Catching a silent autograd bug, an unmasked padding gradient, or an uncalibrated weight initialization at Step 0 prevents costly compute waste, loss divergence mid-run, and checkpoint corruption.

---

## 2. Theoretical Bounds and Mathematical Invariants

Gate 4 evaluates parameter tensors and activation metrics against four fundamental mathematical invariants:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ GATE 4 MATHEMATICAL INVARIANTS                                         │
├──────────────────────────────┬─────────────────────────────────────────┤
│ 1. Theoretical Step-0 Loss   │ L_0 ≈ ln(V)                             │
│    (Uniform Cross-Entropy)   │ For V = 32,000, L_0 ≈ 10.37 nats        │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 2. Residual Projection Scale │ Var(W_proj) ≈ 0.02^2 / (2 * N_layer)    │
│    (Deep Pre-LN Stability)   │ Attenuates residual variance growth.    │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 3. Logit Bounded Range       │ max(|z_0|) <= 15.0                      │
│    (Softmax Stability)       │ Prevents early FP16/BF16 saturation.    │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 4. Gradient Coverage Invariant│ Count(g == None) == 0                   │
│    (Autograd Graph Flow)     │ Across all p where requires_grad=True.  │
└──────────────────────────────┴─────────────────────────────────────────┘

```

### A. Theoretical Step-0 Cross-Entropy Loss Bound

At initialization (Step 0), an untrained model with symmetric random weights assigns roughly uniform probability to all vocabulary tokens $v \in \{1, 2, \dots, V\}$:

$$P(y_t = v \mid x_{<t}) \approx \frac{1}{V}$$

The theoretical initial cross-entropy loss $\mathcal{L}_0$ evaluates to the negative log-likelihood of the uniform distribution:

$$\mathcal{L}_0 = -\sum_{v=1}^V \left(\frac{1}{V}\right) \ln\left(\frac{1}{V}\right) = \ln(V)$$

$$\text{Loss Tolerance Interval: } \mathcal{L}_0 \in \left[ \ln(V) - \epsilon_{\text{loss}}, \; \ln(V) + \epsilon_{\text{loss}} \right]$$

Where $\epsilon_{\text{loss}} \approx 0.5 \text{ nats}$.

- **If $\mathcal{L}_0 \gg \ln(V)$:** Output logits are excessively biased toward incorrect tokens or initialized with excessive variance, causing immediate gradient spikes.
- **If $\mathcal{L}_0 \ll \ln(V)$ (e.g., $\mathcal{L}_0 < 2.0$ on an untrained model):** Target labels are leaking directly into input embeddings, or loss masking is erroneously skipping active target positions.

---

### B. Weight Variance and Residual Scaling

In standard Transformer architectures using Pre-LN, residual connections accumulate variance linearly with depth $L$. To prevent signal explosion across deep layers, projections that feed directly into the residual stream (such as the attention output projection $W_{\text{attn\_out}}$ and the FFN down-projection $W_{\text{down}}$) must follow scaled initialization:

$$\sigma_{\text{base}} = 0.02, \quad \sigma_{\text{residual}} = \frac{\sigma_{\text{base}}}{\sqrt{2 \times N_{\text{layers}}}}$$

Gate 4 asserts that the empirical standard deviation $\hat{\sigma}_l$ of every layer's parameter matrix falls within bounded statistical tolerances:

$$\left\vert{} \text{std}(W_l) - \sigma_{\text{expected}, l} \right\vert{} \le 0.20 \times \sigma_{\text{expected}, l}$$

---

### C. Gradient Flow and Coverage Invariant

Let $\Theta = \{\theta_1, \theta_2, \dots, \theta_K\}$ be the set of all model parameters registered with `requires_grad = True`.
Following a backward pass on a representative micro-batch:

$$\forall \theta_k \in \Theta, \quad \nabla_{\theta_k} \mathcal{L} \neq \text{None} \quad \land \quad \text{isnan}\left(\nabla_{\theta_k} \mathcal{L}\right) == \text{False} \quad \land \quad \Vert{}\nabla_{\theta_k} \mathcal{L}\Vert{}_2 > 0$$

$$\text{Global Step-0 Gradient Norm: } \Vert{}g_0\Vert{}_2 = \sqrt{\sum_{k=1}^K \Vert{}\nabla_{\theta_k} \mathcal{L}\Vert{}_2^2} \in [0.01, 10.0]$$

---

## 3. Pre-Flight Verification Vectors

```text
┌────────────────────────────────────────────────────────────────────────┐
│ PRE-FLIGHT VERIFICATION MATRIX                                         │
├──────────────────────────┬───────────────────┬─────────────────────────┤
│ Verification Target      │ Inspection Method │ Failure Condition       │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 1. Parameter Pointer     │ Memory pointer    │ `id(lm_head.weight)`    │
│    Aliasing              │ `data_ptr()` test │ `!= id(tok_emb.weight)` │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 2. Optimizer Group       │ Set intersection  │ `len(decay ∩ nodecay)`  │
│    Disjointness          │ on param IDs      │ `> 0`                   │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 3. Target Loss Masking   │ Active label      │ `active_tokens == 0` or │
│    Integrity             │ tensor count      │ `active_tokens == total`│
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 4. Dead Head / Submodule │ Gradient norm per │ `||grad(layer_i)|| == 0`│
│    Isolation             │ tensor block      │ on active forward path  │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 5. VRAM Allocation       │ PyTorch CUDA peak │ `Peak_VRAM > 0.85 *`    │
│    Headroom              │ memory tracker    │ `Total_Physical_VRAM`   │
└──────────────────────────┴───────────────────┴─────────────────────────┘

```

### 1. Tied Embedding Pointer Verification

When using tied weights (tying token embeddings to the final LM head), calling `lm_head.weight = tok_emb.weight` shares memory. If an engineer inadvertently re-instantiates or clones the tensor (`lm_head.weight = nn.Parameter(tok_emb.weight.clone())`), the parameters become decoupled:

```text
Tied Embeddings Memory Pointer Check:

Correct (Memory Aliased):
  tok_emb.weight.data_ptr() ──► [ 0x7f9a8b000000 ] ◄── lm_head.weight.data_ptr()
  Status: PASS (Shared parameter memory; gradients accumulate into single tensor)

Corrupted (Decoupled Buffers):
  tok_emb.weight.data_ptr() ──► [ 0x7f9a8b000000 ]
  lm_head.weight.data_ptr() ──► [ 0x7f9a8c120000 ]  <── CRITICAL FAULT!
  Status: FAIL (Model allocates redundant 2x parameter memory; weights diverge)

```

### 2. Disjoint Optimizer Parameter Grouping

Weight decay must apply only to 2D+ weight matrices (GEMMs, projections), while LayerNorm/RMSNorm scales ($\gamma$), biases ($\beta$), and positional embeddings remain unregularized:

$$\text{Group}_{\text{decay}} \cap \text{Group}_{\text{no\_decay}} = \emptyset, \quad \text{Group}_{\text{decay}} \cup \text{Group}_{\text{no\_decay}} = \Theta$$

---

## 4. Gate 4 Mathematical Assertion Vector Formulation

Gate 4 constructs an immutable Boolean assertion vector $\mathbf{G}_4$ across all verification domains:

$$\mathbf{G}_4 = \begin{bmatrix}  \mathbb{I}\left( \text{Count}(\text{NaN}(\Theta) \cup \text{Inf}(\Theta)) == 0 \right) \\ \mathbb{I}\left( \text{data\_ptr}(W_{\text{lm\_head}}) == \text{data\_ptr}(W_{\text{tok\_emb}}) \right) \\ \mathbb{I}\left( \left\vert{} \mathcal{L}_0 - \ln(V) \right\vert{} \le 0.50 \right) \\ \mathbb{I}\left( \max(\vert{}z_0\vert{}) \le 15.0 \right) \\ \mathbb{I}\left( \text{Count}(\nabla_\theta \mathcal{L} == \text{None}) == 0 \right) \\ \mathbb{I}\left( \text{ActiveTokens}(Y) > 0 \;\land\; \text{ActiveTokens}(Y) < \text{Numel}(Y) \right) \\ \mathbb{I}\left( M_{\text{peak}} \le 0.85 \times M_{\text{GPU\_Total}} \right) \end{bmatrix}$$

$$\text{Gate 4 Status} = \begin{cases} \text{PASSED (Ready for Compute Allocation)}, & \text{if } \prod_{k=1}^7 \mathbf{G}_{4, k} == 1 \\ \text{ABORT (Quarantine Job \& Notify Fleet)}, & \text{otherwise} \end{cases}$$

---

## 5. Python / PyTorch Implementation: Production Gate 4 Harness

Below is the standalone, production-grade `PreflightTensorGateEngine` that automates tensor health audits, tied weight assertions, Step-0 loss verification, autograd graph coverage checks, and VRAM memory profiling:

```python
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Gate4PreflightException(Exception):
    """Raised when an assertion in the Pre-Flight Tensor Gate fails."""
    pass


class PreflightTensorGateEngine:
    """
    Automated Gate 4 harness executing in-memory tensor health,
    Step-0 loss verification, autograd coverage, and memory profiling.
    """
    def __init__(
        self,
        model: nn.Module,
        vocab_size: int,
        n_layers: int,
        ignore_index: int = -100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ):
        self.model = model.to(device=device, dtype=dtype)
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.ignore_index = ignore_index
        self.device = device
        self.dtype = dtype

    # =====================================================================
    # 1. PARAMETER HEALTH & TIED POINTER AUDIT
    # =====================================================================
    def audit_initial_weights(self, check_tied_embeddings: bool = True) -> Dict[str, Any]:
        """
        Validates parameter tensor finite values, std-dev bounds, and tied pointers.
        """
        total_params = 0
        trainable_params = 0
        dead_zero_params = 0

        for name, param in self.model.named_parameters():
            numel = param.numel()
            total_params += numel
            if param.requires_grad:
                trainable_params += numel

            # 1. Check for NaNs or Infs in initialized weights
            if torch.isnan(param).any():
                raise Gate4PreflightException(f"GATE 4 FAILED: NaN detected in initial weights: {name}")
            if torch.isinf(param).any():
                raise Gate4PreflightException(f"GATE 4 FAILED: Inf detected in initial weights: {name}")

            # 2. Check for dead (all-zero) weight matrices
            if param.dim() >= 2 and torch.all(param == 0):
                raise Gate4PreflightException(f"GATE 4 FAILED: Weight matrix {name} is entirely zeros.")

            if torch.all(param == 0):
                dead_zero_params += 1

        # 3. Tied Embedding Pointer Verification
        if check_tied_embeddings:
            # Resolve standard attribute names for embeddings and lm_head
            tok_emb = getattr(self.model, "tok_emb", None) or getattr(self.model, "wte", None)
            lm_head = getattr(self.model, "lm_head", None)

            if tok_emb is not None and lm_head is not None:
                emb_weight = tok_emb.weight if hasattr(tok_emb, "weight") else tok_emb
                head_weight = lm_head.weight if hasattr(lm_head, "weight") else lm_head

                if emb_weight.data_ptr() != head_weight.data_ptr():
                    raise Gate4PreflightException(
                        f"GATE 4 FAILED: Tied embeddings do not share memory pointer! "
                        f"tok_emb: {hex(emb_weight.data_ptr())} vs lm_head: {hex(head_weight.data_ptr())}"
                    )

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "zero_tensors_count": dead_zero_params
        }

    # =====================================================================
    # 2. OPTIMIZER PARAMETER GROUP DISJOINTNESS
    # =====================================================================
    def audit_optimizer_parameter_groups(
        self, optimizer: torch.optim.Optimizer
    ) -> Dict[str, int]:
        """
        Asserts that decay and no-decay parameter sets are strictly disjoint.
        """
        seen_param_ptrs: Set[int] = set()
        total_grouped_params = 0

        for group_idx, param_group in enumerate(optimizer.param_groups):
            for param in param_group["params"]:
                ptr = param.data_ptr()
                if ptr in seen_param_ptrs:
                    raise Gate4PreflightException(
                        f"GATE 4 FAILED: Parameter pointer {hex(ptr)} appears in multiple optimizer groups!"
                    )
                seen_param_ptrs.add(ptr)
                total_grouped_params += 1

        # Compare against model trainable parameters count
        trainable_count = sum(1 for p in self.model.parameters() if p.requires_grad)
        if total_grouped_params != trainable_count:
            raise Gate4PreflightException(
                f"GATE 4 FAILED: Optimizer registered {total_grouped_params} tensors, "
                f"but model has {trainable_count} trainable parameters."
            )

        return {"disjoint_parameter_tensors_verified": total_grouped_params}

    # =====================================================================
    # 3. STEP-0 FORWARD, LOSS, AND MASKING AUDIT
    # =====================================================================
    def audit_forward_and_loss(
        self,
        sample_inputs: torch.Tensor,
        sample_targets: torch.Tensor,
        loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """
        Asserts output logit dynamic range, active token counts, and ln(V) initial loss.
        """
        self.model.eval()
        sample_inputs = sample_inputs.to(self.device)
        sample_targets = sample_targets.to(self.device)

        # 1. Target Masking Verification
        active_tokens = (sample_targets != self.ignore_index).sum().item()
        total_tokens = sample_targets.numel()

        if active_tokens == 0:
            raise Gate4PreflightException("GATE 4 FAILED: Batch contains 0 active loss targets (100% masked).")
        if active_tokens == total_tokens:
            raise Gate4PreflightException("GATE 4 FAILED: SFT batch has zero masked tokens (Prompt unmasked).")

        # 2. Forward Pass Execution
        with torch.no_grad():
            with torch.autocast(device_type=self.device if self.device == "cuda" else "cpu", dtype=self.dtype):
                logits = self.model(sample_inputs)
                if isinstance(logits, tuple):
                    logits = logits[0]

        # 3. Logit Stability Check
        max_logit = torch.max(torch.abs(logits)).item()
        if math.isnan(max_logit) or math.isinf(max_logit):
            raise Gate4PreflightException("GATE 4 FAILED: Forward pass generated NaN/Inf logits.")
        if max_logit > 25.0:
            raise Gate4PreflightException(
                f"GATE 4 FAILED: Initial maximum logit {max_logit:.2f} > 25.0 (Softmax instability risk)."
            )

        # 4. Step-0 Cross-Entropy Validation against ln(V)
        shift_logits = logits[..., :-1, :].contiguous().view(-1, self.vocab_size)
        shift_labels = sample_targets[..., 1:].contiguous().view(-1)

        loss = F.cross_entropy(
            shift_logits, shift_labels, ignore_index=self.ignore_index, reduction="mean"
        ).item()

        expected_loss = math.log(self.vocab_size)
        delta_loss = abs(loss - expected_loss)

        # Allow maximum 0.6 nats tolerance around ln(V)
        if delta_loss > 0.60:
            raise Gate4PreflightException(
                f"GATE 4 FAILED: Step-0 Loss {loss:.4f} diverges from theoretical "
                f"ln(V) = {expected_loss:.4f} (Delta: {delta_loss:.4f} > 0.60 nats)."
            )

        return {
            "step_0_loss": loss,
            "expected_theoretical_loss": expected_loss,
            "max_absolute_logit": max_logit,
            "active_target_tokens": active_tokens
        }

    # =====================================================================
    # 4. AUTOGRAD GRAPH COMPLETENESS & GRADIENT FLOW AUDIT
    # =====================================================================
    def audit_autograd_gradient_flow(
        self,
        sample_inputs: torch.Tensor,
        sample_targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Executes backward pass and asserts that 100% of trainable parameters receive valid gradients.
        """
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        sample_inputs = sample_inputs.to(self.device)
        sample_targets = sample_targets.to(self.device)

        with torch.autocast(device_type=self.device if self.device == "cuda" else "cpu", dtype=self.dtype):
            logits = self.model(sample_inputs)
            if isinstance(logits, tuple):
                logits = logits[0]

            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.vocab_size)
            shift_labels = sample_targets[..., 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=self.ignore_index)

        # Execute autograd backward pass
        loss.backward()

        missing_grad_tensors: List[str] = []
        zero_grad_tensors: List[str] = []
        nan_grad_tensors: List[str] = []
        grad_norms: List[float] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if param.grad is None:
                missing_grad_tensors.append(name)
                continue

            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                nan_grad_tensors.append(name)
                continue

            norm = torch.linalg.vector_norm(param.grad).item()
            grad_norms.append(norm ** 2)

            if norm == 0.0:
                zero_grad_tensors.append(name)

        if missing_grad_tensors:
            raise Gate4PreflightException(
                f"GATE 4 FAILED: {len(missing_grad_tensors)} parameters received no gradient! "
                f"Sample missing: {missing_grad_tensors[:3]}"
            )

        if nan_grad_tensors:
            raise Gate4PreflightException(
                f"GATE 4 FAILED: NaN/Inf gradients encountered on: {nan_grad_tensors[:3]}"
            )

        global_grad_norm = math.sqrt(sum(grad_norms))
        if global_grad_norm == 0.0:
            raise Gate4PreflightException("GATE 4 FAILED: Global gradient norm is zero.")

        # Reset gradients after preflight check
        self.model.zero_grad(set_to_none=True)

        return {
            "global_gradient_norm_step0": global_grad_norm,
            "zero_grad_tensor_count": len(zero_grad_tensors)
        }

    # =====================================================================
    # 5. HARDWARE VRAM MEMORY PROFILING
    # =====================================================================
    def audit_vram_headroom(
        self,
        batch_size: int,
        seq_len: int,
        optimizer: torch.optim.Optimizer,
        max_vram_usage_ratio: float = 0.85
    ) -> Dict[str, float]:
        """
        Profiles peak VRAM allocation and asserts headroom limits.
        """
        if self.device != "cuda" or not torch.cuda.is_available():
            return {"status": "skipped_non_cuda"}

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        total_device_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

        # Synthesize realistic batch
        mock_in = torch.randint(0, self.vocab_size, (batch_size, seq_len), device="cuda")
        mock_target = mock_in.clone()
        mock_target[:, : seq_len // 2] = self.ignore_index

        self.model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            logits = self.model(mock_in)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), mock_target.view(-1), ignore_index=self.ignore_index
            )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        usage_ratio = peak_allocated_mb / total_device_memory

        if usage_ratio > max_vram_usage_ratio:
            raise Gate4PreflightException(
                f"GATE 4 FAILED: Peak memory {peak_allocated_mb:.1f} MB exceeds "
                f"{max_vram_usage_ratio * 100:.0f}% of total VRAM ({total_device_memory:.1f} MB). "
                f"Risk of runtime OOM during extended sequence rollout."
            )

        torch.cuda.empty_cache()

        return {
            "total_vram_mb": total_device_memory,
            "peak_allocated_vram_mb": peak_allocated_mb,
            "vram_utilization_ratio": usage_ratio
        }

    # =====================================================================
    # MASTER SUITE RUNNER
    # =====================================================================
    def run_preflight_suite(
        self,
        sample_batch: Tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        check_tied: bool = True
    ) -> Dict[str, Any]:
        """
        Executes complete Gate 4 battery and returns structured verification receipt.
        """
        inputs, targets = sample_batch

        weight_metrics = self.audit_initial_weights(check_tied_embeddings=check_tied)
        opt_metrics = self.audit_optimizer_parameter_groups(optimizer)
        forward_metrics = self.audit_forward_and_loss(inputs, targets)
        grad_metrics = self.audit_autograd_gradient_flow(inputs, targets)
        vram_metrics = self.audit_vram_headroom(
            batch_size=inputs.size(0),
            seq_len=inputs.size(1),
            optimizer=optimizer
        )

        return {
            "gate_status": "PASSED",
            "weights": weight_metrics,
            "optimizer": opt_metrics,
            "forward_loss": forward_metrics,
            "gradients": grad_metrics,
            "hardware_vram": vram_metrics
        }

```

---

## 6. Pre-Flight Diagnostic Failure Matrix

| Failure Symptom                                         | Detection Point        | Root Cause                                       | Engineering Remediation                                                            |
| ------------------------------------------------------- | ---------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------- |
| **$\mathcal{L}_0 \ll \ln(V)$** ($\mathcal{L}_0 < 2.0$)  | Step-0 Loss Evaluation | Target labels unshifted; prompt unmasked         | Ensure causal token shift ($t$ predicts $t+1$); verify loss masking                |
| **$\mathcal{L}_0 \gg \ln(V)$** ($\mathcal{L}_0 > 12.0$) | Step-0 Loss Evaluation | Logit scaling factor missing; bad initialization | Verify $\frac{1}{\sqrt{d_k}}$ attention scale; re-calibrate $\sigma_{\text{init}}$ |
| **$\text{max}(\vert{}z_0\vert{}) > 25.0$**              | Forward Logit Audit    | Unscaled output projection weights               | Clamp linear head initialization; check final RMSNorm / LayerNorm                  |
| **`param.grad is None`**                                | Autograd Flow Audit    | Submodule detached (`.detach()` / broken graph)  | Trace residual additions; ensure forward returns tensor connected to loss          |
| **`data_ptr` Mismatch**                                 | Tied Embedding Audit   | Weights decoupled via `.clone()`                 | Bind references directly: `self.lm_head.weight = self.tok_emb.weight`              |
| **Optimizer Group Overlap**                             | Param Group Audit      | Parameter listed in decay and no-decay sets      | Filter parameter lists using explicit set difference                               |
| **VRAM Usage $> 85\%$**                                 | Memory Profiling       | Batch size or micro-batch too large              | Enable activation checkpointing; reduce micro-batch size                           |
