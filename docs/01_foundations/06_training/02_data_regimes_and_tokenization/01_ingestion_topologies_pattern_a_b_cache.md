# VRAM and Precision Budgeting

## 1. Numerical Precision Formats in Deep Learning

Training and deploying Transformer architectures requires managing two competing constraints: **numerical dynamic range** (to prevent underflow and overflow during gradient and loss calculations) and **memory footprint** (to maximize batch size, sequence length, and model capacity within hardware limits).

Floating-point numbers in modern computing are partitioned into three fundamental components defined by the IEEE 754 and OCP (Open Compute Project) standards: the **Sign bit** ($s$), the **Exponent bits** ($e$, governing dynamic range and magnitude), and the **Mantissa / Fraction bits** ($m$, governing numerical precision and granularity).

$$\text{Value} = (-1)^s \times 2^{e - \text{bias}} \times \left(1 + \frac{m}{2^{\text{bits}}}\right)$$

For integer formats ($\text{INT}$), values are encoded directly using two's complement without an exponent or mantissa, trading dynamic floating range for uniform arithmetic density and lower silicon area.

```text
FP64 (Double Precision - 64 bits):
┌───┬───────────────┬─────────────────────────────────────────────────────────────┐
│ s │ Exponent (11b)│ Mantissa / Fraction (52b)                                   │
└───┴───────────────┴─────────────────────────────────────────────────────────────┘

FP32 (Single Precision - 32 bits):
┌───┬───────────────┬───────────────────────────────────────────────┐
│ s │ Exponent (8b) │ Mantissa / Fraction (23b)                     │
└───┴───────────────┴───────────────────────────────────────────────┘

TF32 (TensorFloat-32 - 19 bits inside 32-bit register):
┌───┬───────────────┬───────────────────────┐
│ s │ Exponent (8b) │ Mantissa (10b)        │
└───┴───────────────┴───────────────────────┘

FP16 (Half Precision - 16 bits):
┌───┬───────────────┬───────────────────────┐
│ s │ Exponent (5b) │ Mantissa (10b)        │
└───┴───────────────┴───────────────────────┘

BF16 (Brain Floating Point - 16 bits):
┌───┬───────────────┬───────────────────────┐
│ s │ Exponent (8b) │ Mantissa (7b)         │
└───┴───────────────┴───────────────────────┘

FP8 E4M3 (8-bit Forward Activations/Weights - OCP):
┌───┬───────────────┬───────────────┐
│ s │ Exponent (4b) │ Mantissa (3b) │
└───┴───────────────┴───────────────┘

FP8 E5M2 (8-bit Backward Gradients - OCP):
┌───┬───────────────┬───────────────┐
│ s │ Exponent (5b) │ Mantissa (2b) │
└───┴───────────────┴───────────────┘

FP4 / NVFP4 / MXFP4 (4-bit Microscaling - E2M1):
┌───┬───────────────┬───────┐
│ s │ Exponent (2b) │ M(1b) │
└───┴───────────────┴───────┘

INT8 / INT4 (Signed Two's Complement Integers):
┌───┬───────────────────────────────────────────────────────────────┐
│ s │ Magnitude Bits (7b for INT8, 3b for INT4)                     │
└───┴───────────────────────────────────────────────────────────────┘

NF4 (NormalFloat-4):
┌───────────────────────────────────────────────────────────────────┐
│ 4-bit non-linear quantile index mapping normal distribution N(0,1)│
└───────────────────────────────────────────────────────────────────┘

```

---

### Precision Comparison Matrix

| Format          | Type  | Total Bits | Exponent Bits | Mantissa Bits | Exponent Bias | Dynamic Range ($[\text{Min}_{\text{norm}}, \text{Max}]$)         | Memory (Bytes / Param) | Primary Role in LLMOps Lifecycle                                                          |
| --------------- | ----- | ---------- | ------------- | ------------- | ------------- | ---------------------------------------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------- |
| **FP64**        | Float | 64         | 11            | 52            | 1023          | $\approx 2.23 \times 10^{-308} \text{ to } 1.80 \times 10^{308}$ | $8\text{ bytes}$       | High-precision scientific simulation (rarely used in LLMs due to memory overhead).        |
| **FP32**        | Float | 32         | 8             | 23            | 127           | $\approx 1.18 \times 10^{-38} \text{ to } 3.40 \times 10^{38}$   | $4\text{ bytes}$       | Golden baseline; AdamW optimizer states ($m_t, v_t$), master weights, loss scaling.       |
| **TF32**        | Float | 19         | 8             | 10            | 127           | $\approx 1.18 \times 10^{-38} \text{ to } 3.40 \times 10^{38}$   | $4\text{ bytes}$       | Hardware-accelerated GEMM operations on NVIDIA Ampere/Hopper/Blackwell Tensor Cores.      |
| **FP16**        | Float | 16         | 5             | 10            | 15            | $\approx 6.10 \times 10^{-5} \text{ to } 65,504$                 | $2\text{ bytes}$       | Legacy mixed-precision training (requires dynamic loss scaling due to narrow exponent).   |
| **BF16**        | Float | 16         | 8             | 7             | 127           | $\approx 1.18 \times 10^{-38} \text{ to } 3.39 \times 10^{38}$   | $2\text{ bytes}$       | Gold standard for modern LLM pre-training & SFT; matches FP32 range without loss scaling. |
| **FP8 (E4M3)**  | Float | 8          | 4             | 3             | 7             | $\approx 0.0156 \text{ to } 448$                                 | $1\text{ byte}$        | Forward pass activations and weights in FP8 mixed-precision training (Hopper/Blackwell).  |
| **FP8 (E5M2)**  | Float | 8          | 5             | 2             | 15            | $\approx 6.10 \times 10^{-5} \text{ to } 57,344$                 | $1\text{ byte}$        | Backward pass gradient representations in FP8 training (prioritizes dynamic range).       |
| **FP4 / NVFP4** | Float | 4          | 2             | 1             | 1             | $\approx 0.5 \text{ to } 6.0$ (scaled via MX block)              | $0.5\text{ bytes}$     | Ultra-high throughput inference and microscaling (MX) matrix multiplication (Blackwell).  |
| **INT8**        | Int   | 8          | —             | —             | —             | $[-128, 127]$                                                    | $1\text{ byte}$        | Post-Training Quantization (PTQ), `bitsandbytes` LLM.int8(), KV-cache quantization.       |
| **INT4**        | Int   | 4          | —             | —             | —             | $[-8, 7]$ (or $[0, 15]$ unsigned)                                | $0.5\text{ bytes}$     | Weight-only quantization (AWQ, GPTQ) for edge & consumer GPU inference serving.           |
| **NF4**         | Quant | 4          | —             | —             | —             | 16 quantile points for $\mathcal{N}(0, 1)$                       | $0.5\text{ bytes}$     | QLoRA fine-tuning; information-theoretically optimal for normally distributed weights.    |

---

### Deep-Dive Analysis of Format Mechanics

#### 1. Why BF16 Replaced FP16 in Modern LLM Training

- **Underflow and Overflow Immunity:** FP16 allocates only 5 bits to the exponent, capping its maximum representable value at $65,504$ and its minimum non-zero normalized value at $6.10 \times 10^{-5}$. During backpropagation in deep Transformer stacks, gradient magnitudes easily underflow to zero or overflow to $\infty$, causing loss divergence (`NaN`). Standard FP16 training requires complex dynamic loss scaling (`torch.cuda.amp.GradScaler`) to artificially shift gradient ranges into the representable window.
- **FP32 Dynamic Range Parity:** BF16 maintains the exact same 8-bit exponent as FP32, providing an identical dynamic range ($10^{-38}$ to $10^{38}$) at half the memory footprint. While BF16 trades off precision (7 mantissa bits vs. 10 in FP16), empirical optimization confirms that deep neural networks are robust to minor mantissa truncation but highly sensitive to exponent clipping.

#### 2. TensorFloat-32 (TF32): Transparent Hardware Acceleration

- **Execution Paradigm:** TF32 is not a persistent storage format (tensors remain in FP32 in VRAM), but an internal compute mode on modern Tensor Cores (NVIDIA Ampere, Hopper, Blackwell).
- **The Best-of-Both Compromise:** TF32 consumes 19 bits: the 8-bit exponent of FP32 (preventing underflow/overflow) coupled with the 10-bit mantissa of FP16. Matrix multiplications ($C = A \cdot B$) execute up to $4\times$ faster on Tensor Cores with zero changes to FP32 user code (`torch.set_float32_matmul_precision('high')`).

#### 3. FP8 Dual-Format Specification (E4M3 vs. E5M2)

Modern 8-bit floating-point standards (OCP / NVIDIA Hopper / AMD MI300) establish two distinct formats to balance the competing demands of the forward and backward passes:

- **FP8 E4M3 (Higher Precision):** Allocates 4 exponent bits and 3 mantissa bits, preserving maximum numerical precision with a bounded range (up to $448$). This format is used for **weights and activations** in the forward pass where values are naturally bounded by LayerNorm/RMSNorm.
- **FP8 E5M2 (Higher Dynamic Range):** Allocates 5 exponent bits and 2 mantissa bits, matching the dynamic range of FP16 (up to $57,344$). This format is used for **gradients** in the backward pass, where gradient vectors span several orders of magnitude.

#### 4. Quantized Integer and Information-Theoretic Formats (INT8, INT4, NF4)

- **INT8 / INT4 Quantization:** Continuous weights $W_{\text{float}}$ are mapped into uniform discrete grids via a scaling factor $S$ and zero-point $Z$:

$$W_{\text{int}} = \text{round}\left(\frac{W_{\text{float}}}{S}\right) + Z$$

- **NF4 (NormalFloat4):** Standard INT4 assumes a uniform distribution of values. However, trained neural network weights follow a zero-mean Gaussian distribution $\mathcal{W} \sim \mathcal{N}(0, \sigma^2)$. NF4 builds a non-uniform 4-bit lookup table where each of the 16 bin boundaries represents an equal area under the standard normal distribution curve, maximizing information retention and outperforming standard INT4 quantization in QLoRA parameter-efficient fine-tuning.

### Why BF16 Replaced FP16 in Modern LLM Training

- **Underflow and Overflow Immunity:** FP16 allocates only 5 bits to the exponent, capping its maximum representable value at $65,504$ and its minimum non-zero value at $6.10 \times 10^{-5}$. During backpropagation in deep networks, small gradient magnitudes easily underflow to zero, while large activation values overflow to $\infty$, causing loss divergence (`NaN`). Standard FP16 training requires complex dynamic loss scaling (`torch.cuda.amp.GradScaler`) to artificially shift gradient ranges.

- **FP32 Dynamic Range Parity:** BF16 maintains the exact same 8-bit exponent as FP32, providing an identical dynamic range ($10^{-38}$ to $10^{38}$) at half the memory footprint. While BF16 trades off precision (7 mantissa bits vs. 10 in FP16), empirical evidence confirms that gradient descent is robust to minor mantissa truncation but highly sensitive to exponent clipping.

---

## 2. Static Memory Footprint ($M_{\text{static}}$)

The static memory footprint consists of state tensors whose sizes remain fixed regardless of batch size or sequence length.

During standard mixed-precision training using the AdamW optimizer, four distinct tensor structures must reside concurrently in VRAM for every learnable parameter $\theta$:

```text
Parameter Memory Breakdown (Mixed-Precision AdamW):
┌────────────────────────────────────────────────┐
│ Model Weights (BF16 / FP16):  2 Bytes / param  │
├────────────────────────────────────────────────┤
│ Gradients (BF16 / FP16):      2 Bytes / param  │
├────────────────────────────────────────────────┤
│ AdamW 1st Moment m_t (FP32):   4 Bytes / param │
├────────────────────────────────────────────────┤
│ AdamW 2nd Moment v_t (FP32):   4 Bytes / param │
├────────────────────────────────────────────────┤
│ Master Weights (FP32):        4 Bytes / param  │
└────────────────────────────────────────────────┘
Total Static Allocation: 16 Bytes / Parameter

```

### The 16-Bytes-per-Parameter Rule

$$M_{\text{static}} = M_{\text{weights}} + M_{\text{gradients}} + M_{\text{optimizer\_states}} + M_{\text{master\_weights}}$$

$$M_{\text{static}} = (2 \times \Phi) + (2 \times \Phi) + (4 \times \Phi + 4 \times \Phi) + (4 \times \Phi) = 16 \times \Phi \text{ bytes}$$

Where $\Phi$ represents the total number of trainable model parameters.

### Component Allocations

| Component                       | Precision   | Size per Parameter    | Allocation for 0.5B Model ($\Phi = 5 \times 10^8$) | Allocation for 7B Model ($\Phi = 7 \times 10^9$) |
| ------------------------------- | ----------- | --------------------- | -------------------------------------------------- | ------------------------------------------------ |
| **Model Weights ($\theta$)**    | BF16 / FP16 | $2\text{ bytes}$      | $1.0\text{ GB}$                                    | $14.0\text{ GB}$                                 |
| **Gradients ($\nabla \theta$)** | BF16 / FP16 | $2\text{ bytes}$      | $1.0\text{ GB}$                                    | $14.0\text{ GB}$                                 |
| **AdamW Momentum ($m_t$)**      | FP32        | $4\text{ bytes}$      | $2.0\text{ GB}$                                    | $28.0\text{ GB}$                                 |
| **AdamW Variance ($v_t$)**      | FP32        | $4\text{ bytes}$      | $2.0\text{ GB}$                                    | $28.0\text{ GB}$                                 |
| **FP32 Master Weights**         | FP32        | $4\text{ bytes}$      | $2.0\text{ GB}$                                    | $28.0\text{ GB}$                                 |
| **Total Static Memory**         | —           | **$16\text{ bytes}$** | **$8.0\text{ GB}$**                                | **$112.0\text{ GB}$**                            |

---

## 3. Dynamic Memory: Activation Footprint ($M_{\text{act}}$)

Dynamic activation memory consists of intermediate tensor states computed during the forward pass and held in VRAM to evaluate partial derivatives during the backward pass.

Unlike static memory, activation memory scales dynamically with **Batch Size ($B$)**, **Sequence Length ($T$)**, **Model Dimension ($d_{\text{model}}$)**, and **Number of Layers ($L$)**.

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Activation Memory per Transformer Layer (Standard Pre-LN Block):       │
│                                                                        │
│ 1. Self-Attention Sub-Layer:                                           │
│    • Q, K, V Linear Projections:        3 x (B x T x d_model)          │
│    • Q @ K^T Matrix Multiplication:     h x (B x T x T)                │
│    • Softmax Attention Weights:         h x (B x T x T)                │
│    • Softmax Dropout Mask:              h x (B x T x T)                │
│    • Value Aggregation (A @ V):         B x T x d_model                │
│    • Output Projection (W_O):           B x T x d_model                │
│                                                                        │
│ 2. Feed-Forward Sub-Layer (FFN):                                       │
│    • Pre-FFN LayerNorm:                 B x T x d_model                │
│    • Linear Expansion (W_1 to 4d):      4 x (B x T x d_model)          │
│    • Activation Non-Linearity (GELU):   4 x (B x T x d_model)          │
│    • Linear Contraction (W_2 to d):     B x T x d_model                │
│    • Post-FFN Dropout:                  B x T x d_model                │
└────────────────────────────────────────────────────────────────────────┘

```

### Quadratic Attention Scaling

For standard attention kernels, the attention score matrix $S = Q K^T$ requires storing an explicit $B \times h \times T \times T$ tensor in memory.

- As sequence length $T$ doubles, the memory required for self-attention activations **quadruples ($\mathcal{O}(T^2)$)**.
- **FlashAttention Mitigation:** FlashAttention eliminates the intermediate $\mathcal{O}(T^2)$ HBM memory writes by fusing the softmax computation into SRAM tiles, reducing the peak activation memory footprint from $\mathcal{O}(T^2)$ to **$\mathcal{O}(T)$ linear scaling**.

---

## 4. Total VRAM Budget Formulation

The total hardware memory required to execute a training step is governed by:

$$\text{VRAM}_{\text{total}} = M_{\text{static}} + M_{\text{activations}} + M_{\text{buffers}} + M_{\text{overhead}}$$

```text
Total Training Memory Allocation Breakdown:
┌────────────────────────────────────────────────────────────────────────┐
│ Static Memory: M_static = 16 Bytes x Parameters (Weights, Grads, AdamW)│
├────────────────────────────────────────────────────────────────────────┤
│ Dynamic Memory: M_act = f(Batch Size B, Sequence T, Hidden d, Layers L)│
├────────────────────────────────────────────────────────────────────────┤
│ Workspace Buffers: CUDA kernels, temporary GEMM scratchpads (~1-2 GB)  │
├────────────────────────────────────────────────────────────────────────┤
│ PyTorch Runtime Overhead: Memory allocator fragmentation (~10-15%)     │
└────────────────────────────────────────────────────────────────────────┘

```

### Worked Budget Example: Pedagogical vs. Production Scale

#### A. Pedagogical Model (MiniGPT on Shakespeare)

- **Configuration:** $\Phi \approx 0.8\text{M params}$, $B = 64$, $T = 256$, $d_{\text{model}} = 128$, $L = 4$.

- **Static Footprint:** $16 \text{ bytes} \times 800,000 = 12.8\text{ MB}$.
- **Activation Footprint:** $\approx 45\text{ MB}$.
- **Total VRAM Consumption:** $< 1.0\text{ GB}$ (easily trains on consumer GPUs such as NVIDIA T4 or Apple Silicon unified RAM).

#### B. Enterprise Model (`Qwen2.5-0.5B-Instruct`)

- **Configuration:** $\Phi = 490\text{M params}$, $B = 4$, $T = 2048$, $d_{\text{model}} = 896$, $L = 24$.
- **Static Footprint:** $16 \text{ bytes} \times 4.9 \times 10^8 = 7.84\text{ GB}$.
- **Activation Footprint (without FlashAttention):** $\approx 4.60\text{ GB}$.
- **Buffers & CUDA Overhead:** $\approx 1.50\text{ GB}$.
- **Total VRAM Required:** $\approx 13.94\text{ GB}$ (requires a 16 GB or 24 GB GPU, such as an NVIDIA RTX 4090 or A10G).

---

## 5. Memory Optimization Techniques

When physical VRAM is insufficient to accommodate the target configuration, training pipelines apply specific architectural and runtime optimizations:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ MEMORY REDUCTION TOOLKIT                                               │
├──────────────────────────────┬─────────────────────────────────────────┤
│ Optimization Technique       │ Primary Mechanism & Impact              │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 1. Gradient Accumulation     │ Shrinks micro-batch B_micro to fit VRAM;│
│                              │ steps optimizer every K micro-batches.  │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 2. Activation Checkpointing  │ Discards intermediate activations;      │
│                              │ recomputes forward pass during backward.│
│                              │ Saves up to 70% dynamic VRAM (+33% FLOP)│
├──────────────────────────────┼─────────────────────────────────────────┤
│ 3. 8-Bit AdamW (bitsandbytes)│ Quantizes m_t and v_t states to 8-bit.  │
│                              │ Reduces static state from 16B -> 6B/param│
├──────────────────────────────┼─────────────────────────────────────────┤
│ 4. zero_grad(set_to_none=True│ Deallocates gradient memory buffers     │
│                              │ rather than writing zeros in-place.     │
└──────────────────────────────┴─────────────────────────────────────────┘

```

### PyTorch In-Memory Optimization

```python
# 1. Zero-grad memory optimization: deallocates gradient buffers rather than zeroing
optimizer.zero_grad(set_to_none=True)

# 2. Mixed precision context manager (BF16 / FP16 execution)
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    logits, loss = model(xb, yb)

# 3. Backward pass computes gradients in BF16/FP16
loss.backward()

# 4. In-place gradient clipping prevents memory duplication
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 5. Optimizer update executes in FP32 master precision
optimizer.step()

```
