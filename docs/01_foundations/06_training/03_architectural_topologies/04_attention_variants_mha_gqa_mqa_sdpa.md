# Attention Mechanisms: Multi-Head (MHA), Multi-Query (MQA), Grouped-Query (GQA), and FlashAttention / SDPA

## 1. The Inference Memory-Bandwidth Bottleneck

During autoregressive inference, generating tokens is fundamentally **memory-bandwidth bound**, rather than compute bound.

For each newly generated token, the model must load all active model parameters and retrieve all historical Key and Value vectors stored in high-bandwidth memory (HBM) via the **KV-Cache**.

$$\text{KV-Cache Memory per Token} = 2 \times 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times d_{\text{head}} \times \text{Bytes per Parameter}$$

Where:

- The first factor of $2$ accounts for both Keys and Values.
- The second factor of $2$ accounts for 16-bit precision (FP16/BF16 = 2 bytes).
- $N_{\text{layers}}$ is network depth.
- $N_{\text{kv\_heads}}$ is the number of Key-Value attention heads.
- $d_{\text{head}} = d_{\text{model}} / N_{\text{query\_heads}}$ is the dimension of each head.

```text
KV-Cache Memory Footprint for LLaMA-7B (L=32, H=32, d_head=128, FP16) across 8k Context:
  • Single Sequence (Batch Size 1, SeqLen 8,192):  ~1.07 GB VRAM
  • Production Serving (Batch Size 64, SeqLen 8,192): ~68.7 GB VRAM (Exceeds a single 80GB GPU)

```

To enable large batch sizes, long context windows, and real-time generation speeds, modern Transformer architectures replace standard Multi-Head Attention with **KV-head compression topologies**.

---

## 2. Taxonomy of Attention Topologies

The relationship between the number of Query heads ($N_q$) and Key/Value heads ($N_{kv}$) establishes three distinct architectural paradigms:

```text
1. Multi-Head Attention (MHA)       2. Grouped-Query Attention (GQA)       3. Multi-Query Attention (MQA)
      (e.g., GPT-2, GPT-3)                   (e.g., Llama-3, Qwen-2.5)                (e.g., Falcon, StarCoder)

    Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8               Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8               Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8
    │  │  │  │  │  │  │  │               └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘               └──────────┬───────────┘
    ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼                  ▼       ▼       ▼       ▼                             ▼
    K1 K2 K3 K4 K5 K6 K7 K8                KV1     KV2     KV3     KV4                           KV1
    V1 V2 V3 V4 V5 V6 V7 V8

    N_q = 8, N_kv = 8                     N_q = 8, N_kv = 4 (Group Ratio = 2)   N_q = 8, N_kv = 1
    KV-Cache: 100% (Baseline)             KV-Cache: 50% Memory Reduction        KV-Cache: 87.5% Memory Reduction

```

### A. Multi-Head Attention (MHA)

- **Configuration:** $N_q = N_{kv} = H$. Every query head possesses an independent, dedicated key and value head.
- **Characteristics:** Maximum representational capacity and multi-feature expressive power.
- **Bottleneck:** High KV-cache memory consumption during inference, limiting maximum serving concurrency and context length.

### B. Multi-Query Attention (MQA)

- **Configuration:** $N_q = H, N_{kv} = 1$. All $H$ query heads share a single key head and a single value head across the entire layer.
- **Characteristics:** Reduces KV-cache memory consumption and bandwidth pressure by a factor of $H\times$ (up to $90\%\text{--}95\%$ reduction).
- **Trade-Off:** Can result in minor quality degradation, higher perplexity, and training instability on complex reasoning or multi-task workloads due to restricted key-value expressivity.

### C. Grouped-Query Attention (GQA)

- **Configuration:** $1 < N_{kv} < N_q$, where the query heads are divided into $G = N_{kv}$ groups, each containing $\text{Group Size} = N_q / N_{kv}$ query heads sharing a single key/value pair.
- **Characteristics:** The production sweet spot. Achieves the generation speed and memory reduction of MQA while preserving the representation quality, task accuracy, and perplexity of full MHA.

---

## 3. Comparison Matrix: MHA vs. GQA vs. MQA

| Architectural Dimension        | Multi-Head Attention (MHA)                               | Grouped-Query Attention (GQA)                                                                     | Multi-Query Attention (MQA)                                                             |
| ------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Query Heads ($N_q$)**        | $H$ (e.g., $32$)                                         | $H$ (e.g., $32$)                                                                                  | $H$ (e.g., $32$)                                                                        |
| **Key/Value Heads ($N_{kv}$)** | $H$ (e.g., $32$)                                         | $G$ (e.g., $8$)                                                                                   | $1$                                                                                     |
| **Linear Projection Matrices** | $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$              | $W_Q \in \mathbb{R}^{d \times d}$, $W_K, W_V \in \mathbb{R}^{d \times (G \cdot d_{\text{head}})}$ | $W_Q \in \mathbb{R}^{d \times d}$, $W_K, W_V \in \mathbb{R}^{d \times d_{\text{head}}}$ |
| **KV-Cache Size per Step**     | $2 \cdot H \cdot d_{\text{head}} \cdot \text{precision}$ | $2 \cdot G \cdot d_{\text{head}} \cdot \text{precision}$                                          | $2 \cdot 1 \cdot d_{\text{head}} \cdot \text{precision}$                                |
| **Serving Throughput**         | Baseline ($1\times$)                                     | **$3\times\text{--}6\times$ Higher Throughput**                                                   | **$5\times\text{--}8\times$ Higher Throughput**                                         |
| **Quality Retention**          | 100% (Reference)                                         | **99.5% - 100% Parity with MHA**                                                                  | 96% - 98% (Slight Perplexity Penalty)                                                   |
| **Adoption in Modern LLMs**    | GPT-2, GPT-3, Original LLaMA                             | **LLaMA-3, Qwen-2.5, Mistral-7B**                                                                 | Falcon-40B, StarCoder                                                                   |

---

## 4. Hardware-Fused Kernels: PyTorch SDPA & FlashAttention

In standard attention, calculating $\text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$ requires materializing the intermediate $B \times H \times T \times T$ attention score matrix in GPU High-Bandwidth Memory (HBM).

```text
Standard Attention Execution (Memory-Bound Bottleneck):
  1. Load Q, K from HBM ──► Compute S = Q @ K^T ──► Write S to HBM (O(T^2) memory footprint)
  2. Load S from HBM    ──► Compute P = Softmax(S) ──► Write P to HBM (O(T^2) memory footprint)
  3. Load P, V from HBM ──► Compute O = P @ V     ──► Write O to HBM

```

### The FlashAttention Solution

FlashAttention restructures the attention algorithm to be **IO-Aware**:

```text
GPU Hardware Architecture:
┌────────────────────────────────────────────────────────────────────────┐
│ Global Memory (HBM): Large (~80GB), Slow Bandwidth (~2-3 TB/s)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Tiled Block Transfers
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ On-Chip SRAM (Shared Memory): Tiny (~100-256KB/SM), Ultra-Fast (~19TB/s) │
│  • Tiles Q, K, V blocks into fast SRAM cache.                            │
│  • Computes Softmax online incrementally using Safe-Softmax statistics.  │
│  • Never writes the O(T^2) matrix back to HBM.                           │
└──────────────────────────────────────────────────────────────────────────┘

```

### PyTorch Scaled Dot-Product Attention (`F.scaled_dot_product_attention`)

PyTorch provides native C++/CUDA kernel acceleration via `torch.nn.functional.scaled_dot_product_attention` (SDPA).

At runtime, SDPA inspects the tensor shape, device architecture, and dtype to automatically select the fastest hardware kernel:

1. **FlashAttention-2/3:** Active on NVIDIA Ampere/Hopper/Blackwell architectures with FP16/BF16 dtypes and unmasked/causal flags.
2. **Memory-Efficient Attention (Cutlass/xFormers):** Fallback for custom masks or legacy GPU hardware.
3. **Optimized C++ Math:** Fallback for CPU execution or unsupported tensor layouts.

---

## 5. PyTorch Implementation: Production GQA Module with Native SDPA

Below is the implementation of a modular **Grouped-Query Attention (GQA)** layer supporting arbitrary query-to-KV head ratios, automatic tensor broadcasting, and hardware acceleration via `torch.nn.functional.scaled_dot_product_attention`:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    """
    Production-grade Grouped-Query Attention (GQA) with native PyTorch SDPA acceleration.

    Supports:
      • Multi-Head Attention (MHA):  num_heads == num_kv_heads
      • Grouped-Query Attention (GQA): num_heads > num_kv_heads > 1
      • Multi-Query Attention (MQA):  num_kv_heads == 1
    """
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        num_kv_heads: int,
        block_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        assert n_head % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.n_embd = n_embd
        self.n_head = n_head
        self.num_kv_heads = num_kv_heads
        self.head_dim = n_embd // n_head
        self.num_queries_per_kv = n_head // num_kv_heads
        self.dropout_p = dropout

        # Linear projections
        # Q projects to full hidden dimension: (n_head * head_dim)
        self.q_proj = nn.Linear(n_embd, self.n_head * self.head_dim, bias=False)
        # K and V project to the compressed KV dimension: (num_kv_heads * head_dim)
        self.k_proj = nn.Linear(n_embd, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, self.num_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = True,
        attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, C = x.shape  # Batch size, Sequence length, Embedding dim

        # 1. Project Q, K, V
        q = self.q_proj(x) # (B, T, n_head * head_dim)
        k = self.k_proj(x) # (B, T, num_kv_heads * head_dim)
        v = self.v_proj(x) # (B, T, num_kv_heads * head_dim)

        # 2. Reshape into multi-head tensor layouts: (B, num_heads, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. GQA Expansion: Broadcast KV heads across their assigned query groups
        if self.num_queries_per_kv > 1:
            # Repeat KV heads along the head dimension to match Q shape: (B, n_head, T, head_dim)
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # 4. Hardware-Fused FlashAttention / SDPA Kernel Execution
        # If is_causal=True and attn_mask is None, PyTorch routes to FlashAttention-2 causal kernel
        dropout_rate = self.dropout_p if self.training else 0.0

        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=dropout_rate,
            is_causal=is_causal and (attn_mask is None)
        )

        # 5. Concatenate heads and apply final output projection
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

```

---

## 6. Architectural Decision Framework

When configuring attention topologies for continuous training workflows:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ ATTENTION TOPOLOGY SELECTION HEURISTIC                                   │
├──────────────────────────────┬───────────────────────────────────────────┤
│ Target Scenario              │ Recommended Topology & Configuration      │
├──────────────────────────────┼───────────────────────────────────────────┤
│ 1. Small Micro-Models        │ Full Multi-Head Attention (MHA)           │
│    (Params < 500M, L < 2048) │ • KV-cache fits easily in consumer VRAM.  │
│                              │ • Preserves maximum representational      │
│                              │   capacity on small datasets.             │
├──────────────────────────────┼───────────────────────────────────────────┤
│ 2. Production Foundation LLMs│ Grouped-Query Attention (GQA)             │
│    (Params ≥ 1B, L ≥ 8192)   │ • Set N_kv = 8 with N_q = 32 (4:1 ratio)  │
│                              │ • Cuts KV-cache memory by 75%.            │
│                              │ • Delivers 100% downstream task parity.   │
├──────────────────────────────┼───────────────────────────────────────────┤
│ 3. Edge / Latency-Critical   │ Multi-Query Attention (MQA)               │
│    (Mobile / Real-Time Spec) │ • Single KV-head maximizes inference      │
│                              │   token-generation throughput.            │
└──────────────────────────────┴───────────────────────────────────────────┘

```
