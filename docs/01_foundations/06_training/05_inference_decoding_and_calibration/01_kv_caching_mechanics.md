# Key-Value (KV) Caching Mechanics and Inference Optimization

## 1. The Autoregressive Generation Bottleneck

During autoregressive text generation, a language model emits tokens sequentially. To predict token $x_{t+1}$, the model conditions on the full historical context $x_{\le t} = (x_1, x_2, \dots, x_t)$.

In a naive decoder implementation without state retention, each generation step passes the entire accumulated sequence through the network from scratch.

```text
Naive Autoregressive Generation (Without KV Cache - Quadratic O(T^2) FLOPs):

Step 1: Input [ "The" ]                   ──► Compute Attn on [ "The" ]              ──► Emit "sky"
Step 2: Input [ "The", "sky" ]            ──► Recompute Attn on [ "The", "sky" ]     ──► Emit "is"
Step 3: Input [ "The", "sky", "is" ]       ──► Recompute Attn on [ "The", "sky", "is"]──► Emit "blue"
Step 4: Input [ "The", "sky", "is", "blue" ]──► Recompute Attn on ALL 4 tokens       ──► Emit "."

Total Attention Dot-Products across T steps: 1 + 2 + 3 + ... + T = O(T^2) Redundant Matrix Multiplications

```

Because Transformer layers are causal, the Key ($K$) and Value ($V$) projections for past tokens $x_1, \dots, x_{t-1}$ do not change when a new token $x_t$ is appended. Recomputing these representations at every step wastes compute cycles and scales execution latency quadratically ($\mathcal{O}(T^2)$).

**Key-Value Caching (KV-Caching)** stores past Key and Value projection tensors in high-speed GPU memory. At step $t$, the model processes only the single newly generated token $x_t$, retrieving past $K$ and $V$ vectors directly from the cache to achieve **linear $\mathcal{O}(T)$ total generation compute**.

---

## 2. Mathematical Mechanics of the KV Cache

Let $x_t \in \mathbb{R}^{1 \times d_{\text{model}}}$ be the token embedding at step $t$. Within an attention head of dimension $d_k$:

$$Q_t = x_t W_Q \in \mathbb{R}^{1 \times d_k}, \quad K_t = x_t W_K \in \mathbb{R}^{1 \times d_k}, \quad V_t = x_t W_V \in \mathbb{R}^{1 \times d_k}$$

```text
KV Cache Update and Single-Query Attention Step:

1. New Token Projection:
   x_t ──► Q_t (1 x d_k), K_t (1 x d_k), V_t (1 x d_k)

2. Cache Concatenation (In-Memory Tensor Extension):
   K_past ( (t-1) x d_k )  +  K_t ( 1 x d_k )  ──►  K_cached ( t x d_k )
   V_past ( (t-1) x d_k )  +  V_t ( 1 x d_k )  ──►  V_cached ( t x d_k )

3. Single-Row Attention Dot-Product:
   Score Vector:  S_t = (Q_t @ K_cached^T) / sqrt(d_k)       ──► Shape: (1 x t)
   Softmax:       P_t = Softmax(S_t)                         ──► Shape: (1 x t)
   Output Vector: O_t = P_t @ V_cached                       ──► Shape: (1 x d_k)

```

$$\text{Attention}(Q_t, K_{\le t}, V_{\le t}) = \text{Softmax}\left(\frac{Q_t K_{\le t}^T}{\sqrt{d_k}}\right) V_{\le t}$$

### Key Mathematical Invariants

* **Query Vector ($Q_t$):** Remains a single-row vector of shape $(1 \times d_k)$ representing only the current decoding position.
* **Cached Keys ($K_{\le t}$):** Has shape $(t \times d_k)$, containing representations from position $1$ to $t$.
* **Attention Score Vector ($S_t$):** Evaluates to a $1 \times t$ row vector rather than a dense $t \times t$ matrix.
* **Causal Mask Redundancy:** Because $Q_t$ only attends to keys at positions $j \le t$ (all of which exist in the past), explicit causal lower-triangular masking is not required during single-token decoding steps.

---

## 3. Two Operational Phases: Prefill vs. Decode

Autoregressive inference executes across two distinct computational regimes with different hardware utilization profiles:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 1. PREFILL PHASE (Prompt Ingestion)                                    │
│    • Input: Entire prompt sequence X = (x_1, ..., x_N)                 │
│    • Computation: Large General Matrix Multiplications (GEMM)          │
│    • Hardware Bound: Compute-bound (Tensor Core utilization ~80-95%)   │
│    • Action: Populates initial KV cache entries for all N prompt tokens│
├────────────────────────────────────────────────────────────────────────┤
│ 2. DECODE PHASE (Autoregressive Token-by-Token Rollout)                │
│    • Input: Single token x_t at each step                              │
│    • Computation: General Matrix-Vector Multiplications (GEMV)         │
│    • Hardware Bound: Memory-bandwidth bound (Arithmetic intensity low) │
│    • Action: Reads full KV cache from HBM, appends 1 token, writes back│
└────────────────────────────────────────────────────────────────────────┘

```

### Computational Characteristics Comparison

| Dimension | Prefill Phase (Prompt Ingestion) | Decode Phase (Token Generation) |
| --- | --- | --- |
| **Input Shape per Batch** | $(B, T_{\text{prompt}}, d_{\text{model}})$ | $(B, 1, d_{\text{model}})$ |
| **Primary Math Operation** | Matrix-Matrix Multiplication (GEMM) | Matrix-Vector Multiplication (GEMV) |
| **GPU Bottleneck** | **Compute Bound** (TFLOPs / Tensor Cores) | **Memory-Bandwidth Bound** (HBM TB/s) |
| **Attention Matrix Shape** | $(T_{\text{prompt}} \times T_{\text{prompt}})$ | $(1 \times T_{\text{current}})$ |
| **Arithmetic Intensity** | High (Many FLOPs per byte loaded) | Low ($\approx 1\text{--}2\text{ FLOPs per byte loaded}$) |
| **Causal Mask** | Requires 2D lower-triangular mask | No mask needed ($Q$ attends to all past $K$) |

---

## 4. KV Cache Memory Footprint & Capacity Sizing

While the KV cache eliminates redundant compute, it introduces a substantial high-bandwidth memory (HBM) footprint that grows dynamically with batch size and context length.

### The KV Cache Memory Formula

For a model with $N_{\text{layers}}$ layers, $N_{\text{kv\_heads}}$ Key-Value heads, head dimension $d_{\text{head}}$, context length $L$, and batch size $B$, running in 16-bit precision (FP16 or BF16 = 2 bytes per element):

$$\text{Memory}_{\text{KV}} = 2 \times 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times d_{\text{head}} \times L \times B \quad \text{(in Bytes)}$$

Where:

* The first factor of $2$ accounts for storing both **Keys** and **Values**.
* The second factor of $2$ accounts for 2 bytes per parameter (16-bit precision).

$$\text{Memory per Token per Batch} = 4 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times d_{\text{head}} \quad \text{Bytes/token}$$

```text
KV Cache Memory Footprint across Context Lengths (Batch Size B = 1, 16-bit precision):

Model Architecture              Heads (Q / KV)  Layers  Head Dim  4K Context   32K Context  128K Context
────────────────────────────────────────────────────────────────────────────────────────────────────────
Llama-2 7B (Standard MHA)       32 / 32         32      128       1.00 GB      8.00 GB      32.00 GB
Llama-3 8B (Grouped-Query GQA)  32 / 8          32      128       0.25 GB      2.00 GB       8.00 GB
Llama-3 70B (Grouped-Query GQA) 64 / 8          80      128       0.62 GB      5.00 GB      20.00 GB
Mistral-7B (GQA + SWA 4K)       32 / 8          32      128       0.25 GB      0.25 GB*      0.25 GB*

```

**Mistral with Sliding Window Attention (SWA) bounds the cache footprint to the window size (4,096 tokens).*

### The Multi-Tenant Serving Impact

At a batch size of $B = 64$ with an $8\text{K}$ context window on a Llama-2 7B model (MHA), the KV cache alone demands:

$$\text{Memory} = 64 \times 8,192 \times (4 \times 32 \times 32 \times 128) \text{ Bytes} \approx 137.4 \text{ GB VRAM}$$

This requirement exceeds the physical memory of an 80GB GPU before accounting for the model's static weights (14 GB), making **Grouped-Query Attention (GQA)** and **PagedAttention** essential for production serving.

---

## 5. Interaction with Modern Architectural Topologies

### 1. Rotary Position Embeddings (RoPE) Integration

Because RoPE applies coordinate rotations based on absolute token positions $m$:

$$K_{\text{rot}, m} = R_{\Theta, m} K_m$$

Keys must be rotated **prior to insertion into the KV cache**:

* The cached key tensor stores the rotated representation $K_{\text{rot}, m}$.
* During generation at step $t$, the model computes $Q_{\text{rot}, t} = R_{\Theta, t} Q_t$, fetches cached $K_{\text{rot}, \le t}$, and computes the dot product directly.
* This avoids re-rotating past keys at each decoding step.

### 2. Grouped-Query Attention (GQA) Memory Savings

In standard Multi-Head Attention ($N_{\text{kv\_heads}} = N_{\text{q\_heads}} = 32$), 32 separate key/value tensors are cached per layer. In GQA ($N_{\text{kv\_heads}} = 8$), only 8 key/value tensors are stored:

$$\text{GQA Memory Reduction Factor} = \frac{N_{\text{q\_heads}}}{N_{\text{kv\_heads}}} = \frac{32}{8} = 4\times \text{ (75\% Memory Reduction)}$$

---

## 6. PyTorch Implementation: Dynamic KV-Cache Causal Attention

Below is the standalone implementation of a production-grade causal self-attention module with dynamic Key-Value caching, supporting both Prefill and Decode execution paths:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    """
    Manages in-memory storage and dynamic concatenation of Key and Value tensors.
    """
    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def update(
        self, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Appends new Key and Value tensors along the sequence dimension.
        Tensor shape: (Batch, Num_KV_Heads, Seq_Len, Head_Dim)
        """
        if self.k is None or self.v is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
            
        return self.k, self.v

    def clear(self):
        """Frees cache memory buffers."""
        self.k = None
        self.v = None


class CausalSelfAttentionWithCache(nn.Module):
    """
    Multi-Head / Grouped-Query Attention with dynamic KV Cache support.
    """
    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        num_kv_heads: int = 2,
        block_size: int = 2048,
        dropout: float = 0.0
    ):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        assert n_head % num_kv_heads == 0, "n_head must be divisible by num_kv_heads"

        self.d_model = d_model
        self.n_head = n_head
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // n_head
        self.num_queries_per_kv = n_head // num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        use_cache: bool = False
    ) -> tuple[torch.Tensor, KVCache | None]:
        B, T, C = x.shape

        # 1. Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 2. Update and retrieve Key/Value representations from cache
        if use_cache:
            if kv_cache is None:
                kv_cache = KVCache()
            k, v = kv_cache.update(k, v)

        # 3. GQA Expansion: Broadcast KV heads to match Query head count
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # 4. Scaled Dot-Product Attention
        # Prefill phase (T > 1) requires causal masking; Decode phase (T == 1) does not
        is_causal = (T > 1) and (not use_cache or q.shape[2] == k.shape[2])

        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )

        # 5. Project back to residual stream
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), kv_cache


class CachedGenerationEngine:
    """
    Demonstrates token-by-token autoregressive generation using KV Caching.
    """
    def __init__(self, model_attn: CausalSelfAttentionWithCache):
        self.attn = model_attn

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        self.attn.eval()
        cache = KVCache()

        # Step 1: PREFILL PHASE - Ingest full prompt, populate cache
        _, cache = self.attn(prompt_tokens, kv_cache=cache, use_cache=True)
        current_token = prompt_tokens[:, -1:] # (B, 1, C)

        generated = [prompt_tokens]

        # Step 2: DECODE PHASE - Roll out tokens one by one
        for _ in range(max_new_tokens):
            # Pass ONLY the newest single token
            out, cache = self.attn(current_token, kv_cache=cache, use_cache=True)
            
            # Simulated next-token selection (for illustration, uses the output directly)
            current_token = out[:, -1:, :]
            generated.append(current_token)

        return torch.cat(generated, dim=1)

```

---

## 7. Advanced KV Cache Optimizations in Modern Inference Engines

Production inference engines (e.g., vLLM, TensorRT-LLM, SGLang) implement advanced memory management architectures to mitigate KV cache bottlenecks:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ MODERN KV CACHE OPTIMIZATION PARADIGMS                                 │
├──────────────────────────────┬─────────────────────────────────────────┤
│ PagedAttention (vLLM)        │ Partitions KV cache into non-contiguous │
│                              │ fixed-size memory blocks (pages),       │
│                              │ eliminating virtual memory fragmentation│
│                              │ and reducing VRAM waste to < 4%.        │
├──────────────────────────────┼─────────────────────────────────────────┤
│ Sliding Window Attention     │ Restricts cache retention to the most   │
│ (Mistral SWA)                │ recent W tokens (e.g., W = 4,096). Past │
│                              │ tokens beyond window W are evicted.     │
├──────────────────────────────┼─────────────────────────────────────────┤
│ Prompt / Prefix Caching      │ Hashes common system prompts to reuse   │
│ (Multi-Turn Optimization)    │ precomputed KV tensors across requests, │
│                              │ reducing prefill latency to zero.       │
├──────────────────────────────┼─────────────────────────────────────────┤
│ KV Cache Quantization        │ Quantizes cached K and V tensors from   │
│ (FP8 / INT4 KV)              │ FP16 (16-bit) to FP8 (8-bit) or INT4,   │
│                              │ reducing VRAM footprint by 50% to 75%.  │
└──────────────────────────────┴─────────────────────────────────────────┘

```