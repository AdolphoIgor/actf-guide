# Positional Encodings: Absolute, RoPE, ALiBi, and Length Generalization

## 1. The Positional Dilemma in Self-Attention

Because standard Scaled Dot-Product Self-Attention is permutation equivariant, token representations contain zero intrinsic awareness of temporal or spatial order.

To model syntax, narrative progression, and causal logic, spatial coordinates must be injected into the network. The evolution of positional representations reflects a transition from static absolute coordinates toward relative, rotation-based, and bias-driven formulations designed for context length generalization.

```text
Generation 1: Absolute Coordinate Injection (Input Layer)
  • Learned Absolute Embeddings (GPT-2, BERT): E_total = E_tok + E_pos
  • Fixed Sinusoidal Embeddings (Vaswani et al.): Static sin/cos coordinate lookup

Generation 2: Relative Distance Matrices (Attention Scores)
  • Relative Position Representations (Shaw et al., T5): Modulates attention logits with learned distance matrices

Generation 3: Dynamic Relative Coordinate Transforms (Q/K Projections & Attention Biases)
  • Rotary Position Embedding (RoPE): Multiplicative 2D rotation of Query and Key vectors
  • Attention with Linear Biases (ALiBi): Non-learned geometric distance penalties in attention logits

```

---

## 2. Absolute Encodings vs. Relative Paradigms

### A. Absolute Learned Positional Embeddings

In standard early architectures (GPT-2, BERT), positions are parameterized as a static lookup table $W_{\text{pos}} \in \mathbb{R}^{L_{\max} \times d_{\text{model}}}$.

- **Structural Failure:** The model can never evaluate sequences longer than the predefined context limit $L_{\max}$. Attempting inference on $T = L_{\max} + 1$ causes an index out-of-bounds error.
- **Lack of Shift Invariance:** The model does not inherently know that the semantic relationship between positions $(1, 2)$ is identical to $(101, 102)$. It must learn translation invariance purely through empirical training exposure.

### B. The Relative Formulation

Relative positional schemes enforce the mathematical inductive bias that the attention weight between token $i$ and token $j$ should depend strictly on their relative displacement $\Delta = i - j$, rather than their absolute index coordinates:

$$\text{Attn}(i, j) = f(q_i, k_j, i - j)$$

---

## 3. Rotary Position Embedding (RoPE)

Introduced by Su et al. (RoFormer) and adopted by modern foundational models (LLaMA-3, Qwen-2.5, Mistral, Gemma), **Rotary Position Embedding (RoPE)** encodes positional coordinates by rotating Query and Key vectors in 2D sub-vector planes.

```text
Query / Key Vector (Dimension d_head = 64):
  Divided into d/2 (32) independent 2D sub-vectors:
  [ (q_0, q_1), (q_2, q_3), (q_4, q_5), ..., (q_{d-2}, q_{d-1}) ]
         │             │             │
         ▼             ▼             ▼
  Rotate by m*θ_0  Rotate by m*θ_1  Rotate by m*θ_2

```

### Mathematical Formulation

Given a token at position $m$ with a 2D sub-vector $x = (x^{(1)}, x^{(2)})^T \in \mathbb{R}^2$, RoPE applies an orthogonal rotation matrix $R_{\Theta, m}^2$:

$$R_{\Theta, m}^2 = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}$$

For a full head dimension $d$, the rotation matrix is block-diagonal:

$$R_{\Theta, m}^d = \text{diag}\left( R_{\Theta, m, 1}^2, R_{\Theta, m, 2}^2, \dots, R_{\Theta, m, d/2}^2 \right)$$

Where the frequency parameters $\theta_i$ scale geometrically across channel dimensions:

$$\theta_i = \text{base}^{-2(i-1)/d}, \quad i \in \left\{1, 2, \dots, \frac{d}{2}\right\}$$

Conventionally, $\text{base} = 10,000$ (or up to $500,000\text{--}1,000,000$ in extended-context LLMs).

### The Inner Product Invariant

When computing the attention score between query $q$ at position $m$ and key $k$ at position $n$:

$$\langle R_{\Theta, m}^d q, \, R_{\Theta, n}^d k \rangle = q^T \left( R_{\Theta, m}^d \right)^T R_{\Theta, n}^d k = q^T R_{\Theta, n - m}^d k$$

Because $\left(R_{\Theta, m}\right)^T R_{\Theta, n} = R_{\Theta, n - m}$, the dot product between the transformed query and key encodes relative displacement $(m - n)$ directly.

### Computational Optimization: 2D Complex Multiplication

Materializing a full $d \times d$ sparse rotation matrix is computationally inefficient. In practice, RoPE is implemented via vectorized element-wise multiplication:

$$R_{\Theta, m}^d x = \begin{pmatrix} x^{(1)} \\ x^{(2)} \\ x^{(3)} \\ x^{(4)} \\ \vdots \end{pmatrix} \odot \begin{pmatrix} \cos(m\theta_1) \\ \cos(m\theta_1) \\ \cos(m\theta_2) \\ \cos(m\theta_2) \\ \vdots \end{pmatrix} + \begin{pmatrix} -x^{(2)} \\ x^{(1)} \\ -x^{(4)} \\ x^{(3)} \\ \vdots \end{pmatrix} \odot \begin{pmatrix} \sin(m\theta_1) \\ \sin(m\theta_1) \\ \sin(m\theta_2) \\ \sin(m\theta_2) \\ \vdots \end{pmatrix}$$

---

## 4. Attention with Linear Biases (ALiBi)

Introduced by Press et al. (Train Short, Test Long), **ALiBi** discards positional embedding tables and rotary transformations entirely.

Instead, it injects a static, non-learned linear penalty directly into the raw self-attention score matrix prior to the Softmax operation.

```text
Standard Self-Attention Logits:
  S_{i, j} = (q_i · k_j) / sqrt(d_k)

ALiBi Modified Logits:
  S_{i, j} = (q_i · k_j) / sqrt(d_k) - m · (i - j)
  where (i - j) is token distance and m is a head-specific slope.

```

```text
ALiBi Linear Bias Matrix (Head Slope m = 1.0, Causal Dec):
         Tok 1   Tok 2   Tok 3   Tok 4
Tok 1  [   0      -inf    -inf    -inf  ]
Tok 2  [  -1        0     -inf    -inf  ]
Tok 3  [  -2       -1       0     -inf  ]
Tok 4  [  -3       -2      -1       0   ]

```

### Head-Specific Geometric Slopes

To allow different attention heads to track different context spans, each head $h \in \{1, \dots, H\}$ receives an immutable geometric slope $m$:

$$m_h = 2^{-\frac{8 h}{H}}$$

For a model with $H = 8$ heads, the slope ratios evaluate to:

$$\left\{ \frac{1}{2^1}, \frac{1}{2^2}, \frac{1}{2^3}, \frac{1}{2^4}, \frac{1}{2^5}, \frac{1}{2^6}, \frac{1}{2^7}, \frac{1}{2^8} \right\} = \left\{ \frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \dots, \frac{1}{256} \right\}$$

Heads with large slopes focus narrowly on immediate local context, while heads with small slopes retain sensitivity across long-range contexts.

---

## 5. Architectural Comparison Matrix

| Dimension                          | Absolute Learned (GPT-2)                      | Sinusoidal (Vaswani) | ALiBi (MPT, Bloom)            | RoPE (Llama-3, Qwen-2.5)                  |
| ---------------------------------- | --------------------------------------------- | -------------------- | ----------------------------- | ----------------------------------------- |
| **Injection Point**                | Input Embeddings                              | Input Embeddings     | Attention Matrix ($Q K^T$)    | Q and K Projections                       |
| **Parameter Overhead**             | $L_{\max} \times d_{\text{model}}$ parameters | 0 parameters         | 0 parameters                  | 0 parameters                              |
| **Relative Invariant**             | No                                            | Partially            | Yes (Linear additive)         | Yes (Rotational inner product)            |
| **Zero-Shot Length Extrapolation** | Hard failure at $T > L_{\max}$                | Poor generalization  | High extrapolation            | Moderate (Requires scaling like YaRN/NTK) |
| **FlashAttention Compatibility**   | Native                                        | Native               | Requires custom additive bias | Native (Q/K transformed pre-kernel)       |
| **KV-Cache Memory Overhead**       | 0%                                            | 0%                   | 0%                            | 0% (Keys cached post-rotation)            |

---

## 6. PyTorch Implementation: Rotary Position Embedding (RoPE)

Below is the standalone implementation of Rotary Positional Embeddings designed for integration with standard multi-head and grouped-query attention blocks:

```python
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Position Embedding (RoPE) over 2D sub-vectors.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "Head dimension must be even for RoPE"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # 1. Compute theta frequencies: theta_i = base^(-2(i-1)/dim)
        # Shape: (dim // 2,)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 2. Precompute cos and sin lookup cache up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        # Generate position coordinates: [0, 1, ..., seq_len - 1]
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)

        # Outer product: (seq_len, dim // 2)
        freqs = torch.outer(t, self.inv_freq)

        # Duplicate frequencies to cover full head dimension: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Precompute cos and sin buffers: (1, 1, seq_len, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Splits tensor in half along channel dimension and rotates: [-x2, x1]."""
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies RoPE rotation to Query and Key tensors.
        Input shapes: (Batch, Num_Heads, Seq_Len, Head_Dim)
        """
        seq_len = q.shape[2]
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        cos = self.cos_cached[:, :, :seq_len, :].to(q.dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(q.dtype)

        # Vectorized 2D rotation
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot

```

---

## 7. Context Length Extension Dynamics (NTK-Aware Scaling & YaRN)

When fine-tuning a pre-trained RoPE model on sequences longer than its original pre-training context ($L_{\text{target}} > L_{\text{pretrain}}$), naive positional extrapolation causes high perplexity because high-frequency Fourier components enter unobserved rotation domains.

### 1. Linear Position Interpolation (PI)

Downscales position indices linearly:

$$m' = m \times \frac{L_{\text{pretrain}}}{L_{\text{target}}}$$

While preventing out-of-distribution rotation angles, linear interpolation compresses high frequencies excessively, impairing the model's ability to distinguish immediately adjacent tokens.

### 2. NTK-Aware Scaling

Neural Tangent Kernel (NTK) interpolation scales the base frequency rather than the input positions:

$$\text{base}' = \text{base} \times \alpha^{\frac{d}{d - 2}}, \quad \text{where } \alpha = \frac{L_{\text{target}}}{L_{\text{pretrain}}}$$

By modifying the base frequency dynamically, low-frequency channels (which govern broad global positioning) are interpolated, while high-frequency channels (which govern local syntax and token order) remain unscaled. This allows modern models to extend context windows from $8\text{k}$ to $128\text{k}+$ tokens with minimal fine-tuning compute.
