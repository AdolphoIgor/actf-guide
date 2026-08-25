# Attention (Communication) vs. FFN (Computation) Anatomy

## 1. The Division of Labor in a Transformer Block

Every standard Transformer block is structured around a strict division of labor between two complementary sub-layers:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Multi-Head Attention (Social Routing / Communication)               │
│    • Moves information ACROSS sequence positions (Time dimension T).   │
│    • Allows tokens to interact, contextualize, and share state.        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Residual Connection
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Feed-Forward Network (Pointwise Reasoning / Computation)            │
│    • Processes information WITHIN each token independently (Channel C) │
│    • Acts as a key-value associative memory and non-linear scratchpad. │
└────────────────────────────────────────────────────────────────────────┘

```

Without Multi-Head Attention, tokens remain isolated islands unable to exchange context. Without the Feed-Forward Network, the model reduces to linear transformations of dynamic weighted averages, lacking the parameter capacity and non-linear depth required for reasoning and factual recall.

---

## 2. Multi-Head Attention (MHA): The Council of Experts

Rather than computing a single set of attention weights across the entire hidden dimension $d_{\text{model}}$, Multi-Head Attention splits the embedding into $h$ independent subspaces, where each head has dimension:

$$d_k = \frac{d_{\text{model}}}{h}$$

```text
Input Activation x (B x T x d_model)
                     │
      ┌──────────────┼──────────────┬──────────────┐
      ▼              ▼              ▼              ▼
    Head 1         Head 2         Head 3         Head h
 (d_k = d/h)    (d_k = d/h)    (d_k = d/h)    (d_k = d/h)
      │              │              │              │
      └──────────────┼──────────────┴──────────────┘
                     ▼
        Concatenate: (B x T x d_model)
                     │
                     ▼
        Output Projection: W_O (d_model x d_model)

```

### Why Multiple Heads?

A single attention head can only focus on one dominant relationship per token at a time (e.g., matching a pronoun to its antecedent noun). By allocating $h$ heads in parallel, different heads specialize simultaneously across distinct linguistic, syntactic, and semantic features:

* **Head 1:** Syntactic dependency (verbs attending to direct objects).


* **Head 2:** Positional adjacency (attending strictly to immediate predecessor tokens).
* **Head 3:** Coreference resolution (pronouns attending to character names).


* **Head 4:** Semantic and thematic affinity (nouns attending to associated adjectives).



### Mathematical Formulation

For $h$ heads, each head $i$ computes:

$$\text{head}_i = \text{Attention}(X W_Q^{(i)}, X W_K^{(i)}, X W_V^{(i)})$$

The individual head outputs are concatenated along the feature dimension and unified through a linear projection matrix $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W_O$$

The output projection $W_O$ mixes the independent findings of all heads back into a coherent token representation.

---

## 3. The Feed-Forward Network (FFN): The Pointwise Engine

Following cross-token communication in the attention layer, the token representations pass into a pointwise **Feed-Forward Network (FFN)**.

Unlike self-attention, the FFN applies the exact same set of linear transformations and non-linearities to every token position **identically and independently**:

$$\text{FFN}(x) = \sigma(x W_1 + b_1) W_2 + b_2$$

```text
Token Representation x:           (B x T x d_model)
                                          │
                                          ▼
Linear Expansion W_1:           (d_model -> 4 * d_model)
                                          │
                                          ▼
Non-Linear Activation σ:       GELU() / SwiGLU() / ReLU()
                                          │
                                          ▼
Linear Contraction W_2:        (4 * d_model -> d_model)
                                          │
                                          ▼
Dropout Regularization:               Dropout(p)

```

### 1. The $4 \times d_{\text{model}}$ Expansion (The Scratchpad Space)

In canonical architectures (e.g., GPT-2, LLaMA, BERT), the inner dimension of the FFN expands by a factor of 4 ($d_{\text{ff}} = 4 \times d_{\text{model}}$).

* **Associative Memory Storage:** Research indicates that the first linear layer ($W_1$) acts as an associative key memory, while the second linear layer ($W_2$) acts as a value retriever. The expanded intermediate space allows the network to store and recall vast amounts of factual knowledge without increasing context window computation.
* **Non-Linear Disentanglement:** Expanding the representation space allows non-linear activation functions to partition complex, multi-modal semantic relationships that cannot be separated in the narrower $d_{\text{model}}$ space.



### 2. Evolution of Activation Functions: ReLU to GELU to SwiGLU

* **ReLU (Rectified Linear Unit):** $\text{ReLU}(z) = \max(0, z)$. Fast to compute, but hard-zeros all negative activations, leading to "dead neurons" where gradients evaluate to zero and stop updating.


* **GELU (Gaussian Error Linear Unit):** $\text{GELU}(z) = z \cdot \Phi(z) = z \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{z}{\sqrt{2}}\right)\right]$. Smoothly curves for slightly negative values, allowing low-magnitude gradients to propagate through deep residual connections and preventing gradient starvation.


* **SwiGLU (Swish Gated Linear Unit):** $\text{SwiGLU}(x) = (x W_{\text{gate}} \cdot \text{SiLU}(x W_1)) W_2$. Utilized in modern architectures (LLaMA-3, Qwen-2.5) to introduce dynamic gating, rescaling the FFN hidden dimension to $\frac{8}{3} d_{\text{model}}$ to maintain parameter parity.

---

## 4. Architectural Anatomy of a Transformer Block

Modern Transformer implementations employ **Pre-LayerNorm** (or Pre-RMSNorm) residual connections:

```text
    Input Tensor: x
          │
          ├─── [ Skip Connection 1 ] ────────────────────────┐
          │                                                  │
          ▼                                                  │
    LayerNorm / RMSNorm                                      │
          │                                                  │
          ▼                                                  │
    Multi-Head Attention (Communication)                     │
          │                                                  │
          ▼                                                  │
          + <────────────────────────────────────────────────┘
          │
          ├─── [ Skip Connection 2 ] ────────────────────────┐
          │                                                  │
          ▼                                                  │
    LayerNorm / RMSNorm                                      │
          │                                                  │
          ▼                                                  │
    Feed-Forward Network (Computation / Scratchpad)          │
          │                                                  │
          ▼                                                  │
          + <────────────────────────────────────────────────┘
          │
    Output Tensor: x_out (B x T x d_model)

```

In Pre-LN topologies, the residual stream remains completely un-normalized, allowing error gradients during backpropagation to travel directly through the skip additions back to the initial embedding layers without numerical degradation.

---

## 5. Complete PyTorch Implementation

Below is the implementation of `MultiHeadAttention`, `FeedForward`, and the unified `Block` module:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """A single scaled dot-product attention head with causal masking."""
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # Scaled dot-product
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of attention running in parallel with projection."""
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate outputs from all heads along channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Pointwise MLP expanding to 4x hidden dimension with GELU activation."""
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    """Transformer Block: Interleaving Communication (MHA) and Computation (FFN)."""
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Residual Connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

```

---

## 6. Parameter and FLOP Distribution Breakdown

In a standard Transformer block with hidden dimension $d = d_{\text{model}}$:

| Component | Sub-Layer Matrices | Parameter Count (Excl. Biases) | Computational FLOPs (per token) | Primary Function |
| --- | --- | --- | --- | --- |
| **Attention (MHA)** | $W_Q, W_K, W_V, W_O$ | $4 d^2$ | $4 d^2 + 2 T d$ | Inter-token communication and routing. |
| **Feed-Forward (FFN)** | $W_1 (d \rightarrow 4d), W_2 (4d \rightarrow d)$ | $8 d^2$ | $8 d^2$ | Factual knowledge storage and reasoning. |
| **LayerNorm (LN1, LN2)** | $\gamma_1, \beta_1, \gamma_2, \beta_2$ | $4 d$ | $4 d$ | Numerical signal stabilization. |
| **Total per Block** | — | $\approx 12 d^2$ | $\approx 12 d^2 + 2 T d$ | Complete Transformer unit. |

The Feed-Forward layer contains **two-thirds ($66.7\%$) of the total parameter budget** in every Transformer block, while Attention handles the dynamic, sequence-dependent routing.
