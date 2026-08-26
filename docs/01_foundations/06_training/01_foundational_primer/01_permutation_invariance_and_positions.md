# The Permutation Invariance Dilemma and Positional Need

## 1. The Set Processing Nature of Self-Attention

A standard Multi-Layer Perceptron (MLP) processes inputs at fixed index positions, while a Recurrent Neural Network (RNN) processes tokens sequentially step-by-step ($h_t = f(h_{t-1}, x_t)$). In contrast, the core mechanism of the Transformer architecture—**Scaled Dot-Product Self-Attention**—is fundamentally an operation over an **unordered set of vectors**.

Without explicit positional intervention, self-attention exhibits **permutation equivariance**: shuffling the input sequence reorders the resulting output vectors identically without altering the pairwise contextual relationships computed between tokens.

```text
Sequence A: "The dog bit the man"
Sequence B: "The man bit the dog"

Without Positional Encodings:
  • "dog" attends to "bit" with the EXACT same attention weight in both sentences.
  • "man" attends to "bit" with the EXACT same attention weight in both sentences.
  • The model cannot distinguish subject from object.

```

If a language model cannot differentiate word order, it cannot learn grammar, syntax, causality, or sequential logic.

---

## 2. Mathematical Proof of Permutation Equivariance

Let an input sequence matrix be $X \in \mathbb{R}^{T \times d}$, where $T$ is sequence length and $d$ is embedding dimension. Let $P \in \mathbb{R}^{T \times T}$ be an arbitrary permutation matrix (a binary orthogonal matrix where $P^T P = I$).

The linear projections for Queries, Keys, and Values are computed via weight matrices $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

The standard self-attention equation is defined as:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

If we permute the input rows of $X$ by applying $P X$:

$$Q_P = (P X) W_Q = P Q$$

$$K_P = (P X) W_K = P K$$

$$V_P = (P X) W_V = P V$$

Computing the attention scores for the permuted input:

$$Q_P K_P^T = (P Q)(P K)^T = P Q K^T P^T$$

Applying the row-wise $\text{Softmax}$ operator (which commutes with permutation matrices across rows):

$$\text{Softmax}\left(\frac{P Q K^T P^T}{\sqrt{d_k}}\right) = P \, \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) P^T$$

Multiplying by the permuted value matrix $V_P$:

$$\text{Attention}(Q_P, K_P, V_P) = \left( P \, \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) P^T \right) (P V)$$

Since $P^T P = I$:

$$\text{Attention}(Q_P, K_P, V_P) = P \left( \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V \right) = P \, \text{Attention}(Q, K, V)$$

Because the permutation matrix $P$ factors out completely, the self-attention operator possesses zero inherent inductive bias regarding temporal or spatial order.

---

## 3. Positional Injection: Token GPS Coordinates

To break permutation equivariance, spatial coordinate signals must be explicitly injected into each token representation prior to entering the first attention layer.

In standard architectures (such as GPT-2 and canonical Transformer decoders), this is accomplished by defining a dedicated **Positional Embedding Matrix** and adding it directly to the token embedding matrix:

$$E_{\text{combined}} = E_{\text{token}}(x_t) + E_{\text{position}}(t)$$

```text
Token IDs:           [   1042,      581,    9124,    1042,      841   ]
                           │         │        │        │         │
Token Embeddings:    [  e_tok0,   e_tok1,  e_tok2,  e_tok3,   e_tok4  ] (Shape: B x T x d_model)
                           +         +        +        +         +
Position Indices:    [     0,        1,       2,       3,        4    ]
                           │         │        │        │         │
Position Embeddings: [  e_pos0,   e_pos1,  e_pos2,  e_pos3,   e_pos4  ] (Shape: 1 x T x d_model)
                           │         │        │        │         │
                           ▼         ▼        ▼        ▼         ▼
Input to Block 0:    [    x_0,      x_1,     x_2,     x_3,      x_4   ] (Shape: B x T x d_model)

```

### Why Vector Addition Instead of Concatenation?

Concatenating a position vector of size $d_p$ to a token vector of size $d_t$ increases the activation tensor dimensionality to $(d_t + d_p)$, forcing all downstream projection matrices ($W_Q, W_K, W_V, W_O, W_{\text{FFN}}$) to allocate additional memory and compute.

Vector addition preserves tensor shape ($B \times T \times d_{\text{model}}$) while utilizing the high-dimensional properties of vector spaces: in high dimensions (e.g., $d_{\text{model}} \ge 128$), independent subspaces can naturally represent semantic identity and positional offset simultaneously without destructive interference.

---

## 4. Implementation: Absolute Learned Embeddings in PyTorch

In a foundational decoder-only Transformer (such as MiniGPT), positional encodings are implemented as a learned lookup table (`nn.Embedding`) parameterized by the maximum context window (`block_size`) and hidden dimension (`n_embd`).

```python
import torch
import torch.nn as nn

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, block_size: int):
        super().__init__()
        # Token Identity Table (What token is this?)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Positional Coordinate Table (Where is this token located?)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx shape: (Batch_Size, Sequence_Length) -> (B, T)
        B, T = idx.shape

        # 1. Retrieve semantic token embeddings: (B, T, n_embd)
        tok_emb = self.token_embedding_table(idx)

        # 2. Generate integer range [0, 1, 2, ..., T-1] matching sequence length
        positions = torch.arange(T, device=idx.device) # Shape: (T,)

        # 3. Retrieve spatial position embeddings: (T, n_embd)
        pos_emb = self.position_embedding_table(positions)

        # 4. Broadcast addition: (B, T, n_embd) + (T, n_embd) -> (B, T, n_embd)
        x = tok_emb + pos_emb
        return x

```

### Step-by-Step Execution Trace

1. **Input:** A batch of tokenized sequences $\text{idx} \in \mathbb{Z}^{B \times T}$ where each integer represents a vocabulary index $0 \le i < V$.
2. **Token Lookup:** `tok_emb` indexes the embedding weights, retrieving a $d$-dimensional representation for each token ($B \times T \times d$).
3. **Coordinate Generation:** `torch.arange(T)` generates a 1D tensor representing index coordinates $[0, 1, \dots, T-1]$.
4. **Positional Lookup:** `pos_emb` fetches the learned position vectors for each coordinate index ($T \times d$).
5. **Broadcasting & Fusion:** PyTorch automatically broadcasts the $(T \times d)$ position tensor across the batch dimension $B$, producing an input tensor $x \in \mathbb{R}^{B \times T \times d}$ injected with spatial identity.

---

## 5. Architectural Trade-Offs & Limitations of Absolute Learned Embeddings

While absolute learned positional embeddings are computationally efficient and simple to implement, they impose strict structural limitations:

| Attribute                  | Absolute Learned Embeddings                                      | Operational Impact                                                                                                                                               |
| -------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Context Extrapolation**  | Hard failure at $T > \text{block\_size}$                         | The model cannot process sequences longer than the defined table size. Inference on $T = \text{block\_size} + 1$ throws an `IndexError`.                         |
| **Relative Distance Bias** | No explicit relative formulation                                 | The model must empirically learn that $\text{dist}(p_1, p_2) = \text{dist}(p_3, p_4) = 1$. It does not encode shift invariance $\Delta = i - j$ mathematically.  |
| **Parameter Overhead**     | Scales linearly: $O(\text{block\_size} \times d_{\text{model}})$ | For long-context models ($L = 32\text{k}\text{--}128\text{k}$), static positional lookup tables consume excessive memory.                                        |
| **Downstream Evolution**   | Replaced by RoPE / ALiBi in modern LLMs                          | Contemporary architectures (e.g., Llama-3, Qwen-2.5) replace learned absolute embeddings with Rotary Position Embeddings (RoPE) to enable length generalization. |
