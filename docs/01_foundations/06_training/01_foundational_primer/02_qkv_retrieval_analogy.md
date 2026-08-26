# The Query, Key, Value (Q, K, V) Retrieval Analogy

## 1. The Database Retrieval Mental Model

In traditional computing, an information retrieval system searches a database using three components:

1. **Query ($Q$):** The search term submitted by a user.
2. **Key ($K$):** The indexing tags associated with each record in the database.
3. **Value ($V$):** The actual underlying content stored inside the record.

When a query matches a key, the database returns the corresponding value.

```text
Database Analogy:
  Search Query (Q)  ──► [ Match Index with Keys (K) ] ──► [ Retrieve Content (V) ]

Transformer Analogy:
  Token Seeking Context (Q) ──► [ Scaled Dot-Product (Q · K^T) ] ──► [ Weighted Sum of Values (V) ]

```

In the Transformer architecture, **every token simultaneously acts as a Query, a Key, and a Value**:

- **Query ($Q$):** What the token is looking for (_"I am a pronoun; where is my antecedent noun?"_).

- **Key ($K$):** What the token offers to other tokens (_"I am a singular masculine proper noun"_).
- **Value ($V$):** The substantive semantic and contextual payload that will be extracted and routed if a match occurs (_"Semantic payload: 'Romeo'"_).

Unlike traditional databases that return binary (hit/miss) results, the Transformer executes a **soft, continuous retrieval**: every token queries all other accessible tokens, calculates continuous affinity scores, and computes a weighted average of all values.

---

## 2. Linear Projections: Constructing $Q, K, V$

An incoming token representation $x \in \mathbb{R}^{d_{\text{model}}}$ contains general embedding information. To participate in attention, it must be projected into specialized subspaces for querying, key matching, and value aggregation.

For an input sequence matrix $X \in \mathbb{R}^{T \times d_{\text{model}}}$ (where $T$ is sequence length and $d_{\text{model}}$ is hidden dimension), the projections are computed via learnable weight matrices $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

In most implementations, projection biases are set to `False` to maintain linear scaling properties and reduce parameter counts:

$$\text{self.query} = \text{nn.Linear}(d_{\text{model}}, d_k, \text{bias}=\text{False})$$

$$\text{self.key} = \text{nn.Linear}(d_{\text{model}}, d_k, \text{bias}=\text{False})$$

$$\text{self.value} = \text{nn.Linear}(d_{\text{model}}, d_v, \text{bias}=\text{False})$$

```text
Input Matrix X:               (T x d_model)
                                   │
           ┌───────────────────────┼───────────────────────┐
           ▼                       ▼                       ▼
    Linear(d_model, d_k)    Linear(d_model, d_k)    Linear(d_model, d_v)
           │                       │                       │
           ▼                       ▼                       ▼
    Query Matrix Q:          Key Matrix K:           Value Matrix V:
    (T x d_k)                (T x d_k)               (T x d_v)

```

---

## 3. Scaled Dot-Product Attention: Mathematical Formulation

The complete mathematical operation of a single attention head is defined as:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

This equation executes in five sequential mathematical operations:

```text
Step 1: Compute Raw Affinity Scores         S = Q · K^T                    (T x T)
Step 2: Scale Variance                      S_scaled = S / sqrt(d_k)       (T x T)
Step 3: Apply Causal / Structural Mask      S_masked = Mask(S_scaled)      (T x T)
Step 4: Normalize to Probabilities          A = Softmax(S_masked)          (T x T)
Step 5: Weighted Value Aggregation          Out = A · V                    (T x d_v)

```

### Step 1: Raw Affinity Calculation ($Q K^T$)

The dot product between query vector $q_i$ and key vector $k_j$ measures directional alignment in the projection subspace:

$$S_{i, j} = q_i \cdot k_j = \sum_{m=1}^{d_k} q_{i, m} k_{j, m}$$

Higher dot-product values represent stronger semantic relevance between token $i$ and token $j$.

### Step 2: The Scaling Factor ($\frac{1}{\sqrt{d_k}}$)

Without scaling, as the head dimension $d_k$ increases, the variance of the dot products grows significantly.

**Variance Proof:**
Assume the components of $q$ and $k$ are independent random variables with mean $\mu = 0$ and variance $\sigma^2 = 1$. The dot product is:

$$S_{i, j} = \sum_{m=1}^{d_k} q_m k_m$$

The expected value and variance are:

$$\mathbb{E}[S_{i, j}] = \sum_{m=1}^{d_k} \mathbb{E}[q_m] \mathbb{E}[k_m] = 0$$

$$\text{Var}(S_{i, j}) = \sum_{m=1}^{d_k} \text{Var}(q_m k_m) = \sum_{m=1}^{d_k} \left( \text{Var}(q_m)\text{Var}(k_m) \right) = \sum_{m=1}^{d_k} (1 \times 1) = d_k$$

When $d_k$ is large (e.g., $d_k = 64$ or $128$), the standard deviation reaches $\sqrt{d_k} = 8$ or $11.3$. Large magnitude inputs push the $\text{Softmax}$ function into regions with near-zero gradients ($\frac{\partial \text{Softmax}}{\partial z} \approx 0$), leading to **vanishing gradients** and stalled learning.

Dividing by $\sqrt{d_k}$ normalizes the variance back to $1$:

$$\text{Var}\left(\frac{S_{i, j}}{\sqrt{d_k}}\right) = \frac{\text{Var}(S_{i, j})}{d_k} = \frac{d_k}{d_k} = 1$$

### Step 3: Causal Masking (Autoregressive Decoders)

In decoder-only language models, tokens must not attend to future positions. A lower-triangular mask matrix fills invalid future positions ($j > i$) with $-\infty$:

$$S_{\text{masked}}[i, j] = \begin{cases} S_{\text{scaled}}[i, j], & \text{if } j \le i \\ -\infty, & \text{if } j > i \end{cases}$$

### Step 4: Softmax Normalization

The Softmax function normalizes attention scores into a valid probability distribution across rows:

$$A_{i, j} = \frac{\exp(S_{\text{masked}}[i, j])}{\sum_{n=1}^T \exp(S_{\text{masked}}[i, n])}$$

Because $\exp(-\infty) = 0$, all future tokens receive an attention probability of exactly $0.0$, strictly enforcing causality.

### Step 5: Weighted Value Aggregation

The attention weight matrix $A \in \mathbb{R}^{T \times T}$ multiplies the Value matrix $V \in \mathbb{R}^{T \times d_v}$:

$$\text{Out}_i = \sum_{j=1}^T A_{i, j} v_j$$

The resulting vector $\text{Out}_i \in \mathbb{R}^{d_v}$ is a contextually enriched representation of token $i$, containing information gathered from all relevant tokens across the sequence.

---

## 4. Tensor Shape Transformation Lifecycle

Tracking tensor dimensions across batch size ($B$), sequence length ($T$), model dimension ($C = d_{\text{model}}$), and head dimension ($d_k$):

| Stage               | Operation / Expression                 | Input Shape                      | Output Shape  | Description                                |
| ------------------- | -------------------------------------- | -------------------------------- | ------------- | ------------------------------------------ |
| **1. Input**        | Raw activations $x$                    | —                                | $(B, T, C)$   | Context vectors entering attention head.   |
| **2. Projections**  | $Q = x W_Q, K = x W_K, V = x W_V$      | $(B, T, C)$                      | $(B, T, d_k)$ | Projections into query, key, value spaces. |
| **3. Dot Product**  | $Q @ K^T$                              | $(B, T, d_k) \times (B, d_k, T)$ | $(B, T, T)$   | Pairwise affinity score matrix.            |
| **4. Scale & Mask** | $(Q K^T / \sqrt{d_k}) + M$             | $(B, T, T)$                      | $(B, T, T)$   | Variance normalization and future masking. |
| **5. Softmax**      | $\text{Softmax}(\cdot, \text{dim}=-1)$ | $(B, T, T)$                      | $(B, T, T)$   | Normalized attention probability weights.  |
| **6. Aggregation**  | $A @ V$                                | $(B, T, T) \times (B, T, d_k)$   | $(B, T, d_k)$ | Contextually aggregated output vectors.    |

---

## 5. Implementation: Self-Attention Head in PyTorch

Below is the clean PyTorch implementation of a single self-attention head with causal masking support:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttentionHead(nn.Module):
    """
    A single attention head executing Scaled Dot-Product Attention
    with causal masking.
    """
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        # Linear projection matrices (W_q, W_k, W_v) without bias
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Lower-triangular causal mask buffer (registered into state_dict)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # Stochastic regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B = Batch size, T = Sequence length, C = Embedding dimension (n_embd)
        B, T, C = x.shape

        # 1. Project inputs to Query, Key, Value spaces
        q = self.query(x) # (B, T, head_size)
        k = self.key(x)   # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # 2. Compute affinity scores: (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # Scaled by 1 / sqrt(head_size)
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)

        # 3. Apply causal mask: fill future positions (where tril == 0) with -inf
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # 4. Normalize scores into a probability distribution
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # 5. Aggregate values: (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        out = wei @ v
        return out

```

---

## 6. Self-Attention vs. Cross-Attention

While self-attention queries its own input sequence, the QKV formulation naturally extends to multi-modal and encoder-decoder architectures (e.g., MiniT5) through **Cross-Attention**:

| Feature                | Self-Attention (Decoder / Encoder)       | Cross-Attention (Seq2Seq Decoder)                          |
| ---------------------- | ---------------------------------------- | ---------------------------------------------------------- |
| **Query Source ($Q$)** | Derived from current sequence $X$<br>    | Derived from current decoder sequence $X_{\text{dec}}$<br> |
| **Key Source ($K$)**   | Derived from current sequence $X$<br>    | Derived from upstream encoder output $X_{\text{enc}}$<br>  |
| **Value Source ($V$)** | Derived from current sequence $X$<br>    | Derived from upstream encoder output $X_{\text{enc}}$<br>  |
| **Causal Masking**     | Active in decoders; inactive in encoders | Inactive (decoder attends to all encoder tokens)           |
| **Operational Role**   | Contextual intra-sequence token routing  | Inter-sequence conditioning and translation bridge         |
