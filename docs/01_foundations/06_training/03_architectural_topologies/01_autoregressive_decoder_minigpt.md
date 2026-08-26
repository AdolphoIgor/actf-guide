# Autoregressive Decoder-Only Transformers: MiniGPT Architecture

## 1. The Autoregressive Language Modeling Paradigm

The fundamental objective of an autoregressive language model is to estimate the joint probability distribution $P(X)$ over a sequence of discrete tokens $X = (x_1, x_2, \dots, x_T)$.

By applying the mathematical probability chain rule, the joint distribution factors into a sequence of conditional next-token probabilities:

$$P(X) = P(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P(x_t \mid x_1, x_2, \dots, x_{t-1}) = \prod_{t=1}^T P(x_t \mid x_{<t})$$

```text
Autoregressive Factorization:
  P("The", "dog", "barked") = P("The") * P("dog" | "The") * P("barked" | "The", "dog")

```

In a **Decoder-Only Transformer** (such as GPT-2, GPT-4, LLaMA-3, and Mistral), every token prediction at step $t$ is conditioned strictly on the preceding context $x_{<t}$. The model is architecturally forbidden from attending to future tokens $x_{>t}$.

---

## 2. Enforcing Causality: The Lower-Triangular Mask (`tril`)

To compute predictions across all sequence positions simultaneously during training without leaking future information, decoder architectures apply a **Causal Attention Mask** to the raw affinity matrix before the $\text{Softmax}$ operation:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M_{\text{causal}}\right) V$$

Where $M_{\text{causal}} \in \{0, -\infty\}^{T \times T}$ is defined as:

$$M_{\text{causal}}[i, j] = \begin{cases} 0, & \text{if } j \le i \\ -\infty, & \text{if } j > i \end{cases}$$

```text
Raw Dot-Product Matrix (Q @ K^T / sqrt(d_k)):
         Tok 1   Tok 2   Tok 3   Tok 4
Tok 1  [  2.41    0.82   -1.15    3.10  ]
Tok 2  [  1.12    1.95    0.45   -0.20  ]
Tok 3  [ -0.50    0.30    2.80    1.15  ]
Tok 4  [  0.90    1.40   -0.80    2.10  ]

Causal Mask Addition (+ M_causal):
         Tok 1   Tok 2   Tok 3   Tok 4
Tok 1  [  2.41   -inf    -inf    -inf   ]  <-- Tok 1 sees only Tok 1
Tok 2  [  1.12    1.95   -inf    -inf   ]  <-- Tok 2 sees Tok 1, Tok 2
Tok 3  [ -0.50    0.30    2.80   -inf   ]  <-- Tok 3 sees Tok 1, Tok 2, Tok 3
Tok 4  [  0.90    1.40   -0.80    2.10  ]  <-- Tok 4 sees all tokens

Softmax Normalization (exp(-inf) = 0.0):
         Tok 1   Tok 2   Tok 3   Tok 4
Tok 1  [  1.00    0.00    0.00    0.00  ]
Tok 2  [  0.30    0.70    0.00    0.00  ]
Tok 3  [  0.03    0.07    0.90    0.00  ]
Tok 4  [  0.15    0.25    0.05    0.55  ]

```

Because $\exp(-\infty) = 0$, all future key positions receive zero attention weight, preserving causality while enabling full parallel computation across all $T$ positions during training.

---

## 3. Shift-by-One Training Alignment (Teacher Forcing)

During training, the decoder processes the entire sequence in a single forward pass. This is achieved by creating two views of the token stream offset by exactly one position:

- **Input Sequence ($X$):** The prompt and context from index $0$ to $T-1$.
- **Target Sequence ($Y$):** The expected next tokens from index $1$ to $T$.

```text
Raw Corpus Stream: [ "ROMEO", ":", " ", "Shall", " ", "I", " ", "speak" ]

Input Tensor (X):  [ "ROMEO", ":", " ", "Shall", " ", "I", " "        ] (Positions 0 to T-1)
Target Tensor (Y): [ ":",     " ", "Shall", " ", "I", " ", "speak"    ] (Positions 1 to T)

Step-by-Step Training Prediction Pairs:
  • Given ["ROMEO"]                 ──► Predict ":"
  • Given ["ROMEO", ":"]            ──► Predict " "
  • Given ["ROMEO", ":", " "]       ──► Predict "Shall"
  • Given ["ROMEO", ":", " ", ... ] ──► Predict "speak"

```

In PyTorch, the batching function samples random slices of text and constructs $X$ and $Y$ using tensor offsets:

```python
def get_batch(split: str, train_data: torch.Tensor, val_data: torch.Tensor, batch_size: int, block_size: int, device: str):
    data = train_data if split == 'train' else val_data
    # Sample random starting indices
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Slice inputs (x) and targets shifted by 1 (y)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)

```

---

## 4. The Complete MiniGPT Architectural Blueprint

A canonical decoder-only Transformer stacks $N$ identical Pre-LN Transformer blocks between an initial embedding layer and a final linear projection head:

```text
Input Token IDs (B x T)
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ EMBEDDING LAYER                                        │
│ • Token Embeddings:    nn.Embedding(vocab_size, n_embd)│
│ • Position Embeddings: nn.Embedding(block_size, n_embd)│
│ • Fusion: x = tok_emb + pos_emb                        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ N x TRANSFORMER BLOCKS (Residual Highway)              │
│ ┌────────────────────────────────────────────────────┐ │
│ │ Sub-Block 1: Causal Multi-Head Self-Attention      │ │
│ │   x = x + MHA( LayerNorm_1(x) )                    │ │
│ ├────────────────────────────────────────────────────┤ │
│ │ Sub-Block 2: Pointwise Feed-Forward Network (FFN)  │ │
│ │   x = x + FFN( LayerNorm_2(x) )                    │ │
│ └────────────────────────────────────────────────────┘ │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ OUTPUT STAGE                                           │
│ • Final Normalization: ln_f = nn.LayerNorm(n_embd)     │
│ • Language Model Head: lm_head = nn.Linear(n_embd, V)  │
│ • (Optional) Weight Tying: lm_head.weight = wte.weight │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
               Logits (B x T x vocab_size)

```

---

## 5. PyTorch Implementation: MiniGPT from Scratch

Below is the standalone, production-grade implementation of a causal decoder-only language model (`MiniGPT`), featuring Pre-LN residual connections, GELU non-linearities, symmetric weight tying, and temperature-calibrated decoding:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with lower-triangular masking.
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Combined Q, K, V linear projections in a single matrix
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Register lower-triangular causal mask buffer
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # Batch size, Sequence length, Embedding dim (n_embd)

        # 1. Project Q, K, V and split across heads
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(C, dim=2)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 2. Scaled Dot-Product Attention with Causal Mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 3. Aggregate Values and project back to residual stream
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """
    Pointwise Feed-Forward Network with 4x expansion and GELU activation.
    """
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    Transformer Block: Pre-LN Attention + Pre-LN Feed-Forward.
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))  # Residual Highway 1
        x = x + self.mlp(self.ln_2(x))   # Residual Highway 2
        return x

class MiniGPT(nn.Module):
    """
    Complete Autoregressive Decoder-Only Transformer.
    """
    def __init__(
        self,
        vocab_size: int = 512,
        block_size: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        # Embedding tables
        self.wte = nn.Embedding(vocab_size, n_embd)  # Token embeddings
        self.wpe = nn.Embedding(block_size, n_embd)  # Positional embeddings
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])

        # Final LayerNorm and Output Head
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight Tying: Share weights between token embeddings and output projection
        if tie_weights:
            self.lm_head.weight = self.wte.weight

        # Weight Initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        # 1. Forward embeddings (Token ID + Spatial GPS Position)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)  # (B, T, n_embd)
        pos_emb = self.wpe(pos)  # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)

        # 2. Forward through Transformer backbone
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # 3. Compute output logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 4. Calculate Cross-Entropy Loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten tensors: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None
    ) -> torch.Tensor:
        """
        Autoregressive generation loop.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context if it exceeds max block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # Forward the model to obtain logits
            logits, _ = self(idx_cond)

            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature

            # Optional Top-K truncation
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Softmax to obtain probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token ID from categorical distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled token to sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

```

---

## 6. Autoregressive Rollout Dynamics

During inference, generating text executes token-by-token in a sequential loop:

```text
Initial Prompt: "ROMEO:" (Length T = 6)
  1. Pass "ROMEO:" through model ──► Logits for Position 5 ──► Sample: " "
  2. Append: "ROMEO: "
  3. Pass "ROMEO: " through model ──► Logits for Position 6 ──► Sample: "I"
  4. Append: "ROMEO: I"
  5. Pass "ROMEO: I" through model ──► Logits for Position 7 ──► Sample: " "
  6. Terminate when max_new_tokens is reached or <|im_end|> is emitted.

```

### Computational Complexity: The Quadratic Inference Bottleneck

In the standard generation loop shown above, predicting each new token re-computes the entire forward pass across all previous tokens $1 \dots t$, resulting in **$\mathcal{O}(T^2)$ computational complexity** for a sequence of length $T$.

This quadratic inference cost is eliminated in production systems via **Key-Value Caching (KV-Caching)**, which caches past Key and Value projection vectors in memory to achieve **linear $\mathcal{O}(T)$ inference time**.
