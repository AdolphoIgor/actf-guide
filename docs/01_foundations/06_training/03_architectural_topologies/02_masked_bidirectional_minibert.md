# Masked Bidirectional Encoders: MiniBERT Architecture

## 1. The Masked Language Modeling (MLM) Paradigm

While autoregressive decoder architectures (such as MiniGPT) factorize sequence probability strictly from left to right ($P(X) = \prod P(x_t \mid x_{<t})$), this unidirectional constraint limits the model's ability to build holistic contextual representations. In tasks such as sentence classification, named entity recognition (NER), semantic search, and extractive question answering, a token's meaning depends equally on both its preceding context (the past) and its succeeding context (the future).

**BERT** (Bidirectional Encoder Representations from Transformers) resolves this by abandoning causal generation in favor of the **Masked Language Modeling (MLM)** objective (the _Cloze_ task).

```text
Autoregressive Decoder (Unidirectional Context):
  Context: "The [dog] [bit] [the] ---> [ ? ]"
  • Can only look to the left.

Masked Bidirectional Encoder (Full Context):
  Corrupted Context: "ROMEO: I [MASK] thee, Julia!"
  • Looks both left ("ROMEO: I") and right ("thee, Julia!") simultaneously.
  • Reconstructs the corrupted token.

```

Instead of predicting the next token, the encoder is presented with a sequence where a fraction of tokens (typically 15%) are replaced by a special `[MASK]` token. The model optimizes the bidirectional cross-entropy loss required to reconstruct the original identity of the masked positions:

$$\mathcal{L}_{\text{MLM}}(\theta) = -\sum_{i \in \mathcal{M}} \log P_\theta(x_i \mid \tilde{X})$$

Where $\mathcal{M}$ represents the set of masked token indices and $\tilde{X}$ is the corrupted input sequence.

---

## 2. Removing the Causal Mask: Full Bidirectional Attention

In an encoder, the lower-triangular causal mask (`tril`) is removed entirely. Every token computes raw dot-product affinity scores against all other tokens in the sequence simultaneously:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

```text
Unmasked Bidirectional Affinity Matrix (T x T):
         Tok 1   Tok 2  [MASK]   Tok 4
Tok 1  [  0.40    0.25    0.15    0.20  ]  <-- Tok 1 attends to all positions
Tok 2  [  0.10    0.60    0.20    0.10  ]  <-- Tok 2 attends to all positions
[MASK] [  0.30    0.20    0.10    0.40  ]  <-- [MASK] extracts context from left and right
Tok 4  [  0.15    0.15    0.20    0.50  ]  <-- Tok 4 attends to all positions

```

Without the lower-triangular restriction, the attention matrix is dense ($T \times T$), allowing representation vectors at every layer to synthesize information across the entire sequence.

---

## 3. Masking Dynamics & Vocabulary Extension

To implement MLM, the base vocabulary must be expanded by allocating a dedicated index for the `[MASK]` control token:

$$\text{vocab\_size}_{\text{encoder}} = \text{len}(\text{chars}) + 1 \quad \implies \quad \text{mask\_token\_id} = \text{vocab\_size} - 1$$

```python
# Vocabulary setup for MiniBERT
chars = sorted(list(set(raw_text)))
vocab_size = len(chars) + 1  # +1 reserved for the special [MASK] token
mask_token_id = vocab_size - 1

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
itos[mask_token_id] = "[MASK]"

```

### The 15% Dynamic Masking Pipeline

During batch creation, the training tensor $X$ is cloned from target sequence $Y$. A boolean noise mask is sampled uniformly at a 15% probability threshold:

$$M_{i, j} \sim \text{Bernoulli}(p = 0.15)$$

```text
Target Sentence (Y):   [ "R", "O", "M", "E", "O", ":", " ", "I", " ", "l", "o", "v", "e" ]
Random Mask Masking:   [  0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1  ] (15% random selection)
Corrupted Input (X):   [ "R", "O", "M", "E", "O", ":", " ", "I", " ", [M], [M], [M], [M] ]

```

```python
def get_batch(split: str, train_data: torch.Tensor, val_data: torch.Tensor, batch_size: int, block_size: int, device: str):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))

    # Target sequence Y is the clean, uncorrupted slice
    y = torch.stack([data_split[i:i + block_size] for i in ix])
    x = y.clone()

    # 15% chance to replace a token with the [MASK] token ID
    mask_arr = torch.rand(x.shape) < 0.15
    x[mask_arr] = mask_token_id

    return x.to(device), y.to(device)

```

---

## 4. MiniBERT Architectural Blueprint

MiniBERT stacks bidirectional Pre-LN Transformer blocks where each sub-layer uses unmasked attention:

```text
Corrupted Input IDs with Masks: X (B x T)
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│ EMBEDDING LAYER                                        │
│ • Token Embeddings:    nn.Embedding(vocab_size, n_embd)│
│ • Position Embeddings: nn.Embedding(block_size, n_embd)│
│ • Fusion: x = tok_emb + pos_emb                        │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│ N x ENCODER BLOCKS (Bidirectional Residual Highway)    │
│ ┌────────────────────────────────────────────────────┐ │
│ │ Sub-Block 1: Unmasked Bidirectional Self-Attention │ │
│ │   x = x + MHA_Bidirectional( LayerNorm_1(x) )      │ │
│ ├────────────────────────────────────────────────────┤ │
│ │ Sub-Block 2: Pointwise Feed-Forward Network (FFN)  │ │
│ │   x = x + FFN( LayerNorm_2(x) )                    │ │
│ └────────────────────────────────────────────────────┘ │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│ OUTPUT STAGE                                           │
│ • Final Normalization: ln_f = nn.LayerNorm(n_embd)     │
│ • Reconstruction Head: lm_head = nn.Linear(n_embd, V)  │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
  Reconstruction Logits: (B x T x vocab_size)
                     │
                     ▼
  Cross-Entropy Loss evaluated against clean targets Y

```

---

## 5. PyTorch Implementation: MiniBERT from Scratch

Below is the complete implementation of a bidirectional encoder model (`MiniBERT`) designed for masked token reconstruction:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalHead(nn.Module):
    """
    A single attention head with unmasked bidirectional receptive fields.
    """
    def __init__(self, n_embd: int, head_size: int, dropout: float = 0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # Full dense attention matrix: NO lower-triangular causal masking applied
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v # (B, T, head_size)
        return out

class BidirectionalMultiHeadAttention(nn.Module):
    """
    Parallel bidirectional attention heads concatenated and projected.
    """
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([
            BidirectionalHead(n_embd, head_size, dropout) for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    """
    Pointwise MLP with 4x hidden expansion.
    """
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EncoderBlock(nn.Module):
    """
    Bidirectional Transformer Block: Pre-LN Attention + Pre-LN FFN.
    """
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.sa = BidirectionalMultiHeadAttention(n_embd, n_head, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniBERT(nn.Module):
    """
    Pure Masked Bidirectional Transformer Encoder.
    """
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        n_layer: int = 3,
        n_head: int = 4,
        n_embd: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[
            EncoderBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape

        # 1. Identity Embeddings + Spatial GPS Coordinates
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        # 2. Bidirectional Encoder Backbone
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # Shape: (B, T, vocab_size)

        # 3. Loss Calculation Across Sequence
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

```

---

## 6. Mask Restoration & Inference Mechanics

Because MiniBERT is non-causal, it cannot perform open-ended text generation in the style of GPT. Instead, inference operates in **Detective Mode**: parsing a corrupted sequence, querying context around the masked slots, and outputting highest-probability reconstructions.

```python
@torch.no_grad()
def restore_sentence(
    model: nn.Module,
    masked_sentence: str,
    stoi: dict[str, int],
    itos: dict[int, str],
    mask_token_id: int,
    device: str
) -> str:
    """
    Fills [MASK] slots in a string using MiniBERT's contextual predictions.
    """
    model.eval()

    # 1. Parse string and replace "[MASK]" with mask_token_id
    tokens = []
    i = 0
    while i < len(masked_sentence):
        if masked_sentence[i:i + 6] == "[MASK]":
            tokens.append(mask_token_id)
            i += 6
        else:
            tokens.append(stoi.get(masked_sentence[i], 0))
            i += 1

    x = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, T)

    # 2. Forward pass through Bidirectional Encoder
    logits, _ = model(x)  # (1, T, vocab_size)

    # 3. Extract highest probability predictions
    probs = F.softmax(logits, dim=-1)
    predicted_ids = torch.argmax(probs, dim=-1).squeeze(0)

    # 4. Reconstruct sentence
    result = ""
    for orig_id, pred_id in zip(tokens, predicted_ids):
        if orig_id == mask_token_id:
            # Highlight restored token
            result += f"\033[92m{itos[pred_id.item()]}\033[0m"
        else:
            result += itos[orig_id]

    return result

```

```text
Sample Execution:
Input:  "ROMEO: I [MASK] thee, Julia!"
Output: "ROMEO: I love thee, Julia!"

```

---

## 7. Structural Comparison: Decoder (MiniGPT) vs. Encoder (MiniBERT)

| Architectural Dimension       | Decoder-Only (MiniGPT)                            | Masked Encoder (MiniBERT)                                               |
| ----------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------- |
| **Attention Receptive Field** | Causal Lower-Triangular (`tril` enforced)         | **Full Bidirectional (Unconstrained dense $T \times T$)**               |
| **Training Objective**        | Next-Token Prediction: $\prod P(x_t \mid x_{<t})$ | **Masked Language Modeling: $P(x_{\mathcal{M}} \mid \tilde{X})$**       |
| **Input / Target Structure**  | Inputs $X_{0 \dots T-1}$, Targets $Y_{1 \dots T}$ | **Inputs $X_{\text{noisy}}$, Targets $Y_{\text{clean}}$ (same length)** |
| **Generative Capability**     | Autoregressive text synthesis                     | **Cannot generate open-ended sequences**                                |
| **Primary Production Role**   | Dialogue, code generation, reasoning              | **Embeddings, classification, re-ranking, NER**                         |
