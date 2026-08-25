# Sequence-to-Sequence Encoder-Decoder Transformers: MiniT5 Architecture

## 1. The Sequence-to-Sequence (Seq2Seq) Paradigm

While decoder-only models (MiniGPT) specialize in open-ended autoregressive generation and encoder-only models (MiniBERT) specialize in bidirectional representation extraction, many language tasks are fundamentally **transduction problems**: transforming an input sequence in one domain $X = (x_1, x_2, \dots, x_{T_{\text{enc}}})$ into a target sequence in another domain $Y = (y_1, y_2, \dots, y_{T_{\text{dec}}})$.

In standard Seq2Seq tasks (machine translation, document summarization, text-to-SQL, algorithmic transformations), the input and output lengths are typically asymmetric ($T_{\text{enc}} \ne T_{\text{dec}}$).

**Encoder-Decoder Transformers** (such as T5, BART, and the original Vaswani et al. architecture) decouple the architecture into two communicating engines:

1. **The Bidirectional Encoder:** Ingests and contextualizes the source prompt using unmasked, bidirectional attention.
2. **The Autoregressive Decoder:** Generates the target sequence token-by-token, continuously querying the encoder's representations via **Cross-Attention**.

```text
Source Input: "BANANA" (Length = 6)
                  │
                  ▼
      ┌─────────────────────────┐
      │  Bidirectional Encoder  │ (Unmasked Self-Attention)
      └───────────┬─────────────┘
                  │ Continuous Contextual Embeddings (H_enc)
                  ▼
      ┌─────────────────────────┐
      │ Autoregressive Decoder  │ (Causal Self-Attention + Cross-Attention)
      └───────────┬─────────────┘
                  │
                  ▼
Target Output: "ANANAB" (Length = 6)

```

---

## 2. The Cross-Attention Bridge: Mathematical Formulation

The core innovation connecting the encoder to the decoder is the **Cross-Attention Layer** (also known as Encoder-Decoder Attention).

In standard self-attention, Queries ($Q$), Keys ($K$), and Values ($V$) all originate from the same input tensor. In Cross-Attention:

* **Queries ($Q$):** Projected from the **current decoder hidden states** ($X_{\text{dec}}$).
* **Keys ($K$):** Projected from the **final encoder representations** ($H_{\text{enc}}$).
* **Values ($V$):** Projected from the **final encoder representations** ($H_{\text{enc}}$).

```text
    Decoder State (X_dec)               Encoder Output (H_enc)
            │                                      │
            ▼                                      ▼
   Linear Projection (W_Q)           Linear Projections (W_K, W_V)
            │                                      │
            ▼                                      ▼
         Queries (Q)                     Keys (K), Values (V)
            │                                      │
            └───────────────────┬──────────────────┘
                                ▼
          Scaled Dot-Product: Softmax( Q @ K^T / sqrt(d_k) ) @ V
                                │
                                ▼
                  Cross-Attention Output Vectors

```

### Mathematical Formulation

Let $X_{\text{dec}} \in \mathbb{R}^{T_{\text{dec}} \times d_{\text{model}}}$ and $H_{\text{enc}} \in \mathbb{R}^{T_{\text{enc}} \times d_{\text{model}}}$. The cross-attention projections are computed as:

$$Q = X_{\text{dec}} W_Q^{\text{cross}}, \quad K = H_{\text{enc}} W_K^{\text{cross}}, \quad V = H_{\text{enc}} W_V^{\text{cross}}$$

$$\text{CrossAttention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

### Key Structural Properties of Cross-Attention

* **Asymmetric Matrix Shape:** The affinity score matrix $S = Q K^T$ has dimensions $(T_{\text{dec}} \times T_{\text{enc}})$. Each row $i$ represents how much decoder token $i$ attends to every source token in the encoder.
* **No Causal Masking:** Cross-Attention does not apply a causal triangular mask. At every decoding step $t$, the decoder has full visibility across all encoder tokens $1 \dots T_{\text{enc}}$.
* **Static Keys and Values:** During autoregressive generation, $H_{\text{enc}}$ is computed only once. The resulting $K$ and $V$ matrices remain static while the decoder steps forward.

---

## 3. Training Dynamics: Teacher Forcing & Right-Shifted Targets

Training an encoder-decoder architecture in parallel requires three synchronized tensors:

1. **`X_enc` (Encoder Input):** The uncorrupted source sequence (e.g., `"BANANA"`).
2. **`Y` (Target Labels):** The true output sequence to be predicted (e.g., `"ANANAB"`).
3. **`X_dec` (Decoder Input - Shifted Right):** The target sequence shifted right by one position, prepended with a Start-of-Sequence (`<SOS>` or Space) token.

```text
Source Input (X_enc):     [ "B", "A", "N", "A", "N", "A" ]
Target Labels (Y):        [ "A", "N", "A", "N", "A", "B" ]

Decoder Input (X_dec):    [ <SOS>, "A", "N", "A", "N", "A" ] (Shifted Right)
                               │     │    │    │    │    │
                               ▼     ▼    ▼    ▼    ▼    ▼
Decoder Predictions:      [  "A",  "N", "A", "N", "A", "B" ]

```

### Why Teacher Forcing?

Under **Teacher Forcing**, the decoder receives the ground-truth previous tokens ($X_{\text{dec}}$) during training rather than its own potentially erroneous predictions.

This enables the entire target sequence to be trained in a single parallel forward pass using causal masking, avoiding step-by-step rollout during backpropagation.

```python
def get_seq2seq_batch(items: list[str], block_size: int, stoi: dict[str, int], device: str):
    # Encoder input: original text
    X_enc = torch.stack([torch.tensor([stoi[c] for c in s]) for s in items])
    
    # Target: reversed string
    Y = torch.stack([torch.tensor([stoi[c] for c in s[::-1]]) for s in items])
    
    # Decoder input: shifted right with <SOS> (index 0) prepended
    X_dec = torch.cat([torch.zeros((len(items), 1), dtype=torch.long), Y[:, :-1]], dim=1)
    
    return X_enc.to(device), X_dec.to(device), Y.to(device)

```

---

## 4. MiniT5 Block Anatomy: The 3 Sub-Layers

While an encoder block contains 2 sub-layers (Self-Attention + FFN), a **Decoder Block** contains 3 distinct sub-layers wrapped in Pre-LN residual connections:

```text
    Decoder Input: x_dec
          │
          ├─── [ Residual Highway 1 ] ─────────────────────────┐
          │                                                    │
          ▼                                                    │
    LayerNorm 1                                                │
          │                                                    │
          ▼                                                    │
    Sub-Layer 1: Causal Self-Attention (Dec-to-Dec Context)    │
          │                                                    │
          ▼                                                    │
          + <──────────────────────────────────────────────────┘
          │
          ├─── [ Residual Highway 2 ] ─────────────────────────┐
          │                                                    │
          ▼                                                    │
    LayerNorm 2                                                │
          │                                                    │
          ▼                                                    │
    Sub-Layer 2: Cross-Attention (Dec-to-Enc Bridge)           │
          │       H_enc (from Encoder)                         │
          │                                                    │
          ▼                                                    │
          + <──────────────────────────────────────────────────┘
          │
          ├─── [ Residual Highway 3 ] ─────────────────────────┐
          │                                                    │
          ▼                                                    │
    LayerNorm 3                                                │
          │                                                    │
          ▼                                                    │
    Sub-Layer 3: Pointwise Feed-Forward Network (FFN)          │
          │                                                    │
          ▼                                                    │
          + <──────────────────────────────────────────────────┘
          │
    Output: x_out (B x T_dec x d_model)

```

---

## 5. Complete PyTorch Implementation: MiniT5 from Scratch

Below is the implementation of the `MiniT5` architecture, featuring dedicated `EncoderBlock` and `DecoderBlock` modules with Pre-LN normalization and cross-attention routing:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    """
    Versatile Attention Head supporting Self-Attention (Causal/Bidirectional)
    and Cross-Attention.
    """
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.1, is_cross: bool = False):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.is_cross = is_cross

        # Causal mask only required for decoder self-attention
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.query(x)

        # In Cross-Attention, Keys and Values originate from Encoder representations
        if self.is_cross and enc_output is not None:
            k = self.key(enc_output)
            v = self.value(enc_output)
        else:
            k = self.key(x)
            v = self.value(x)

        # Scaled dot-product attention
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)

        # Apply causal mask strictly for decoder self-attention
        if not self.is_cross:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multi-Head wrapper for Self- and Cross-Attention."""
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1, is_cross: bool = False):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([
            AttentionHead(n_embd, head_size, block_size, dropout, is_cross=is_cross)
            for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor | None = None) -> torch.Tensor:
        out = torch.cat([h(x, enc_output) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    """Standard Pointwise FFN."""
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EncoderBlock(nn.Module):
    """Bidirectional Encoder Block (2 Sub-layers)."""
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout, is_cross=False)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class DecoderBlock(nn.Module):
    """Causal Decoder Block with Cross-Attention (3 Sub-layers)."""
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout, is_cross=False)
        self.ca = MultiHeadAttention(n_embd, n_head, block_size, dropout, is_cross=True)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor) -> torch.Tensor:
        # Sub-Layer 1: Causal Self-Attention
        x = x + self.sa(self.ln1(x))
        # Sub-Layer 2: Cross-Attention over Encoder representations
        x = x + self.ca(self.ln2(x), enc_output=enc_output)
        # Sub-Layer 3: Pointwise Feed-Forward
        x = x + self.ffwd(self.ln3(x))
        return x

class MiniT5(nn.Module):
    """Complete Sequence-to-Sequence Encoder-Decoder Architecture."""
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 64,
        n_layer: int = 3,
        n_head: int = 4,
        n_embd: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.block_size = block_size
        
        # Shared Embedding and Positional Tables
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Encoder and Decoder Stacks
        self.encoder = nn.ModuleList([
            EncoderBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(
        self, idx_enc: torch.Tensor, idx_dec: torch.Tensor, targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T_enc = idx_enc.shape
        _, T_dec = idx_dec.shape

        # 1. Forward Pass through Encoder (Source Sequence)
        pos_enc = torch.arange(T_enc, device=idx_enc.device)
        enc_x = self.token_embedding_table(idx_enc) + self.position_embedding_table(pos_enc)
        for block in self.encoder:
            enc_x = block(enc_x)

        # 2. Forward Pass through Decoder (Target Sequence with Cross-Attention)
        pos_dec = torch.arange(T_dec, device=idx_dec.device)
        dec_x = self.token_embedding_table(idx_dec) + self.position_embedding_table(pos_dec)
        for block in self.decoder:
            dec_x = block(dec_x, enc_output=enc_x)

        # 3. Project to vocabulary logits
        logits = self.lm_head(self.ln_f(dec_x))

        # 4. Cross-Entropy Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        return logits, loss

```

---

## 6. Inference Mechanics: Autoregressive Rollout

During inference, generating an output sequence from an encoder-decoder model proceeds in two distinct operational phases:

```text
Phase 1: Ingest Source Sequence (Run ONCE)
  Input "BANANA" ──► Encoder ──► Generates H_enc (Frozen in RAM)

Phase 2: Autoregressive Decoding Loop (Run per Token)
  Step 0: Feed [<SOS>] + H_enc         ──► Predicts "A"
  Step 1: Feed [<SOS>, "A"] + H_enc    ──► Predicts "N"
  Step 2: Feed [<SOS>, "A", "N"] + H_enc ──► Predicts "A"
  ...
  Terminates when max_new_tokens is reached or </s> is emitted.

```

```python
@torch.no_grad()
def generate_seq2seq(
    model: MiniT5,
    source_str: str,
    stoi: dict[str, int],
    itos: dict[int, str],
    block_size: int,
    device: str
) -> str:
    model.eval()

    # 1. Encode source text into tensor
    tokens_enc = [stoi[c] for c in source_str]
    enc_in = torch.tensor([tokens_enc], dtype=torch.long, device=device)

    # 2. Initialize decoder with <SOS> (index 0)
    dec_in = torch.zeros((1, 1), dtype=torch.long, device=device)

    # 3. Autoregressively roll out generation
    for _ in range(block_size - 1):
        logits, _ = model(enc_in, dec_in)
        
        # Greedy argmax selection on last position
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        dec_in = torch.cat([dec_in, next_token], dim=1)

    generated_ids = dec_in[0].tolist()
    return "".join([itos[i] for i in generated_ids]).strip()

```

---

## 7. Paradigm Comparison: GPT vs. BERT vs. T5

| Architectural Dimension | Autoregressive Decoder (MiniGPT) | Masked Encoder (MiniBERT) | Seq2Seq Encoder-Decoder (MiniT5) |
| --- | --- | --- | --- |
| **Attention Topologies** | Causal Masked Self-Attention | Unmasked Bidirectional Attention | **Bidirectional (Enc) + Causal (Dec) + Cross-Attention** |
| **Primary Objective** | Next-Token Prediction ($P(x_t \mid x_{<t})$) | Masked Reconstruction ($P(x_{\mathcal{M}} \mid \tilde{X})$) | **Conditional Generation ($P(Y \mid X)$)** |
| **Input / Output Structure** | Single continuous sequence | Noisy input $\rightarrow$ Clean output | **Source Sequence $X$ $\rightarrow$ Target Sequence $Y$** |
| **Cross-Attention Bridge** | None | None | **Active in all Decoder blocks** |
| **FLOP Allocation** | $100\%$ on generation | $100\%$ on representation | **Split across Encoding ($H_{\text{enc}}$) and Generation** |
| **Target Use Cases** | Open-ended text, dialogue, code generation | Embeddings, classification, NER, search | **Translation, summarization, structured transduction** |
