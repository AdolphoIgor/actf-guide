```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TiedTransformerLM(nn.Module):
    """
    Autoregressive Transformer with explicit Symmetric Weight Tying
    between Token Embeddings (wte) and the Output Linear Head (lm_head).
    """
    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 128,
        block_size: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size

        # 1. Embedding Tables
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        # 2. Transformer Backbone
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True  # Pre-LN Topology
            )
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # 3. Output Language Model Head (No Bias for pure inner products)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 4. SYMMETRIC WEIGHT TYING: Direct Memory Pointer Aliasing
        # Binds the memory address of the output head weights to the embedding table
        self.lm_head.weight = self.wte.weight

        # 5. Initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch# Symmetric Weight Tying (Tied Embeddings)

## 1. The Duality of Reading and Writing Tokens

In standard language model architectures, two distinct operations interface directly with the discrete vocabulary $V$:

```text
1. Token Ingestion ("Reading"):
   Discrete Token ID (t) ──► [ Input Embedding Lookup (W_emb) ] ──► Continuous Vector h_0 ∈ R^{d_model}

2. Logit Generation ("Writing"):
   Final Hidden Vector h_L ∈ R^{d_model} ──► [ Output Linear Projection (W_head) ] ──► Unnormalized Logits z ∈ R^V

```

In an untied architecture:

* The input embedding matrix is parameterized as $W_{\text{emb}} \in \mathbb{R}^{V \times d_{\text{model}}}$.
* The output language model head is parameterized as $W_{\text{head}} \in \mathbb{R}^{V \times d_{\text{model}}}$ (or $W_{\text{head}} \in \mathbb{R}^{d_{\text{model}} \times V}$ depending on orientation).

**Symmetric Weight Tying** (introduced by Press & Wolf and Inan et al.) enforces a direct parameter-sharing constraint:

$$W_{\text{head}} = W_{\text{emb}} = W_{\text{shared}} \in \mathbb{R}^{V \times d_{\text{model}}}$$

Under this formulation, the model uses the exact same geometric vector to represent a word when reading it from context as when computing the probability of predicting it at the output.

---

## 2. Mathematical Formulation and Gradient Mechanics

### Forward Pass Token Flow

When a batch of integer token IDs $X \in \mathbb{Z}^{B \times T}$ enters the network, row indexing retrieves the input embeddings:

$$E = W_{\text{shared}}[X] \in \mathbb{R}^{B \times T \times d_{\text{model}}}$$

After processing through $L$ Transformer blocks and applying final normalization ($\text{LN}_f$), the output hidden state tensor is $H_L \in \mathbb{R}^{B \times T \times d_{\text{model}}}$.

The output logits $Z \in \mathbb{R}^{B \times T \times V}$ are computed via transposed matrix multiplication against the shared embedding weights:

$$Z = H_L W_{\text{shared}}^T$$

The conditional probability of predicting token $v \in \{1, \dots, V\}$ at sequence position $t$ is:

$$P(x_{t+1} = v \mid x_{\le t}) = \frac{\exp(h_{L, t} \cdot w_v)}{\sum_{j=1}^V \exp(h_{L, t} \cdot w_j)}$$

Where $w_v \in \mathbb{R}^{d_{\text{model}}}$ is the $v$-th row vector of $W_{\text{shared}}$.

```text
┌────────────────────────────────────────────────────────────────────────┐
│ SYMMETRIC WEIGHT TYING FORWARD & BACKWARD FLOW                         │
├────────────────────────────────────────────────────────────────────────┤
│ Forward Pass:                                                          │
│   Input Token ID v ──► Lookup Row v in W_shared                        │
│   Hidden State h_L ──► Dot Product with Row v in W_shared ──► Logit z_v│
├────────────────────────────────────────────────────────────────────────┤
│ Backward Pass:                                                         │
│   Accumulated Gradient: ∇ W_shared = ∇ W_input + ∇ W_output            │
└────────────────────────────────────────────────────────────────────────┘

```

### Backward Pass Gradient Accumulation

During backpropagation, the loss function $\mathcal{L}$ computes gradients with respect to both operations. Because both paths point to the identical memory address, the autograd engine sums the gradient contributions:

$$\frac{\partial \mathcal{L}}{\partial W_{\text{shared}}} = \underbrace{\frac{\partial \mathcal{L}}{\partial Z} \frac{\partial Z}{\partial W_{\text{shared}}}}_{\text{Output Logit Gradient}} + \underbrace{\sum_{t=1}^T \frac{\partial \mathcal{L}}{\partial E_t} \frac{\partial E_t}{\partial W_{\text{shared}}}}_{\text{Input Embedding Gradient}}$$

$$\nabla W_{\text{shared}} = \left( \nabla_Z \mathcal{L} \right)^T H_L + \text{ScatterAdd}\left( \nabla_E \mathcal{L}, X \right)$$

This joint update ensures that every parameter update simultaneously refines the word's input semantic representation and its output predictive probability.

---

## 3. Geometric & Representational Implications

### 1. Vector Space Alignment

In an untied network, input embeddings and output projection vectors exist in separate coordinate spaces. The model must learn two independent manifolds that approximate similar semantic geometries.

Weight tying collapses these spaces into a **single metric space**:

* If two words $w_a$ and $w_b$ are semantically similar, their input vectors $w_a, w_b$ are close under cosine similarity.
* When the hidden state $h_L$ approaches the cluster around $w_a$, the dot product $h_L \cdot w_a$ produces a high logit, making the model predict tokens that share the same latent neighborhood.

### 2. Regularization in Small Data Regimes

In constrained corpora (such as TinyShakespeare or domain-specific code datasets), rare tokens appear infrequently:

* **Untied Models:** The input embedding vector for a rare token updates only when that token appears in the prompt, while its output projection vector updates only when that token is the target. Both matrices remain under-trained and prone to overfitting.
* **Tied Models:** Every time a rare token appears as an input or an output target, the same underlying vector $w_{\text{shared}}[v]$ receives gradient updates. This doubles the effective sample efficiency per parameter for the vocabulary.

---

## 4. Parameter Efficiency: Micro-Models vs. Large Scale

The parameter reduction from weight tying depends directly on the ratio of vocabulary size $V$ to total model parameters $\Phi$:

$$\Delta_{\text{parameters}} = V \times d_{\text{model}}$$

```text
Parameter Budget Breakdown with vs. without Weight Tying:

A. Small Pedagogical Model (V = 50,257, d_model = 128, L = 4):
   • Untied Footprint: Embeddings (6.43M) + LM Head (6.43M) + Core (0.80M) = 13.66M params
   • Tied Footprint:   Shared W_emb (6.43M) + Core (0.80M) = 7.23M params
   • Parameter Savings: 47.1% Total Model Reduction

B. Medium Foundation Model (V = 32,000, d_model = 4,096, L = 32):
   • Untied Footprint: Embeddings (131M) + LM Head (131M) + Core (6.8B) = ~7.06B params
   • Tied Footprint:   Shared W_emb (131M) + Core (6.8B) = ~6.93B params
   • Parameter Savings: ~1.85% Total Model Reduction

```

### Quantitative Comparison Matrix

| Model Architecture | Parameters ($\Phi$) | Vocab Size ($V$) | Hidden Dim ($d$) | Parameter Savings from Tying | Architectural Choice |
| --- | --- | --- | --- | --- | --- |
| **MiniGPT** | $\approx 1\text{M}$ | $512$ | $128$ | **$5.9\%$** ($0.065\text{M}$ params) | **Tied** (Critical for small regimes) |
| **GPT-2 Small** | $124\text{M}$ | $50,257$ | $768$ | **$24.1\%$** ($38.6\text{M}$ params) | **Tied** |
| **Gemma-2B / 7B** | $2.5\text{B} / 8.5\text{B}$ | $256,000$ | $2048 / 3072$ | **$17.3\% / 8.4\%$** | **Tied** |
| **LLaMA-3 8B** | $8.0\text{B}$ | $128,256$ | $4096$ | **$6.1\%$** ($525\text{M}$ params) | **Untied** |
| **Mistral-7B** | $7.2\text{B}$ | $32,000$ | $4096$ | **$1.8\%$** ($131\text{M}$ params) | **Untied** |

---

## 5. PyTorch Implementation and Memory Pointer Verification

To implement weight tying in PyTorch, assign the weight parameter of the output `nn.Linear` layer directly to the weight parameter of `nn.Embedding`.

Setting `bias=False` on the linear layer is mandatory to maintain mathematical symmetry.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TiedTransformerLM(nn.Module):
    """
    Decoder-only language model enforcing Symmetric Weight Tying
    between the input token embedding and the output LM head.
    """
    def __init__(self, vocab_size: int, n_embd: int, block_size: int, n_layer: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size

        # 1. Instantiate Input Embedding Table
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # 2. Transformer Core Blocks
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd, 
                nhead=4, 
                dim_feedforward=4*n_embd, 
                activation="gelu", 
                batch_first=True, 
                norm_first=True
            )
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)

        # 3. Instantiate Output LM Head without Bias
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # 4. ENFORCE SYMMETRIC WEIGHT TYING:
        # Reassign the weight Parameter object to share the exact underlying storage pointer
        self.lm_head.weight = self.token_embedding.weight

        # 5. Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        assert T <= self.block_size

        # Token + Positional Encodings
        tok_emb = self.token_embedding(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        # Backbone
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)

        # Output Logits via Tied Projection
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    def verify_weight_tying(self) -> bool:
        """
        Pre-flight test verifying that the input embedding and output head
        share the exact same memory storage pointer.
        """
        # Pointer check: must point to the identical object in RAM/VRAM
        is_same_object = self.lm_head.weight is self.token_embedding.weight
        is_same_data_ptr = self.lm_head.weight.data_ptr() == self.token_embedding.weight.data_ptr()

        if not (is_same_object and is_same_data_ptr):
            raise AssertionError("Weight tying failed: Tensors reside in distinct memory buffers.")
        
        return True

```

### Optimizer Deduplication Behavior

When passing parameters to an optimizer (`torch.optim.AdamW(model.parameters(), lr=1e-3)`), PyTorch relies on object identity (`id(param)`).

Because `self.lm_head.weight` is an identical reference to `self.token_embedding.weight`, the `model.parameters()` generator yields the tensor **exactly once**. The optimizer allocates only a single set of first and second moment buffers ($m_t, v_t$), avoiding double updates or redundant memory consumption in static VRAM.

---

## 6. The Untying Dilemma in Modern Frontier LLMs

While weight tying is standard in small-to-medium models, modern multi-billion parameter architectures (such as LLaMA-3 and Mistral) frequently leave embeddings **untied**.

```text
┌────────────────────────────────────────────────────────────────────────┐
│ WHY FRONTIER MODELS OPT TO UNTIE WEIGHTS                               │
├──────────────────────────────┬─────────────────────────────────────────┤
│ Asymmetric Representation    │ "Reading" tokens benefits from broad,   │
│ Demands                      │ smooth semantic clustering; "Writing"   │
│                              │ logits benefits from sharp, high-margin │
│                              │ competitive classification boundaries.  │
├──────────────────────────────┼─────────────────────────────────────────┤
│ Minimal Relative Overhead    │ At scale (7B+ params), an untied vocab  │
│                              │ accounts for less than 3-5% of total    │
│                              │ parameters, making savings negligible.  │
├──────────────────────────────┼─────────────────────────────────────────┤
│ Distributed Tensor Parallel  │ Sharding an untied LM head across TP    │
│ Flexibility                  │ ranks avoids cross-node communication   │
│                              │ synchronization with the input layer.   │
└──────────────────────────────┴─────────────────────────────────────────┘

```

### When to Tie vs. When to Untie

* **Enforce Weight Tying When:** Training micro-models ($\Phi < 1\text{B}$), operating in data-constrained or pedagogical regimes (TinyShakespeare, synthetic tasks), using wide vocabularies ($V > 32\text{k}$) on small models, or deploying on edge devices with limited VRAM.
* **Keep Weights Untied When:** Training frontier models ($\Phi \ge 7\text{B}$) on multi-trillion token datasets where parameter capacity is not bottlenecked by the embedding table, and maximum logit expressivity is required.