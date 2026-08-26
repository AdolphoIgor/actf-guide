# Supervised Fine-Tuning: Cross-Entropy Formulation and Target-Only Loss Masking

## 1. The Supervised Fine-Tuning (SFT) Paradigm

In self-supervised pre-training, a model optimizes next-token prediction across every token in an unstructured document:

$$\mathcal{L}_{\text{PT}}(\theta) = -\sum_{t=1}^T \log P_\theta(x_t \mid x_{<t})$$

In **Supervised Fine-Tuning (SFT)**, the objective shifts from generic document completion to conditional instruction following. Given a prompt $X = (x_1, x_2, \dots, x_N)$ containing system instructions, context, and user queries, the model must generate the corresponding target completion $Y = (y_1, y_2, \dots, y_M)$:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^M \log P_\theta(y_t \mid X, y_{<t})$$

```text
Pre-Training Paradigm:
  [ Document Header ] ──► [ Paragraph 1 ] ──► [ Paragraph 2 ] ──► [ Code Snippet ]
  └───────────────────────────────┬─────────────────────────┘
              Loss computed over 100% of tokens in sequence

Supervised Fine-Tuning Paradigm:
  [ System Prompt + User Query (X) ] ──► [ Assistant Response (Y) ]
  └────────────────┬───────────────┘     └────────────┬───────────┘
          Conditioning Context                 Active Loss Targets
          (Loss Masked = 0)                   (Gradients Flow = 1)

```

### The Pitfall of Full-Sequence Loss in SFT

If standard cross-entropy is applied across both prompt $X$ and target $Y$:

1. **Prompt Memorization:** The model expends parameter capacity learning how users phrase questions rather than learning how to solve them.
2. **Gradient Distortion:** In multi-turn dialogues with long system prompts, prompt tokens frequently outnumber completion tokens $3:1$ or $10:1$. The optimizer updates weights predominantly based on prompt syntax, degrading response quality.
3. **Catastrophic Style Overfitting:** The model learns fixed transitions inside the prompt formatting headers, resulting in brittle instruction following during zero-shot inference.

---

## 2. Mathematical Formulation of Masked Cross-Entropy

Let an input sequence have total length $T = \vert{}X\vert{} + \vert{}Y\vert{}$. The model maps hidden activations $h_t \in \mathbb{R}^{d_{\text{model}}}$ to unnormalized vocabulary logits $z_t \in \mathbb{R}^V$ via the output projection matrix $W_{\text{head}} \in \mathbb{R}^{V \times d_{\text{model}}}$:

$$z_t = W_{\text{head}} h_t$$

The conditional probability of emitting the true target token $y_t \in \{1, \dots, V\}$ is computed via the $\text{Softmax}$ operator:

$$P_\theta(y_t \mid x_{\le t}) = \frac{\exp(z_{t, y_t})}{\sum_{j=1}^V \exp(z_{t, j})}$$

### Numerical Stability: The Log-Sum-Exp Formulation

Computing raw exponentials over large vocabulary dimensions ($V \ge 32,000$) risks arithmetic overflow. PyTorch computes the negative log-likelihood (NLL) using the numerically stable **Log-Sum-Exp (LSE)** trick:

$$\log P_\theta(y_t \mid x_{\le t}) = z_{t, y_t} - \text{LSE}(z_t) = z_{t, y_t} - \left( c + \log \sum_{j=1}^V \exp(z_{t, j} - c) \right)$$

Where $c = \max_j (z_{t, j})$.

### The Active Target Mask Matrix

To isolate assistant completions, we introduce a binary mask indicator $M \in \{0, 1\}^T$:

$$M_t = \begin{cases} 1, & \text{if } t \in \text{Assistant Completion Tokens} \\ 0, & \text{if } t \in \text{System Prompt, User Query, or Padding} \end{cases}$$

The masked Cross-Entropy loss over a batch of size $B$ evaluates to:

$$\mathcal{L}_{\text{SFT}} = -\frac{\sum_{b=1}^B \sum_{t=1}^T M_{b, t} \cdot \log P_\theta(y_{b, t} \mid x_{b, <t})}{\sum_{b=1}^B \sum_{t=1}^T M_{b, t}}$$

In PyTorch, setting masked token indices to **`-100`** instructs `nn.CrossEntropyLoss(ignore_index=-100)` to set $M_{b, t} = 0$ internally, eliminating those positions from both the loss numerator and the normalization denominator.

---

## 3. Structural Anatomy of Loss Masking across Dialogue Turns

In multi-turn conversations (e.g., using OpenAI ChatML or Llama-3 formatting), formatting headers and delimiters must be categorized accurately as either context or targets:

```text
Turn Index / Content                                  Input Token (x_t)        Target Label (y_t)        Gradient Status
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
1. System Frame: <|im_start|>system\n                [ 151644, 8948, 198 ]    [ -100, -100, -100 ]       Masked (No loss)
2. System Content: You are a helpful bot.<|im_end|>  [ 2610, ..., 151645 ]    [ -100, ..., -100 ]        Masked (No loss)
3. User Frame: <|im_start|>user\n                    [ 151644, 872, 198 ]     [ -100, -100, -100 ]       Masked (No loss)
4. User Content: What is 2+2?<|im_end|>\n            [ 3838, ..., 151645 ]    [ -100, ..., -100 ]        Masked (No loss)
5. Assistant Frame: <|im_start|>assistant\n          [ 151644, 77091, 198 ]   [ -100, -100, -100 ]       Masked (No loss)
6. Assistant Content: 2+2 is 4.                      [ 17, 10, 17, 374, 19 ]  [ 17, 10, 17, 374, 19 ]    ACTIVE TARGET
7. Assistant Delimiter: <|im_end|>\n                 [ 151645, 198 ]          [ 151645, 198 ]            ACTIVE TARGET

```

```text
┌────────────────────────────────────────────────────────────────────────┐
│ CRITICAL BOUNDARY RULES IN SFT MASKING                                 │
├──────────────────────────────┬─────────────────────────────────────────┤
│ The Turn Header Rule         │ ALWAYS mask `<|im_start|>assistant\n`.  │
│                              │ The prompt sets up the turn; the model  │
│                              │ should not be penalized for the header. │
├──────────────────────────────┼─────────────────────────────────────────┤
│ The Turn Terminator Rule     │ NEVER mask `<|im_end|>` or `</s>`.      │
│                              │ The model MUST learn the stopping       │
│                              │ condition to prevent infinite decoding. │
├──────────────────────────────┼─────────────────────────────────────────┤
│ The Multi-Turn Past Rule     │ In Turn N, mask all prior assistant     │
│                              │ turns (Turns 1 to N-1) as context.      │
│                              │ Compute loss only on the current turn.  │
└──────────────────────────────┴─────────────────────────────────────────┘

```

---

## 4. Normalization Topologies: Token-Averaged vs. Sample-Averaged Loss

When aggregating cross-entropy across batches with variable-length assistant completions, the choice of denominator changes sample weighting:

```text
Batch Item 1 (Short Code Fix): [ Prompt (500 tokens) ] ──► [ Fix: "x = 1" (5 tokens) ]
Batch Item 2 (Long Analysis):  [ Prompt (100 tokens) ] ──► [ Report (500 tokens) ]

```

```text
Topology A: Global Token-Averaged Loss (Standard PyTorch Default)
  L_global = (Sum of ALL Active Losses in Batch) / (Total Active Tokens in Batch)
  • Batch Item 2 has 100x more weight in the gradient update than Batch Item 1.
  • Short, concise instructions are dominated by lengthy responses.

Topology B: Sample-Averaged (Normalized) Loss
  L_sample = (1 / B) * Sum_{b=1}^B [ (Sum of Active Losses in Sample b) / (Active Tokens in Sample b) ]
  • Every conversation exerts equal gradient magnitude regardless of output length.
  • Prevents long-tail generations from dominating optimization.

```

### Mathematical Comparison

| Normalization Scheme | Formula                                                              | Behavioral Bias                                          | Optimal Use Case                              |
| -------------------- | -------------------------------------------------------------------- | -------------------------------------------------------- | --------------------------------------------- |
| **Token-Averaged**   | $\frac{\sum_b \sum_t \ell_{b, t}}{\sum_b N_b}$                       | Biased toward long responses (more tokens = more weight) | Pre-training, standard sequence packing       |
| **Sample-Averaged**  | $\frac{1}{B} \sum_b \left( \frac{1}{N_b} \sum_t \ell_{b, t} \right)$ | Uniform weighting per instruction                        | Instruction tuning, reasoning, multi-task SFT |

---

## 5. PyTorch Implementation: SFT Loss Engine with Masking

Below is the implementation of an SFT training step demonstrating dynamic label construction, shift-by-one causal alignment, `-100` prompt masking, and sample-normalized loss computation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SFTLossEngine(nn.Module):
    """
    Computes masked Cross-Entropy loss for Supervised Fine-Tuning (SFT).
    Supports token-averaged and sample-averaged loss normalization.
    """
    def __init__(self, ignore_index: int = -100, loss_mode: str = "token_average"):
        super().__init__()
        self.ignore_index = ignore_index
        assert loss_mode in ["token_average", "sample_average"]
        self.loss_mode = loss_mode

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            logits: Unnormalized network predictions (Batch, Seq_Len, Vocab_Size)
            labels: Synchronized target IDs with -100 mask (Batch, Seq_Len)
        """
        # 1. Shift logits and labels for causal next-token alignment
        # Predict token t+1 given tokens 0...t
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        B, T_minus_1, V = shift_logits.shape

        if self.loss_mode == "token_average":
            # Flatten tensors across batch and sequence dimensions
            flat_logits = shift_logits.view(-1, V)
            flat_labels = shift_labels.view(-1)

            # Standard PyTorch Cross-Entropy with ignore_index
            loss = F.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=self.ignore_index,
                reduction="mean"
            )
        else:
            # Sample-averaged loss: compute per-token NLL without reduction
            flat_logits = shift_logits.view(-1, V)
            flat_labels = shift_labels.view(-1)

            per_token_loss = F.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=self.ignore_index,
                reduction="none"
            ).view(B, T_minus_1)

            # Mask indicator: 1 for active targets, 0 for -100
            active_mask = (shift_labels != self.ignore_index).float()
            tokens_per_sample = active_mask.sum(dim=-1).clamp(min=1.0)

            # Average loss per sample, then average across batch
            sample_loss = (per_token_loss * active_mask).sum(dim=-1) / tokens_per_sample
            loss = sample_loss.mean()

        # Compute tracking metrics
        with torch.no_grad():
            active_tokens = (shift_labels != self.ignore_index).sum().item()
            total_tokens = shift_labels.numel()
            active_ratio = active_tokens / max(1, total_tokens)
            perplexity = torch.exp(loss.detach()).item()

        metrics = {
            "sft_loss": loss.item(),
            "target_perplexity": perplexity,
            "active_token_ratio": active_ratio,
            "active_token_count": active_tokens
        }

        return loss, metrics

```

---

## 6. Diagnostic Metrics & Failure Modes

Monitoring SFT convergence requires isolating metrics to active target tokens:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ SFT DIAGNOSTIC SUITE                                                   │
├──────────────────────────────┬─────────────────────────────────────────┤
│ Target Perplexity (PPL)      │ PPL = exp(L_SFT). Tracks uncertainty    │
│                              │ strictly on assistant response tokens.  │
├──────────────────────────────┼─────────────────────────────────────────┤
│ Active Token Ratio           │ active_tokens / total_tokens.           │
│                              │ Must remain between 0.15 and 0.65.      │
│                              │ < 0.05 indicates over-masking.          │
├──────────────────────────────┼─────────────────────────────────────────┤
│ EOS Emission Entropy         │ Negative log-prob of `<|im_end|>`.      │
│                              │ High values indicate risk of repetitive │
│                              │ runaway generation loops during serving.│
└──────────────────────────────┴─────────────────────────────────────────┘

```

### Common Failure Modes and Root Causes

- **The Infinite Generation Bug:** The model generates high-quality responses during inference but fails to terminate, outputting random tokens until hitting `max_tokens`.
- _Root Cause:_ The closing delimiter (`<|im_end|>` or `</s>`) was masked with `-100` during training, preventing the model from receiving gradients on the sequence termination signal.

- **Loss Evaluates to `NaN`:** During early training steps, the batch loss immediately crashes to `NaN`.
- _Root Cause:_ A batch containing entirely empty assistant responses was ingested, resulting in $\sum M_t = 0$. The cross-entropy denominator becomes zero. Always guard dynamic collation with `clamp(min=1.0)` or Gate 4 assertions.

- **Prompt Echoing:** The model repeats user questions verbatim during inference before answering.
- _Root Cause:_ System and user prompts were included in the cross-entropy target mask ($M_{\text{prompt}} = 1$), teaching the model autoregressively to reconstruct the prompt.
