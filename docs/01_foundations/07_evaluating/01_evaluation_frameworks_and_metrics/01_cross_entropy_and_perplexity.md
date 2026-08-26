# Cross-Entropy Loss, Perplexity Dynamics, and Information-Theoretic Evaluation

## 1. Information-Theoretic Foundations of Language Modeling

Autoregressive language modeling evaluates how well a parameterized probability distribution $Q_\theta$ approximates the true, unknown data distribution $P$ over a sequence of discrete tokens $X = (x_1, x_2, \dots, x_T)$.

To understand the mathematical objective of training, we analyze the relationship between **Surprisal**, **Shannon Entropy**, **Cross-Entropy**, and **Kullback-Leibler (KL) Divergence**:

```text
Information-Theoretic Hierarchy:

1. Surprisal (Self-Information of an event x_t):
   I(x_t) = -log Q_theta(x_t | x_<t)
   • Measures the informational shock of observing token x_t.
   • Highly probable tokens carry low surprisal; rare tokens carry high surprisal.

2. Shannon Entropy (Theoretical compression limit of true distribution P):
   H(P) = - Sum_x P(x) log P(x)
   • The minimum expected number of bits required to encode information from P.

3. Cross-Entropy (Average bits needed using model Q_theta to encode true data P):
   H(P, Q_theta) = - Sum_x P(x) log Q_theta(x)

4. Kullback-Leibler (KL) Divergence (Relative Entropy / Inefficiency Penalty):
   D_KL(P || Q_theta) = H(P, Q_theta) - H(P) >= 0

```

```text
┌────────────────────────────────────────────────────────────────────────┐
│ THE OPTIMIZATION EQUIVALENCE                                           │
├────────────────────────────────────────────────────────────────────────┤
│ Because the true data entropy H(P) is constant with respect to model   │
│ parameters theta, minimizing Cross-Entropy is mathematically identical │
│ to minimizing the KL Divergence between true data P and model Q_theta: │
│                                                                        │
│   arg min_theta H(P, Q_theta) <===> arg min_theta D_KL(P || Q_theta)   │
└────────────────────────────────────────────────────────────────────────┘

```

When training on an empirical corpus $\mathcal{D}$, the true distribution $P$ is represented by the empirical sample frequencies (a Dirac delta / one-hot indicator over the ground-truth token). Cross-entropy reduces to the sample average of the **Negative Log-Likelihood (NLL)**.

---

## 2. Cross-Entropy Loss Formulation in Autoregressive Decoders

Let $z_t \in \mathbb{R}^V$ be the unnormalized logit vector produced by the language model head at sequence step $t$, where $V$ is the vocabulary size.

The conditional probability of predicting the correct target token $y_t \in \{1, \dots, V\}$ is computed via the $\text{Softmax}$ transformation:

$$Q_\theta(y_t \mid x_{<t}) = \frac{\exp(z_{t, y_t})}{\sum_{j=1}^V \exp(z_{t, j})}$$

The token-level loss $\ell_t$ is the negative log-likelihood of the ground-truth target:

$$\ell_t = -\log Q_\theta(y_t \mid x_{<t}) = -\log \left( \frac{\exp(z_{t, y_t})}{\sum_{j=1}^V \exp(z_{t, j})} \right) = -z_{t, y_t} + \log \left( \sum_{j=1}^V \exp(z_{t, j}) \right)$$

### Numerical Stability: The Log-Sum-Exp (LSE) Formulation

Directly evaluating $\sum \exp(z_j)$ risks severe floating-point overflow when logits exceed $\approx 88.7$ in Float32 or $\approx 11.0$ in Float16.

PyTorch computes cross-entropy via the stable **Log-Sum-Exp** formulation by factoring out the maximum logit $c = \max_j (z_{t, j})$:

$$\log \left( \sum_{j=1}^V \exp(z_{t, j}) \right) = c + \log \left( \sum_{j=1}^V \exp(z_{t, j} - c) \right)$$

```text
Raw Logits:          [ 1050.2,  1048.1,  1052.8 ]  ──► exp(1052.8) = Overflow (Inf)!
Shifted (c = 1052.8):[   -2.6,    -4.7,     0.0 ]  ──► exp(0.0) = 1.0 (Numerically Stable)

```

### Batch Reduction Modes & Target Loss Masking

Given a batch of $B$ sequences of length $T$, where $M_{b, t} \in \{0, 1\}$ is a binary mask ($0$ for prompt/padding tokens mapped to `-100`, $1$ for active target tokens):

$$\mathcal{L}_{\text{CE}} = \frac{\sum_{b=1}^B \sum_{t=1}^T M_{b, t} \cdot \ell_{b, t}}{\sum_{b=1}^B \sum_{t=1}^T M_{b, t}}$$

```text
Input Sequence:   [ <|im_start|>, user, \n, Hi, <|im_end|>, \n, <|im_start|>, asst, \n, Hello, <|im_end|> ]
Target Labels:    [     -100,     -100, -100, -100,  -100,  -100,    -100,     -100, -100, Hello, <|im_end|> ]
Active Mask M:    [        0,        0,    0,    0,     0,     0,       0,        0,    0,     1,          1 ]
                           └───────────────────────┬───────────────────────┘               └────────┬────────┘
                                            Loss = 0.0                                 Active Gradients Flow

```

---

## 3. Perplexity (PPL): Mathematical Derivation and Geometric Intuition

While Cross-Entropy loss is measured in natural units (nats) or bits per token, **Perplexity (PPL)** exponentiates the cross-entropy loss to project the error metric back into the vocabulary space.

### Mathematical Formulation

Given an evaluation corpus $X = (x_1, x_2, \dots, x_N)$ containing $N$ active evaluation tokens:

$$\text{PPL}(X) = \exp\left( \mathcal{L}_{\text{CE}} \right) = \exp\left( -\frac{1}{N} \sum_{t=1}^N \log Q_\theta(x_t \mid x_{<t}) \right)$$

Using the properties of logarithms, perplexity represents the geometric mean of the inverse probabilities assigned to the true tokens:

$$\text{PPL}(X) = \left( \prod_{t=1}^N \frac{1}{Q_\theta(x_t \mid x_{<t})} \right)^{\frac{1}{N}}$$

```text
Geometric Intuition: The Effective Branching Factor

Perplexity represents the number of equally probable choices the model is choosing among
at each step of sequence generation.

Case 1: Perfect Model (Zero Loss)
  • Probability of true token: Q(x_t) = 1.0 for all t
  • Cross-Entropy Loss: L_CE = -log(1.0) = 0.0 nats
  • Perplexity: PPL = exp(0.0) = 1.0
  • Interpretation: The model has zero uncertainty; effective branching factor is 1.

Case 2: Uniform Random Baseline (Zero Knowledge)
  • Vocabulary Size: V = 32,000
  • Probability assigned to each token: Q(x_t) = 1 / 32,000
  • Cross-Entropy Loss: L_CE = -log(1 / 32000) = log(32000) ≈ 10.373 nats
  • Perplexity: PPL = exp(10.373) = 32,000
  • Interpretation: The model is guessing blindly among all 32,000 vocabulary tokens.

Case 3: Well-Tuned LLM (e.g., PPL = 8.5)
  • Cross-Entropy Loss: L_CE = ln(8.5) ≈ 2.14 nats
  • Interpretation: At each step, the model is as uncertain as if it were choosing
    uniformly among 8.5 equally plausible candidate tokens.

```

---

## 4. The Tokenizer Incomparability Hazard & Bits-per-Byte (BPB)

A critical vulnerability in LLM evaluation is comparing raw Perplexity across models that use different tokenizers. **Perplexity is strictly token-dependent** and cannot be used to compare models with different vocabulary sizes or tokenization algorithms.

```text
The Vocabulary Compression Paradox:

Text Snippet: "The quick brown fox jumps" (25 raw ASCII bytes)

Tokenizer A (Character-Level, Vocab Size V = 256):
  • Encodes into 25 tokens.
  • Average Loss per token = 1.2 nats.
  • Token-Level Perplexity = exp(1.2) = 3.32.

Tokenizer B (Subword BPE, Vocab Size V = 128,000):
  • Encodes into 5 tokens (e.g., ["The", " quick", " brown", " fox", " jumps"]).
  • Average Loss per token = 2.8 nats (predicting large multi-character words is harder per step).
  • Token-Level Perplexity = exp(2.8) = 16.44.

Misleading Conclusion: Tokenizer A appears 5x better (PPL 3.32 vs 16.44).
Mathematical Reality:  Tokenizer B compressed 25 bytes into 5 tokens; total sequence entropy is lower!

```

### The Solution: Bits-per-Byte (BPB) and Bits-per-Character (BPC)

To establish a tokenizer-agnostic evaluation metric, cross-entropy loss must be normalized by the **physical byte length** or **character length** of the un-tokenized source text.

$$\text{Total Entropy (in bits)} = \frac{\sum_{t=1}^N \ell_t (\text{in nats})}{\ln(2)}$$

$$\text{Bits-per-Byte (BPB)} = \frac{\text{Total Entropy (bits)}}{\text{Total Raw UTF-8 Bytes}} = \frac{\mathcal{L}_{\text{total (nats)}}}{\ln(2) \times \text{Byte Count}}$$

$$\text{Bits-per-Character (BPC)} = \frac{\text{Total Entropy (bits)}}{\text{Total Unicode Characters}} = \frac{\mathcal{L}_{\text{total (nats)}}}{\ln(2) \times \text{Character Count}}$$

```text
┌────────────────────────────────────────────────────────────────────────┐
│ TOKENIZER-AGNOSTIC BENCHMARKING RULE                                   │
├────────────────────────────────────────────────────────────────────────┤
│ • Use Perplexity (PPL) ONLY when comparing checkpoints trained with    │
│   the EXACT same tokenizer vocabulary and merge rules.                 │
│ • Use Bits-per-Byte (BPB) when benchmarking models across different    │
│   tokenizers, vocabulary sizes, or architectural paradigms.            │
└────────────────────────────────────────────────────────────────────────┘

```

---

## 5. Perplexity Dynamics Across Training Stages

Tracking validation perplexity trajectories reveals critical phase transitions during model training:

```text
Perplexity Trajectories Across Training Phases:

PPL
 ▲
 │  10,000 ──► Pre-Training Step 0 (Random weights, PPL ≈ Vocab Size)
 │
 │   1,000 ──► Step 500 (Basic syntactic n-grams & whitespace memorization)
 │
 │     100 ──► Step 5,000 (Lexical associations, common words calibrated)
 │
 │      20 ──► Step 50,000 (Grammatical structure, contextual reasoning)
 │
 │      10 ──► Pre-Training Convergence Basin (PPL ≈ 6 - 12 on RedPajama/SlimPajama)
 │
 │       3 ──► Supervised Fine-Tuning (SFT) on Target Responses (PPL ≈ 2 - 4)
 └────────────────────────────────────────────────────────────────────────► Training Steps

```

### Diagnostic Signatures of Perplexity Curves

- **Healthy Convergence:** Training loss and validation loss decrease smoothly in parallel. Validation perplexity plateaus asymptotically.
- **Overfitting Divergence:** Training PPL continues to drop (e.g., $3.0 \to 1.5$) while Validation PPL reverses direction and begins climbing ($5.2 \to 6.8$). Immediate early stopping trigger.
- **Perplexity Spike Anomaly:** A sudden exponential jump in PPL (e.g., $8.2 \to 450.0$) indicates:

1. A corrupt or unmasked training batch containing repetitive formatting tokens.
2. Numerical overflow in attention logits (loss scale explosion in FP16).
3. AdamW second-moment ($v_t$) buffer corruption following an unclipped gradient update.

---

## 6. Python / PyTorch Implementation: Evaluation and Metrics Engine

Below is the standalone evaluation engine implementing numerically stable loss calculation, target-only loss masking, sample-weighted validation perplexity aggregation, and tokenizer-agnostic Bits-per-Byte (BPB) calculation:

```python
import math
from typing import Any, Dict, Iterator, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricsEvaluationEngine:
    """
    Evaluates Cross-Entropy Loss, Perplexity (PPL), and Bits-per-Byte (BPB)
    with strict loss masking and correct sample-weighted aggregation.
    """
    def __init__(self, ignore_index: int = -100):
        self.ignore_index = ignore_index

    @torch.no_grad()
    def evaluate_batch(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        raw_byte_counts: torch.Tensor | None = None
    ) -> Dict[str, Any]:
        """
        Computes exact loss and token counts for a single forward pass.

        Args:
            logits: Unnormalized predictions of shape (Batch, Seq_Len, Vocab_Size)
            labels: Target IDs with ignore_index of shape (Batch, Seq_Len)
            raw_byte_counts: 1D tensor of byte lengths per sample (optional)
        """
        # 1. Causal Shift Alignment: Predict token t+1 given tokens 0...t
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        B, T, V = shift_logits.shape

        # 2. Compute unreduced Negative Log-Likelihood per token
        loss_unreduced = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction="none"
        ).view(B, T)

        # 3. Mask active evaluation tokens
        active_mask = (shift_labels != self.ignore_index)
        active_tokens_per_batch = active_mask.sum().item()

        total_loss_nats = (loss_unreduced * active_mask.float()).sum().item()

        # 4. Byte count summation for BPB
        total_bytes = 0
        if raw_byte_counts is not None:
            total_bytes = raw_byte_counts.sum().item()

        return {
            "total_loss_nats": total_loss_nats,
            "active_tokens": active_tokens_per_batch,
            "total_bytes": total_bytes
        }

    @torch.no_grad()
    def evaluate_dataset(
        self,
        model: nn.Module,
        data_loader: Iterator[Tuple[torch.Tensor, torch.Tensor, List[int]]],
        num_batches: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ) -> Dict[str, float]:
        """
        Runs comprehensive evaluation loop across validation batches.
        Aggregates metrics using total token-weighted sums to prevent batch size bias.
        """
        model.eval()

        cumulative_loss_nats = 0.0
        cumulative_tokens = 0
        cumulative_bytes = 0

        for _ in range(num_batches):
            try:
                batch_data = next(data_loader)
            except StopIteration:
                break

            # Unpack inputs, targets, and raw byte lengths
            inputs, targets, byte_lengths = batch_data
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            byte_tensor = torch.tensor(byte_lengths, device=device) if byte_lengths else None

            with torch.autocast(device_type=device if device == "cuda" else "cpu", dtype=dtype):
                logits = model(inputs)
                if isinstance(logits, tuple):
                    logits = logits[0]

            batch_metrics = self.evaluate_batch(logits, targets, byte_tensor)

            cumulative_loss_nats += batch_metrics["total_loss_nats"]
            cumulative_tokens += batch_metrics["active_tokens"]
            cumulative_bytes += batch_metrics["total_bytes"]

        if cumulative_tokens == 0:
            raise ValueError("Zero active evaluation tokens encountered in validation dataset.")

        # 1. Compute Global Token-Averaged Cross-Entropy Loss
        avg_loss_nats = cumulative_loss_nats / cumulative_tokens
        avg_loss_bits = avg_loss_nats / math.log(2)

        # 2. Compute Mathematical Perplexity: PPL = exp(mean_nats) = 2^(mean_bits)
        perplexity = math.exp(avg_loss_nats)

        # 3. Compute Bits-per-Byte (BPB) if byte lengths were tracked
        bits_per_byte = None
        if cumulative_bytes > 0:
            total_entropy_bits = cumulative_loss_nats / math.log(2)
            bits_per_byte = total_entropy_bits / cumulative_bytes

        results = {
            "val_loss_nats": avg_loss_nats,
            "val_loss_bits": avg_loss_bits,
            "perplexity": perplexity,
            "bits_per_byte": bits_per_byte,
            "total_evaluated_tokens": cumulative_tokens,
            "total_evaluated_bytes": cumulative_bytes
        }

        return results

```

---

## 7. Metrics Comparison Matrix

| Evaluation Metric                             | Mathematical Definition                                              | Base Unit                      | Tokenizer Dependent? | Primary Diagnostic Role                                             |
| --------------------------------------------- | -------------------------------------------------------------------- | ------------------------------ | -------------------- | ------------------------------------------------------------------- |
| **Cross-Entropy ($\mathcal{L}_{\text{CE}}$)** | $-\frac{1}{N} \sum \log Q(x_t)$                                      | nats / token                   | Yes                  | Direct optimization objective for gradient backpropagation          |
| **Perplexity (PPL)**                          | $\exp(\mathcal{L}_{\text{CE}})$                                      | Effective choices              | Yes                  | Intuitive tracking of model uncertainty across training checkpoints |
| **Bits-per-Byte (BPB)**                       | $\frac{\mathcal{L}_{\text{total (bits)}}}{\text{Total UTF-8 Bytes}}$ | bits / byte                    | **No (Universal)**   | Fair benchmark comparison across different models and tokenizers    |
| **Bits-per-Character (BPC)**                  | $\frac{\mathcal{L}_{\text{total (bits)}}}{\text{Total Characters}}$  | bits / char                    | **No (Universal)**   | Cross-model comparison on fixed natural language corpora            |
| **Target Accuracy (Top-1)**                   | $\frac{1}{N} \sum \mathbb{I}(\arg\max z_t == y_t)$                   | Percentage ($0\text{--}100\%$) | Yes                  | Tracking hard classification accuracy on next-token prediction      |
