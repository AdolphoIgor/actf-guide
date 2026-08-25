# Temperature Scaling, Top-k, and Top-p (Nucleus) Sampling

## 1. The Mechanics of Stochastic Distribution Shaping

At each autoregressive generation step, a language model projects its final hidden state $h_t \in \mathbb{R}^{d_{\text{model}}}$ through the output language modeling head $W_{\text{head}} \in \mathbb{R}^{V \times d_{\text{model}}}$ to produce an unnormalized logit vector $z_t \in \mathbb{R}^V$:

$$z_t = W_{\text{head}} h_t$$

In stochastic decoding, these raw logits are reshaped into a constrained probability distribution over the vocabulary $V$. The goal of sampling techniques is to control **entropy** and eliminate the **unreliable probability tail**:

```text
Raw Logits (z_t)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ STEP 1: Temperature Scaling                            │
│   z'_i = z_i / T                                       │
│   Modulates distribution sharpness / entropy.          │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ STEP 2: Top-k Truncation                               │
│   Masks all logits outside the top-k highest values.   │
│   Bounds candidate search space to fixed size k.       │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ STEP 3: Softmax Normalization                          │
│   P(i) = exp(z'_i) / Sum_j exp(z'_j)                   │
│   Maps filtered logits to probability simplex.         │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ STEP 4: Top-p (Nucleus) Filtering                      │
│   Isolates smallest set of tokens where Sum P(i) >= p. │
│   Dynamically adapts candidate pool size to context.   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ STEP 5: Categorical Sampling                           │
│   y_t ~ Multinomial( P_renormalized )                  │
└────────────────────────────────────────────────────────┘

```

Without distribution shaping, sampling directly from standard softmax probabilities introduces low-probability tokens from the long tail of the vocabulary, causing semantic drift, grammatical decay, and hallucinations.

---

## 2. Temperature Scaling ($T$)

Temperature scaling is a post-processing operation that modifies the variance of the logit distribution before the $\text{Softmax}$ transformation:

$$P(y_t = i \mid y_{<t}) = \frac{\exp(z_{t, i} / T)}{\sum_{j=1}^V \exp(z_{t, j} / T)}$$

Where $T \in (0, \infty)$ is the temperature parameter.

```text
Effect of Temperature Scaling on Logit Distribution:

Low Temperature (T = 0.2):            Neutral (T = 1.0):             High Temperature (T = 1.8):
Peak-Sharpened / Low Entropy          Base Model Distribution        Flattened / High Entropy

            █                                      █                               
            █                                  █   █                               ▄   █   ▄   ▃
            █   ▂                          █   █   █   ▂                       █   █   █   █   █
    ────────┴───┴───                   ────┴───┴───┴───┴───                ────┴───┴───┴───┴───┴───
    Token 1  2   3                     Token 1  2   3   4                  Token 1  2   3   4   5
    P(1) ≈ 0.98, P(2) ≈ 0.02           P(1) = 0.55, P(2) = 0.25            P(1) ≈ 0.28, P(2) ≈ 0.22

```

### Mathematical Asymptotes of Temperature

1. **Greedy Limit ($T \to 0$):**

$$\lim_{T \to 0^+} P(y_t = i) = \begin{cases} 1, & \text{if } z_i = \max_j z_j \\ 0, & \text{otherwise} \end{cases}$$



As $T$ approaches zero, the softmax output converges to a one-hot indicator (Dirac delta distribution) centered on the argmax token, making generation purely deterministic.
2. **Base Calibration ($T = 1.0$):**
The distribution reflects the unmodified predictive confidence of the pre-trained model.
3. **Uniform Limit ($T \to \infty$):**

$$\lim_{T \to \infty} P(y_t = i) = \frac{\exp(0)}{\sum_{j=1}^V \exp(0)} = \frac{1}{\vert{}V\vert{}}$$



As $T$ approaches infinity, differences between logits become negligible, and the distribution converges to a uniform distribution over all $\vert{}V\vert{}$ vocabulary tokens.

---

## 3. Top-$k$ Truncation

Introduced by Fan et al., **Top-$k$ sampling** imposes a hard upper bound on the number of candidate tokens considered at each step.

### Mathematical Formulation

Given a sorted logit vector $z_{(1)} \ge z_{(2)} \ge \dots \ge z_{(V)}$, the retained vocabulary subset $V^{(k)}$ is defined as:

$$V^{(k)} = \{ (1), (2), \dots, (k) \}$$

Logits outside this top-$k$ partition are set to negative infinity prior to computing probabilities:

$$\tilde{z}_i = \begin{cases} z_i, & \text{if } i \in V^{(k)} \\ -\infty, & \text{if } i \notin V^{(k)} \end{cases}$$

$$P'(y_t = i) = \frac{\exp(\tilde{z}_i / T)}{\sum_{j \in V^{(k)}} \exp(\tilde{z}_j / T)}$$

```text
Top-k Truncation (k = 3):

Full Sorted Logits: [ "Paris": 8.2, "France": 6.5, "city": 5.1, "the": 3.8, "apple": 0.2 ]
Retained Pool (k=3): [ "Paris": 8.2, "France": 6.5, "city": 5.1 ]
Truncated to -inf:   [ "the": -inf, "apple": -inf ]

```

### Inherent Flaws of Static Top-$k$

```text
Flaw 1: Fixed k on Confident Contexts (Tail Pollution)
Context: "The Eiffel Tower is located in the city of [ ? ]"
True Distribution: P("Paris") = 0.99, P("France") = 0.005, ...
With k = 50: Forces 48 improbable tail tokens into the active candidate pool.

Flaw 2: Fixed k on Uncertain Contexts (Premature Truncation)
Context: "She opened the door and saw a [ ? ]"
True Distribution: Flat distribution across 150 plausible nouns (P ≈ 0.006 each).
With k = 50: Drops 100 valid linguistic continuations arbitrarily.

```

---

## 4. Top-$p$ (Nucleus) Sampling

Introduced by Holtzman et al., **Top-$p$ (Nucleus) sampling** solves the static limitation of Top-$k$ by dynamically sizing the candidate pool according to cumulative probability mass.

### Mathematical Formulation

Let $P(y_t = i)$ be the sorted probability distribution such that $P_{(1)} \ge P_{(2)} \ge \dots \ge P_{(V)}$.

The nucleus $V^{(p)} \subset V$ is the smallest subset of tokens whose cumulative probability satisfies the threshold $p \in (0, 1]$:

$$\sum_{i \in V^{(p)}} P_{(i)} \ge p$$

Where $V^{(p)} = \{ (1), (2), \dots, (k^*) \}$ and $k^*$ is the minimum index satisfying:

$$k^* = \min \left\{ k \in \{1, \dots, V\} \;\Bigg\vert{}\; \sum_{i=1}^k P_{(i)} \ge p \right\}$$

The remaining tokens are truncated, and the distribution is renormalized across the nucleus:

$$P'(y_t = i) = \begin{cases} \frac{P(y_t = i)}{\sum_{j \in V^{(p)}} P(y_t = j)}, & \text{if } i \in V^{(p)} \\ 0, & \text{otherwise} \end{cases}$$

```text
Dynamic Nucleus Behavior (Threshold p = 0.90):

Scenario A: High-Confidence Context
Tokens:         [ "Paris": 0.85, "France": 0.08, "the": 0.04, "city": 0.02, "car": 0.01 ]
Cumulative P:   [   0.85,          0.93,          0.97,         0.99,        1.00   ]
                  └─────────────────┬┘
Retained Pool:  k* = 2 tokens (Cumulative Mass = 0.93 >= 0.90). Tail safely eliminated.

Scenario B: Low-Confidence Context
Tokens:         [ "red": 0.15, "blue": 0.12, "green": 0.11, "yellow": 0.10, ..., "white": 0.02 ]
Cumulative P:   [  0.15,        0.27,         0.38,           0.48,   ...,   0.91        ]
                  └─────────────────────────────────────────────────────────────┬┘
Retained Pool:  k* = 12 tokens expand dynamically to encompass the 0.90 probability mass.

```

---

## 5. Pipeline Ordering and Interaction Dynamics

The structural order in which sampling transforms are applied directly impacts mathematical outcomes and GPU computational efficiency.

```text
Optimal Sampling Transformation Pipeline:
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Repetition / Presence Penalty Modification                          │
│    Operates on raw logits in-place before any non-linear transforms.   │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Temperature Division (z / T)                                        │
│    Scales logit spacing prior to sorting and thresholding.             │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Top-k Logit Truncation (torch.topk)                                 │
│    Reduces sorting complexity from O(V log V) down to O(V + k log k).  │
│    Sets non-top-k logits to -inf.                                      │
├────────────────────────────────────────────────────────────────────────┤
│ 4. Softmax Transformation                                              │
│    Converts the top-k subset into probability space.                   │
├────────────────────────────────────────────────────────────────────────┤
│ 5. Top-p Nucleus Truncation                                            │
│    Evaluates cumulative sum strictly over the non-zero probabilities.  │
├────────────────────────────────────────────────────────────────────────┤
│ 6. Renormalization & Categorical Sampling (torch.multinomial)          │
│    Draws discrete token ID from the final normalized distribution.     │
└────────────────────────────────────────────────────────────────────────┘

```

### Computational Advantage of Top-$k$ Pre-Filtering

Sorting the full vocabulary ($V \ge 128,000$ in modern LLMs like Llama-3) is memory-bandwidth intensive. Applying `topk` on logits before `softmax` and `top_p` allows the GPU to run a fast partial selection algorithm, passing only $k \ll V$ elements to subsequent cumulative-sum operations.

---

## 6. PyTorch Implementation: High-Performance Sampler

Below is the standalone PyTorch implementation of an optimized sampling pipeline executing Temperature scaling, Top-$k$, Top-$p$, and final categorical selection:

```python
import torch
import torch.nn.functional as F

class UnifiedSampler:
    """
    Production-grade sampling pipeline executing Temperature scaling,
    Top-k truncation, and Top-p (Nucleus) filtering.
    """
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.90
    ):
        self.temperature = max(1e-5, temperature)
        self.top_k = max(0, top_k)
        self.top_p = min(1.0, max(0.0, top_p))

    @torch.no_grad()
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: FloatTensor of shape (Batch_Size, Vocab_Size)
        Returns:
            sampled_ids: LongTensor of shape (Batch_Size, 1)
        """
        # 1. Temperature Scaling
        scaled_logits = logits / self.temperature

        # 2. Top-k Pre-filtering
        if self.top_k > 0 and self.top_k < scaled_logits.size(-1):
            top_k_vals, _ = torch.topk(scaled_logits, self.top_k, dim=-1)
            # Find the minimum value among the top-k candidates per batch row
            min_val = top_k_vals[:, -1, None]
            # Mask all elements below the k-th highest logit
            scaled_logits = torch.where(
                scaled_logits < min_val,
                torch.full_like(scaled_logits, float("-inf")),
                scaled_logits
            )

        # 3. Softmax Normalization
        probs = F.softmax(scaled_logits, dim=-1)

        # 4. Top-p (Nucleus) Filtering
        if self.top_p < 1.0:
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Mask tokens where cumulative probability exceeds the nucleus threshold
            sorted_mask = cumulative_probs > self.top_p
            # Shift mask right by 1 to always retain the first token exceeding threshold
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False

            # Zero out probabilities outside the nucleus
            sorted_probs[sorted_mask] = 0.0

            # Re-scatter filtered probabilities back to original vocabulary positions
            probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)

        # 5. Renormalize Probabilities
        probs_sum = torch.sum(probs, dim=-1, keepdim=True).clamp(min=1e-8)
        probs = probs / probs_sum

        # 6. Categorical Multinomial Sampling
        next_tokens = torch.multinomial(probs, num_samples=1)
        return next_tokens

```

---

## 7. Comparative Analysis & Hyperparameter Presets

### Sampling Strategy Comparison

| Strategy | Tuning Parameter | Space Level | Truncation Type | Failure Mode on Misconfiguration |
| --- | --- | --- | --- | --- |
| **Temperature** | $T \in (0, \infty)$ | Logit Space | Smooth Rescaling | $T \to 0$: Repetition; $T > 1.5$: Gibberish |
| **Top-$k$** | $k \in [1, \vert{}V\vert{}]$ | Logit Space | Fixed Count | $k$ too small: Monotonous; $k$ too large: Tail noise |
| **Top-$p$** | $p \in (0, 1.0]$ | Probability Space | Dynamic Mass | $p$ too small: Truncation; $p \to 1.0$: Tail noise |

### Recommended Production Parameter Presets

```text
┌────────────────────────────────────────────────────────────────────────┐
│ RECOMMENDED SAMPLING PROFILES                                          │
├──────────────────────────────┬─────────────────────────────────────────┤
│ 1. Deterministic / Exact     │ Temperature: 0.0 (Argmax)               │
│    (Code, SQL, Math, JSON)   │ Top-k: 1                                │
│                              │ Top-p: 1.0                              │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 2. Factual & Reasoning QA    │ Temperature: 0.2                        │
│    (Chain-of-Thought, SFT)   │ Top-k: 20                               │
│                              │ Top-p: 0.80                             │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 3. Balanced Conversational   │ Temperature: 0.7                        │
│    (General Dialogue, Chat)  │ Top-k: 50                               │
│                              │ Top-p: 0.90                             │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 4. Creative Writing          │ Temperature: 0.9                        │
│    (Brainstorming, Fiction)  │ Top-k: 100                              │
│                              │ Top-p: 0.95                             │
└──────────────────────────────┴─────────────────────────────────────────┘

```