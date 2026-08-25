# Deterministic vs. Stochastic Sampling Strategies in LLM Inference

## 1. The Logit-to-Token Decoding Pipeline

At each step of autoregressive generation, a decoder-only Transformer produces an unnormalized logit vector $z_t \in \mathbb{R}^V$ over the vocabulary $V$ from the final hidden state $h_t \in \mathbb{R}^{d_{\text{model}}}$:

$$z_t = W_{\text{head}} h_t$$

Converting this continuous logit vector into a discrete output token requires passing $z_t$ through a decoding strategy. Decoding algorithms fall into two broad mathematical paradigms:

```text
Logit Output Vector (z_t in R^V)
               │
               ├─────────────────────────────────────────┐
               ▼                                         ▼
┌─────────────────────────────┐           ┌─────────────────────────────┐
│ 1. Deterministic Decoding   │           │ 2. Stochastic Sampling      │
│  • Greedy Search (Argmax)   │           │  • Temperature Scaling      │
│  • Beam Search              │           │  • Top-k Truncation         │
│  • Contrastive Search       │           │  • Top-p (Nucleus) Filtering│
│                             │           │  • Min-p Filtering          │
│ Property: P(Y|X) is static; │           │                             │
│ identical outputs per run.  │           │ Property: Samples from P(Y);│
│                             │           │ non-deterministic outputs.  │
└─────────────────────────────┘           └─────────────────────────────┘

```

The choice of decoding strategy controls the balance between **precision** (syntactic validity, factual consistency, logical coherence) and **entropy** (lexical diversity, expressive variation, multi-hypothesis exploration).

---

## 2. Deterministic Search Strategies

Deterministic algorithms select tokens by maximizing sequence likelihood without introducing pseudorandom sampling.

```text
Greedy Search: Path follows maximum immediate logit at every step
  Step 1: "The" (P=0.9) ──► Step 2: "capital" (P=0.8) ──► Step 3: "of" (P=0.95)

Beam Search (Width B = 2): Tracks top-2 full sequence hypotheses across time
  Step 1:
    ├── Path A: "The" (P=0.9)
    └── Path B: "Paris" (P=0.08)
  Step 2:
    ├── Path A1: "The capital" (0.9 * 0.8 = 0.72)
    ├── Path A2: "The city"    (0.9 * 0.1 = 0.09)
    ├── Path B1: "Paris is"    (0.08 * 0.95 = 0.076)
    └── Path B2: "Paris has"   (0.08 * 0.04 = 0.0032)
    ──► Prune to Top-2: Retain Path A1 (0.72) and Path A2 (0.09)

```

### A. Greedy Decoding (Argmax Search)

Greedy search selects the token with the highest conditional probability at every individual time step:

$$y_t = \arg\max_{v \in V} z_{t, v} = \arg\max_{v \in V} P(v \mid y_{<t}, X)$$

* **Computational Complexity:** $\mathcal{O}(T)$ with zero memory or branching overhead.
* **Failure Modes:**
* **Myopic Optimization:** Greedy search makes decisions locally. An optimal token choice early on can lead directly to low-probability valleys later in the sequence.
* **Degenerate Repetition Loops:** On open-ended generation tasks, greedy decoding frequently falls into self-reinforcing repetitive attractor loops (e.g., repeating phrases indefinitely).



### B. Beam Search

Beam search maintains a fixed number $B$ (beam width) of active candidate sequences (hypotheses) simultaneously. At step $t$, it expands all $B$ paths across all $V$ vocabulary options ($B \times V$ candidates), calculates cumulative sequence log-probabilities, and retains only the top $B$ paths:

$$\mathcal{S}_t = \text{Top-}B \left( \left\{ \mathbf{y}_{<t}^{(b)} \circ v \;\Big\vert{}\; \mathbf{y}_{<t}^{(b)} \in \mathcal{S}_{t-1}, \; v \in V \right\}, \; \text{score}(\mathbf{y}) \right)$$

$$\text{score}(\mathbf{y}_{1:t}) = \frac{1}{t^\alpha} \sum_{i=1}^t \log P(y_i \mid y_{<i}, X)$$

Where $\alpha \in [0.6, 1.0]$ is a length normalization penalty preventing the search from systematically biasing toward shorter sequences.

* **Where Beam Search Excels:** Closed-ended transduction problems with narrow output spaces (Machine Translation, Abstractive Summarization, Text-to-SQL).
* **Where Beam Search Degrades:** Open-ended natural dialogue and creative reasoning. As demonstrated by Holtzman et al., high-probability beam search paths in open domains correspond to unnatural, bland, and repetitive text because human language does not consistently maximize point-wise token likelihood.

---

## 3. Stochastic Sampling & Distribution Reshaping

Stochastic decoding treats the normalized logit vector as a categorical probability distribution and samples tokens pseudorandomly.

### A. Temperature Scaling ($T$)

Temperature scaling modulates the entropy of the probability distribution by scaling raw logits prior to applying the Softmax function:

$$P(y_t = i \mid y_{<t}) = \frac{\exp(z_{t, i} / T)}{\sum_{j=1}^V \exp(z_{t, j} / T)}$$

Where $T > 0$ is the temperature scalar.

```text
Logit Transformation Under Temperature Scaling:

High Temperature (T = 1.5):           Standard (T = 1.0):            Low Temperature (T = 0.2):
Uniform / High Entropy               Unmodified Distribution         Peak-Sharpened / Near-Greedy

      █   ▄   ▂                             █                               █
  █   █   █   █   ▄                     █   █   ▄   ▂                   █   │   │   │   │
  ─────────────────                     ─────────────────           ─────────────────
  Increases diversity / randomness      Natural model distribution   Suppresses low-probability tail

```

* **$T \to 0$:** The distribution collapses to a Dirac delta function centered at the maximum logit ($\arg\max$). Equivalent to Greedy Search.
* **$T = 1.0$:** Standard unmodified probability distribution as parameterized during pre-training.
* **$T \to \infty$:** The distribution flattens into a uniform distribution $\mathcal{U}(1, V)$, maximizing entropy and hallucinations.

---

### B. Top-$k$ Truncation

Top-$k$ sampling (Fan et al.) restricts the sampling pool strictly to the $k$ tokens with the highest probabilities, redistributing the remaining probability mass among them:

$$V^{(k)} = \text{top\_k\_indices}(P, k)$$

$$P'(y_t = i) = \begin{cases} \frac{P(y_t = i)}{\sum_{j \in V^{(k)}} P(y_t = j)}, & \text{if } i \in V^{(k)} \\ 0, & \text{otherwise} \end{cases}$$

```text
Top-k Truncation (k = 3):
Original Tokens:    [ "cat": 0.50, "dog": 0.30, "fish": 0.15, "car": 0.04, "tree": 0.01 ]
Retained Pool:      [ "cat": 0.50, "dog": 0.30, "fish": 0.15 ]  (Drop "car", "tree")
Renormalized:       [ "cat": 0.526, "dog": 0.316, "fish": 0.158 ]

```

* **Limitation:** Top-$k$ uses a static threshold regardless of model confidence:
* **When the model is confident:** (e.g., $P(\text{"Paris"}) = 0.98$), setting $k=50$ forces the sampler to include 49 improbable tail tokens.
* **When the model is uncertain:** (flat distribution across 200 plausible options), setting $k=50$ truncates valid candidates prematurely.



---

### C. Top-$p$ (Nucleus) Sampling

Top-$p$ sampling (Holtzman et al.) dynamically scales the candidate pool based on cumulative probability mass. It isolates the smallest set of tokens $V^{(p)}$ whose cumulative sum exceeds threshold $p \in (0, 1]$:

$$\sum_{i \in V^{(p)}} P(y_t = i) \ge p$$

```text
Nucleus (Top-p) Dynamic Vocabulary Scaling (p = 0.90):

Scenario A: High Confidence Context (Sharp Distribution)
Sorted Tokens:      [ "Paris": 0.85, "France": 0.08, "the": 0.04, "city": 0.02, ... ]
Cumulative Sum:     [   0.85,          0.93,          0.97,         0.99       ]
                      └─────────────────┬┘
Retained Pool:      Only 2 tokens selected (Mass = 0.93 >= 0.90). Tail safely eliminated.

Scenario B: High Uncertainty Context (Flat Distribution)
Sorted Tokens:      [ "red": 0.12, "blue": 0.11, "green": 0.10, "yellow": 0.09, ..., "black": 0.01 ]
Retained Pool:      Expands dynamically to 18 tokens to reach cumulative 0.90 mass.

```

---

### D. Min-$p$ Sampling

While Top-$p$ scales dynamically, it remains vulnerable to truncating valid options in the long-tail when the top token has low probability, or retaining noisy tokens when temperature is elevated.

**Min-$p$ sampling** sets a dynamic probability threshold relative to the highest-probability candidate $P_{\max} = \max_j P(y_t = j)$:

$$\text{Threshold} = p_{\text{base}} \times P_{\max}$$

$$V^{(\text{min-}p)} = \{ i \in V \mid P(y_t = i) \ge p_{\text{base}} \times P_{\max} \}$$

```text
Min-p Sampling Mechanics (p_base = 0.10):

Case 1: Confident Prediction (P_max = 0.80)
  Threshold = 0.10 * 0.80 = 0.08
  • Any token with P < 0.08 is truncated.
  • Prevents low-probability hallucinations during confident steps.

Case 2: Ambiguous Prediction (P_max = 0.15)
  Threshold = 0.10 * 0.15 = 0.015
  • Candidate pool automatically broadens to include all viable hypotheses.

```

Min-$p$ provides a cleaner separation of signal and noise across varying temperatures than Top-$p$, and is widely supported in modern inference systems (such as `llama.cpp` and `vLLM`).

---

## 4. Repetition and Frequency Penalties

To prevent repetitive phrasing without reducing vocabulary expressivity, inference engines modify logits based on token presence in historical context.

Let $c(v)$ be the count of occurrences of token $v$ in the generated context window $y_{<t}$.

```text
┌────────────────────────────────────────────────────────────────────────┐
│ PENALTY FORMULATIONS                                                   │
├──────────────────────────────┬─────────────────────────────────────────┤
│ 1. Multiplicative Repetition │ If z_v > 0:  z'_v = z_v / theta         │
│    Penalty (Keskar et al.)   │ If z_v <= 0: z'_v = z_v * theta         │
│                              │ (Standard: theta in [1.05, 1.20])       │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 2. Additive Frequency Penalty│ z'_v = z_v - (alpha_freq * c(v))        │
│    (Scales with occurrences) │ Penalizes repeated usage proportionally.│
├──────────────────────────────┼─────────────────────────────────────────┤
│ 3. Additive Presence Penalty │ z'_v = z_v - (alpha_pres * I(c(v) > 0)) │
│    (Binary penalty)          │ One-time penalty for any prior mention. │
└──────────────────────────────┴─────────────────────────────────────────┘

```

---

## 5. Comparative Sampling Matrix

| Decoding Strategy | Deterministic? | Computational Cost | Diversity Score | Primary Production Use Case |
| --- | --- | --- | --- | --- |
| **Greedy Search** | Yes | Lowest ($\mathcal{O}(1)$ selection) | Minimal | Code syntax, SQL, JSON extraction, arithmetic |
| **Beam Search** | Yes | High ($\mathcal{O}(B)$ forward states) | Low | Machine translation, formal summarization |
| **Top-$k$ Sampling** | No | Low ($\text{Top-}k$ sort) | Moderate | Legacy text generation recipes |
| **Top-$p$ (Nucleus)** | No | Moderate (Full sort + prefix sum) | High | Conversational assistants, roleplay, creative writing |
| **Min-$p$ Sampling** | No | Low (Threshold mask) | High | Modern LLM generation, extended reasoning, general chat |

---

## 6. PyTorch Implementation: Production Generation Sampler Engine

Below is the complete PyTorch implementation of a modular generation sampler supporting logit penalties, temperature scaling, Top-$k$, Top-$p$, Min-$p$, and deterministic fallbacks:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductionSampler(nn.Module):
    """
    Inference sampling engine supporting Temperature, Top-k, Top-p,
    Min-p, and Multiplicative Repetition Penalties.
    """
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0
    ):
        super().__init__()
        self.temperature = max(1e-5, temperature)
        self.top_k = max(0, top_k)
        self.top_p = min(1.0, max(0.0, top_p))
        self.min_p = min(1.0, max(0.0, min_p))
        self.repetition_penalty = repetition_penalty

    def apply_repetition_penalty(
        self, logits: torch.Tensor, generated_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies multiplicative repetition penalty to previously emitted tokens.
        """
        if self.repetition_penalty == 1.0 or generated_tokens.numel() == 0:
            return logits

        B, V = logits.shape
        for b in range(B):
            unique_tokens = torch.unique(generated_tokens[b])
            token_logits = logits[b, unique_tokens]
            
            # If logit is positive, divide by penalty; if negative, multiply
            updated_logits = torch.where(
                token_logits > 0,
                token_logits / self.repetition_penalty,
                token_logits * self.repetition_penalty
            )
            logits[b, unique_tokens] = updated_logits

        return logits

    def filter_top_k(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Filters out all tokens outside the top-k highest logits."""
        if k <= 0 or k >= logits.size(-1):
            return logits
        
        top_k_values, _ = torch.topk(logits, k, dim=-1)
        min_top_k_val = top_k_values[..., -1, None]
        return torch.where(logits < min_top_k_val, float("-inf"), logits)

    def filter_top_p(self, probs: torch.Tensor, p: float) -> torch.Tensor:
        """Filters out tokens outside the top-p cumulative probability nucleus."""
        if p >= 1.0:
            return probs

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Shift cumulative probabilities to retain the first token exceeding p
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Zero out removed probabilities and re-scatter to original indices
        sorted_probs[sorted_indices_to_remove] = 0.0
        filtered_probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
        return filtered_probs

    def filter_min_p(self, probs: torch.Tensor, min_p_ratio: float) -> torch.Tensor:
        """Filters out tokens with probability less than min_p_ratio * max_prob."""
        if min_p_ratio <= 0.0:
            return probs

        max_probs = torch.max(probs, dim=-1, keepdim=True).values
        threshold = min_p_ratio * max_probs
        filtered_probs = torch.where(probs < threshold, torch.zeros_like(probs), probs)
        return filtered_probs

    @torch.no_grad()
    def sample(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor | None = None,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Samples next token IDs from raw logits.
        
        Args:
            logits: Unnormalized logits tensor of shape (Batch, Vocab_Size)
            generated_tokens: Past token IDs of shape (Batch, Context_Len)
            deterministic: Force greedy argmax decoding
        """
        # 1. Apply Repetition Penalty
        if generated_tokens is not None:
            logits = self.apply_repetition_penalty(logits, generated_tokens)

        # 2. Deterministic Shortcut (Greedy Search)
        if deterministic or self.temperature < 1e-4:
            return torch.argmax(logits, dim=-1, keepdim=True)

        # 3. Temperature Scaling
        scaled_logits = logits / self.temperature

        # 4. Top-K Logit Truncation
        if self.top_k > 0:
            scaled_logits = self.filter_top_k(scaled_logits, self.top_k)

        # 5. Softmax to Probability Space
        probs = F.softmax(scaled_logits, dim=-1)

        # 6. Min-p Dynamic Filtering
        if self.min_p > 0.0:
            probs = self.filter_min_p(probs, self.min_p)

        # 7. Top-p (Nucleus) Filtering
        if self.top_p < 1.0:
            probs = self.filter_top_p(probs, self.top_p)

        # 8. Renormalize Probabilities Post-Filtering
        sum_probs = torch.sum(probs, dim=-1, keepdim=True).clamp(min=1e-8)
        probs = probs / sum_probs

        # 9. Multinomial Categorical Sampling
        next_tokens = torch.multinomial(probs, num_samples=1)
        return next_tokens

```

---

## 7. Recommended Sampling Configurations by Task

```text
┌────────────────────────────────────────────────────────────────────────┐
│ RECOMMENDED SAMPLING PROFILES                                          │
├──────────────────────────────┬─────────────────────────────────────────┤
│ 1. Deterministic Extraction  │ Temperature: 0.0 (Greedy)               │
│    (Code, SQL, JSON, Math)   │ Repetition Penalty: 1.0                 │
│                              │ Target: Zero hallucination, exact syntax│
├──────────────────────────────┼─────────────────────────────────────────┤
│ 2. Formal Reasoning & SFT    │ Temperature: 0.2 - 0.4                  │
│    (Chain-of-Thought, QA)    │ Min-p: 0.05                             │
│                              │ Target: Focused coherence, low drift    │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 3. Conversational / Chat     │ Temperature: 0.7                        │
│    (General Assistant)       │ Min-p: 0.05 or Top-p: 0.90              │
│                              │ Repetition Penalty: 1.10                │
│                              │ Target: Natural pacing, balanced lexical│
├──────────────────────────────┼─────────────────────────────────────────┤
│ 4. Creative Writing          │ Temperature: 0.85 - 1.0                 │
│    (Brainstorming, Narrative)│ Min-p: 0.08                             │
│                              │ Repetition Penalty: 1.15                │
│                              │ Target: High entropy, non-repetitive    │
└──────────────────────────────┴─────────────────────────────────────────┘

```