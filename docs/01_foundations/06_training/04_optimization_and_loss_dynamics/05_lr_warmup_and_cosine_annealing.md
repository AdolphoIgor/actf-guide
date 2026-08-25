# Learning Rate Warmup, Cosine Annealing, and Schedule Topologies

## 1. The Learning Rate Scheduling Problem in Transformers

In deep Transformer optimization, using a constant learning rate throughout training leads to severe failure modes at both ends of the optimization trajectory:

```text
Early Training (Steps 0 to T_warmup):
  • Uninitialized weights generate high-variance, chaotic gradients.
  • AdamW variance buffers (v_t) are uncalibrated (relying heavily on bias correction).
  • High learning rates at step 0 cause catastrophic gradient updates, knocking weights
    into flat or sub-optimal local regions (loss divergence / NaN).

Late Training (Steps T_warmup to T_max):
  • As parameters approach optimal loss valleys, high learning rates oscillate across
    ravines and bounce out of sharp minima.
  • Constant learning rates prevent the model from settling into flat, generalizable basins.

```

To achieve stable optimization and optimal generalization, the learning rate $\eta_t$ must be dynamically modulated over time: warming up slowly from zero to allow internal statistics to calibrate, then gradually annealing toward a minimal floor to facilitate convergence.

```text
Learning Rate Trajectory (Linear Warmup + Cosine Annealing):

  LR (η)
   ▲
   │           Peak Learning Rate (η_max)
   │                  ┌─────────────────┐
   │                 /                   \
   │                /                     \
   │               /                       \
   │  Linear      /                         \  Cosine Decay Phase
   │  Warmup     /                           \
   │  Phase     /                             \
   │           /                               \
   │          /                                 \___  Min Learning Rate (η_min)
   └─────────┴────────────────────────────────────────┴───────► Optimization Steps (t)
           t = 0     t = T_warmup                   t = T_max

```

---

## 2. Linear Warmup Mechanics

During the **Linear Warmup** phase, the learning rate scales linearly from zero (or an initial base rate $\eta_{\text{init}}$) to the peak target learning rate $\eta_{\max}$ across a predefined number of steps $T_{\text{warmup}}$.

### Mathematical Formulation

For step $t \in [0, T_{\text{warmup}}]$:

$$\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}$$

If a non-zero initial learning rate $\eta_{\text{init}}$ is configured:

$$\eta_t = \eta_{\text{init}} + (\eta_{\max} - \eta_{\text{init}}) \cdot \frac{t}{T_{\text{warmup}}}$$

### Why Warmup Stabilizes AdamW

1. **Moment Buffer Calibration:** At initialization, AdamW's first moment $m_0$ and second moment $v_0$ are initialized to zero tensors. In the initial steps ($t < 100$), the bias correction factor $(1 - \beta_2^t)^{-1}$ heavily inflates the variance estimates. Restricting $\eta_t$ during this window prevents high-gain updates while the variance vector $v_t$ populates.
2. **Attention Alignment Stabilization:** In early steps, Query-Key dot products produce high-entropy attention weight distributions. As the self-attention heads organize into distinct functional roles, low learning rates protect the fragile residual stream from destabilizing perturbations.

---

## 3. Cosine Annealing Formulation

Introduced by Loshchilov & Hutter and widely adopted across LLM pre-training (such as GPT-3, LLaMA, Chinchilla, and Mistral), **Cosine Annealing** decays the learning rate following the curve of a half-cosine period.

### Mathematical Formulation

For step $t$ where $T_{\text{warmup}} \le t \le T_{\text{max}}$:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left( 1 + \cos\left( \pi \cdot \frac{t - T_{\text{warmup}}}{T_{\text{max}} - T_{\text{warmup}}} \right) \right)$$

For steps beyond the scheduled training horizon ($t > T_{\text{max}}$):

$$\eta_t = \eta_{\min}$$

Where:

* $\eta_{\max}$ is the peak learning rate (e.g., $3 \times 10^{-4}$).
* $\eta_{\min}$ is the minimum learning rate floor (conventionally set to $0.1 \times \eta_{\max}$ or $0$).
* $T_{\text{warmup}}$ is the duration of the linear warmup phase.
* $T_{\text{max}}$ is the total number of scheduled training steps.

### The Minimum Learning Rate Floor ($\eta_{\min}$)

Rather than decaying learning rate completely to zero ($\eta_{\min} = 0$), production recipes set $\eta_{\min} = 0.1 \times \eta_{\max}$ (a 10% floor).

* Decaying to absolute zero halts parameter movement entirely, preventing the model from continuing to learn from final data shards.
* Maintaining a 10% floor ensures ongoing gradient mobility, allowing weight decay to continue regularizing weights while fine-tuning final layer representations.

---

## 4. Schedule Comparison Matrix

| Schedule Type | Mathematical Curve | Peak Stability | Extensibility / Continuous Training | Convergence Speed | Primary Use Case |
| --- | --- | --- | --- | --- | --- |
| **Cosine Annealing with Warmup** | Smooth harmonic decay | High | Poor (Requires knowing $T_{\max}$ upfront) | Optimal | Standard fixed-budget pre-training & SFT |
| **Linear Decay with Warmup** | Piecewise linear descent | Moderate | Poor (Tied to fixed $T_{\max}$) | Fast | Quick fine-tuning sweeps, BERT-style tasks |
| **Inverse Square Root** | $\eta_t \propto 1 / \sqrt{t}$ | High | High (Decays indefinitely without target $T_{\max}$) | Moderate | Seq2Seq translation (T5, original Transformer) |
| **Warmup-Stable-Decay (WSD)** | Trapezoidal (Flat plateau + rapid decay) | High | **Optimal (Can extend training arbitrarily)** | Fast during decay | Modern Continuous Training (CT) and dynamic LLM runs |

---

## 5. PyTorch Implementation: Production Cosine Scheduler

Below is the standalone Python implementation of a customized Cosine Annealing schedule with Linear Warmup, built on top of `torch.optim.lr_scheduler.LambdaLR`:

```python
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Creates a learning rate schedule that increases linearly from 0.0 to 1.0
    over num_warmup_steps, then decreases following a cosine curve to min_lr_ratio
    over the remainder of num_training_steps.
    """
    def lr_lambda(current_step: int) -> float:
        # 1. Linear Warmup Phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 2. Beyond scheduled training horizon -> hold at min_lr_ratio
        if current_step > num_training_steps:
            return min_lr_ratio

        # 3. Cosine Annealing Phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

class CosineWarmupLREngine:
    """
    Self-contained schedule manager for continuous training pipelines.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
        min_lr_ratio = min_lr / max_lr
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=min_lr_ratio
        )

    def step(self):
        """Advances the scheduler by one optimization step."""
        self.scheduler.step()

    def get_current_lr(self) -> float:
        """Extracts current learning rate for logging."""
        return self.optimizer.param_groups[0]["lr"]

```

---

## 6. The Continuous Training Dilemma: Cosine vs. WSD (Warmup-Stable-Decay)

While Cosine Annealing delivers strong empirical results on fixed datasets, it introduces a major structural constraint for **Continuous Training (CT)** pipelines:

```text
The Cosine Fixed-Horizon Problem:
  • The cosine decay trajectory depends on a pre-committed total step count T_max.
  • If a team decides to train for an additional 50,000 steps on newly curated data,
    extending T_max alters the schedule retrospectively.
  • Restarting cosine annealing from the low floor causes severe optimization shocks.

```

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ WSD (Warmup-Stable-Decay) Architecture: The Continuous Training Solution │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LR (η)                                                                  │
│   ▲                                                                      │
│   │           Peak Plateau (η_max)                                       │
│   │        ┌───────────────────────────────────────┐                     │
│   │       /                                         \                    │
│   │      /                                           \                   │
│   │     /  Warmup                                     \  Fast Decay      │
│   │    /   Phase                                       \ (Last 10-20%)   │
│   │   /                                                 \___ (η_min)     │
│   └──┴──────────────────────────────────────────────────┴──────► Steps   │
│     t=0  t=T_warmup                      t=Decay_Start  t=T_final        │
└──────────────────────────────────────────────────────────────────────────┘

```

### The Three Phases of WSD

1. **Warmup Phase ($0 \le t < T_{\text{warmup}}$):** Linear scaling up to $\eta_{\max}$ (typically $1\%\text{--}5\%$ of the initial expected run).
2. **Stable Phase ($T_{\text{warmup}} \le t < T_{\text{decay\_start}}$):** The learning rate remains constant at $\eta_{\max}$ indefinitely. New data streams can be added continuously without altering the schedule.
3. **Decay Phase ($T_{\text{decay\_start}} \le t \le T_{\text{final}}$):** When the compute budget or dataset horizon is reached, the learning rate decays rapidly (via cosine or linear descent) over the final $10\%\text{--}20\%$ of steps.

Empirical studies (e.g., MiniCPM, Llama-3 technical insights) demonstrate that models trained under WSD achieve final loss parity with pure Cosine schedules while offering complete flexibility to extend pre-training runs indefinitely.

---

## 7. Hyperparameter Calibration Guidelines

| Regime / Task Scale | Warmup Duration ($T_{\text{warmup}}$) | Peak Learning Rate ($\eta_{\max}$) | Min Learning Rate ($\eta_{\min}$) | Decay Duration |
| --- | --- | --- | --- | --- |
| **Micro-Models ($\Phi \le 10\text{M}$)** | $100\text{--}500\text{ steps}$ | $5 \times 10^{-4} \text{ to } 1 \times 10^{-3}$ | $0.1 \times \eta_{\max}$ | Full run balance |
| **Small Baseline ($\Phi \approx 500\text{M}$)** | $1,000\text{--}2,000\text{ steps}$ | $3 \times 10^{-4} \text{ to } 6 \times 10^{-4}$ | $0.1 \times \eta_{\max}$ | Full run balance |
| **Enterprise SFT ($\Phi \approx 7\text{B}$)** | $100\text{--}500\text{ steps}$ ($3\%\text{--}5\%$) | $1 \times 10^{-5} \text{ to } 5 \times 10^{-5}$ | $0.1 \times \eta_{\max}$ | Over total epochs |
| **Continuous Training (WSD)** | $2,000\text{ steps}$ | $3 \times 10^{-4}$ | $0.05 \times \eta_{\max}$ | Final $10\%\text{--}15\%$ of run |