# AdamW Optimization and Selective Weight Decay

## 1. The Adam vs. AdamW Decoupled Weight Decay Formulation

In standard Stochastic Gradient Descent (SGD), $L_2$ regularization is mathematically equivalent to weight decay. Adding an $L_2$ penalty term $\frac{1}{2} \lambda \Vert{}\theta\Vert{}_2^2$ to the loss function $\mathcal{L}(\theta)$ modifies the gradient:

$$\nabla \mathcal{L}_{\text{total}}(\theta) = \nabla \mathcal{L}(\theta) + \lambda \theta$$

When applied to standard SGD with learning rate $\eta$:

$$\theta_{t} = \theta_{t-1} - \eta \left( \nabla \mathcal{L}(\theta_{t-1}) + \lambda \theta_{t-1} \right) = (1 - \eta \lambda)\theta_{t-1} - \eta \nabla \mathcal{L}(\theta_{t-1})$$

In adaptive gradient methods like **Adam** (Adaptive Moment Estimation), this equivalence breaks down.

```text
Standard Adam with L2 Regularization (Coupled / Distorted):
  Gradient with L2:    g_t = ∇L(θ_{t-1}) + λ θ_{t-1}
  Variance Tracking:   v_t = β_2 v_{t-1} + (1 - β_2) g_t^2   <-- Regularization enters variance!
  Parameter Update:    θ_t = θ_{t-1} - η * [ m_t / (sqrt(v_t) + ε) ]
  Problem: Parameters with large historic gradients receive LESS weight decay,
           while parameters with small gradients receive EXCESSIVE weight decay.

AdamW (Decoupled Weight Decay):
  Loss Gradient:       g_t = ∇L(θ_{t-1})                     <-- Pure objective gradient
  Variance Tracking:   v_t = β_2 v_{t-1} + (1 - β_2) g_t^2   <-- Tracks true gradient variance
  Decoupled Decay:     θ_t = θ_{t-1} - η λ θ_{t-1} - η * [ m_t / (sqrt(v_t) + ε) ]
  Result: Weight decay is applied uniformly and independently of gradient variance.

```

Loshchilov & Hutter demonstrated that coupling $L_2$ regularization with adaptive moments causes weights with frequent, large gradients to be regularized less than weights with infrequent gradients. **AdamW** resolves this by decoupling weight decay from the gradient update step, applying decay directly to the parameter tensor.

---

## 2. Mathematical Mechanics of the AdamW Optimizer

The AdamW algorithm tracks two running moments for every trainable parameter: the exponentially decaying average of past gradients (**First Moment $m_t$**) and past squared gradients (**Second Moment $v_t$**).

```text
At each optimization step t:
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Compute Objective Gradient:     g_t = ∇_θ L_t(θ_{t-1})              │
│ 2. Update Biased 1st Moment:       m_t = β_1 m_{t-1} + (1 - β_1) g_t   │
│ 3. Update Biased 2nd Moment:       v_t = β_2 v_{t-1} + (1 - β_2) g_t^2 │
│ 4. Compute Bias-Corrected 1st:     m_hat_t = m_t / (1 - β_1^t)         │
│ 5. Compute Bias-Corrected 2nd:     v_hat_t = v_t / (1 - β_2^t)         │
│ 6. Apply Decoupled Decay & Step:   θ_t = θ_{t-1} - η_t λ θ_{t-1}       │
│                                    -η_t * m_hat_t / (sqrt(v_hat_t) + ε)│
└────────────────────────────────────────────────────────────────────────┘

```

### 1. First Moment ($m_t$) — Momentum

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$


Tracks the directional velocity of the descent trajectory. Standard configuration is $\beta_1 = 0.9$.

### 2. Second Moment ($v_t$) — Adaptive Variance

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$


Tracks the uncentered variance (energy) of each parameter coordinate. Standard configurations are $\beta_2 = 0.999$ (for small models/pre-training baselines) or $\beta_2 = 0.95$ (standard in modern LLM training like LLaMA to shorten memory retention of gradient spikes).

### 3. Initialization Bias Correction ($\hat{m}_t, \hat{v}_t$)

Because $m_0$ and $v_0$ are initialized to zero vectors, both moments are biased toward zero in initial steps. Dividing by $(1 - \beta_1^t)$ and $(1 - \beta_2^t)$ corrects for this initialization bias:

$$\lim_{t \to \infty} (1 - \beta^t) = 1.0$$

At step $t = 1$ with $\beta_1 = 0.9$, $(1 - 0.9^1) = 0.1$, scaling $m_1$ up by $10\times$ to reflect the true gradient magnitude.

---

## 3. The Necessity of Selective Weight Decay

Applying weight decay indiscriminately across all parameters in a Transformer damages model stability and representation capacity. Parameters must be segregated into **decay** and **no-decay** groups.

```text
Model Parameter Tensors
           │
           ├─── Dimension >= 2 (Matrices & Embeddings) ──► Apply Weight Decay (λ = 0.01 - 0.1)
           │    • Attention Projections (W_q, W_k, W_v, W_o)
           │    • Feed-Forward Projections (W_gate, W_up, W_down)
           │    • Token Embedding Table (W_emb)
           │
           └─── Dimension < 2 (Vectors & Scalars) ────────► ZERO Weight Decay (λ = 0.0)
                • LayerNorm / RMSNorm scale gains (γ)
                • Additive Biases (b)
                • Positional Scale Scalars

```

### Why 1D Parameters Must NOT Decay

1. **Normalization Scale Factors ($\gamma$):** In LayerNorm and RMSNorm, $\gamma$ is a 1D scaling vector initialized to $1.0$. If weight decay is applied, $\gamma$ is penalized and driven toward $0.0$. Shrinking $\gamma$ artificially dampens activations across the residual stream, fighting the normalization layer's ability to maintain unit variance.
2. **Bias Vectors ($b$):** Biases shift decision boundaries without increasing model capacity in high-dimensional feature spaces. Penalizing biases does not prevent overfitting and can introduce systematic activation offsets.
3. **Representational Degradation:** 2D weight matrices parameterize complex multi-dimensional manifolds where $L_2$ shrinkage acts as a genuine capacity regularizer. 1D parameters govern statistical calibration and scale alignment; penalizing them degrades optimization dynamics without providing generalization benefits.

---

## 4. Parameter Partitioning Architecture

To implement selective weight decay in PyTorch, model parameters are partitioned into two disjoint parameter groups passed to `torch.optim.AdamW`:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ PARAMETER GROUP CONFIGURATION                                          │
├──────────────────────────────┬─────────────────────────────────────────┤
│ Group 1: Decayed Parameters  │ • Condition: param.dim() >= 2           │
│                              │ • Weight Decay: λ = 0.1 (Configurable)  │
│                              │ • Includes: Linear weights, Embeddings  │
├──────────────────────────────┼─────────────────────────────────────────┤
│ Group 2: Non-Decayed Params  │ • Condition: param.dim() < 2            │
│                              │ • Weight Decay: λ = 0.0                 │
│                              │ • Includes: LayerNorm/RMSNorm weights, b│
└──────────────────────────────┴─────────────────────────────────────────┘

```

### Parameter Count Verification Assertion

A robust continuous training engine must verify that every parameter is allocated to **exactly one** group:

$$\vert{}\Theta_{\text{total}}\vert{} = \vert{}\Theta_{\text{decay}}\vert{} + \vert{}\Theta_{\text{no\_decay}}\vert{}$$

$$\Theta_{\text{decay}} \cap \Theta_{\text{no\_decay}} = \emptyset$$

---

## 5. PyTorch Implementation: `configure_optimizers` Engine

Below is the production-grade optimizer configuration method for Transformer architectures:

```python
import torch
import torch.nn as nn

def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.1,
    learning_rate: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    device_type: str = "cuda"
) -> torch.optim.AdamW:
    """
    Configures AdamW optimizer with strict selective weight decay:
    - 2D+ tensors (Linear weights, Embedding tables) -> weight decay enabled
    - 1D tensors (Normalization scales, Biases) -> weight decay disabled
    """
    # Separate parameters into decay and no-decay pools
    decay_params = []
    nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Frozen parameters are skipped

        # 2D+ parameters decay; 1D parameters (biases, norms) do not
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    # Build optimizer parameter groups
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]

    # Calculate statistics for pre-flight assertions
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    total_params = num_decay_params + num_nodecay_params

    print(f"Optimizer Parameter Allocation:")
    print(f"  • Decayed parameter tensors:     {len(decay_params):3d} ({num_decay_params:,} parameters)")
    print(f"  • Non-decayed parameter tensors: {len(nodecay_params):3d} ({num_nodecay_params:,} parameters)")
    print(f"  • Total trainable parameters:    {total_params:,}")

    # Use fused AdamW kernel if available on CUDA (Ampere/Hopper/Blackwell)
    use_fused = (device_type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
    extra_args = dict(fused=True) if use_fused else dict()

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
        **extra_args
    )

    return optimizer

```

---

## 6. Hyperparameter Calibration & Numerical Stability

| Hyperparameter | Standard Default | Extended Context / LLM Scale | Impact of Misconfiguration |
| --- | --- | --- | --- |
| **Learning Rate ($\eta$)** | $3 \times 10^{-4}$ | $1 \times 10^{-4} \text{ to } 6 \times 10^{-4}$ | Too high: Loss divergence (`NaN`). Too low: Stalled convergence. |
| **Momentum ($\beta_1$)** | $0.90$ | $0.90$ | Too high: Overshoots sharp loss valleys. Too low: Slow traversal of flat plateaus. |
| **Variance ($\beta_2$)** | $0.999$ | **$0.95 \text{ to } 0.98$** | $0.999$ retains gradient spikes too long; $0.95$ recovers rapidly from outlier batches. |
| **Denominator Epsilon ($\epsilon$)** | $10^{-8}$ | **$10^{-6} \text{ to } 10^{-5}$** | In mixed precision (FP16/BF16), $\epsilon = 10^{-8}$ can underflow, causing division by zero. |
| **Weight Decay ($\lambda$)** | $0.01$ | **$0.10$** | Too high: Underfitting / excessive parameter shrinkage. Too low: Overfitting on small data. |

### Optimizer Comparison Matrix

| Dimension | SGD with Momentum | Standard Adam (L2) | AdamW (Decoupled) | Lion (EvoLved Sign) |
| --- | --- | --- | --- | --- |
| **State Memory** | 1 buffer ($m_t$, 4B/param) | 2 buffers ($m_t, v_t$, 8B/param) | **2 buffers ($m_t, v_t$, 8B/param)** | 1 buffer ($m_t$, 4B/param) |
| **Decoupled Decay** | N/A (Equivalent) | No (Distorted by $v_t$) | **Yes (Pure uniform decay)** | Yes (Sign-based decay) |
| **Sensitivity to Scale** | High (Requires exact LR tuning) | Low (Self-normalizing) | **Low (Robust across layers)** | Low (Sign operation bounded) |
| **LLM Pre-Training Standard** | Rarely used in modern LLMs | Deprecated | **Universal Industry Standard** | Experimental alternative |