# Early Stopping, State Serialization, and Checkpoint Snapshotting

## 1. The Generalization Horizon and Overfitting Trajectory

During continuous pre-training and supervised fine-tuning (SFT), empirical training loss decreases monotonically as parameters adapt to the training distribution. However, optimization eventually crosses an inflection point beyond which the model transitions from learning generalized semantic representations to memorizing corpus-specific artifacts and noisy sequence patterns.

```text
Loss Trajectory Over Optimization Steps:

  Loss (L)
   ▲
   │  \
   │   \  Underfitting Regime             Inflection Point (t*)
   │    \                                          │
   │     \      Validation Loss (L_val)            │      Overfitting Regime
   │      \    \                              _ - -▼- - _
   │       \    \                         _ -             - _  Validation Loss Diverges
   │        \    \                    _ -                     - _ ──► Generalization Loss
   │         \    \               _ -
   │          \    \_ _ _ _ _ _ -
   │           \
   │            \_________________________________________________  Training Loss (L_train)
   │                                                                 Monotonic Descent
   └───────────────────────────────────────────────────────────────► Steps (t)
                                 Optimal Snapshot Window

```

Terminating training strictly based on a fixed step count risks either **underfitting** (stopping before convergence) or **overfitting** (training past $t^*$).

**Early stopping** continuously evaluates out-of-sample validation loss ($\mathcal{L}_{\text{val}}$) over an isolated evaluation split, terminating the training run when generalization gains plateau. Concurrently, **checkpoint snapshotting** persists the optimal model weights and execution states to non-volatile storage.

---

## 2. Early Stopping Mathematical Mechanics

Early stopping evaluates whether the current validation loss $\mathcal{L}_{\text{val}}^{(t)}$ achieves a statistically meaningful improvement over the historic minimum $\mathcal{L}_{\text{best}}$.

```text
                       [ Evaluate Validation Loss: L_val(t) ]
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Improvement Threshold Check:          │
                     │ L_val(t) < L_best - delta_min ?       │
                     └───────────────────┬───────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │ YES                                       │ NO
                   ▼                                           ▼
       ┌────────────────────────┐                  ┌─────────────────────────┐
       │ 1. L_best = L_val(t)   │                  │ 1. patience_counter += 1│
       │ 2. patience_counter = 0│                  └───────────┬─────────────┘
       │ 3. Save Best Snapshot  │                              │
       └────────────────────────┘                              ▼
                                                   ┌────────────────────────┐
                                                   │ patience >= Max Limit ?│
                                                   └───────────┬────────────┘
                                                               │
                                             ┌─────────────────┴─────────────────┐
                                             │ YES                               │ NO
                                             ▼                                   ▼
                                 [ TRIGGER EARLY STOPPING ]             [ Continue Training ]

```

### Mathematical Formulation

Let $\mathcal{L}_{\text{val}}^{(t)}$ be the validation loss computed at evaluation interval $t \in \{1, 2, \dots, T_{\text{eval}}\}$. An improvement is registered if and only if:

$$\mathcal{L}_{\text{val}}^{(t)} < \mathcal{L}_{\text{best}} - \delta_{\text{min}}$$

Where:

- $\mathcal{L}_{\text{best}} = \min_{i < t} \mathcal{L}_{\text{val}}^{(i)}$ is the lowest validation loss recorded prior to step $t$.
- $\delta_{\text{min}} \ge 0$ is the minimum absolute delta required to qualify as a valid improvement (e.g., $\delta_{\text{min}} = 10^{-4}$).

The patience counter $p_t$ updates according to the recursive relation:

$$p_t = \begin{cases} 0, & \text{if } \mathcal{L}_{\text{val}}^{(t)} < \mathcal{L}_{\text{best}} - \delta_{\text{min}} \\ p_{t-1} + 1, & \text{otherwise} \end{cases}$$

When $p_t \ge P_{\max}$ (where $P_{\max}$ is the configured maximum patience threshold), optimization halts, and the pipeline restores weights from the checkpoint saved at $\mathcal{L}_{\text{best}}$.

### Exponential Moving Average (EMA) Smoothing

Validation loss curves on mini-batches often exhibit high variance due to sequence length variability or outlier batches. To prevent false-positive early stopping triggers, early stopping can evaluate an Exponential Moving Average of validation loss:

$$\bar{\mathcal{L}}_{\text{val}}^{(t)} = \alpha \mathcal{L}_{\text{val}}^{(t)} + (1 - \alpha) \bar{\mathcal{L}}_{\text{val}}^{(t-1)}$$

Where $\alpha \in (0, 1]$ is the smoothing factor (typically $\alpha = 0.3$).

---

## 3. Comprehensive State Serialization Anatomy

Saving only the model parameter weights (`model.state_dict()`) is insufficient for fault-tolerant continuous training. If a node crashes or training resumes from a preemption, missing optimizer moments or scheduler states will destabilize the training run upon restart.

```text
Complete Checkpoint Archive (checkpoint.pt):
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Model State Dictionary (model_state_dict)                           │
│    • Weights, projections, embeddings, normalization gains             │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Optimizer State Dictionary (optimizer_state_dict)                   │
│    • First moments (m_t), second moments (v_t), step counters per param│
├────────────────────────────────────────────────────────────────────────┤
│ 3. Learning Rate Scheduler State (scheduler_state_dict)                │
│    • Current step, current learning rate factor, cycle indices         │
├────────────────────────────────────────────────────────────────────────┤
│ 4. Mixed Precision Scaler State (scaler_state_dict)                    │
│    • Dynamic loss scale factor S, consecutive clean step counts        │
├────────────────────────────────────────────────────────────────────────┤
│ 5. Pseudorandom Number Generator States (rng_states)                   │
│    • torch.get_rng_state(), torch.cuda.get_rng_state_all(), python rng │
├────────────────────────────────────────────────────────────────────────┤
│ 6. Metadata & Hyperparameters (metadata)                               │
│    • Global step, epoch, best validation loss, git commit hash, config │
└────────────────────────────────────────────────────────────────────────┘

```

### Why RNG States Must Be Serialized

If data loaders shuffle dynamically during continuous training, resuming from a checkpoint without restoring RNG states causes the sampler to regenerate previously seen data sequences or alter dropout masks, breaking exact training run reproducibility.

---

## 4. Atomic Persistence and Rolling Top-K Management

### 1. Atomic Persistence (Temp-Write-Rename Pattern)

Writing a multi-gigabyte checkpoint directly to its destination path risks generating a corrupted file if the process is terminated mid-write (e.g., spot instance preemption or power failure).

To guarantee disk consistency, checkpoints must follow the **Atomic Write Pattern**:

1. Write the state payload to a temporary file: `checkpoint_step_1000.pt.tmp`.
2. Flush and synchronize I/O buffers to physical storage (`os.fsync`).
3. Atomically rename the temporary file to the target destination: `os.replace("checkpoint_step_1000.pt.tmp", "checkpoint_step_1000.pt")`.

```text
Storage IO Sequence:
  PyTorch Tensor Stream ──► [ checkpoint_step_1000.pt.tmp ] ──► os.fsync() ──► os.replace() ──► [ checkpoint_step_1000.pt ]

```

### 2. Rolling Top-K Checkpoint Retention

Persisting checkpoints at every evaluation interval quickly exhausts local disk space. A production checkpoint manager maintains:

- **The Best Checkpoint (`best_model.pt`):** The historical minimum validation loss snapshot.
- **The Latest Checkpoint (`latest_checkpoint.pt`):** The most recent state for crash recovery.
- **Top-K Best Checkpoints (`ckpt_top_1.pt`, `ckpt_top_2.pt`, ...):** A ranked priority queue of the $K$ best checkpoints, automatically pruning older, inferior snapshots from storage.

---

## 5. Python Implementation: Production Checkpointing and Early Stopping

Below is the complete PyTorch implementation of an atomic checkpoint manager and early stopping coordinator:

```python
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler


class EarlyStoppingCoordinator:
    """
    Tracks validation metrics, manages patience counters,
    and signals training loop termination.
    """
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        smoothing_alpha: float = 1.0
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_alpha = smoothing_alpha

        self.best_loss = float("inf")
        self.smoothed_loss: Optional[float] = None
        self.patience_counter = 0
        self.should_stop = False

    def step(self, current_val_loss: float) -> bool:
        """
        Updates state with latest validation loss.
        Returns True if the current score is a new historical best.
        """
        # Apply optional exponential smoothing
        if self.smoothed_loss is None:
            self.smoothed_loss = current_val_loss
        else:
            self.smoothed_loss = (
                self.smoothing_alpha * current_val_loss
                + (1.0 - self.smoothing_alpha) * self.smoothed_loss
            )

        eval_metric = self.smoothed_loss

        # Check for meaningful improvement
        if eval_metric < (self.best_loss - self.min_delta):
            self.best_loss = eval_metric
            self.patience_counter = 0
            is_best = True
        else:
            self.patience_counter += 1
            is_best = False

        if self.patience_counter >= self.patience:
            self.should_stop = True

        return is_best


class CheckpointManager:
    """
    Manages atomic serialization, full state snapshotting,
    and rolling top-K checkpoint retention.
    """
    def __init__(
        self,
        save_dir: str,
        max_to_keep: int = 3
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep

        # Track top-K checkpoints as a list of tuples: (val_loss, file_path)
        self.top_checkpoints: List[tuple[float, Path]] = []

    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        val_loss: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[GradScaler] = None,
        config: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> Path:
        """
        Atomically saves complete training state to disk.
        """
        checkpoint_filename = f"checkpoint_step_{step:07d}_loss_{val_loss:.4f}.pt"
        final_path = self.save_dir / checkpoint_filename
        temp_path = self.save_dir / f"{checkpoint_filename}.tmp"

        # 1. Capture exact RNG states
        rng_states = {
            "python_rng": random.getstate(),
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }

        # 2. Extract unwrapped model state dict if wrapped in DDP/Compiled
        raw_model = getattr(model, "module", model)
        raw_model = getattr(raw_model, "_orig_mod", raw_model)

        # 3. Assemble complete state dictionary
        state = {
            "step": step,
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "rng_states": rng_states,
            "config": config or {}
        }

        # 4. Atomic Write: Save to temp file first
        torch.save(state, temp_path)
        with open(temp_path, "a+") as f:
            os.fsync(f.fileno())  # Ensure bytes hit physical non-volatile storage

        # Atomic Rename
        os.replace(temp_path, final_path)

        # 5. Manage Best Checkpoint Symlink/Copy
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            shutil.copyfile(final_path, best_path)

        # 6. Manage Latest Checkpoint Copy
        latest_path = self.save_dir / "latest_checkpoint.pt"
        shutil.copyfile(final_path, latest_path)

        # 7. Prune older snapshots exceeding max_to_keep
        self._prune_checkpoints(val_loss, final_path)

        return final_path

    def _prune_checkpoints(self, val_loss: float, path: Path):
        """Keeps only the top-K checkpoints with lowest validation loss."""
        self.top_checkpoints.append((val_loss, path))
        # Sort ascending by validation loss (lowest loss is best)
        self.top_checkpoints.sort(key=lambda x: x[0])

        if len(self.top_checkpoints) > self.max_to_keep:
            worst_loss, worst_path = self.top_checkpoints.pop()
            if worst_path.exists() and worst_path != path:
                worst_path.unlink()

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[GradScaler] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Loads state dictionary and restores all optimizer, scheduler, and RNG states.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Restore model weights
        raw_model = getattr(model, "module", model)
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        raw_model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer states
        if optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler states
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore AMP scaler
        if scaler and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore RNG states
        rng = checkpoint.get("rng_states")
        if rng:
            random.setstate(rng["python_rng"])
            torch.set_rng_state(rng["torch_cpu_rng"])
            if torch.cuda.is_available() and rng["torch_cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(rng["torch_cuda_rng"])

        return {
            "step": checkpoint.get("step", 0),
            "epoch": checkpoint.get("epoch", 0),
            "val_loss": checkpoint.get("val_loss", float("inf")),
            "config": checkpoint.get("config", {})
        }

```

---

## 6. Pre-Flight Checkpoint Verification and Recovery Invariants

Before training resumes from an existing checkpoint, the recovery pipeline must verify the integrity of the saved state:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ CHECKPOINT RECOVERY VERIFICATION CHECKLIST                             │
├──────────────────────────────┬─────────────────────────────────────────┤
│ 1. Parameter Key Match       │ Assert: len(set(model.keys())           │
│                              │       ^ set(ckpt.keys())) == 0          │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 2. Device Mapping            │ Ensure tensors map cleanly to target    │
│                              │ GPU IDs (avoid GPU 0 memory bottlenecks)│
├──────────────────────────────┼─────────────────────────────────────────┤
│ 3. Optimizer State Alignment │ Verify optimizer momentum tensors match │
│                              │ parameter shapes exactly.               │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 4. Scheduler Synchrony       │ Verify scheduler step count equals      │
│                              │ checkpoint global step counter.         │
└──────────────────────────────┴─────────────────────────────────────────┘

```

### Common Checkpoint Recovery Pitfalls

- **Resuming Without Optimizer Moments:** Loading only `model.load_state_dict()` and initializing a fresh optimizer resets $m_t = 0$ and $v_t = 0$. At step $t_{\text{resume}}$, the fresh optimizer applies large, uncalibrated updates that destabilize the pre-trained weights.
- **GPU Memory Spikes on Load:** Calling `torch.load("checkpoint.pt")` without `map_location="cpu"` defaults to loading all tensors onto GPU 0 first before redistributing, triggering an unexpected CUDA Out-of-Memory exception. Always load to CPU first, then transfer parameters to device ranks.
