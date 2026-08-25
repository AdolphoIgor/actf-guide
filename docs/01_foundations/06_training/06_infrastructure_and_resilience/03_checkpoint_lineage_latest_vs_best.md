# Checkpoint Lineage: Latest vs. Best State Management

## 1. Dual-Track Checkpoint Architecture

In continuous training and long-running distributed Supervised Fine-Tuning (SFT), checkpoint serialization serves two conflicting operational objectives:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 1. FAULT-TOLERANT EXECUTION RECOVERY (The "Latest" Track)              │
│    • Objective: Zero loss of compute time upon node preemption.        │
│    • Target: The exact, uncorrupted state of step t.                   │
│    • Payload: Heavy (Weights + Optimizer Moments + Schedulers + RNG).  │
│    • Retention: Short-lived rolling FIFO queue (last 1-2 steps).       │
├────────────────────────────────────────────────────────────────────────┤
│ 2. MODEL GENERALIZATION & AUDIT (The "Best" Track)                     │
│    • Objective: Optimal validation metric for production promotion.    │
│    • Target: Step t* where Validation Loss (or Eval Score) was minimal.│
│    • Payload: Lightweight/Inference (Weights + Config + Provenance).   │
│    • Retention: Persistent / Long-term Immutable Registry.             │
└────────────────────────────────────────────────────────────────────────┘

```

```text
Training Trajectory across Time:

  Loss
   ▲
   │  \
   │   \                                    Step t* (Global Best)
   │    \       Validation Metric (L_val)            │
   │     \     \                               _ - -▼- - _
   │      \     \                          _ -             - _  Validation Overfitting
   │       \     \                     _ -                     - _  (Divergence)
   │        \     \                _ -                            \
   │         \     \_ _ _ _ _ _ -                                  \      Step t_crash
   │          \                                                     \          │
   └───────────┴─────────────────────────────────────────────────────┴─────────▼─────► Steps
               t=0                 t* (Saved as best_model.pt)       t_curr (latest.pt)

Actions on Failure:
  • Resume Training: Restore from "latest.pt" (Step t_curr) to retain optimizer moments.
  • Deploy / Audit:  Promote "best_model.pt" (Step t*) to Gatekeeper evaluation.

```

Conflating these two objectives causes operational failure:

* **Promoting `latest` to production:** Exposes users to overfitted, degraded models that trained past optimal generalization.
* **Resuming optimization from `best`:** Resets the training step backward in time, causing data duplication, learning rate schedule discontinuities, and corrupted momentum dynamics.

---

## 2. State Payloads and Recovery Semantics

The internal schema of a checkpoint payload differs based on its target track:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ STATE PAYLOAD COMPARISON MATRIX                                        │
├──────────────────────────────┬───────────────────┬─────────────────────┤
│ Component                    │ Latest (Recovery) │ Best (Promotion)    │
├──────────────────────────────┼───────────────────┼─────────────────────┤
│ Model Parameter Weights (θ)  │ Yes (FP32 / BF16) │ Yes (FP16 / BF16)   │
├──────────────────────────────┼───────────────────┼─────────────────────┤
│ AdamW 1st & 2nd Moments (m,v)│ Yes (Mandatory)   │ Stripped (Optional) │
├──────────────────────────────┼───────────────────┼─────────────────────┤
│ LR Scheduler State Dict      │ Yes (Step sync)   │ Stripped            │
├──────────────────────────────┼───────────────────┼─────────────────────┤
│ AMP GradScaler Scale Factor  │ Yes (FP16 only)   │ Stripped            │
├──────────────────────────────┼───────────────────┼─────────────────────┤
│ CPU & CUDA RNG Tensors       │ Yes (Exact replay)│ Stripped            │
├──────────────────────────────┼───────────────────┼─────────────────────┤
│ Dataset Shard & Batch Index  │ Yes (No repeats)  │ Stripped            │
├──────────────────────────────┼───────────────────┼─────────────────────┤
│ Evaluation Benchmark Scores  │ Optional          │ Yes (Audit Trail)   │
├──────────────────────────────┼───────────────────┼─────────────────────┤
│ Cryptographic Provenance BOM │ Yes               │ Yes (Mandatory)     │
└──────────────────────────────┴───────────────────┴─────────────────────┘

```

### Recovery Semantics

When an automated orchestrator (e.g., Slurm, Kubernetes Job, Ray Train) restarts a preempted container:

1. **Identify `latest_checkpoint.pt`:** Inspect the lineage metadata file (`lineage.json`) to resolve the latest valid step.
2. **Re-instantiate Computation Graph:** Load model parameters into GPU VRAM.
3. **Restore Optimizer State:** Populate AdamW parameter momentum buffers ($m_t, v_t$).
4. **Advance Scheduler & RNG:** Align learning rate scalar $\eta_t$ and random seeds to step $t$.
5. **Fast-Forward DataLoader:** Skip exactly $t \times B_{\text{effective}}$ records in the dataset shard iterator to prevent data repetition.

---

## 3. Atomic Pointer and Symlink Architecture

Writing multi-gigabyte checkpoints directly to fixed filenames (`latest.pt` or `best.pt`) risks creating race conditions or corrupting storage if a process dies mid-write.

On POSIX storage systems, dual-track management uses **Atomic Symlink Swapping**:

```text
Storage Directory Structure:
  checkpoints/
  ├── ckpt_step_00045000_loss_1.8210.pt     <-- Immutable physical file
  ├── ckpt_step_00050000_loss_1.7942.pt     <-- Immutable physical file (Global Best)
  ├── ckpt_step_00055000_loss_1.8105.pt     <-- Immutable physical file (Current Latest)
  │
  ├── latest.pt  ──(Atomic Symlink)──► ckpt_step_00055000_loss_1.8105.pt
  ├── best.pt    ──(Atomic Symlink)──► ckpt_step_00050000_loss_1.7942.pt
  └── lineage.json                          <-- Master tracking manifest

```

```text
Atomic Pointer Update Sequence:
  1. Write checkpoint to temporary file: "ckpt_step_00060000.pt.tmp"
  2. Flush OS buffers to physical disk:   os.fsync()
  3. Rename temporary to canonical:      os.replace("...tmp", "ckpt_step_00060000.pt")
  4. Create temporary symlink:           os.symlink("ckpt_step_00060000.pt", "latest.pt.tmp")
  5. Atomically swap symlink pointer:    os.replace("latest.pt.tmp", "latest.pt")

```

Because `os.replace` is an atomic system call on POSIX filesystems, external monitoring processes reading `latest.pt` will always receive either the complete prior checkpoint or the complete new checkpoint, never a corrupted intermediate state.

---

## 4. The Checkpoint Lineage Directed Acyclic Graph (DAG)

In continuous training, pipelines do not run in isolation. Models undergo multiple branching runs, SFT updates, and architecture adjustments. A comprehensive tracking system structures checkpoints as nodes in a **Lineage DAG**:

```text
Lineage Graph Representation:

  [ Foundation Pre-Train Checkpoint ] (Step 500k, Hash: a1b2c3)
                  │
                  ▼
  [ Continuous Domain Adaptation ] (Step 550k, Hash: d4e5f6)
                  │
         ┌────────┴────────────────────────┐
         ▼                                 ▼
  [ SFT Run A: General Chat ]       [ SFT Run B: Code Specialization ]
  (Step 600k, Hash: 7g8h9i)         (Step 600k, Hash: 0j1k2l)
         │                                 │
         ▼                                 ▼
   best_model.pt                     best_model.pt
  (Parent: d4e5f6)                  (Parent: d4e5f6)

```

### The Lineage Manifest Schema (`lineage.json`)

The lineage manifest tracks parent-child relationships, training durations, metrics, and physical file locations:

```json
{
  "project_id": "minigpt-continuous-v2",
  "active_latest_pointer": "ckpt_step_00055000_loss_1.8105.pt",
  "active_best_pointer": "ckpt_step_00050000_loss_1.7942.pt",
  "best_validation_metric": 1.7942,
  "best_step": 50000,
  "history": [
    {
      "step": 45000,
      "epoch": 2,
      "val_loss": 1.8210,
      "parent_checkpoint_hash": "a1b2c3d4e5f6",
      "artifact_path": "ckpt_step_00045000_loss_1.8210.pt",
      "timestamp": "2026-08-23T01:00:00Z"
    },
    {
      "step": 50000,
      "epoch": 3,
      "val_loss": 1.7942,
      "parent_checkpoint_hash": "b2c3d4e5f6a1",
      "artifact_path": "ckpt_step_00050000_loss_1.7942.pt",
      "timestamp": "2026-08-23T01:30:00Z"
    },
    {
      "step": 55000,
      "epoch": 3,
      "val_loss": 1.8105,
      "parent_checkpoint_hash": "c3d4e5f6a1b2",
      "artifact_path": "ckpt_step_00055000_loss_1.8105.pt",
      "timestamp": "2026-08-23T02:00:00Z"
    }
  ]
}

```

---

## 5. Python Implementation: Dual-Track Lineage & Checkpoint Engine

Below is the standalone Python implementation of an atomic, dual-track checkpoint manager that maintains rolling recovery state, tracks the global best model, and logs lineage DAG transitions:

```python
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler


class DualTrackCheckpointEngine:
    """
    Manages dual-track serialization:
      1. Latest Track: Full state (Model + Optimizer + Sched + RNG) for crash recovery.
      2. Best Track: Minimum validation loss checkpoint for production promotion.
      3. Rolling Top-K & Atomic Pointer Management.
    """
    def __init__(
        self,
        checkpoint_dir: str,
        max_latest_to_keep: int = 2,
        max_best_to_keep: int = 3
    ):
        self.dir = Path(checkpoint_dir).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        
        self.max_latest = max_latest_to_keep
        self.max_best = max_best_to_keep
        self.manifest_path = self.dir / "lineage.json"

        # Initialize or load lineage manifest
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "active_latest_pointer": None,
            "active_best_pointer": None,
            "best_validation_loss": float("inf"),
            "best_step": 0,
            "latest_snapshots": [],
            "best_snapshots": []
        }

    def _save_manifest(self):
        temp_manifest = self.manifest_path.with_suffix(".tmp")
        with open(temp_manifest, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)
        os.replace(temp_manifest, self.manifest_path)

    def _update_symlink(self, target_filename: str, link_name: str):
        """Atomically updates a symbolic link on POSIX filesystems."""
        link_path = self.dir / link_name
        temp_link_path = self.dir / f"{link_name}.tmp"

        if temp_link_path.exists() or temp_link_path.is_symlink():
            temp_link_path.unlink()

        # Create relative symlink
        os.symlink(target_filename, temp_link_path)
        os.replace(temp_link_path, link_path)

    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        val_loss: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[GradScaler] = None,
        parent_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Path, bool]:
        """
        Saves step state, manages Latest FIFO queue, and updates Best if val_loss improves.
        """
        filename = f"ckpt_step_{step:07d}_loss_{val_loss:.4f}.pt"
        final_path = self.dir / filename
        temp_path = self.dir / f"{filename}.tmp"

        # 1. Capture Full Recovery State Payload
        raw_model = getattr(model, "module", model)
        raw_model = getattr(raw_model, "_orig_mod", raw_model)

        rng_states = {
            "python_rng": random.getstate(),
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }

        payload = {
            "step": step,
            "epoch": epoch,
            "val_loss": val_loss,
            "parent_checkpoint_hash": parent_hash,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "rng_states": rng_states,
            "metadata": metadata or {}
        }

        # 2. Atomic Physical Write
        torch.save(payload, temp_path)
        with open(temp_path, "a+") as f:
            os.fsync(f.fileno())
        os.replace(temp_path, final_path)

        # 3. Update Latest Track
        self.manifest["active_latest_pointer"] = filename
        self.manifest["latest_snapshots"].append({"step": step, "path": filename})
        self._update_symlink(filename, "latest.pt")
        self._prune_latest_snapshots()

        # 4. Check and Update Best Track
        is_new_best = val_loss < self.manifest["best_validation_loss"]
        if is_new_best:
            self.manifest["best_validation_loss"] = val_loss
            self.manifest["best_step"] = step
            self.manifest["active_best_pointer"] = filename
            self.manifest["best_snapshots"].append({"step": step, "loss": val_loss, "path": filename})
            
            self._update_symlink(filename, "best.pt")
            
            # Export stripped inference-only model artifact
            inference_path = self.dir / "best_model_inference.pt"
            torch.save(
                {
                    "step": step,
                    "val_loss": val_loss,
                    "model_state_dict": raw_model.state_dict(),
                    "metadata": metadata or {}
                },
                inference_path
            )
            self._prune_best_snapshots()

        self._save_manifest()
        return final_path, is_new_best

    def _prune_latest_snapshots(self):
        """Removes older recovery snapshots beyond max_latest limit."""
        while len(self.manifest["latest_snapshots"]) > self.max_latest:
            oldest = self.manifest["latest_snapshots"].pop(0)
            p = self.dir / oldest["path"]
            # Do not delete if the file is currently registered as active Best
            if p.exists() and oldest["path"] != self.manifest["active_best_pointer"]:
                p.unlink()

    def _prune_best_snapshots(self):
        """Retains only top-K historical best checkpoints."""
        if len(self.manifest["best_snapshots"]) > self.max_best:
            # Sort ascending by loss
            self.manifest["best_snapshots"].sort(key=lambda x: x["loss"])
            pruned = self.manifest["best_snapshots"].pop() # Remove worst
            p = self.dir / pruned["path"]
            # Do not delete if the file is currently registered as active Latest
            if p.exists() and pruned["path"] != self.manifest["active_latest_pointer"]:
                p.unlink()

    def load_recovery_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[GradScaler] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Restores training state from latest.pt.
        """
        latest_link = self.dir / "latest.pt"
        if not latest_link.exists():
            raise FileNotFoundError(f"No recovery checkpoint found at {latest_link}")

        checkpoint = torch.load(latest_link, map_location=device)

        # Restore Model
        raw_model = getattr(model, "module", model)
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        raw_model.load_state_dict(checkpoint["model_state_dict"])

        # Restore Optimizer
        if optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore Scheduler
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore Scaler
        if scaler and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore RNG
        rng = checkpoint.get("rng_states")
        if rng:
            random.setstate(rng["python_rng"])
            torch.set_rng_state(rng["torch_cpu_rng"])
            if torch.cuda.is_available() and rng["torch_cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(rng["torch_cuda_rng"])

        print(f"Restored recovery state from step {checkpoint['step']} (Val Loss: {checkpoint['val_loss']:.4f})")
        return {
            "step": checkpoint["step"],
            "epoch": checkpoint["epoch"],
            "val_loss": checkpoint["val_loss"],
            "metadata": checkpoint.get("metadata", {})
        }

```

---

## 6. Pre-Flight Verification & Release Invariants

Before triggering downstream pipeline stages, the orchestrator applies hard assertion checks:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ CHECKPOINT LINEAGE INVARIANTS                                          │
├──────────────────────────────┬─────────────────────────────────────────┤
│ Invariant 1: Pointer Symlink │ Assert: os.path.islink("latest.pt")     │
│ Resolvability                │ Assert: os.path.exists(readlink("...")) │
├──────────────────────────────┼─────────────────────────────────────────┤
│ Invariant 2: Metric Monotony │ Best validation score must strictly     │
│                              │ evaluate to min(Historical Eval Losses).│
├──────────────────────────────┼─────────────────────────────────────────┤
│ Invariant 3: Clean State     │ Ingested recovery checkpoint must match │
│ Disjointness                 │ parameter names 100% (zero missing keys)│
├──────────────────────────────┼─────────────────────────────────────────┤
│ Invariant 4: Inference Pure  │ Best inference artifact must NOT retain │
│ Artifact Sizing              │ AdamW states (File size <= 50% of full).│
└──────────────────────────────┴─────────────────────────────────────────┘

```

| Operation Target | Primary File Source | State Requirement | Downstream Destination |
| --- | --- | --- | --- |
| **Crash / Preemption Recovery** | `latest.pt` | Full state (Model, Moments, Sched, RNG) | Training Loop continuation |
| **Gate 5 Offline Audit** | `best.pt` | Model parameters + Tokenizer + Metadata | Gate 5 Verification Runner |
| **Production Serving Fleet** | `best_model_inference.pt` | Weights-only (Inference optimized) | Model Registry / vLLM Cluster |
| **Governance & Lineage Audit** | `lineage.json` | Hash chain + Training BOM metadata | Central Compliance Archive |