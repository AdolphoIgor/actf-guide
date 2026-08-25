# Fault-Tolerant Compute Lifecycle and Resilient Distributed Training

## 1. The Economics and Instability of Large-Scale Compute

In modern foundation model pre-training and continuous fine-tuning pipelines, compute infrastructure is subject to frequent hardware, network, and scheduling disruptions. Operating across large GPU clusters (hundreds or thousands of accelerators) introduces high failure frequencies dictated by cluster **Mean Time Between Failures (MTBF)**:

$$\text{MTBF}_{\text{cluster}} = \frac{\text{MTBF}_{\text{node}}}{N_{\text{nodes}}}$$

```text
Cluster Failure Probabilities across Scaling Regimes:

Cluster Scale       Typical Node MTBF     Effective Cluster MTBF    Expected Disruptions
────────────────────────────────────────────────────────────────────────────────────────
8 GPUs (1 Node)     ~1,000 Days           1,000 Days                Rare (< 1 per year)
64 GPUs (8 Nodes)   ~1,000 Days           125 Days                  Periodic (3 per year)
512 GPUs (64 Nodes) ~1,000 Days           15.6 Days                 Bi-weekly interruptions
4096 GPUs (512 Nodes)~1,000 Days          1.95 Days                 Daily failure events

```

```text
┌────────────────────────────────────────────────────────────────────────┐
│ PRIMARY CAUSES OF DISTRIBUTED COMPUTE INTERRUPTION                     │
├──────────────────────────────┬─────────────────────────────────────────┤
│ Spot / Preemptible Evictions │ Cloud providers reclaim spot instances  │
│                              │ with short notice (30s to 120s warning).│
├──────────────────────────────┼─────────────────────────────────────────┤
│ GPU Hardware Faults          │ Uncorrectable ECC memory errors, NVLink │
│                              │ connection drops, thermal throttling.   │
├──────────────────────────────┼─────────────────────────────────────────┤
│ Inter-Node Network Timeouts  │ InfiniBand packet drops, RoCE congestion│
│                              │ collapses, silent NCCL collective hangs.│
├──────────────────────────────┼─────────────────────────────────────────┤
│ Node Eviction & Maintenance  │ Slurm job time-limit expirations,       │
│                              │ Kubernetes node drains, host kernel OOMs│
└──────────────────────────────┴─────────────────────────────────────────┘

```

Relying on uninterrupted runtime execution is mathematically non-viable at scale. A robust distributed training framework must treat compute resources as **ephemeral**, orchestrating a fault-tolerant lifecycle capable of intercepting termination signals, persisting atomic recovery states within short preemption windows, and resuming without loss of gradient momentum or dataset synchronization.

---

## 2. Signal Handling and Graceful Preemption Workflows

When an orchestrator (such as Kubernetes, AWS Batch, or Slurm) schedules an instance eviction or hits a job allocation timeout, it issues an operating system signal prior to sending an uncatchable `SIGKILL`.

```text
Orchestrator Preemption Timeline:

t = 0s                        t = 1s                            t = 25s                     t = 30s
──┬─────────────────────────────┬─────────────────────────────────┬───────────────────────────┬──►
  │                             │                                 │                           │
  ▼                             ▼                                 ▼                           ▼
[ Preemption Signal Sent ]   [ OS Signal Intercepted ]         [ Atomic Checkpoint Saved ]  [ SIGKILL ]
  • Slurm: SIGUSR1             • Complete current micro-step     • Weights + Opt Moments    Hard termination
  • K8s / AWS: SIGTERM         • Halt forward execution          • Flush to disk via fsync  by hypervisor
                               • Sync barriers across ranks      • Update 'latest.pt' symlink

```

### The Signal Interception Contract

1. **Signal Interception:** Register handlers for `SIGTERM`, `SIGINT`, and `SIGUSR1` immediately upon training process initialization.
2. **Current Step Finalization:** Do not abort the process mid-backward pass. Allow the currently running micro-step or accumulation window to conclude to avoid writing inconsistent gradient buffers to disk.
3. **Collective State Synchronization:** Issue a non-blocking NCCL barrier check across all distributed ranks to confirm uniform execution boundaries.
4. **Emergency State Serialization:** Write complete model parameters, optimizer moments, scheduler states, and random number generator (RNG) seeds to persistent shared storage.
5. **Clean Exit Code:** Terminate with exit code `0` or `143` (`128 + 15` for `SIGTERM`) to notify the external cluster autoscaler that the job exited gracefully and should be automatically re-queued.

---

## 3. State Reconstruction and Fast-Forward Data Recovery

When a preempted or failed job restarts on a fresh set of hardware nodes, it must reconstruct the exact mathematical state of the optimization graph without re-reading millions of previously processed data records.

```text
Fast-Forwarding vs. Index Tracking:

Naive Approach: Ingest from Step 0 and discard (Wasteful)
  Step 0 ──► (Forward / Discard) ──► Step 10,000 ──► (Forward / Discard) ──► Step 50,000 (1 hour wasted!)

Deterministic Resumption via State Metadata:
  1. Load weights θ and moments (m_t, v_t) from "latest.pt"
  2. Restore RNG states: torch.set_rng_state(), random.setstate()
  3. Seek Dataset Shard Reader:
     • Shard Index = Step * B_effective // Shard_Capacity
     • In-Shard Offset = (Step * B_effective) % Shard_Capacity
  4. Resume immediate training at Step 50,001 with ZERO discarded forward passes.

```

### The Fast-Forward Shard Pointer Formula

Let $t$ be the recovered global training step, $B_{\text{effective}}$ be the effective global batch size (sequences per step), and $S_{\text{capacity}}$ be the number of tokenized sequences per storage shard file.

$$\text{Total Sequences Processed} = t \times B_{\text{effective}}$$

$$\text{Target Shard Index} = \left\lfloor \frac{t \times B_{\text{effective}}}{S_{\text{capacity}}} \right\rfloor$$

$$\text{Intra-Shard Sequence Offset} = (t \times B_{\text{effective}}) \pmod{S_{\text{capacity}}}$$

This index lookup allows the dataset loader to seek directly to the exact target byte position in object storage or network-attached filesystems (NFS/Lustre) within milliseconds of process boot.

---

## 4. Collective Health Monitoring and Deadlock Mitigation

In multi-node distributed training using PyTorch Distributed Data Parallel (DDP) or Fully Sharded Data Parallel (FSDP), the primary failure mode is the **Silent NCCL Deadlock**: a single GPU rank encounters a silent CUDA kernel crash, ECC error, or out-of-memory condition, leaving all other ranks blocked indefinitely at the next `AllReduce` communication collective.

```text
Silent NCCL Deadlock Scenario:

GPU Rank 0: Executing Backward Pass ──► Waiting at AllReduce Barrier ... (BLOCKED INDEFINITELY)
GPU Rank 1: Executing Backward Pass ──► Waiting at AllReduce Barrier ... (BLOCKED INDEFINITELY)
GPU Rank 2: CUDA Kernel Fault / Panic ──► Process Hanging / Terminated
GPU Rank 3: Executing Backward Pass ──► Waiting at AllReduce Barrier ... (BLOCKED INDEFINITELY)

Resolution: NCCL Health Watchdog Daemon terminates cluster after timeout (e.g., 600s),
            triggering cluster scheduler auto-restart and recovery from latest.pt.

```

### Watchdog Configuration Invariants

To prevent runaway compute billing on deadlocked clusters:

* **`NCCL_ASYNC_ERROR_HANDLING=1` (or `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`):** Instructs the PyTorch runtime to hook asynchronous CUDA errors and crash all collective ranks instead of hanging.
* **`TORCH_DISTRIBUTED_DEBUG=INFO`:** Logs detailed communication graph topologies when collective timeouts occur.
* **Custom Step Heartbeat Watchdog:** Tracks wall-clock time between consecutive optimizer steps. If elapsed time exceeds $3\times$ the running average step duration, the watchdog throws an exception and triggers emergency shutdown.

---

## 5. Python Implementation: Production Fault-Tolerant Compute Engine

Below is the standalone implementation of a resilient training harness equipped with OS signal trapping (`SIGTERM`, `SIGUSR1`), emergency state persistence, collective health monitoring, and fast-forward dataset resumption:

```python
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler


class PreemptionHandler:
    """
    Intercepts OS termination signals (SIGTERM, SIGUSR1, SIGINT)
    and coordinates graceful shutdown flags across training ranks.
    """
    def __init__(self):
        self.received_shutdown_signal = False
        self.signal_received_timestamp: Optional[float] = None
        
        # Register signal hooks
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        # SIGUSR1 is standard for Slurm preemption warnings
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, self._handle_signal)

    def _handle_signal(self, signum: int, frame: Any):
        self.received_shutdown_signal = True
        self.signal_received_timestamp = time.time()
        print(
            f"WARNING: Process {os.getpid()} intercepted OS Signal {signum} "
            f"at timestamp {self.signal_received_timestamp}. Initiating graceful exit sequence."
        )

    def should_exit(self) -> bool:
        return self.received_shutdown_signal


class ResilientTrainingEngine:
    """
    Orchestrates fault-tolerant execution with state recovery,
    heartbeat monitoring, and preemption-safe synchronization.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: GradScaler,
        checkpoint_dir: str,
        rank: int = 0,
        world_size: int = 1,
        max_step_timeout_sec: float = 300.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.rank = rank
        self.world_size = world_size
        self.max_step_timeout_sec = max_step_timeout_sec
        self.preemption_handler = PreemptionHandler()
        
        self.last_step_heartbeat = time.time()

    def save_emergency_state(
        self, step: int, epoch: int, val_loss: float, dataset_state: Dict[str, Any]
    ) -> Path:
        """
        Executes immediate atomic checkpoint persistence upon preemption warning.
        """
        temp_path = self.checkpoint_dir / f"emergency_step_{step:07d}.pt.tmp"
        final_path = self.checkpoint_dir / f"emergency_step_{step:07d}.pt"

        raw_model = getattr(self.model, "module", self.model)
        raw_model = getattr(raw_model, "_orig_mod", raw_model)

        state = {
            "step": step,
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "dataset_state": dataset_state,
            "rng_states": {
                "python": random.getstate(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            },
            "timestamp": time.time()
        }

        # Save only on Rank 0 to prevent filesystem write contention
        if self.rank == 0:
            torch.save(state, temp_path)
            with open(temp_path, "a+") as f:
                os.fsync(f.fileno())
            os.replace(temp_path, final_path)
            
            # Update latest symlink
            link_path = self.checkpoint_dir / "latest.pt"
            temp_link = self.checkpoint_dir / "latest.pt.tmp"
            if temp_link.exists() or temp_link.is_symlink():
                temp_link.unlink()
            os.symlink(final_path.name, temp_link)
            os.replace(temp_link, link_path)
            
            print(f"SUCCESS: Emergency checkpoint synchronized and persisted to {final_path}")

        if self.world_size > 1:
            dist.barrier()

        return final_path

    def check_heartbeat(self, current_step: int):
        """
        Detects silent collective hangs and GPU kernel stalls.
        """
        now = time.time()
        duration = now - self.last_step_heartbeat
        if duration > self.max_step_timeout_sec:
            raise TimeoutError(
                f"DEADLOCK DETECTED on Rank {self.rank}: Step {current_step} took "
                f"{duration:.2f}s (Threshold: {self.max_step_timeout_sec}s). Terminating process."
            )
        self.last_step_heartbeat = now

    def execute_resilient_loop(
        self,
        start_step: int,
        total_steps: int,
        data_iterator: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        loss_fn: nn.Module
    ):
        """
        Main execution loop wrapped in preemption and health checks.
        """
        self.model.train()
        
        for step in range(start_step, total_steps):
            # 1. Evaluate Heartbeat Watchdog
            self.check_heartbeat(step)

            # 2. Check for Preemption Signal
            if self.preemption_handler.should_exit():
                print(f"Rank {self.rank}: Halting loop at step boundary {step} for graceful exit.")
                self.save_emergency_state(
                    step=step,
                    epoch=0,
                    val_loss=0.0,
                    dataset_state={"consumed_steps": step}
                )
                # Exit cleanly to prompt cluster re-queue
                sys.exit(0)

            # 3. Training Step Execution
            x, y = next(data_iterator)
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                logits = self.model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # 4. Periodic Snapshotting (Every 1000 steps)
            if step > 0 and step % 1000 == 0:
                self.save_emergency_state(
                    step=step,
                    epoch=0,
                    val_loss=loss.item(),
                    dataset_state={"consumed_steps": step}
                )

```

---

## 6. Failure Recovery and Diagnostic Protocols

```text
┌────────────────────────────────────────────────────────────────────────┐
│ INCIDENT RESPONSE & AUTOMATED RECOVERY PROTOCOLS                       │
├──────────────────────────┬───────────────────────┬─────────────────────┤
│ Disruption Event         │ Detection Mechanism   │ Automated Action    │
├──────────────────────────┼───────────────────────┼─────────────────────┤
│ Spot Node Preemption     │ SIGTERM / SIGUSR1     │ Save emergency state│
│                          │ signal trap           │ and exit code 0.    │
├──────────────────────────┼───────────────────────┼─────────────────────┤
│ CUDA Out of Memory (OOM) │ RuntimeError: CUDA    │ Empty cache, down-  │
│                          │ out of memory caught  │ scale micro-batch.  │
├──────────────────────────┼───────────────────────┼─────────────────────┤
│ NCCL Collective Deadlock │ Step heartbeat watchdog│ Terminate hanging  │
│                          │ timeout (> 300s)      │ job; restart node.  │
├──────────────────────────┼───────────────────────┼─────────────────────┤
│ Unrecoverable ECC Error  │ Host kernel dmesg /   │ Cordon failed node; │
│                          │ NVML health check     │ migrate to standby. │
├──────────────────────────┼───────────────────────┼─────────────────────┤
│ Loss Explosion (NaN/Inf) │ Loss tensor sanity    │ Revert to step t-1k │
│                          │ assertion failure     │ with reduced LR.    │
└──────────────────────────┴───────────────────────┴─────────────────────┘

```

### The Post-Crash Recovery Sequence

When the automated infrastructure orchestrator detects process exit:

1. **Node Validation Check:** Inspect `nvidia-smi --query-gpu=pstate,memory.used,ecc.errors.uncorrected.aggregate.total` on all assigned nodes to quarantine unhealthy hosts.
2. **Lineage Pointer Resolution:** Query `checkpoints/lineage.json` and verify that `checkpoints/latest.pt` is a valid, readable symbolic link.
3. **Hardware Cluster Re-allocation:** Re-bind the exact target GPU topology ($N_{\text{nodes}} \times N_{\text{gpus}}$).
4. **State Deserialization:** Ingest `latest.pt`, populate optimizer momentum tensors, seek the dataset cursor forward by $t_{\text{recovered}} \times B_{\text{effective}}$, and resume training without loss of gradient trajectory.