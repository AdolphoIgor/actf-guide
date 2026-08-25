# Telemetry Streaming, Real-Time Observability, and Automated Training Circuit Breakers

## 1. Real-Time Observability in Distributed Continuous Training

In large-scale continuous pre-training and automated Supervised Fine-Tuning (SFT), training jobs run unattended across hundreds of distributed GPU nodes for days or weeks. Relying on passive logging (such as synchronous stdout prints or deferred log parsing) introduces two major operational failure modes:

```text
1. Host-Device Synchronization Bottlenecks:
   Calling `tensor.item()` or `print()` synchronously inside the inner training loop forces
   the CPU to block until the GPU pipeline flushes all pending CUDA operations, destroying
   kernel pipelining and reducing Model FLOPs Utilization (MFU) from >50% to <20%.

2. Runaway Spend on Diverging Optimizations:
   When an optimization trajectory destabilizes (due to corrupt data batches, exploding loss cliffs,
   or FP16 gradient underflow), a job left unchecked can consume millions of dollars in compute
   generating completely corrupted NaN/Inf parameter matrices.

```

To achieve high hardware efficiency and operational safety, distributed training pipelines decouple metric collection into an **asynchronous, non-blocking telemetry streaming engine** integrated with **automated training circuit breakers**.

```text
Training Loop (CUDA Execution Graph)
          │
          ├──► Fast Tensor Operations (Forward, Backward, Optimizer Step)
          │
          └──► Non-Blocking Metric Ingestion (Pushed to Thread-Safe Ring Buffer)
                     │
                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ ASYNCHRONOUS TELEMETRY ENGINE (Background Worker Thread)               │
│  • Reads metric events without blocking GPU compute.                   │
│  • Computes rolling statistical distributions (EWMA, variance, z-score)│
│  • Streams time-series telemetry to external sinks (OTLP, Prometheus). │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ AUTOMATED CIRCUIT BREAKER STATE MACHINE                                │
│  • Evaluates trip conditions against real-time anomaly rules.          │
│  • State: CLOSED (Nominal) ──► OPEN (Tripped) ──► HALF-OPEN (Recovering)│
│  • Executes automated mitigation: Rollback, LR Decay, or Emergency Exit│
└────────────────────────────────────────────────────────────────────────┘

```

---

## 2. Core Telemetry Signals and Metric Extraction

A production observability harness captures telemetry vectors across four operational layers at every global step $t$:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ TELEMETRY EXTRACTION TAXONOMY                                          │
├──────────────────────────┬───────────────────┬─────────────────────────┤
│ Metric Category          │ Monitored Signals │ Anomaly Indication      │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 1. Optimization Dynamics │ • Cross-Entropy   │ Sudden loss spikes,     │
│                          │ • Target PPL      │ plateau stagnation,     │
│                          │ • Active Token %  │ over-masking anomalies  │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 2. Gradient Mechanics    │ • Global L2 Norm  │ Gradient explosion,     │
│                          │ • Layer-Wise Norms│ dead layers, vanishing  │
│                          │ • Grad Sparsity % │ gradients in bottom blocks│
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 3. State & Representation│ • Weight Norms    │ Weight explosion,       │
│                          │ • AdamW v_t RMS   │ variance buffer poison, │
│                          │ • Activation RMS  │ logit scale drift       │
├──────────────────────────┼───────────────────┼─────────────────────────┤
│ 4. Hardware & Cluster    │ • Step Latency    │ Straggler GPU nodes,    │
│                          │ • Allocated VRAM  │ NCCL deadlocks, memory  │
│                          │ • MFU / TFLOPS    │ fragmentation / leaks   │
└──────────────────────────┴───────────────────┴─────────────────────────┘

```

### A. Mathematical Metric Formulations

#### 1. Global Gradient Norm ($L_2$)

Tracks the total magnitude of the parameter gradient vector across all $K$ trainable parameter tensors:

$$\Vert{}g_t\Vert{}_2 = \sqrt{\sum_{k=1}^K \sum_{i} (g_{k, i})^2}$$

#### 2. Parameter Weight Norm

Monitors the drift of model weights $\theta$:

$$\Vert{}\theta_t\Vert{}_2 = \sqrt{\sum_{k=1}^K \Vert{}\theta_k\Vert{}_2^2}$$

A sudden drop in $\Vert{}\theta_t\Vert{}_2$ indicates destructive weight decay or corrupted optimizer state; an uncontrolled exponential increase signals unregularized parameter divergence.

#### 3. AdamW Second-Moment Energy

Tracks the root-mean-square of the uncentered gradient variance buffers $v_t$:

$$\text{RMS}(v_t) = \sqrt{\frac{1}{P} \sum_{i=1}^P v_{t, i}}$$

Where $P$ is the total parameter count. If $\text{RMS}(v_t)$ spikes by orders of magnitude, a single outlier batch has corrupted the optimizer's historical memory.

#### 4. Model FLOPs Utilization (MFU)

Measures the ratio of observed training throughput against the theoretical maximum compute capacity of the underlying hardware:

$$\text{MFU} = \frac{\text{FLOPs}_{\text{step}}}{\Delta t_{\text{step}} \times \text{Peak FLOPs}_{\text{GPU}} \times N_{\text{GPUs}}}$$

For a decoder-only Transformer with sequence length $L$, batch size $B$, and parameter count $\Phi$:

$$\text{FLOPs}_{\text{step}} \approx 6 \times \Phi \times B \times L + 12 \times N_{\text{layers}} \times N_{\text{heads}} \times d_{\text{head}} \times B \times L^2$$

---

## 3. Circuit Breaker State Machine & Anomaly Trip Conditions

The training circuit breaker implements a fault-tolerant state machine to protect cluster budgets and model weights from irrecoverable divergence.

```text
Circuit Breaker State Machine Topology:

             ┌──────────────────────────────────────────────────┐
             │                                                  │
             ▼                                                  │
   ┌───────────────────┐      Trip Condition Triggered     ┌───────────────────┐
   │      CLOSED       │──────────────────────────────────►│       OPEN        │
   │ (Normal Execution)│                                   │(Compute Suspended)│
   └───────────────────┘                                   └───────────────────┘
             ▲                                                       │
             │                                                       │ Automated Action:
             │ All Recovery Probes Pass                              │ 1. Load Checkpoint t-k
             │                                                       │ 2. LR Backoff (0.5x)
             │                                                       │ 3. Skip Bad Shard
             │               ┌───────────────────┐                   │
             └───────────────│     HALF-OPEN     │◄──────────────────┘
                             │ (Trial Validation)│
                             └───────────────────┘
                                       │
                                       │ Verification Fails (Persistent Fault)
                                       ▼
                             [ HARD EMERGENCY SHUTDOWN ]
                             • Flush Diagnostic Dump
                             • Cordon Failing Node / Release GPU Pool

```

### The Four Automated Trip Conditions

```text
┌────────────────────────────────────────────────────────────────────────┐
│ CIRCUIT BREAKER TRIP SPECIFICATIONS                                    │
├──────────────────────────────┬─────────────────────────────────────────┤
│ 1. Loss Anomaly Breaker      │ • is_nan(L_t) or is_inf(L_t)            │
│    (Immediate Hard Trip)     │ • L_t > Rolling_Mean(L) + k * Sigma(L)  │
│                              │   (where k = 4.0 over 100-step window)  │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 2. Gradient Explosion /      │ • ||g_t||_2 > 25.0 * Max_Grad_Norm      │
│    Collapse Breaker          │ • ||g_t||_2 < 1e-7 for >= 10 steps      │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 3. Hardware Straggler &      │ • Step_Time > 3.0 * Rolling_Mean(Time)  │
│    Stall Breaker             │ • Step_Time > Hard_Timeout (e.g. 600s)  │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 4. Target Masking Integrity  │ • Active_Token_Count == 0 (100% masked) │
│    Breaker                   │ • Active_Token_Ratio < 0.05             │
└──────────────────────────────┴─────────────────────────────────────────┘

```

#### Trip Condition 1: Dynamic Rolling Z-Score Loss Spikes

Rather than using static loss thresholds, the system tracks an Exponentially Weighted Moving Average (EWMA) and standard deviation of validation/training loss:

$$\mu_t = \alpha \mathcal{L}_t + (1 - \alpha) \mu_{t-1}, \quad \sigma_t^2 = \alpha (\mathcal{L}_t - \mu_t)^2 + (1 - \alpha) \sigma_{t-1}^2$$

$$\text{Z-Score}_t = \frac{\mathcal{L}_t - \mu_{t-1}}{\sigma_{t-1} + \epsilon}$$

$$\text{Condition: } \text{Z-Score}_t > 4.0 \implies \text{Trip Breaker (Loss Spike)}$$

#### Trip Condition 2: Gradient Vanishing / Dead Subnetworks

If backpropagation yields zero or near-zero gradient norms across multiple steps:

$$\sum_{i=t-N}^t \mathbb{I}\left( \Vert{}g_i\Vert{}_2 < 10^{-7} \right) \ge N \quad (N = 10) \implies \text{Trip Breaker (Gradient Starvation)}$$

---

## 4. Non-Blocking Telemetry Streaming Architecture

To avoid CUDA synchronization stalls, metric tensors are detached, cloned to pinned host memory asynchronously, and handed off to a dedicated background worker via lock-free or bounded thread-safe queues.

```text
Main Training Process (GPU Rank 0)                Background Telemetry Thread
──────────────────────────────────                ───────────────────────────
1. Compute Forward & Loss                         
2. Scalar Loss = loss.detach()                    
3. Non-blocking Host Copy:                        
   metric_event = {                               
     "step": step,                                
     "loss": loss_tensor,                         
     "grad_norm": norm_tensor                     
   }                                              
4. Metric_Queue.put_nowait(metric_event) ──────► 1. metric_event = Metric_Queue.get()
5. Continue immediately to Step t+1               2. Convert tensors to CPU floats
   (Zero CUDA synchronization latency!)           3. Update EWMA Statistics
                                                  4. Evaluate Circuit Breaker Rules
                                                  5. If Tripped: Signal Main Process
                                                  6. Flush to Time-Series DB (Prometheus)

```

---

## 5. Python Implementation: Telemetry Engine and Circuit Breaker

Below is the standalone, production-grade telemetry harness and automated circuit breaker engine:

```python
from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn as nn


class BreakerState(str, Enum):
    CLOSED = "closed"        # Normal operational state
    OPEN = "open"            # Tripped; compute suspended
    HALF_OPEN = "half_open"  # Recovery trial in progress


@dataclass
class TelemetryEvent:
    step: int
    loss: float
    grad_norm: float
    active_tokens: int
    total_tokens: int
    step_duration_sec: float
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class AutomatedCircuitBreaker:
    """
    Real-time safety engine evaluating mathematical stability trip conditions.
    """
    def __init__(
        self,
        z_score_threshold: float = 4.0,
        max_grad_norm_multiplier: float = 25.0,
        base_max_grad_norm: float = 1.0,
        max_step_time_multiplier: float = 3.0,
        history_window: int = 100,
        min_active_token_ratio: float = 0.05
    ):
        self.z_thresh = z_score_threshold
        self.max_grad_multiplier = max_grad_norm_multiplier
        self.base_max_grad_norm = base_max_grad_norm
        self.max_time_multiplier = max_step_time_multiplier
        self.min_active_ratio = min_active_token_ratio
        
        self.state = BreakerState.CLOSED
        self.loss_history = deque(maxlen=history_window)
        self.step_time_history = deque(maxlen=history_window)
        self.trip_reason: Optional[str] = None

    def evaluate_step(self, event: TelemetryEvent) -> Tuple[BreakerState, Optional[str]]:
        """
        Evaluates step metrics against all trip conditions.
        Returns the updated state and trip reason if an anomaly is detected.
        """
        # 1. NaN / Inf Hard Trip Check
        if math.isnan(event.loss) or math.isinf(event.loss):
            self._trip(f"Loss arithmetic overflow detected: {event.loss}")
            return self.state, self.trip_reason

        if math.isnan(event.grad_norm) or math.isinf(event.grad_norm):
            self._trip(f"Gradient norm arithmetic overflow detected: {event.grad_norm}")
            return self.state, self.trip_reason

        # 2. Dynamic Loss Spike Z-Score Check
        if len(self.loss_history) >= 20:
            mean_loss = sum(self.loss_history) / len(self.loss_history)
            variance = sum((x - mean_loss) ** 2 for x in self.loss_history) / len(self.loss_history)
            std_loss = math.sqrt(variance) + 1e-6

            z_score = (event.loss - mean_loss) / std_loss
            if z_score > self.z_thresh:
                self._trip(
                    f"Loss spike anomaly: Z-Score = {z_score:.2f} > {self.z_thresh} "
                    f"(Current: {event.loss:.4f}, Window Mean: {mean_loss:.4f})"
                )
                return self.state, self.trip_reason

        # 3. Gradient Explosion Check
        grad_ceiling = self.base_max_grad_norm * self.max_grad_multiplier
        if event.grad_norm > grad_ceiling:
            self._trip(
                f"Gradient explosion: ||g|| = {event.grad_norm:.2f} exceeds ceiling {grad_ceiling:.2f}"
            )
            return self.state, self.trip_reason

        # 4. Target Masking Corruption Check
        active_ratio = event.active_tokens / max(1, event.total_tokens)
        if active_ratio < self.min_active_ratio:
            self._trip(
                f"Target loss over-masking: Active Token Ratio = {active_ratio:.4f} < {self.min_active_ratio}"
            )
            return self.state, self.trip_reason

        # 5. Hardware Straggler / Stall Check
        if len(self.step_time_history) >= 10:
            avg_time = sum(self.step_time_history) / len(self.step_time_history)
            if event.step_duration_sec > (avg_time * self.max_time_multiplier):
                self._trip(
                    f"Cluster latency stall: Step took {event.step_duration_sec:.2f}s "
                    f"(Average: {avg_time:.2f}s)"
                )
                return self.state, self.trip_reason

        # Update historical ring buffers
        self.loss_history.append(event.loss)
        self.step_time_history.append(event.step_duration_sec)
        
        return self.state, None

    def _trip(self, reason: str):
        self.state = BreakerState.OPEN
        self.trip_reason = reason

    def reset_to_half_open(self):
        """Allows testing recovery from snapshot."""
        self.state = BreakerState.HALF_OPEN
        self.trip_reason = None


class TelemetryDispatcher:
    """
    Non-blocking background telemetry processor.
    Collects events from GPU workers, updates stats, and streams to sinks.
    """
    def __init__(self, circuit_breaker: AutomatedCircuitBreaker):
        self.event_queue: queue.Queue = queue.Queue(maxsize=10000)
        self.breaker = circuit_breaker
        self.stop_signal = threading.Event()
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        
        self.latest_state = BreakerState.CLOSED
        self.latest_reason: Optional[str] = None

    def push_metric(
        self,
        step: int,
        loss_val: float,
        grad_norm_val: float,
        active_tokens: int,
        total_tokens: int,
        step_duration: float
    ):
        """
        Fast, non-blocking enqueue called from main training loop.
        """
        event = TelemetryEvent(
            step=step,
            loss=loss_val,
            grad_norm=grad_norm_val,
            active_tokens=active_tokens,
            total_tokens=total_tokens,
            step_duration_sec=step_duration
        )
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass  # Drop telemetry rather than blocking the compute thread

    def _process_queue(self):
        while not self.stop_signal.is_set() or not self.event_queue.empty():
            try:
                event: TelemetryEvent = self.event_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Evaluate Circuit Breaker Rules in background thread
            state, reason = self.breaker.evaluate_step(event)
            self.latest_state = state
            self.latest_reason = reason

            if state == BreakerState.OPEN:
                print(f"\n[CIRCUIT BREAKER TRIGGERED AT STEP {event.step}]")
                print(f"Reason: {reason}\n")

            self.event_queue.task_done()

    def is_tripped(self) -> Tuple[bool, Optional[str]]:
        return (self.latest_state == BreakerState.OPEN), self.latest_reason

    def shutdown(self):
        self.stop_signal.set()
        self.worker_thread.join(timeout=2.0)

```

---

## 6. Integration Lifecycle with Training Loop and Auto-Recovery

Below is the orchestration pattern linking the training loop, the non-blocking telemetry engine, and automated checkpoint rollback remediation:

```python
def run_monitored_training_step(
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_engine: nn.Module,
    telemetry: TelemetryDispatcher,
    x: torch.Tensor,
    y: torch.Tensor,
    rollback_fn: Callable[[int], None]
):
    t_start = time.perf_counter()

    # 1. Pre-Flight Check: Is the Circuit Breaker Tripped?
    tripped, reason = telemetry.is_tripped()
    if tripped:
        print(f"Execution halted by Circuit Breaker: {reason}")
        # Execute automated recovery action
        rollback_fn(step)
        return None

    # 2. Model Forward & Loss
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss, active_tokens = loss_engine(logits, y)

    # 3. Backward Pass
    loss.backward()

    # 4. Global Norm Clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 5. Optimizer Step
    optimizer.step()

    t_end = time.perf_counter()
    duration = t_end - t_start

    # 6. Non-blocking Telemetry Dispatch (Passing scalar floats)
    telemetry.push_metric(
        step=step,
        loss_val=loss.item(),
        grad_norm_val=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        active_tokens=active_tokens,
        total_tokens=y.numel(),
        step_duration=duration
    )

    return loss.item()

```

---

## 7. Production Circuit Breaker Threshold Matrix

| Monitored Channel | Normal Operating Envelope | Warning Horizon (Log Alert) | Trip Condition (Hard Halt) | Automated Remediation Action |
| --- | --- | --- | --- | --- |
| **Arithmetic Loss Integrity** | $\mathcal{L}_t \in [0.5, 12.0]$ | $\Delta \mathcal{L} > +1.0$ in 1 step | `isnan(L)` or `isinf(L)` | Roll back to checkpoint $t-1\text{k}$; reduce learning rate by $50\%$ |
| **Loss Spike Z-Score** | $\vert{}Z_t\vert{} \le 2.0$ | $2.5 \le Z_t < 4.0$ | $Z_t \ge 4.0$ over 100-step window | Abort current batch; discard dataset shard; revert 1 step |
| **Global Gradient Norm** | $\Vert{}g_t\Vert{}_2 \in [0.1, 1.5]$ | $2.0 \le \Vert{}g_t\Vert{}_2 < 10.0$ | $\Vert{}g_t\Vert{}_2 \ge 25.0$ | Discard optimizer update; zero gradients; decay LR |
| **Gradient Starvation** | $\Vert{}g_t\Vert{}_2 \ge 0.01$ | $\Vert{}g_t\Vert{}_2 < 10^{-4}$ for 3 steps | $\Vert{}g_t\Vert{}_2 < 10^{-7}$ for 10 steps | Re-initialize uncalibrated linear heads; check activation scales |
| **Target Loss Masking** | Active $\% \in [20\%, 60\%]$ | Active $\% < 10\%$ | Active $\%=0\%$ (Zero targets) | Quarantine incoming SFT batch; verify collator delimiter masks |
| **Cluster Node Latency** | $\Delta t_{\text{step}} \le 1.2 \times \mu_{\text{time}}$ | $1.5\times \le \Delta t_{\text{step}} < 3.0\times$ | $\Delta t_{\text{step}} \ge 3.0 \times \mu_{\text{time}}$ or $>600\text{s}$ | Terminate job; cordon high-latency GPU node; restart on standby |