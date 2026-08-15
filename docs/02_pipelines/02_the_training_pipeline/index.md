# The Training Pipeline

## Executive Overview
The **Training Pipeline** executes distributed, multi-GPU parameter optimization over model-ready Gold feature shards. It manages compute resource allocation, handles FSDP2/DeepSpeed ZeRO-3 model state partitioning, and logs training metrics asynchronously.

---

## Module Scope
* **Distributed Training Execution:** Axolotl / PyTorch FSDP2 / DeepSpeed wrappers.
* **Experiment Tracking:** Out-of-band streaming of loss curves, gradient norms, and VRAM utilization to Weights & Biases.