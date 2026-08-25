# Cryptographic Provenance, Git SHA Lineage, and Artifact Auditing

## 1. The Lineage Crisis in Continuous Training

In continuous pre-training and automated supervised fine-tuning (SFT), a serialized model checkpoint (`.pt` or `.safetensors`) represents an opaque binary matrix.

If a model demonstrates a sudden performance regression, hallucinates dangerous tokens, or violates regulatory data governance, auditing the failure requires reconstructing the exact conditions that generated those weights.

```text
The Opaque Checkpoint Failure Mode:
  "checkpoint_step_50000.pt" ──► Who trained it? What code branch?
                                 Which dataset version? Were there uncommitted local changes?
                                 What were the exact random seeds and hyperparameter values?

The Cryptographically Auditable Artifact:
┌────────────────────────────────────────────────────────────────────────┐
│ MODEL CHECKPOINT ARTIFACT                                              │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │ Tensors: Model Parameters (θ), Optimizer Moments (m_t, v_t)        │ │
│ ├────────────────────────────────────────────────────────────────────┤ │
│ │ Cryptographic Lineage Envelope (Metadata BOM):                     │ │
│ │  • Code Lineage:    Git Commit SHA + Clean Tree / Unified Diff     │ │
│ │  • Data Lineage:    Dataset Merkle Root / SHA-256 Manifest Hash    │ │
│ │  • Env Lineage:     PyTorch/CUDA Toolchain, Host Hardware Digest   │ │
│ │  • Execution State: Global Seed, Training Configuration, Timestamp │ │
│ └────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘

```

**Cryptographic Provenance** binds model weights directly to their originating source code revision, dataset snapshots, and execution environment. Without deterministic provenance tracking, training artifacts cannot be reproduced, audited for compliance, or promoted through Gatekeeper deployment pipelines.

---

## 2. The Cryptographic Lineage Triad

Complete artifact reproducibility depends on three immutable lineage vectors:

```text
                          ┌───────────────────────────┐
                          │   CRYPTOGRAPHIC TRIAD     │
                          └─────────────┬─────────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼                              ▼
┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
│ 1. CODE STATE   │            │ 2. DATA STATE   │            │ 3. RUNTIME ENV  │
│ • Git Commit SHA│            │ • Merkle Hash   │            │ • Lockfile Hash │
│ • Tree Status   │            │ • Shard Digests │            │ • CUDA/PyTorch  │
│ • Git Diff Patch│            │ • Tokenizer SHA │            │ • RNG Seeds     │
└─────────────────┘            └─────────────────┘            └─────────────────┘

```

### 1. Code Provenance (Git SHA + Tree State)

* **Git Commit SHA:** The 40-character hexadecimal SHA-1/SHA-256 hash identifying the exact repository commit.
* **Dirty Working Tree Policy:** Running training on uncommitted local modifications destroys reproducibility. A production orchestrator must either:
1. Enforce a clean working tree assertion (`git diff-index --quiet HEAD`).
2. Generate and serialize a unified git patch (`git diff HEAD`) directly into the checkpoint metadata envelope.



### 2. Data Provenance (Dataset Merkle Hash)

* **Dataset Manifest Digest:** A cryptographic hash computed over the exact sequence of training shards.
* **Tokenizer Fingerprint:** SHA-256 hash of the tokenizer vocabulary and merge configuration (`vocab.json`, `merges.txt`, or `tokenizer.json`).

### 3. Runtime & Hardware Environment

* **Dependency Manifest:** Hash of the locked dependency graph (`uv.lock`, `poetry.lock`, or pinned `requirements.txt`).
* **Hardware & Toolchain Metadata:** CUDA runtime version, cuDNN version, NVIDIA driver revision, GPU microarchitecture (e.g., Hopper H100 SXM5), and PyTorch build commit.
* **Deterministic Seeds:** Base global pseudorandom number generator (PRNG) seeds for Python `random`, `numpy`, and `torch`.

---

## 3. Merkle Trees & Sharded Dataset Hashing

Large-scale continuous training datasets span hundreds of gigabytes across sharded storage formats (JSONL, Parquet, or memory-mapped binary tokens). Computing a monolithic SHA-256 hash across terabytes of data during every job startup causes extreme I/O bottlenecks.

Instead, pipelines utilize **Hierarchical Shard Hashing (Merkle Trees)**:

```text
                          Top Dataset Root Hash: H_root
                         SHA-256( H_shard_01 || H_shard_23 )
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        H_shard_01 = SHA(H0 || H1)              H_shard_23 = SHA(H2 || H3)
              ┌─────┴─────┐                           ┌─────┴─────┐
              ▼           ▼                           ▼           ▼
           H_shard_0   H_shard_1                   H_shard_2   H_shard_3
         (shard_0.bin)(shard_1.bin)              (shard_2.bin)(shard_3.bin)

```

### Mathematical Formulation

Let a dataset $\mathcal{D}$ consist of $N$ immutable binary shards $S = (s_1, s_2, \dots, s_N)$.
For each shard $s_i$, the shard digest is computed via chunked streaming:

$$h_i = \text{SHA256}(s_i)$$

The total dataset provenance digest $H_{\mathcal{D}}$ is the SHA-256 digest of the concatenated, lexicographically sorted shard hashes:

$$H_{\mathcal{D}} = \text{SHA256}\left( h_1 \parallel h_2 \parallel \dots \parallel h_N \right)$$

This allows worker nodes to verify dataset integrity in parallel, caching intermediate shard hashes in local metadata indices.

---

## 4. The Checkpoint Metadata Envelope

When saving model state, metadata is stored in a structured JSON schema embedded within the root checkpoint payload or alongside a SafeTensors header.

```json
{
  "provenance_version": "1.0",
  "artifact_id": "minigpt-sft-v2-step-40000",
  "timestamp_utc": "2026-08-23T04:10:00Z",
  "code_lineage": {
    "git_commit_sha": "9f8a7c6e5d4b3a210fedcba9876543210abcdef0",
    "git_branch": "main",
    "is_dirty": false,
    "patch_diff_sha256": null
  },
  "data_lineage": {
    "dataset_root_hash": "a8f5f167f44f4964e6c998dee827110c",
    "shard_count": 128,
    "total_token_count": 104857600,
    "tokenizer_hash": "3c59dc048e8850243be8079a5c74d079"
  },
  "environment": {
    "python_version": "3.11.9",
    "torch_version": "2.4.0+cu124",
    "cuda_version": "12.4",
    "gpu_model": "NVIDIA H100 80GB HBM3",
    "device_count": 8
  },
  "execution_parameters": {
    "global_step": 40000,
    "epoch": 3,
    "global_seed": 1337,
    "batch_size_effective": 512,
    "learning_rate_peak": 0.0003,
    "weight_decay": 0.1
  }
}

```

---

## 5. Python Implementation: Automated Provenance Engine

Below is the standalone Python engine that inspects repository status, captures uncommitted diffs, computes streaming dataset hashes, records system environments, and injects the provenance envelope into saved checkpoints:

```python
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


class ProvenanceEngine:
    """
    Automates cryptographic provenance extraction, dataset verification,
    and metadata envelope injection for training artifacts.
    """
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()

    def get_git_provenance(self, allow_dirty: bool = False) -> Dict[str, Any]:
        """
        Extracts Git commit SHA, branch, and uncommitted working tree diffs.
        """
        try:
            # 1. Extract Current Commit SHA
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()

            # 2. Extract Branch Name
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.repo_path,
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()

            # 3. Check for Uncommitted Modifications (Dirty Tree)
            status_output = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()

            is_dirty = len(status_output) > 0
            patch_diff = ""
            patch_sha = None

            if is_dirty:
                if not allow_dirty:
                    raise RuntimeError(
                        f"PROVENANCE GATE FAILED: Working tree contains uncommitted changes. "
                        f"Commit all code before initiating production training run."
                    )
                # Capture unified diff to preserve local modifications
                patch_diff = subprocess.check_output(
                    ["git", "diff", "HEAD"],
                    cwd=self.repo_path,
                    stderr=subprocess.DEVNULL
                ).decode("utf-8")
                patch_sha = hashlib.sha256(patch_diff.encode("utf-8")).hexdigest()

            return {
                "git_commit_sha": sha,
                "git_branch": branch,
                "is_dirty": is_dirty,
                "patch_diff_sha256": patch_sha,
                "patch_diff": patch_diff if is_dirty else None
            }

        except subprocess.CalledProcessError as err:
            raise RuntimeError(f"Failed to execute Git commands: {err}")

    @staticmethod
    def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
        """Computes streaming SHA-256 digest of a single file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    def compute_dataset_merkle_root(self, shard_paths: List[Path]) -> Dict[str, Any]:
        """
        Computes deterministic Merkle root hash across dataset shards.
        """
        sorted_shards = sorted(shard_paths, key=lambda p: p.name)
        shard_digests = []

        for shard in sorted_shards:
            if not shard.is_file():
                continue
            digest = self.compute_file_sha256(shard)
            shard_digests.append((shard.name, digest))

        # Combine all shard hashes into a single root hash
        combined_payload = "".join([f"{name}:{digest}" for name, digest in shard_digests])
        root_hash = hashlib.sha256(combined_payload.encode("utf-8")).hexdigest()

        return {
            "dataset_root_hash": root_hash,
            "shard_count": len(shard_digests),
            "shard_manifest": dict(shard_digests)
        }

    @staticmethod
    def capture_environment_fingerprint() -> Dict[str, Any]:
        """Captures hardware, OS, and toolchain configurations."""
        env_data = {
            "python_version": sys.version.split()[0],
            "os": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        if torch.cuda.is_available():
            env_data["gpu_model"] = torch.cuda.get_device_name(0)
        return env_data

    def build_provenance_envelope(
        self,
        shard_paths: List[Path],
        training_config: Dict[str, Any],
        step: int,
        allow_dirty: bool = False
    ) -> Dict[str, Any]:
        """Assembles complete cryptographic bill of materials."""
        return {
            "provenance_spec_version": "1.0",
            "code_lineage": self.get_git_provenance(allow_dirty=allow_dirty),
            "data_lineage": self.compute_dataset_merkle_root(shard_paths),
            "environment": self.capture_environment_fingerprint(),
            "execution": {
                "step": step,
                "config": training_config
            }
        }

    def save_audited_checkpoint(
        self,
        save_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        envelope: Dict[str, Any]
    ):
        """Persists checkpoint with embedded cryptographic envelope."""
        final_path = Path(save_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "provenance": envelope
        }

        # Atomic persistence
        temp_path = final_path.with_suffix(".tmp")
        torch.save(checkpoint_payload, temp_path)
        os.replace(temp_path, final_path)

        # Write standalone human-readable provenance JSON sidecar
        sidecar_path = final_path.with_suffix(".provenance.json")
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(envelope, f, indent=2)

        print(f"Checkpoint and Provenance Sidecar saved to {final_path}")

```

---

## 6. Pre-Flight and Release Audit Invariants

```text
┌────────────────────────────────────────────────────────────────────────┐
│ PROVENANCE AUDIT VERIFICATION POLICIES                                 │
├──────────────────────────────┬─────────────────────────────────────────┤
│ Gate 0: Pre-Flight Start     │ • Git tree must be clean (or patch saved│
│                              │ • Dataset root hash must match registry │
│                              │ • Refuse execution if uncommitted code  │
├──────────────────────────────┼─────────────────────────────────────────┤
│ Gate 6: Production Registry  │ • Verify provenance envelope exists     │
│ Promotion Audit              │ • Verify Git SHA exists on remote origin│
│                              │ • Re-verify dataset hash reproducibility│
└──────────────────────────────┴─────────────────────────────────────────┘

```

### Verification Rules

| Audit Target | Evaluation Rule | Failure Action |
| --- | --- | --- |
| **Git Working Tree** | `is_dirty == False` (Strict Mode) | Terminate training job before GPU allocation |
| **Dataset Merkle Hash** | Hash matches approved dataset registry manifest | Reject data ingestion; raise DataIntegrityError |
| **Tokenizer Hash** | Hash matches reference tokenizer build | Abort; prevents token ID vocabulary misalignment |
| **Envelope Completeness** | Checkpoint contains all required metadata keys | Reject promotion to production serving registry |