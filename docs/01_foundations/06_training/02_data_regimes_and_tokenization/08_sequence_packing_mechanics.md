# Sequence Packing Mechanics and In-Memory Tensor Serialization

## 1. The Pad Token Tax: The Cost of Sparse Batching

In standard deep learning pipelines, batches of variable-length text sequences are aligned into uniform rectangular matrices using padding tokens (`<pad>` or `0`). While computationally simple, static or dynamic padding introduces severe throughput inefficiencies during Supervised Fine-Tuning (SFT) and pre-training:

```text
Standard Padded Batch (Batch Size B = 3, Context Limit L = 12):
Row 1: [ T1,  T2,  T3,  T4,  T5,  T6,  T7,  <pad>, <pad>, <pad>, <pad>, <pad> ] (41.6% Waste)
Row 2: [ T1,  T2,  T3,  <pad>, <pad>, <pad>, <pad>, <pad>, <pad>, <pad>, <pad>, <pad> ] (75.0% Waste)
Row 3: [ T1,  T2,  T3,  T4,  T5,  T6,  T7,   T8,    T9,   T10,   T11,   T12  ] ( 0.0% Waste)

Total Tokens Processed: 36  |  Actual Informational Tokens: 22  |  Padding Inefficiency: 38.8%

```

Because standard self-attention computes pairwise dot products across the entire sequence length ($\mathcal{O}(L^2)$), the GPU spends considerable compute calculating attention weights between uninformative `<pad>` tokens. In datasets with long-tail sequence length distributions, padding waste routinely consumes **$30\%\text{--}60\%$ of total GPU FLOPs and VRAM allocation**.

---

## 2. Contiguous Sequence Packing (The Multipack Strategy)

**Sequence Packing** (also known as *Multipack* or *Example Packing*) eliminates padding overhead entirely by concatenating multiple independent, variable-length conversations end-to-end into a single contiguous 1D token array, separated only by native delimiter tokens (`<|im_end|>` or `<|end_of_text|>`).

This continuous stream is sliced into dense, uniform $B \times L$ tensor matrices where every single token cell contains valid training data.

```text
Individual Curated Conversations:
  Doc 1 (Len = 5): [ D1_1, D1_2, D1_3, D1_4, <eos> ]
  Doc 2 (Len = 4): [ D2_1, D2_2, D2_3, <eos> ]
  Doc 3 (Len = 6): [ D3_1, D3_2, D3_3, D3_4, D3_5, <eos> ]

Packed Contiguous Stream:
  [ D1_1, D1_2, D1_3, D1_4, <eos>, D2_1, D2_2, D2_3, <eos>, D3_1, D3_2, D3_3, D3_4, D3_5, <eos> ]
                                │
                                ▼
Packed 2D Matrix (Batch Size B = 1, Sequence Limit L = 15):
Row 1: [ D1_1, D1_2, D1_3, D1_4, <eos>, D2_1, D2_2, D2_3, <eos>, D3_1, D3_2, D3_3, D3_4, D3_5, <eos> ]

Total Tokens Processed: 15  |  Actual Informational Tokens: 15  |  Padding Inefficiency: 0.0%

```

### Computational Benefits

* **100% Hardware FLOP Utilization:** Every matrix multiplication executed by Tensor Cores operates on valid training signal.
* **Deterministic VRAM Allocation:** Dynamic activation memory ($M_{\text{act}}$) remains completely constant across iterations, eliminating unexpected Out-of-Memory (OOM) crashes triggered by outlier long batches.
* **Accelerated Training Convergence:** With zero pad tokens, the effective token throughput per GPU second increases by $1.4\times\text{--}2.5\times$.

---

## 3. Attention Contamination & Document Isolation

While packing achieves maximum FLOP efficiency, concatenating unrelated documents into a single row introduces **Cross-Document Attention Contamination**:

```text
Contaminated Causal Attention (Without Isolation):
  Tokens in Doc 2 (e.g., D2_3) attend causally to all preceding tokens,
  including the entirety of Doc 1 (D1_1 through <eos>).

```

If left unmitigated, the model learns invalid statistical dependencies between distinct topics, corrupted syntax transitions across `<eos>` boundaries, and degraded few-shot reasoning.

Resolving this requires **Document Isolation** via one of two mechanisms:

### Method A: Block-Diagonal Causal Attention Masking

A 2D binary attention mask is constructed where query tokens $q_i$ are permitted to attend to key tokens $k_j$ if and only if $j \le i$ **and** both tokens belong to the exact same document partition $\text{Doc}(i) == \text{Doc}(j)$.

```text
Block-Diagonal Causal Mask (Doc 1: Len 3, Doc 2: Len 2):
       D1_1  D1_2  <eos>   D2_1  <eos>
D1_1  [  1     0     0   │   0     0  ]
D1_2  [  1     1     0   │   0     0  ]
<eos> [  1     1     1   │   0     0  ]
──────┼──────────────────┼────────────┤
D2_1  [  0     0     0   │   1     0  ]
<eos> [  0     0     0   │   1     1  ]

```

### Method B: FlashAttention Cumulative Sequence Lengths (`cu_seqlens`)

Modern GPU attention kernels (FlashAttention-2/3 and PyTorch `scaled_dot_product_attention`) support unpadded, variable-length batch execution without materializing dense 2D mask tensors in memory.

The dataset compiles a 1D index array `cu_seqlens` (Cumulative Sequence Lengths) that marks the exact token offsets of document boundaries:

$$\text{cu\_seqlens} = [0, \text{len}(\text{Doc}_1), \text{len}(\text{Doc}_1) + \text{len}(\text{Doc}_2), \dots, L]$$

```text
For Docs of Length [3, 2, 4]:
cu_seqlens = [0, 3, 5, 9]

```

During the forward pass, the GPU kernel loops through each boundary window independently, executing self-attention exclusively within each sub-segment in high-speed SRAM, delivering true $\mathcal{O}(\sum L_i^2)$ compute rather than $\mathcal{O}((\sum L_i)^2)$.

---

## 4. Position ID Reset Dynamics

In standard non-packed sequences, positional coordinates increment monotonically from $0$ to $L-1$. In packed sequences, continuing this linear increment across document boundaries distorts relative distance modeling:

```text
Un-Reset Position IDs (Incorrect):
Tokens:       [ D1_1, D1_2, <eos>, D2_1, D2_2, <eos> ]
Position IDs: [   0,    1,    2,     3,    4,    5   ]
Problem: Doc 2 begins at Position 3; the model believes D2_1 is a continuation of a prior sentence.

Reset Position IDs (Correct):
Tokens:       [ D1_1, D1_2, <eos>, D2_1, D2_2, <eos> ]
Position IDs: [   0,    1,    2,     0,    1,    2   ]
Result: Every document begins at spatial index 0, matching standalone inference conditions.

```

### Impact on Rotary Position Embeddings (RoPE)

Modern architectures rely on relative rotary angle rotations ($\mathcal{R}_{\Theta, m}$). Resetting position IDs ensures that the rotary vector for the first token of Document 2 is computed at coordinate $m = 0$, matching the exact mathematical state the model experiences during production inference when processing a fresh user prompt.

---

## 5. Synchronized Target Loss Masking (`-100`)

When packing instruction and chat dialogues, the target label array must maintain strict token-by-token alignment with the packed input stream. System prompts, user queries, and structural formatting headers must be masked to `-100`, while assistant responses and closing delimiters remain active targets:

```text
Packed Inputs (x):
  [ <|im_start|>, user, \n, Hi, <|im_end|>, \n, <|im_start|>, asst, \n, Hello, <|im_end|>, \n, <|im_start|>, user, \n, 2+2?, <|im_end|>, \n, <|im_start|>, asst, \n, 4, <|im_end|>, \n ]

Synchronized Target Labels (y):
  [     -100,     -100, -100, -100,  -100,  -100,    -100,     -100, -100, Hello, <|im_end|>, \n,    -100,     -100, -100, -100,  -100,  -100,    -100,     -100, -100, 4, <|im_end|>, \n ]
  └───────────────────────────────┬───────────────────────────────┘  └────────────┬────────────┘ └───────────────────────────────┬───────────────────────────────┘  └────────────┬────────────┘
              Doc 1: Prompt Tokens (Loss Masked)                     Doc 1: Target Response                  Doc 2: Prompt Tokens (Loss Masked)                     Doc 2: Target Response

```

---

## 6. PyTorch Implementation: Contiguous Tensor Packing Engine

Below is the production implementation of a zero-waste contiguous sequence packer. It formats multi-turn Silver conversations, performs target-only loss masking, packs tokens into dense fixed $B \times L$ tensors, resets position IDs, and outputs cumulative sequence offsets (`cu_seqlens`):

```python
from typing import Any, Dict, List, Tuple
import torch
from tokenizers import ByteLevelBPETokenizer


class ContiguousSequencePacker:
    """
    Packs variable-length multi-turn dialogues into contiguous B x L matrices
    with synchronized loss masking (-100), position ID resets, and cu_seqlens.
    """
    def __init__(
        self,
        tokenizer: ByteLevelBPETokenizer,
        block_size: int = 2048,
        im_start: str = "<|im_start|>",
        im_end: str = "<|im_end|>"
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.im_start = im_start
        self.im_end = im_end

    def _encode_dialogue(self, messages: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        """Encodes a single dialogue into input_ids and masked label_ids."""
        input_ids = []
        labels = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted_turn = f"{self.im_start}{role}\n{content}{self.im_end}\n"
            turn_tokens = self.tokenizer.encode(formatted_turn).ids

            input_ids.extend(turn_tokens)

            if role == "assistant":
                # Mask the turn header; calculate loss on content and closing delimiter
                header_tokens = self.tokenizer.encode(f"{self.im_start}{role}\n").ids
                header_len = len(header_tokens)
                turn_labels = ([-100] * header_len) + turn_tokens[header_len:]
                labels.extend(turn_labels)
            else:
                # System and User turns are completely masked from backprop
                labels.extend([-100] * len(turn_tokens))

        return input_ids, labels

    def pack_dataset(
        self, conversations: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Concatenates all dialogues and slices into dense B x L tensor matrices.
        """
        all_input_ids: List[int] = []
        all_labels: List[int] = []
        all_position_ids: List[int] = []
        cu_seqlens: List[int] = [0]
        
        current_cumulative_len = 0

        for record in conversations:
            inp_ids, lbl_ids = self._encode_dialogue(record["messages"])
            doc_len = len(inp_ids)

            # Skip single documents that exceed maximum context window
            if doc_len > self.block_size:
                inp_ids = inp_ids[:self.block_size]
                lbl_ids = lbl_ids[:self.block_size]
                doc_len = self.block_size

            all_input_ids.extend(inp_ids)
            all_labels.extend(lbl_ids)
            
            # Reset position IDs to start at 0 for every document
            all_position_ids.extend(list(range(doc_len)))
            
            current_cumulative_len += doc_len
            cu_seqlens.append(current_cumulative_len)

        # Truncate stream to the nearest multiple of block_size
        total_tokens = len(all_input_ids)
        num_blocks = total_tokens // self.block_size
        usable_tokens = num_blocks * self.block_size

        if usable_tokens == 0:
            raise ValueError(
                f"Insufficient tokens ({total_tokens}) to populate a single block_size={self.block_size}"
            )

        truncated_inputs = all_input_ids[:usable_tokens]
        truncated_labels = all_labels[:usable_tokens]
        truncated_positions = all_position_ids[:usable_tokens]

        # Reshape into uniform [B, L] 2D tensor matrices
        tensor_inputs = torch.tensor(truncated_inputs, dtype=torch.long).view(num_blocks, self.block_size)
        tensor_labels = torch.tensor(truncated_labels, dtype=torch.long).view(num_blocks, self.block_size)
        tensor_positions = torch.tensor(truncated_positions, dtype=torch.long).view(num_blocks, self.block_size)

        return {
            "input_ids": tensor_inputs,
            "labels": tensor_labels,
            "position_ids": tensor_positions,
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32)
        }

```

---

## 7. Pre-Flight Verification Checklist (Gate 4 Alignment)

Before feeding packed tensor batches to the model, the data loader must satisfy the **Gate 4 Pre-Flight Invariants**:

* **Tensor Rectangularity:** Matrix shape evaluates strictly to $(B, L)$ where $B = \lfloor \frac{\sum \text{len}(D_i)}{L} \rfloor$.
* **Integer Bounds:** Every integer in `input_ids` falls strictly within the active vocabulary range ($0 \le \text{ID} < V$).
* **Label Complementarity:** Every element in `labels` is either equal to its corresponding `input_ids` token or exactly equal to the ignore index ($-100$).
* **Position Reset Alignment:** The value `0` in `position_ids` occurs at least once per document boundary in every packed row.