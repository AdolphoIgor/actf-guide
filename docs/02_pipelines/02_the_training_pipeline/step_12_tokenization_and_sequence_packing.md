# Step 12: JIT Tokenization & Sequence Packing

## 1. Core Objective & Operational Placement

Executing as the second stage of the **Training Pipeline**, Step 12 converts template-formatted text strings into model-ready integer tensors.

Operating in-memory within `/dev/shm`, Step 12 utilizes multi-threaded Rust tokenization backends to encode sub-word tokens via Byte-Pair Encoding (BPE), creates target-only label masks (setting prompt tokens to $-100$), packs variable-length conversations into fixed-length context windows ($B \times L$), and feeds tensors directly to PyTorch DataLoaders.

```text
 [ Formatted String ] ──► [ Multi-Threaded BPE ] ──► [ Label Masking (-100) ] ──► [ Tensor Matrix BxL ]

```

---

## 2. Theoretical & Architectural Justification

Standard sequence padding wastes substantial compute on null attention tokens. Sequence packing concatenates multiple conversations up to the maximum context length $L$ (e.g., $L = 2048$), maximizing arithmetic intensity per batch.

Simultaneously, Supervised Fine-Tuning (SFT) requires that loss is calculated **strictly over assistant completions**. Step 12 constructs the label tensor dynamically, ensuring user prompts do not distort empirical loss calculations.

---

## 3. Mathematical Formulation & Execution Mechanics

### 1. Byte-Pair Encoding & Sequence Packing

- Encodes strings into 1D integer arrays: $T = [t_1, t_2, \dots, t_N]$.
- Concatenates sequence blocks separated by `<|im_end|>` tokens into 2D matrices of shape $B \times L$.

### 2. Target-Only SFT Label Masking

PyTorch cross-entropy loss ignores label tokens assigned a value of $-100$. Step 12 builds the training label tensor:

$$\text{Labels}[i] = \begin{cases} \text{Input\_ID}[i], & \text{if } i \in Y \text{ (Assistant Completion)} \\ -100, & \text{if } i \in X \text{ (System / User Prompt)} \end{cases}$$

The objective function optimizes strictly over assistant tokens:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{|Y|} \sum_{t \in Y} \log P_\theta(y_t \mid x, y_{<t})$$

### 3. Pre-Flight Verification Handshake (Gates 3 & 4)

- **Gate 3 (Data Leakage & Split Gate):** Asserts zero cryptographic hash overlap between Train and Validation splits while enforcing a $95/5$ ratio.
- **Gate 4 (Pre-Flight Tensor Gate):** Validates matrix dimensions ($B \times L$), vocabulary boundaries ($0 \le \text{Token ID} < V$), and binary attention masks ($\{0, 1\}$) before passing batches to the model forward pass.
