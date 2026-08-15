# Raw Data Ingestion

The goal of this layer is to ingest raw records (such as Parquet files from your MinIO Bronze layer) and convert them into Ray’s internal distributed data structure without overwhelming your cluster's memory.

## How It Works Under the Hood

* **Lazy Loading:** When you call `ray.data.read_parquet()`, Ray does not read your actual text into memory yet. It simply scans the storage metadata (schema, file layout, and total file paths) to generate an execution plan.
* **Block Partitioning:** Ray splits the target files into logical, streamable units called Partitions (or Blocks), aiming for an in-memory size of roughly 100 MB to 512 MB per block.
* **The Object Store:** As the pipeline is actively executed by Airflow, Ray worker processes read these Parquet blocks from MinIO, deserialize them from their on-disk formats into memory-optimized Apache Arrow tables, and place them into Ray's distributed shared-memory object store (Plasma). This allows multiple workers on the same machine to access the raw text blocks directly without wasting time duplicating memory.

---

## Ingestion in the Continuous Training Loop

In an enterprise Continuous Training loop, Ingestion is not just a file reader; it is a **memory-throttling firewall**. If ingestion is designed poorly, a spike in raw database logs will flood the system, trigger an Out-Of-Memory (OOM) panic, and crash your entire cloud pipeline before training ever begins.

Here are the deep architectural mechanics of how Ray Data constructs and executes its ingestion block engine:

### 1. The Two-Phase Execution Plan (Lazy Evaluation)

When Apache Airflow triggers your data ingestion script, Ray splits the work into two distinct runtime phases: **The Planning Phase** and **The Materialization Phase**.

#### Phase A: The Metadata Scan (Planning)

When you invoke `ray.data.read_parquet("s3://minio/bronze/")`, the Ray Driver process initiates a metadata handshake with MinIO.

* **What happens:** Ray does not read the string data. It downloads only the Parquet file footers and schema definitions.
* **The Mechanics:** Parquet footers contain structural metadata—byte offsets, column statistics, row group counts, and dictionary encodings. Ray reads this metadata to calculate exactly how many rows exist and how large the data will be once expanded into raw memory. It uses this to pre-calculate an execution graph of streamable chunks before spinning up a single heavy worker task.

#### Phase B: The Streaming Execution (Materialization)

Data is only pulled from MinIO when a downstream operation (like a filter or a save function) explicitly requests it. Ray executes the ingestion using a Streaming Generator pattern, pulling data off the wire block by block.

### 2. Block Partitioning & Memory Control Mechanics

In-memory data structures occupy vastly more space than compressed files on a disk. A 100 MB Snappy-compressed Parquet file inside MinIO can easily inflate to 500 MB to 1 GB of raw, uncompressed text strings once loaded into Python RAM.

To manage this safely, Ray Data utilizes a target-driven block sizing strategy controlled by the `DataContext`.

**How Ray Determines Block Sizes:**

* **The Target Limit:** By default, Ray tries to size output blocks to roughly 128 MB of in-memory data.
* **File Bundling vs. Splitting:**
* **Bundling:** If your Spark pipeline wrote thousands of tiny 5 MB Parquet files to MinIO, Ray's metadata planner will automatically bundle multiple file references into a single read task to avoid creating downstream processing overhead.
* **Splitting:** If Spark wrote massive 2 GB Parquet files, Ray uses the Parquet footer metadata to read the file in smaller, isolated Row Groups, splitting a single physical file on disk into multiple independent memory blocks.



### 3. The Shared-Memory Object Store (Plasma)

Once a worker node pulls a block from MinIO, it does not hand it over as a standard, isolated Python variable. Doing so would cause a memory bottleneck if multiple operations needed to inspect that data. Instead, Ray loads the block into the Plasma Object Store.

**The Under-the-Hood Mechanics:**

* **Apache Arrow Serialization:** Ray deserializes the raw bytes from MinIO straight into an Apache Arrow columnar format inside memory. Arrow stores strings in contiguous, highly compact byte arrays rather than wrapping each text line in heavy Python string pointers.
* **Shared-Memory Allocations (`/dev/shm`):** The data block is committed directly to a shared-memory segment on the operating system.
* **Zero-Copy Deserialization:** Because the block lives in a shared space, your cleaning scripts, tokenizers, and verification functions can read the exact same memory blocks simultaneously. Multiple worker processes read the data via memory mapping without copying the raw strings between themselves. This reduces your system memory overhead to near-zero during the ingestion step.

---

# Deep Dives: Ingestion & Memory Architecture

## 1. How Apache Arrow Stores Strings in Memory

Standard Python stores lists of strings as an array of pointers pointing to random, scattered locations in memory. This creates massive CPU overhead because the processor has to constantly hop around memory addresses (causing cache misses).

Apache Arrow solves this by storing strings in a completely flat, contiguous physical layout. It represents a string array using three distinct contiguous memory buffers.

### Apache Arrow 3-Buffer Architecture

If you have an Arrow string array containing `["Hey", None, "Robot"]`, Arrow allocates three physical memory blocks side-by-side:

1. **The Validity Bitmap Buffer:** A sequence of bits indicating if a slot is null or valid.
* For 3 items, it allocates a byte where bits are flipped: `1` (Valid), `0` (Null), `1` (Valid) $\rightarrow$ `10100000`.


2. **The Offsets Buffer (Int32 or Int64):** A contiguous array of integers that tell the CPU exactly where each string begins and ends inside the actual data block.
* For our data, the offsets would be: `[0, 3, 3, 8]`.
* String 0 length: $3 - 0 = 3$ bytes.
* String 1 (Null) length: $3 - 3 = 0$ bytes.
* String 2 length: $8 - 3 = 5$ bytes.


3. **The Value / Data Buffer:** A single, unbroken block of bytes containing the raw characters concatenated directly together: `HeyRobot`.

**Why this is blazingly fast:**
When a CPU reads this data, it can stream the entire raw byte block straight into its L1/L2 hardware cache in a single clock cycle. It doesn't follow memory pointers; it simply jumps to `Offsets[i]` to read the text instantly.

---

## 2. Arrow & The Operating System Shared Memory Segment

In a multi-process environment (like your Ray cluster or a system running multiple Python scripts), transferring data between processes usually requires Serialization. Process A transforms its data into bytes, sends it over a network or pipe, and Process B deserializes those bytes back into internal objects. This consumes massive CPU power.

Apache Arrow eliminates this completely through the OS Shared Memory Segment using "Zero-Copy" architecture.

### The Step-by-Step Kernel Mechanics:

1. **Allocation via POSIX (`shm_open`):** When Ray Data initializes its internal object store (Plasma), it asks the Linux/Unix kernel to allocate a specific chunk of virtual memory dedicated to shared files, typically mounted under `/dev/shm` (Shared Memory).
2. **Writing the Arrow Payload:** Process A (the ingestion task) receives data and writes the Arrow 3-Buffer layout (Bitmap, Offsets, Data) directly into this `/dev/shm` virtual memory region.
3. **Memory Mapping (`mmap`):** When Process B (your text cleaning script) needs to process that data, Ray does not send the data over an internal network pipe. Instead, it passes the raw 64-bit memory address pointer and file descriptor to Process B. Process B calls the system kernel's `mmap()` function.
4. **Zero-Copy Reading:** The operating system maps Process B’s virtual memory table to point to the exact same physical RAM pages where Process A wrote the data.

Process B can now read, search, and iterate through the strings inside that shared segment instantly. The data is never copied, serialized, or altered in transit. This allows your Option B pipelines to process gigabytes of data on restricted host hardware without consuming extra RAM.

---

## 3. Does `/dev/shm` Use the OS Virtual Memory Capability?

Yes, absolutely. `/dev/shm` (shared memory) is a virtual filesystem implemented using `tmpfs` (Temporary Filesystem) in the Linux kernel. It treats your system's virtual memory subsystem exactly like a hard drive.

When an application like Ray writes an Apache Arrow table into `/dev/shm`, it is not interacting with physical RAM directly; it is talking to the Linux Virtual Memory Manager (VMM).

### What happens if physical RAM fills up?

Because `/dev/shm` uses the OS virtual memory capabilities, it is completely backed by Swap space.

* **The Safe Overflow:** If your Option B pipeline loads an exceptionally massive raw dataset that exceeds your physical RAM capacity, the Linux kernel will not crash the machine. Instead, its virtual memory paging algorithm will gracefully (but slowly) evict the oldest, least-frequently-used chunks of `/dev/shm` data out of physical RAM and write them to your Swap partition/file on your SSD.
* **The Performance Trade-off:** While your pipeline stays alive, you lose the "Zero-Copy" speed advantage. Reading data from an SSD swap file is orders of magnitude slower than reading straight from electrical RAM registers. This is why properly sizing your data blocks during Ingestion (Step 1) is so critical.

---

## 4. What Happens When a Data Block Exceeds the CPU L1/L2 Cache Space?

First, a tiny hardware correction to ground us in reality: The CPU cannot stream a whole raw dataset into L1/L2 cache in a single clock cycle. The physical wire connection (the memory bus) between RAM and the CPU cache is restricted. Instead, data moves in fixed, tiny physical packets called **Cache Lines**, which are almost universally 64 bytes in size on modern processors.

When your code tells the CPU to iterate through a massive contiguous Apache Arrow string array, a highly choreographed hardware race begins.

### Scenario: The Data is Larger than the L1/L2 Cache

If you are iterating through a 50 MB Arrow block, it cannot fit into your CPU’s ultra-fast 32 KB L1 Data Cache or 512 KB L2 Cache. Here is the mechanical step-by-step of what happens:

1. **The Cache Miss:** The CPU looks at its L1 cache for the next string offset byte. It isn't there. This is an L1 Cache Miss.
2. **The Cache Line Fill:** The CPU halts execution for a brief moment, goes down to the L2 cache, L3 cache, or main RAM, grabs a 64-byte Cache Line containing your target byte plus its immediate neighboring bytes, and copies it into L1.
3. **Exploiting Spatial Locality:** Because Apache Arrow stores strings back-to-back in an unbroken block, when the CPU brings in those 64 bytes, it accidentally gets the next few characters or strings for free! The next few loop cycles score a 100% Cache Hit speed.
4. **The Eviction (The Overfill):** As your code keeps looping through the 50 MB array, new 64-byte cache lines keep flooding the L1/L2 caches. When the cache runs out of physical slots, the CPU hardware invokes an LRU (Least Recently Used) Eviction Policy. It kicks the oldest 64-byte blocks out of L1 back down to L2/L3 to make space for the incoming data.
5. **Hardware Prefetching to the Rescue:** Modern CPUs have a dedicated circuit called a Hardware Prefetcher. It watches your code's memory access pattern. If it notices you are reading the Arrow byte buffer sequentially, it predicts the future. It will proactively fetch the next 64-byte cache lines from main RAM and load them into the L2/L3 cache before your code even asks for them, completely masking the latency of your small cache sizes.

---

# Why JSONL is a Bad Idea

It completely breaks your Ingestion Stage (Step 1), introducing a severe computational tax on CPU, memory, and disk I/O. Here is exactly what changes under the hood when your input files switch from binary column-oriented Parquet to text row-oriented JSON Lines.

## 1. The Death of Zero-Copy Materialization

When you read a `.parquet` file into your Ray Data pipeline, it uses Apache Arrow under the hood. Parquet is natively aligned with Arrow’s columnar structure. The operating system can map the binary pages of the Parquet file directly from disk into your `/dev/shm` shared memory segment with virtually zero CPU parsing cycles.

With `.jsonl`, every single line is an independent ASCII/UTF-8 string.

* **The CPU Parsing Tax:** The CPU cannot simply memory-map raw JSON text into structured registers. A worker core must sequentially read the bytes of each line, find the string delimiters, instantiate a text parser (like Python's `json.loads` or a fast C-based parser like `simdjson`), and translate those raw characters into a structured object.
* **The Memory Expansion Explosion:** JSON text strings are heavily packed. When parsed into a mutable in-memory object (like a Python dictionary), the memory footprint expands significantly. A 512 MB compressed JSONL file can easily explode into 1.5 GB to 2 GB of raw heap memory during the parsing phase alone, immediately threatening your 1.5 GB process allocation threshold.

## 2. Row-Based Scanning vs. Projective Column Filtering

In our exact deduplication design, we only want to hash the specific target column (e.g., `raw_text`).

* **The Parquet Way:** Parquet’s metadata allows the pipeline to perform column projection. Ray Data skips reading the metadata columns or author signatures completely; it extracts only the binary pages of the `raw_text` column from disk.
* **The JSONL Way:** JSONL is strictly row-oriented. To extract the value of the `"text"` key, the process must scan and parse every single byte of the row from left to right. If your JSONL file contains heavy metadata schemas nested alongside the text, your worker cores waste massive cycles deserializing data that will be immediately thrown away.

## 3. A Fix for Those Who Really Need to Use JSONL: Streaming Arrow-JSON Ingestion

To prevent JSONL files from destroying your lean hardware blueprint, you cannot use standard Python file readers (`open().readlines()`). You must enforce a compiled, streaming C++ translation layer at the entry point.

Inside your Ray Data ingestion setup, you utilize `ray.data.read_json()` backed by PyArrow’s Native JSON Reader Engine.

By configuring PyArrow's native C++ reader to ingest the JSONL files in strict 50 MB raw text increments:

* The C++ layer parses the text rows and immediately projects them into a structured, columnar Apache Arrow Table format.
* It drops the metadata columns instantly before they can contaminate the process heap.
* It bundles the remaining text rows into your uniform 512 MB shared memory block inside `/dev/shm`.

---

## The Architectural Verdict

Avoid using this format unless it’s tightly attached to the enterprise constraints. Doing so, your Ingestion Step changes from a fast, low-compute disk-map into an intensive, CPU-bound parsing operation. This requires your node sizing formulas to comfortably support the initial parsing memory expansion before the rows hit the deduplication gate.