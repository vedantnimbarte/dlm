# Technical Specification (`specs.md`)
## System Implementation Blueprint for `flip`

---

## 1. System Technology Stack
*   **Orchestration & Systems Engine:** Rust (edition 2021). Chosen for safe, multi-threaded CPU I/O, deterministic memory management, and zero-cost abstractions.
*   **Machine Learning Core & Hardware Bindings:** PyTorch C++ API (`libtorch`) / Raw CUDA Kernels mapped directly into Rust via FFI bindings.
*   **Python Interface Layer:** `PyO3` + `maturin`. Provides optional high-level scripting hooks, compiling Rust source into an optimized native extension module (`.so` / `.pyd`).
*   **CLI Application Framework:** `clap` crate (v4) with derive features for high-performance argument parsing.
*   **Distributed Networking Transport:** `tonic` (Rust gRPC implementation) using HTTP/2 over TLS/TCP with Protocol Buffers (`prost`).
*   **Asynchronous Runtime Engine:** `tokio` (multi-threaded feature profile).

---

## 2. Low-Level Architecture & Memory Topography

`flip` segments the physical GPU VRAM into three strict memory regions. The structural blueprint of the system layout is configured as follows:
┌────────────────────────────────────────────────────────┐│                      GPU VRAM                          │├───────────────────┬───────────────────┬────────────────┤│   PINNED ZONE     │   STREAMING ZONE  │  CACHE ZONE    ││  • Embedding Block│  • Double-Buffer A│ • Paged KV     ││  • LM Head / Norm │  • Double-Buffer B│ • Intermediate ││  • Draft Model    │   (Asynchronous)  │   Residuals    │└───────────────────┴───────────────────┴────────────────┘


### 2.1 The Pinned Zone
*   **Embedding Layer & LM Head:** Must remain in VRAM permanently. Removing them introduces a severe PCIe bus bottleneck during token conversion.
*   **Draft Model Infrastructure:** Holds a tiny, quantized draft model (1B to 3B parameters) permanently in memory to handle Speculative Decoding loops.

### 2.2 The Streaming Zone
Divided into two distinct physical buffers (`Buffer_A` and `Buffer_B`).
*   While `Buffer_A` is locked and executing operations on the active GPU compute stream, `Buffer_B` uses Direct Memory Access (DMA) to pre-fetch the next set of layers from the host system memory over the PCIe bus.

### 2.3 The Cache Zone
*   **PagedAttention Cache:** Breaks down the model's Key-Value (KV) history matrix into fixed physical blocks, preventing VRAM fragmentation.
*   **Residual Activation Pools:** Stores intermediate mathematical skip-connection arrays so past layer parameters can be freed while still feeding forward into future non-linear transformer blocks.

---

## 3. Core Engine Algorithms & Implementation

### 3.1 Dynamic VRAM Profiling Math Engine
Before launching inference, the engine dynamically determines execution configurations using the following formula:

$$\text{Layers To Load} = \left\lfloor \frac{M_{\text{free}} - M_{\text{safety}} - M_{\text{kv\_total}}}{M_{\text{layer\_weight}}} \right\rfloor$$

Where:
*   $M_{\text{free}}$ = Calculated at runtime using `cudaMemGetInfo()`.
*   $M_{\text{safety}}$ = Explicit cushion parameter (Default = 1.5 GB) to capture mathematical runtime activation spikes.
*   $M_{\text{kv\_total}}$ = $2 \times N_{\text{kv\_heads}} \times D_{\text{head}} \times 2 \text{ bytes} \times L_{\text{context\_target}}$.

#### Rust Memory Calculation Engine Example:
```rust
pub struct VramProfiler {
    pub target_context: u32,
    pub safety_margin_bytes: u64,
}

impl VramProfiler {
    pub fn calculate_optimal_chunks(&self, model_config: &ModelConfig) -> u32 {
        let (free_mem, _total_mem) = unsafe {
            // Native FFI Hook to CUDA runtime API
            cuda_get_mem_info_safe()
        };
        
        let usable_vram = free_mem.saturating_sub(self.safety_margin_bytes);
        
        // Calculate exact size of a single quantized layer block
        let bytes_per_param = 0.5; // Assumes 4-bit AWQ / GPTQ quantization
        let model_total_bytes = model_config.total_params as f64 * bytes_per_param;
        let single_layer_weight_bytes = model_total_bytes / model_config.num_layers as f64;
        
        // Calculate KV Cache memory consumption footprint per layer block
        let head_dim = model_config.hidden_size / model_config.num_attention_heads;
        let kv_bytes_per_token_per_layer = 2 * model_config.num_kv_heads * head_dim * 2; // FP16 KV
        let kv_cache_per_layer_bytes = kv_bytes_per_token_per_layer * self.target_context;
        
        let total_bytes_per_layer = single_layer_weight_bytes + kv_cache_per_layer_bytes as f64;
        
        let calculated_chunks = (usable_vram as f64 / total_bytes_per_layer).floor() as u32;
        
        std::cmp::max(1, std::cmp::min(calculated_chunks, model_config.num_layers))
    }
}
```

### 3.2 Asynchronous Double-Buffering & Memory Pipeline
To prevent the GPU from sitting idle during disk I/O, the execution framework isolates computation and data loading onto separate pipelines using `cudaStream_t`.

Timeline  ────────────────────────────────────────────────────────────────────────►Stream A (Compute):   [ Execute Layers 1-10 ]                     [ Execute Layers 11-20 ]▲                                             ▲│ (Pointer Swap)                              │Stream B (Memory):    [ Load Layers 11-20 into Buffer B ]         [ Load Layers 21-30 into Buffer A ]

1.  **Memory Mapping (`mmap`):** The application maps model files from the NVMe SSD directly into the process memory map, skipping standard OS buffer copies.
2.  **Pinned Host Memory (`cudaHostAlloc`):** Allocates page-locked host memory buffers. This allows the PCIe controller to use asynchronous Direct Memory Access (DMA) to stream weights directly into VRAM without CPU intervention.
3.  **Non-Blocking Stream Sync:** Memory copies use `cudaMemcpyAsync` scheduled on Stream B. Computation kernels run concurrently on Stream A. Synchronization points are managed via explicit `cudaEvent_t` markers to verify data transfers before switching active pointers.

### 3.3 Multi-Device Pipeline Parallelism (PP)
When running across multiple local GPUs, layers are distributed across devices. Data is passed between GPUs through a ring pipeline using NCCL collections:

```rust
pub struct PipelineParallelEngine {
    pub local_gpus: Vec<u32>,
    pub layers_per_gpu: u32,
    pub nccl_comm: NcclCommunicator,
}

impl PipelineParallelEngine {
    pub async fn forward_step(&mut self, stage_id: usize, mut hidden_states: Tensor) -> Tensor {
        if stage_id == 0 {
            // First GPU stage: Processes local layers
            hidden_states = self.compute_local_layers(stage_id, hidden_states).await;
            // Async P2P send to next GPU
            self.nccl_comm.send(&hidden_states, self.local_gpus[stage_id + 1]).await;
        } else if stage_id == self.local_gpus.len() - 1 {
            // Last GPU stage: Receives input, computes final layers
            hidden_states = self.nccl_comm.recv(self.local_gpus[stage_id - 1]).await;
            hidden_states = self.compute_local_layers(stage_id, hidden_states).await;
        } else {
            // Intermediate GPU stage
            hidden_states = self.nccl_comm.recv(self.local_gpus[stage_id - 1]).await;
            hidden_states = self.compute_local_layers(stage_id, hidden_states).await;
            self.nccl_comm.send(&hidden_states, self.local_gpus[stage_id + 1]).await;
        }
        hidden_states
    }
}
```

### 3.4 Multi-Server Distributed Topology (gRPC Framework)
For multi-server cluster orchestration, `flip` relies on a Master-Worker topology over gRPC:

*   **Network Serialization Engine:** Instead of heavy text formatting (JSON), multi-dimensional matrix tensors are converted into raw byte streams wrapped in flat `bytes` fields within Protobuf models.
*   **Streaming RPC Routing:** The master node initiates a bi-directional gRPC channel (`rpc StreamInference(stream InferencePayload) returns (stream TokenResponse)`). Tensors pass through network streams sequentially between server nodes.
*   **Heartbeat Error Recovery:** The master node sends low-overhead TCP ping payloads to worker nodes every 20ms. If a worker node disconnects or stops responding, the coordinator reroutes the next set of layer computations to the local system CPU RAM memory buffer pool. This prevents the inference pipeline from throwing an error or freezing.

---

## 4. Operational Command Interface (CLI Specification)

The standalone compiled application binary will expose a production CLI interface using the following schema:

```bash
flip serve \
    --model-path "/home/user/models/Llama-3-70B-Instruct" \
    --vram-budget-gb 13.5 \
    --context-length 8192 \
    --port 8000 \
    --host 127.0.0.1 \
    --draft-model-path "/home/user/models/Llama-3-3B" \
    --multi-gpu-ids 0,1 \
    --distributed-mode master \
    --worker-nodes 192.168.1.50:9001,192.168.1.51:9001
```

### Command Parameter Matrix:
*   `--model-path`: Absolute file path directory containing Safetensors weight slices and configuration files (`config.json`).
*   `--vram-budget-gb`: Upper hardware bounds cap assigned manually to limit memory footprint.
*   `--context-length`: Sets the targeted structural size configuration for the conversation window buffer.
*   `--draft-model-path`: Optional path targeting a tiny model profile to enable Speculative Decoding execution.
*   `--multi-gpu-ids`: Comma-separated index string to split layers across multiple local graphics adapters.
*   `--distributed-mode`: Toggles server operational configuration state (`standalone`, `master`, or `worker`).