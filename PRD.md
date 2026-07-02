# Product Requirement Document (PRD)
## Project Name: flip (Dynamic Layer-Streaming Inference Engine)

---

## 1. Executive Summary & Objective
`flip` is a high-performance, ultra-low-resource local LLM inference engine built to run massive large language models (70B, 405B+) on highly restricted consumer hardware (e.g., 16GB VRAM GPUs). 

By replacing standard stationary VRAM weight allocation with an asynchronous, multi-layered pipeline streaming engine, `flip` breaks memory limitations. It achieves execution speeds close to native human reading (5–12 tokens/sec) without requiring high-end enterprise server nodes.

The name `flip` represents the core mechanic of the system: constantly and smoothly flipping layer weights in and out of the GPU's active memory workspace at the microsecond level to bypass traditional hardware limitations.

## 2. Core User Personas
*   **Local AI Developers:** Users who want to experiment locally with massive open-source models (e.g., Llama-3 70B/405B, DeepSeek R1) without paying cloud API fees.
*   **Privacy-First Enterprises:** Organizations running inference on highly sensitive data over local air-gapped networks.
*   **Resource-Constrained Research Teams:** Labs running multi-agent benchmarks across varied environments (single consumer GPUs, multi-GPU rigs, or small network clusters).

## 3. Product Features & Scope

### 3.1 Core Architecture & Deployment
*   **Zero-Dependency Local Inference:** Runs entirely locally on user hardware. No cloud requirements.
*   **Pure CLI Execution Model:** Operates standalone from the command line (`flip serve`). Users do not need to install an external Python runtime environment or write orchestration code.
*   **OpenAI-Compatible Local API Server:** Exposes standard endpoints (`/v1/chat/completions`) natively to integrate seamlessly with Open WebUI, AnythingLLM, and agentic frameworks.

### 3.2 Dynamic Hardware Orchestration
*   **Dynamic VRAM Profiling:** Auto-calculates VRAM space in real-time, taking background OS overhead into account, to decide the maximum amount of layers to load simultaneously.
*   **Double-Buffered Asynchronous Streaming:** Uses separate background compute and memory copy paths to load subsequent layers into VRAM while the GPU executes the current block.
*   **Tiered Memory Caching:** Leverages System RAM (CPU) as an intermediate fast buffer cache between the physical NVMe storage and the GPU.

### 3.3 Intelligence & Optimization
*   **Speculative Decoding:** Integrates a small, permanently pinned draft model (e.g., 1B–3B) to generate quick token guesses, which are verified by the streamed target model in a single parallel step.
*   **Continuous Batching:** Schedules incoming user requests at the microsecond level, avoiding bubbles during multi-server or multi-device processing.
*   **Structural Anchoring:** Permanently pins critical non-linear layers (Embedding Layer, LM Head) into VRAM, streaming only intermediate transformer blocks.

### 3.4 Multi-Device & Scaling Configurations
*   **Single-Node Multi-GPU Support:** Implements Horizontal Pipeline Parallelism (PP) across multiple physical GPUs over local PCIe lanes via NCCL/RCCL.
*   **Distributed Multi-Server Architecture:** Establishes a Master-Worker topology over local networks using gRPC (Protobuf binary streaming) for multi-node configurations.
*   **Fault-Tolerant Heartbeats:** Monitors node networks. Instantly switches layers to the host CPU RAM fallback cache if a worker node goes offline mid-inference.

---

## 4. Non-Functional Requirements & Performance Targets
*   **Target Inference Speed:** 5 to 12 tokens per second for a 70B model running on a single PCIe Gen 4 NVMe SSD paired with a 16GB VRAM GPU.
*   **Baseline Resource Footprint:** Standalone Rust binary orchestration overhead must not exceed 50 MB of system memory.
*   **Memory Reliability:** Zero Out-Of-Memory (OOM) crashes across active conversation context windows up to 8,192 tokens.

## 5. Development Milestones & Phase Map

### Phase 1: Local Foundation
*   Develop the Rust storage manager using memory-mapped I/O (`mmap`).
*   Implement real-time dynamic VRAM math profiling.
*   Build the simple linear layer-swapping execution cycle.

### Phase 2: Asynchronous Acceleration
*   Implement double-buffering using isolated CUDA Streams.
*   Introduce PagedAttention memory layouts and Tiered CPU RAM caching.
*   Add the Command Line Interface (CLI) configuration tool.

### Phase 3: Distributed Scalability
*   Integrate Speculative Decoding and Continuous Batching.
*   Build the gRPC multi-server communication networking architecture.
*   Launch the OpenAI-compatible network routing server.
