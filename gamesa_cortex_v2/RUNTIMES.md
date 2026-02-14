# Gamesa Cortex V2: Runtime Configuration Map

This document defines the required runtimes and the logic for selecting the "Best Configuration".

## 1. Supported Runtimes

| Runtime | Purpose | Priority | Hardware Target |
| :--- | :--- | :--- | :--- |
| **Python 3.10+** | Control Plane, Orchestration | 1 (Required) | CPU (Any) |
| **Rust (Cargo)** | Path Planning, Heavy Math | 2 (High) | CPU (Performance) |
| **Vulkan 1.2+** | 3D Voxel Grid, Compute Shaders | 3 (Med) | GPU (Intel Xe, NVIDIA, AMD) |
| **OpenCL 3.0** | General Purpose Compute | 4 (Alt) | GPU/iGPU (Legacy) |
| **OpenVINO** | AI Inference (LLM/Vision) | 5 (AI) | NPU/CPU (Intel) |

## 2. Best Configuration Map

The system probes for hardware and selects the runtime stack:

1.  **Intel 11th Gen+ (Tiger Lake/Rocket Lake)**
    *   **Compute**: Vulkan (on Iris Xe)
    *   **Planning**: Rust (AVX-512 Optimized)
    *   **AI**: OpenVINO (VNNI)
    
2.  **ARM (Jetson Orin/AgX)**
    *   **Compute**: CUDA (if available) or Vulkan
    *   **Planning**: Rust (NEON Optimized)
    *   **AI**: TensorRT (or PyTorch)

3.  **Legacy/Fallback (Standard PC)**
    *   **Compute**: OpenCL (or Python fallback)
    *   **Planning**: Python
    *   **AI**: ONNX Runtime (CPU)

## 3. Dependency Fetching
Use `scripts/fetch_grid_dependencies.py` to auto-resolve these libraries.
