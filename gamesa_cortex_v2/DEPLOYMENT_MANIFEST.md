# Gamesa Cortex V2: Deployment & Runtime Manifest

## 1. Preferred Runtimes & Utilization

To achieve maximum performance and "Better Utilization" of hardware resources, the architecture selects runtimes based on the specific capabilities of the host system.

| Capability | Preferred Runtime | Why? |
| :--- | :--- | :--- |
| **Path Planning** | **Rust (Cargo)** | Memory safety, zero-cost abstractions, and SIMD (AVX/NEON) checks. |
| **3D Grid & Physics** | **Vulkan 1.2+** | Low-level GPU access, explicit control over allocation, and Compute Shaders. |
| **Legacy Compute** | **OpenCL 3.0** | Broad compatibility (FPGA, Older GPUs, iGPUs) where Vulkan is absent. |
| **AI Inference** | **OpenVINO / ONNX** | Optimized for Intel Hardware (NPU/VNNI) integration. |
| **Orchestration** | **Python 3.10** | Glue code, rapid logic, and ecosystem (NumPy/SciPy). |

## 2. Docker & vGPU Framework

The Docker execution environment is designed to be the "Standard Unit of Deployment".

*   **vGPU Manager**: Divides physical GPU resources into virtual slices for containers.
*   **Hardware Passthrough**: Maps `/dev/dri` (Linux DRM) to containers for near-native performance.
*   **Unified Build**: The Dockerfile compiles the **Rust Planner** during build, ensuring the native extension matches the container OS exactly.

## 3. Supported Distributions (Linux)

For "Maximum Compatibility", we target:

*   **Ubuntu 22.04 LTS**: Primary Target (Debian .deb built directly).
*   **Fedora 38+**: Supported via Docker/Podman.
*   **Arch Linux**: Supported via AUR approach (Source build) or Docker.
*   **Debian 12**: Fully supported via .deb.

## 4. Run Instructions

**Via Docker (Recommended)**:
```bash
docker-compose up -d
```

**Via Native (.deb)**:
```bash
sudo dpkg -i gamesa-cortex-v2_2.0.0_amd64.deb
sudo apt-get install -f
gamesa-cortex-v2 --start
```
