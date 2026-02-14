# Use Ubuntu 22.04 as base for broad hardware support (Intel/Vulkan)
FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    cmake build-essential curl \
    vulkan-tools libvulkan-dev \
    ocl-icd-opencl-dev opencl-headers \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 3. Work Directory
WORKDIR /app

# 4. Copy Code
COPY . /app

# 5. Install Python Dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    numpy scipy wgpu pyopencl llama-cpp-python

# 6. Build Rust Planner
WORKDIR /app/gamesa_cortex_v2/rust_planner
RUN cargo build --release

# 7. Reset Workdir
WORKDIR /app

# 8. Define Entrypoint
CMD ["python3", "-m", "gamesa_cortex_v2.src.core.npu_coordinator"]
