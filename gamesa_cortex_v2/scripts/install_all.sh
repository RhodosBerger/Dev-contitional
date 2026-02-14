#!/bin/bash
echo "Gamesa Cortex V2: Installing All Dependencies..."

# 1. Python Dependencies (User Space)
echo "Installing Python Libraries (numpy, scipy, wgpu, pyopencl)..."
pip install numpy scipy wgpu-py pyopencl llama-cpp-python --upgrade || echo "Pip install warning: Check permissions or virtualenv."

# 2. Rust Toolchain (User Space)
if ! command -v cargo &> /dev/null
then
    echo "Rust/Cargo not found. Installing via rustup..."
    # curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    echo "Please install Rust manually: curl https://sh.rustup.rs -sSf | sh"
else
    echo "Rust Cargo detected."
fi

# 3. System Dependencies (Requires Sudo)
echo "----------------------------------------------------------------"
echo "ATTENTION: For full 3D Grid & hardware acceleration, run:"
echo "sudo apt-get install cmake vulkan-tools libvulkan-dev ocl-icd-opencl-dev"
echo "----------------------------------------------------------------"

echo "Setup Complete (Python Layer)."
