#!/bin/bash
echo "Installing Benchmarking Tools..."

# Update Repos
# sudo apt-get update

# Install Sysbench (CPU/Memory)
# sudo apt-get install -y sysbench

# Install GLMark2 (GPU/OpenGL)
# sudo apt-get install -y glmark2

echo "NOTE: 'sudo' commands are commented out. Run manually if needed."
echo "Running Local Verification..."

python3 gamesa_cortex_v2/scripts/verify_latency.py
