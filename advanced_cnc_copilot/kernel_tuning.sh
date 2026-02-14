#!/bin/bash
# Intel Core Optimization Script (11th Gen+)
# Targets: Tiger Lake, Rocket Lake, Alder Lake, Raptor Lake
# Goal: Maximize Vulkan Compute & AVX-512 Performance

echo "Applying Intel Kernel Optimizations..."

# 1. CPU Frequency Governor (Performance)
# 11th Gen+ supports Intel Speed Shift (HWP)
if [ -d "/sys/devices/system/cpu/cpu0/cpufreq" ]; then
    echo "Setting CPU Governor to Performance..."
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
    
    # Set Energy Performance Bias to Performance (0)
    # This allows VNNI/AVX-512 to clock higher
    if [ -f "/sys/devices/system/cpu/cpu0/power/energy_perf_bias" ]; then
        echo 0 | tee /sys/devices/system/cpu/cpu*/power/energy_perf_bias > /dev/null
    fi
fi

# 2. Scheduler Tuning for Hybrid Architecture (P-Cores vs E-Cores)
# Reduce migration cost to allow threads to stick to P-Cores
sysctl -w kernel.sched_migration_cost_ns=500000

# Increase slice for compute tasks (Gamesa Grid)
sysctl -w kernel.sched_min_granularity_ns=10000000
sysctl -w kernel.sched_wakeup_granularity_ns=15000000

# 3. Virtual Memory (Memory Bandwidth)
# Reduce swappiness to keep Neural Tensors in RAM
sysctl -w vm.swappiness=10
sysctl -w vm.vfs_cache_pressure=50

# 4. Intel iGPU (Xe) Optimization
# Increase preemption timeout for long-running Vulkan shaders
# (Note: path varies by kernel/driver version)
if [ -f "/sys/class/drm/card0/engine/rcs0/preempt_timeout_ms" ]; then
    echo 2500 > /sys/class/drm/card0/engine/rcs0/preempt_timeout_ms
fi

echo "Intel Kernel Optimization Applied."
