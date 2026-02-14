# ARM Platform Optimization Guide

## Architecture: ARMv8 (64-bit) & Big.LITTLE

ARM integration requires specific handling of "Performance" vs "Efficiency" cores.

### 1. Resources API & Privileges
To gain maximum speed, the process must request `CAP_SYS_NICE` privileges to set:
*   **Nice Value**: -10 or lower for the NPU Coordinator.
*   **CPU Affinity**: `taskset -c 4-7` (Bind to Big Cores on an 8-core SoC).

### 2. Power Profiles
Linux provides `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`.

| Profile | Governor | Usage |
| :--- | :--- | :--- |
| **Overdrive** | `performance` | Real-Time CNC Control. Locks max freq. |
| **Balanced** | `schedutil` | Standard operation. Scales with load. |
| **Eco** | `powersave` | Idle monitoring. Forces min freq to save watts. |

### 3. Implementation in Cortex V2
The `PowerGovernor` class (in `npu_coordinator.py`) interfaces with these system files to switch modes dynamically based on the `NeuralState`.
