# System Architecture

## Overview
GAMESA/KrystalStack is architected as an **Industrial Organism**. It does not merely "run" code; it metabolizes data and trades resources in an internal economy.

## 1. The Core Trinity

### A. The Economic Engine (Cross-Forex Market)
Most schedulers are dictators; ours is a **Free Market**.
- **Commodities**: `VRAM_BLOCK`, `COMPUTE_CYCLE`, `BANDWIDTH_LANE`.
- **Mechanism**: Processes bid for resources using "credits" earned by delivering successful frames or compute units.
- **Result**: High-value tasks automatically acquire resources without manual tuning.

### B. The Guardian (Immune System)
A bio-mimetic safety system with two distinct layers:
1.  **Spinal Reflex (Layer 1)**:
    - **Speed**: <1ms.
    - **Logic**: Deterministic, hard-coded constraints (e.g., "If Temp > 85°C, Throttle").
    - **Language**: Rust/C.
2.  **Cortical Monitor (Layer 2)**:
    - **Speed**: Seconds/Minutes.
    - **Logic**: Probabilistic AI analysis (Metacognition).
    - **Function**: Analyzes long-term trends and "rewrites" the Spinal Reflex policies if they become obsolete.

### C. 3D Grid Memory
Traditional memory is flat (0x0...0xF). KrystalStack uses a **Spatial Topology**:
- **Dimensions**: Tier (Speed) × Slot (Time) × Depth (Compute Intensity).
- **Optimization**: Data that is semantically related is placed physically closer in the grid, minimizing "travel time" (cache misses).

## 2. Advanced Concepts

### Predictive Pre-Execution
The system uses a Temporal Prediction Layer to execute instructions *before* they are requested.
> *Example: If the player is running towards a door, the system pre-loads the "Room B" shaders into VRAM before the door opens.*

### Zero-Copy Universe
By integrating CPU L3, GPU VRAM, and System RAM into a single **Unified Address Space**, we eliminate data copying. Pointers are universal, and data "teleports" by changing ownership rather than moving bits.

## 3. Deployment Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                       APPLICATION LAYER                      │
│                (Games, AI Models, Scientific Sims)           │
├─────────────────────────────────────────────────────────────┤
│                    MIDDLEWARE (Python/C++)                   │
│         [Market Maker]    [Guardian Policy]   [Grid Manager] │
├─────────────────────────────────────────────────────────────┤
│                      KERNEL (Rust/C)                         │
│       [Spinal Reflex]   [Memory Allocator]    [HAL]          │
└─────────────────────────────────────────────────────────────┘
          │                 │                  │
        [GPU]             [RAM]              [CPU]
```
