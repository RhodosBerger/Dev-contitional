# Gamesa Cortex V2: The Neural Control Plane

**Architecture Review & Theoretical Foundation**

## 1. The Python Control Paradigm
In the Cortex V2 architecture, Python serves not as the "Compute Engine" but as the **Control Plane**. This adheres to the modern "Glue Logic" paradigm where high-level reasoning (Python) orchestrates low-level acceleration (C++/Vulkan/OpenCL).

*   **Role**: Orchestration, State Management, High-Level Logic.
*   **Constraint**: Python's Global Interpreter Lock (GIL) is circumvented by offloading heavy tasks to `VulkanGridEngine` and `OpenCLAccelerator`.

## 2. NPU Serving & Process Scheduling
The **NPU Coordinator** (`npu_coordinator.py`) implements advanced OS scheduling theories to serve AI models efficiently.

### A. The "Neural Accommodation" Theory
Unlike standard CPU schedulers (CFS) that optimize for fairness, an NPU scheduler must optimize for **Throughput** and **Latency** depending on the context.
*   **Accommodating Bursts**: The NPU pre-empts lower priority tasks (e.g., Grid Visualization) when a "Safety Critical" inference (e.g., Collision Detection) is required.

### B. Better Timers & Real-Time Constraints
Standard `time.sleep()` is inaccurate for industrial control (jitter > 1ms). Cortex V2 utilizes **Monotonic High-Resolution Timers** (`CLOCK_MONOTONIC_RAW`).
*   **Isochronous Scheduling**: We enforce strict time slices for the Grid loops to ensure the visualization runs at a locked framerate (e.g., 60Hz = 16.66ms deadline).

## 3. Computer Science Methods
*   **Earliest Deadline First (EDF)**: A dynamic priority scheduling algorithm used in the NPU Coordinator. Tasks closer to their deadline (e.g., "Motion Stop Signal") get immediate NPU access.
*   **Resource Accomodation**: The system "samples" resolution (Dynamic Scaling) to fit the compute workload within the available time budget.
