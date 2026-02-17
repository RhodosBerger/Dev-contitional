# Spectral Roadmap: Neuro-Visual Transduction System

**Project:** Gamesa Cortex V2 / Neural Compositor  
**Status:** Active Prototype (Phase 1 Complete)  
**Date:** Feb 17, 2026  
**Author:** Dušan Kopecký

This roadmap organizes all system features into a **Spectral Hierarchy**, guiding development from low-level kernel operations to high-level user interactions.

---

## 🟣 Violet Band: Kernel & Governance (The Foundation)
*Features related to resource management, hardware abstraction, and stability.*

### ✅ Phase 1: Foundation (Complete)
- [x] **Economic Governor Integration:** Implemented a credit-based budget system to regulate compute usage (`economic_governor.py`).
- [x] **Mock Hardware Hooks:** Simulators for OpenVINO and Vulkan subsystems (`vulkan_learner.py`).

### 🚧 Phase 2: Deep Integration (Q2 2026)
- [ ] **Real Vulkan Interception:** Hook into `libvulkan.so` to capture actual draw calls from running games/apps.
- [ ] **Thermal Throttling:** Connect Economic Governor to system temperature sensors (`lm-sensors`).

### 🔮 Phase 3: Autonomous Kernel (2027)
- [ ] **Self-Repairing Hooks:** Kernel modules that re-attach to game processes if they crash or restart.

---

## 🔵 Blue Band: Cognition & Learning (The Brain)
*Features related to analysis, optimization, and model training.*

### ✅ Phase 1: Foundation (Complete)
- [x] **OpenVINO Simulation:** Accelerator bridge for model optimization (`openvino_accelerator.py`).
- [x] **Vulkan Introspection Strategy:** Logic for learning style from geometry metadata (`vulkan_learner.py`).

### 🚧 Phase 2: Deep Integration (Q2 2026)
- [ ] **Live Training Loop:** Feed captured Vulkan frames continuously into the `NeuralStyleModel` during gameplay.
- [ ] **NPU Deployment:** Compile `sobel_kernel` to run explicitly on Neural Processing Units (NPU).

### 🔮 Phase 3: Augmented Intelligence (2027)
- [ ] **Predictive Rendering:** AI guesses the next frame's composition before the GPU renders it, reducing latency to zero.

---

## 🟢 Green Band: Synthesis & Generation (The Eye)
*Features related to image processing, ASCII conversion, and style transfer.*

### ✅ Phase 1: Foundation (Complete)
- [x] **Neural Art Engine:** Generative script with Sobel/Laplacian kernels (`neural_art_engine.py`).
- [x] **Preset Synthesis:** Procedural generation of "Nature", "Tech", "Architecture" datasets.
- [x] **Dataset Compiler:** Batch processing pipeline (`dataset_compiler.py`).

### 🚧 Phase 2: Deep Integration (Q2 2026)
- [ ] **Video Stream Transduction:** Process `mp4` or Webcam input in real-time (30 FPS).
- [ ] **Custom Style Weights:** Allow users to define their own "Style DNA" (e.g., 70% Cyberpunk, 30% Sketch).

### 🔮 Phase 3: Holographic ASCII (2027)
- [ ] **3D ASCII Clouds:** Generating voxel-based ASCII point clouds for VR headsets.

---

## 🔴 Red Band: Interaction & Feedback (The Hand)
*Features related to user interface, haptics, and sensory feedback.*

### ✅ Phase 1: Foundation (Complete)
- [x] **Conceptual Framework:** Research on Haptic-Text Synesthesia and Audio-Reactivity (`AUGMENTED_REALITY_RESEARCH.md`).

### 🚧 Phase 2: Deep Integration (Q2 2026)
- [ ] **Audio-Reactive Mode:** Modulate character density based on microphone input (Bass = Blocks, Treble = Dots).
- [ ] **Glitch Backpressure UI:** Visual artifacts (red text, tearing) when the Economic Governor denies budget.

### 🔮 Phase 3: Total Immersion (2027)
- [ ] **Haptic Controller Support:** Map ASCII texture density to controller vibration motors.
- [ ] **Brain-Computer Interface (BCI):** Modulate style based on user focus/stress levels (EEG).

---

## 🎯 Summary of Next Steps

1.  **Immediate Priority:** Connect `neural_art_engine.py` to a real **Webcam/Video Feed** to prove real-time capability beyond static datasets.
2.  **Secondary Priority:** Implement the **Audio-Reactive** mode to demonstrate multi-modal input processing.
3.  **Long-Term:** Build the C++ **Vulkan Hook** library.
