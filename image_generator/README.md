# Image Generator Suite

**Engine:** `image_processing.py` (Neural Compositor)
**Framework:** Gamesa Cortex V2 / Neuro-Visual Transduction
**Status:** Standalone Deployment

This folder contains the complete, self-contained **Augmented Reality Image Processing System**.

## 1. Components
*   **`image_processing.py`**: The core Generative AI engine. Uses Sobel/Laplacian kernels to turn images into ASCII.
*   **`system/`**: Modality subsystems for AR integration:
    *   `thermal_monitor.py`: Hardware governance.
    *   `audio_reactor.py`: Sound-reactive modulation.
    *   `video_processor.py`: Camera/Video stream handling.
    *   `main_ar_system.py`: The Master AR integration script.

## 2. Installation
```bash
pip install -r requirements.txt
```

## 3. Usage (The Processor)
Run the core engine in generative mode:
```bash
python3 image_processing.py --mode cyberpunk
```

Run with input image:
```bash
python3 image_processing.py --input my_photo.jpg --mode edge
```

## 4. Usage (The AR Experience)
Launch the full augmented reality system (requires webcam or synthetic mode):
```bash
python3 system/main_ar_system.py
```
