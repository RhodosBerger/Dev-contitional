# Active Neural ASCII Compositor (v1.0)
**Framework:** Gamesa Cortex V2 / FANUC RISE
**Module:** `neural_art_engine.py`

This engine transforms real-world images into detailed ASCII compositions by simulating machine vision feature extraction.

## Features
- **Image Input:** Process any JPG/PNG.
- **Directional Rendering:** Uses Convolutional Kernels (Sobel) to detect line angles (`|`, `/`, `-`, `\`) instead of just brightness.
- **Multiple Modes:**
    - `standard`: High-quality density mapping.
    - `edge`: Technical blueprint style (line detection).
    - `cyberpunk`: High-contrast, glitchy aesthetic.
    - `retro`: Scanline artifacts.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1.  **Run with Sample (Generative Mode):**
    ```bash
    python3 neural_art_engine.py --mode edge
    ```
    *(Generates a fractal noise pattern if no input provided)*

2.  **Run with Real Image:**
    ```bash
    python3 neural_art_engine.py --input path/to/image.jpg --width 120 --mode cyberpunk
    ```

3.  **Run with Output File:**
    ```bash
    python3 neural_art_engine.py --input image.jpg --output result.txt
    ```

## Logic
The engine uses `numpy` to compute gradient magnitudes (brightness changes) and gradient directions (angles). It maps these angles to directional ASCII characters, creating a "sketched" look.
