# Neural Compositor: Implementation Guide

**Framework:** Gamesa Cortex V2 / FANUC RISE
**Module:** `ascii_neural_compositor`
**Project:** Generative ASCII from Visual Reality

This guide explains how to integrate the neural compositor into your workflow to create multiple ASCII interpretations from real images.

## 1. Prerequisites (Setup)
Ensure you have the required Python libraries.

```bash
cd ascii_neural_compositor
pip install -r requirements.txt
```

## 2. Using the Engine (The Core Paradigm)
The engine is designed to take *any* visual input and produce a semantic interpretation.

### Scenario A: Generative Art (No Input)
If you don't provide an image, the engine generates a synthetic dream-state pattern using fractal noise logic.

```bash
python3 neural_art_engine.py --mode edge
python3 neural_art_engine.py --mode cyberpunk
```

### Scenario B: Transduction (Real Input)
To convert a photo (`my_photo.jpg`), run the following commands to get **three distinct interpretations**:

1.  **The Blueprint (Structure):** Focuses on edges and architecture.
    ```bash
    python3 neural_art_engine.py --input my_photo.jpg --mode edge --output structural.txt
    ```

2.  **The Dream (Texture):** Focuses on shading and organic detail.
    ```bash
    python3 neural_art_engine.py --input my_photo.jpg --mode standard --output texture.txt
    ```

3.  **The Simulation (Cyberpunk):** High-contrast, glitch aesthetic.
    ```bash
    python3 neural_art_engine.py --input my_photo.jpg --mode cyberpunk --output glitch.txt
    ```

## 3. Advanced Integration
To use this as a library within another Python script:

```python
from neural_art_engine import load_image, render_ascii

# Load an image object (PIL)
img = load_image("path/to/image.jpg", width=120)

# Generate ASCII string
ascii_data = render_ascii(img, mode='sketch')

# Save or Display
print(ascii_data)
```

## 4. Next Steps
- Experiment with Kernel sizes in `neural_art_engine.py` (line 65) to change detail detection.
- Add new `MODES` by defining custom character sets in `neural_art_engine.py`.
