# Neuro-Visual Transduction Architecture

**System:** Neural Compositor Paradigm  
**Flow:** Reality → Convolution → Semantic Structure

This diagram illustrates the transformation of visual signals into ASCII meaning.

```mermaid
graph TD
    A[REAL WORLD IMAGE] -->|Input Signal| B(Pre-Processing Retina)
    B -->|Grayscale + Norm| C{NEURAL LAYERS}
    
    C -->|Kernel 1: Sobel X/Y| D[Edge Detection Map]
    C -->|Kernel 2: Laplacian| E[Texture Density Map]
    C -->|Kernel 3: Quantize| F[High Contrast Map]
    
    D -->|Directional| G[Structure Synthesis]
    E -->|Intensity| H[Shading Synthesis]
    F -->|Blocky| I[Glitch Synthesis]
    
    G --> J{ASCII MAPPING ENGINE}
    H --> J
    I --> J
    
    J -->|Char Selection| K[FINAL COMPOSITION]
    
    style A fill:#f9f,stroke:#333,stroke-width:4px
    style C fill:#ccf,stroke:#333,stroke-width:2px
    style K fill:#9f9,stroke:#333,stroke-width:4px
```

## Layer Breakdown

1.  **Input Reality:** A standard RGB bitmap (photograph or video frame).
2.  **Retina (Pre-processing):** Converts color space to luminance (grayscale), removes high-frequency noise that would translate to "character jitter."
3.  **Neural Layers (Convolution):**
    *   **Sobel Filters:** Calculate the gradient vector at every point. This tells us *which way* a line is pointing.
    *   **Laplacian Filters:** Calculate the second derivative (rate of change). This identifies fine texture (hair, grass, fabric).
4.  **Synthesis:**
    *   **Sketch Mode:** Uses only the *Direction* data (Sobel). Draws lines.
    *   **Standard Mode:** Uses *Intensity* data (Laplacian). Shades areas.
    *   **Cyberpunk Mode:** Uses *Quantized* data (Posterization). Creates blocks.
