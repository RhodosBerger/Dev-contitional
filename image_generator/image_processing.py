#!/usr/bin/env python3
"""
NEURO-VISUAL ASCII COMPOSITOR v2.0
Framework: Gamesa Cortex V2 / FANUC RISE
Module: Generative Neural Engine

This script transforms visual reality (Real Images) into semantic ASCII structures using
Convolutional Kernel heuristics (Sobel Edge Detection) and Density Mapping.

Usage:
    python3 neural_art_engine.py --input <path/to/image.jpg> --mode <MODE>

Modes:
    - standard:   High-fidelity density mapping (brightness -> char)
    - edge:       Technical blueprint style (gradient magnitude -> char)
    - cyberpunk:  High-contrast, glitch aesthetic
    - retro:      Scanline simulation
    - sketch:     Directional edge mapping (Sobel Angle -> line char)

Author: Dušan Kopecký (Krystal-Stack Framework)
Date: Feb 17, 2026
"""

import argparse
import sys
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import math
import random

# --- Character Sets ---
DENSITY_CHARS = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
EDGE_CHARS = " .:-=+*#%@"
CYBERPUNK_CHARS = " ░▒▓█"
RETRO_CHARS = " .:-=+*#%@"
DIRECTIONAL_CHARS = {
    'vertical': "|",
    'horizontal': "-",
    'diag_ur': "/",
    'diag_dr': "\\",
    'none': " "
}

def load_image(path=None, width=100, preset=None):
    """
    Loads an image from path. If None, generates a pattern based on Preset.
    Presets mimic Pexels categories: 'nature', 'tech', 'architecture'.
    """
    if path:
        try:
            img = Image.open(path).convert("L")
        except Exception as e:
            print(f"Error loading image: {e}")
            sys.exit(1)
    else:
        print(f">> Generating Synthetic Input (Preset: {preset or 'Fractal'})...")
        width_px = width * 8
        height_px = int(width * 4.5)
        
        if preset == 'tech':
            # Grid lines and circuitry
            data = np.zeros((height_px, width_px), dtype=np.uint8)
            # Draw grid
            data[::50, :] = 200
            data[:, ::50] = 200
            # Random blocks
            for _ in range(20):
                x, y = np.random.randint(0, width_px-100), np.random.randint(0, height_px-100)
                data[y:y+100, x:x+100] = 255
            img = Image.fromarray(data, mode='L')
            img = img.filter(ImageFilter.GaussianBlur(radius=1))
            
        elif preset == 'nature':
            # Perlin-ish noise (Simulated with massive blur)
            data = np.random.randint(0, 255, (height_px, width_px), dtype=np.uint8)
            img = Image.fromarray(data, mode='L')
            img = img.filter(ImageFilter.GaussianBlur(radius=10)) # Soft clouds
            
        elif preset == 'architecture':
            # Hard edges, gradient sky
            data = np.zeros((height_px, width_px), dtype=np.uint8)
            # Gradient background
            for i in range(height_px):
                data[i, :] = int((i / height_px) * 255)
            # Buildings (Black blocks)
            for _ in range(5):
                w = np.random.randint(50, 200)
                h = np.random.randint(100, height_px)
                x = np.random.randint(0, width_px - w)
                data[height_px-h:, x:x+w] = 20
            img = Image.fromarray(data, mode='L')
            
        else:
            # Default Fractal Noise
            data = np.random.randint(0, 255, (height_px, width_px), dtype=np.uint8)
            img = Image.fromarray(data, mode='L')
            img = img.filter(ImageFilter.GaussianBlur(radius=random.randint(2, 5)))

    # Resize while maintaining aspect ratio (approx 0.55 char aspect ratio)
    aspect_ratio = img.height / img.width
    new_height = int(aspect_ratio * width * 0.55)
    img = img.resize((width, new_height), Image.Resampling.LANCZOS)
    return img

def apply_sobel_kernels(img_array):
    """
    Simulates a Convolutional Layer (Edge Detection).
    """
    # Simple Sobel implementation using numpy slicing (for speed/portability)
    # Kernel X: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    # Kernel Y: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    
    # Pad image to handle borders
    padded = np.pad(img_array, ((1, 1), (1, 1)), mode='edge').astype(np.float32)
    
    gx = (padded[:-2, 2:] - padded[:-2, :-2] +
          2*padded[1:-1, 2:] - 2*padded[1:-1, :-2] +
          padded[2:, 2:] - padded[2:, :-2])
          
    gy = (padded[2:, :-2] - padded[:-2, :-2] +
          2*padded[2:, 1:-1] - 2*padded[:-2, 1:-1] +
          padded[2:, 2:] - padded[:-2, 2:])
          
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) # Returns radians
    
    # Normalize magnitude to 0-255
    magnitude = (magnitude / magnitude.max()) * 255.0
    
    return magnitude, direction

def map_density(pixel_val, char_set):
    """Maps 0-255 brightness to character index."""
    idx = int((pixel_val / 255.0) * (len(char_set) - 1))
    return char_set[idx]

def map_direction(magnitude, angle, threshold=30):
    """Maps edge direction to ASCII line characters."""
    if magnitude < threshold:
        return " "
    
    # Convert angle to degrees (0-180 for line symmetry)
    deg = math.degrees(angle) % 180
    
    if 67.5 <= deg < 112.5:
        return "|"
    elif 22.5 <= deg < 67.5:
        return "/"
    elif 112.5 <= deg < 157.5:
        return "\\"
    else:
        return "-"

def render_ascii(img, mode='standard'):
    """
    Main render loop.
    """
    pixels = np.array(img).astype(np.float32)
    width, height = img.size
    output = []
    
    # Pre-calculate features (Simulated Neural Layers)
    sobel_mag, sobel_dir = apply_sobel_kernels(pixels)
    
    print(f">> Processing Image: {width}x{height} | Mode: {mode}")

    for y in range(height):
        line = ""
        for x in range(width):
            pixel = pixels[y, x]
            
            if mode == 'standard':
                # Density Mapping (Brightness)
                # Invert logic: Dark pixel = Dense Char (usually better on white bg)
                # Here assuming dark terminal: Bright pixel = Dense Char
                char = map_density(pixel, DENSITY_CHARS)
                
            elif mode == 'edge':
                 # Edge Magnitude (Structure)
                 mag = sobel_mag[y, x]
                 char = map_density(mag, EDGE_CHARS)
                 
            elif mode == 'cyberpunk':
                # Quantized Density (High Contrast)
                q_pixel = (int(pixel / 64) * 64) # Posterize
                char = map_density(q_pixel, CYBERPUNK_CHARS)
                
            elif mode == 'retro':
                 # Scanlines (Every 2nd line is dark)
                 if y % 2 == 0:
                     char = map_density(pixel * 0.5, RETRO_CHARS)
                 else:
                     char = map_density(pixel, RETRO_CHARS)
                     
            elif mode == 'sketch':
                # Directional Mapping (Vector Simulation)
                mag = sobel_mag[y, x]
                angle = sobel_dir[y, x]
                char = map_direction(mag, angle, threshold=40)
                
            else:
                char = "?"
                
            line += char
        output.append(line)
        
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="Neuro-Visual ASCII Compositor Engine")
    parser.add_argument('--input', type=str, help="Path to input image file (jpg/png)")
    parser.add_argument('--output', type=str, help="Path to save output text file")
    parser.add_argument('--width', type=int, default=100, help="Output width in characters")
    parser.add_argument('--mode', type=str, default='standard', 
                        choices=['standard', 'edge', 'cyberpunk', 'retro', 'sketch'],
                        help="Rendering style (model interpretation)")
    
    args = parser.parse_args()
    
    # 1. Load Reality (Image Input)
    img = load_image(args.input, args.width)
    
    # 2. Transduce (Render)
    ascii_art = render_ascii(img, args.mode)
    
    # 3. Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(ascii_art)
        print(f">> Saved composition to {args.output}")
    else:
        print("\n" + ascii_art + "\n")

if __name__ == "__main__":
    main()
