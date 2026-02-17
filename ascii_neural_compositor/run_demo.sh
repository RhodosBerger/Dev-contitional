#!/bin/bash
# Neural Compositor Demo Script
# Runs the Generative ASCII Engine in 3 modes

echo ">> NEURAL COMPOSITOR: Generative ASCII Demo (Synthetic Reality)"

echo "1. Generating Structural Blueprint (Edge Mode)..."
python3 neural_art_engine.py --mode edge --output output_renders/blueprint.txt
echo "   Saved to output_renders/blueprint.txt"

echo "2. Generating Texture Dream (Standard Mode)..."
python3 neural_art_engine.py --mode standard --output output_renders/texture.txt
echo "   Saved to output_renders/texture.txt"

echo "3. Generating Cyberpunk Glitch (Cyberpunk Mode)..."
python3 neural_art_engine.py --mode cyberpunk --output output_renders/glitch.txt
echo "   Saved to output_renders/glitch.txt"

echo ">> Done. Check 'output_renders/' for results."
