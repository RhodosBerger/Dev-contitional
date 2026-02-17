#!/usr/bin/env python3
"""
GREEN BAND: Video Processing System
Framework: Gamesa Cortex V2 / Neural Compositor
Module: visual_acquisition

This module handles video input streams (Webcam, File, or Synthetic).
It integrates the Neural Art Engine to process frames in real-time.

Usage:
    processor = VideoProcessor(source=0) # Webcam
    ascii_frame = processor.get_next_frame(mode="cyberpunk")
"""

import sys
import os
import time
import numpy as np
from PIL import Image, ImageFilter
import cv2 # Try to use OpenCV if available
import logging

# Ensure access to Neural Engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neural_art_engine import render_ascii, load_image

class VideoProcessor:
    def __init__(self, source=0, use_synthetic=False):
        self.logger = logging.getLogger("VideoProcessor")
        self.use_synthetic = use_synthetic
        self.cap = None
        self.frame_count = 0
        
        if not use_synthetic:
            try:
                self.cap = cv2.VideoCapture(source)
                if not self.cap.isOpened():
                    self.logger.warning(f"Could not open video source {source}. Falling back to Synthetic.")
                    self.use_synthetic = True
            except ImportError:
                self.logger.warning("OpenCV not installed. Falling back to Synthetic.")
                self.use_synthetic = True
            except Exception as e:
                self.logger.error(f"Error opening camera: {e}")
                self.use_synthetic = True

    def get_next_frame(self, width=120, mode="standard", audio_meta=None):
        """
        Acquire frame, process with Neural Engine, return ASCII string.
        """
        self.frame_count += 1
        img = None
        
        if not self.use_synthetic and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB (OpenCV uses BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb).convert("L")
                
                # Resize specifically for ASCII (approx 0.55 aspect ratio)
                aspect_ratio = img.height / img.width
                new_height = int(aspect_ratio * width * 0.55)
                img = img.resize((width, new_height), Image.Resampling.NEAREST) # Speed optimization
            else:
                # End of video or error
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop
                return self.get_next_frame(width, mode, audio_meta)
        
        if img is None:
            # Synthetic Frame Generation (Moving Fractal Pattern)
            t = self.frame_count * 0.1
            size = (width * 8, int(width * 4.5))
            
            # Procedural Noise
            if audio_meta and audio_meta['beat']:
                # Flash/Glitch on beat
                noise = np.random.randint(50, 255, size, dtype=np.uint8)
            else:
                noise = np.random.randint(0, 150, size, dtype=np.uint8)
                
            img = Image.fromarray(noise, mode='L')
            if audio_meta:
                # Blur based on Treble
                blur = int(audio_meta['treble'] * 5)
                if blur > 0: img = img.filter(ImageFilter.GaussianBlur(radius=blur))

            # Resize
            aspect_ratio = img.height / img.width
            new_height = int(aspect_ratio * width * 0.55)
            img = img.resize((width, new_height), Image.Resampling.LANCZOS)

        # Transduction
        # Here we could inject 'audio_meta' into render_ascii if extended
        return render_ascii(img, mode=mode)

    def release(self):
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    proc = VideoProcessor(use_synthetic=True) # Force synthetic for testing
    try:
        while True:
            frame = proc.get_next_frame(mode="edge")
            print("\033[H" + frame) # Clear screen and print
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
