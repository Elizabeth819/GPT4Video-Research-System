#!/usr/bin/env python3
"""Test imports and basic functionality"""

import sys
import os

print("Python version:", sys.version)
print("Current working directory:", os.getcwd())

# Test opencv import
try:
    import cv2
    print("✓ OpenCV imported successfully, version:", cv2.__version__)
except ImportError as e:
    print("❌ OpenCV import failed:", e)

# Test moviepy import
try:
    from moviepy.editor import VideoFileClip
    print("✓ MoviePy imported successfully")
except ImportError as e:
    print("❌ MoviePy import failed:", e)

# Test numpy import
try:
    import numpy as np
    print("✓ NumPy imported successfully, version:", np.__version__)
except ImportError as e:
    print("❌ NumPy import failed:", e)

# Check video directory
video_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos"
print(f"\nVideo directory exists: {os.path.exists(video_dir)}")

if os.path.exists(video_dir):
    # Check for specific videos
    target_videos = ["images_1_003.avi", "images_1_006.avi", "images_1_008.avi"]
    for video in target_videos:
        video_path = os.path.join(video_dir, video)
        exists = os.path.exists(video_path)
        print(f"  {video}: {'✓' if exists else '❌'}")

# Check output directory
output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples"
print(f"\nOutput directory exists: {os.path.exists(output_dir)}")
print(f"Output directory writable: {os.access(output_dir, os.W_OK)}")

print("\nEnvironment test complete!")