#!/usr/bin/env python3
"""
Test frame extraction from ghost probing videos
"""

import cv2
import os
from pathlib import Path

def test_single_video():
    # Paths
    video_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos/images_1_002.avi"
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples/multimodal_fsl_frames"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Testing video: {video_path}")
    print(f"Output directory: {output_dir}")
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video")
        cap.release()
        return False
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video properties:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f}s")
    
    if fps <= 0 or total_frames <= 0:
        print("ERROR: Invalid video properties")
        cap.release()
        return False
    
    # Test extracting frame at 5s (ghost probing timestamp)
    timestamp = 5
    frame_number = int(timestamp * fps)
    
    print(f"Extracting frame at {timestamp}s (frame {frame_number})")
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if ret:
        # Save frame
        output_path = os.path.join(output_dir, "test_images_1_002_frame_at_5s.jpg")
        success = cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            print(f"SUCCESS: Frame saved to {output_path}")
            
            # Get file size to verify
            file_size = os.path.getsize(output_path)
            print(f"File size: {file_size} bytes")
            
            cap.release()
            return True
        else:
            print("ERROR: Failed to save frame")
    else:
        print("ERROR: Failed to read frame")
    
    cap.release()
    return False

if __name__ == "__main__":
    print("Testing frame extraction...")
    success = test_single_video()
    if success:
        print("✓ Test passed!")
    else:
        print("✗ Test failed!")