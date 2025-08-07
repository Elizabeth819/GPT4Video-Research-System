#!/usr/bin/env python3
"""
Simple frame extraction without shell dependencies
"""

# First, let's try to import cv2 and verify it works
try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"Failed to import OpenCV: {e}")
    exit(1)

import os
from pathlib import Path

# Create a minimal frame extraction function
def extract_single_frame():
    """Extract a single frame to test the process"""
    
    video_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos/images_1_002.avi"
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples/extracted_frames"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Attempting to extract frame from: {video_path}")
    print(f"Output directory: {output_dir}")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"ERROR: Video file does not exist: {video_path}")
        return False
    
    # Try to open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("ERROR: Could not open video file")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Video properties:")
    print(f"  FPS: {fps}")
    print(f"  Frame count: {frame_count}")
    print(f"  Duration: {duration:.2f}s")
    
    # Extract frame at 5 seconds (ghost probing moment)
    timestamp = 5.0
    frame_number = int(timestamp * fps)
    
    print(f"Seeking to frame {frame_number} (at {timestamp}s)")
    
    # Seek to the specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    
    if not ret:
        print("ERROR: Could not read frame")
        cap.release()
        return False
    
    # Save the frame
    output_path = os.path.join(output_dir, "test_frame_5s.jpg")
    success = cv2.imwrite(output_path, frame)
    
    cap.release()
    
    if success:
        print(f"SUCCESS: Frame saved to {output_path}")
        
        # Check file size
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"File size: {size} bytes")
            return True
    else:
        print("ERROR: Failed to save frame")
    
    return False

# Execute the test
if __name__ == "__main__":
    print("Testing frame extraction...")
    result = extract_single_frame()
    if result:
        print("✓ Frame extraction test PASSED")
    else:
        print("✗ Frame extraction test FAILED")