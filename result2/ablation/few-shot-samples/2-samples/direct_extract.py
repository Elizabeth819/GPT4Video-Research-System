#!/usr/bin/env python3
"""
Direct frame extraction for ghost probing videos
"""

import cv2
import os
from pathlib import Path

def extract_ghost_probing_frames():
    """Extract frames directly using OpenCV"""
    
    # Base paths
    base_dir = Path("/Users/wanmeng/repository/GPT4Video-cobra-auto")
    video_dir = base_dir / "DADA-2000-videos"
    output_dir = base_dir / "result2" / "ablation" / "few-shot-samples" / "2-samples" / "multimodal_fsl_frames"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Video configurations: (filename, ghost_probing_timestamp, sequence_timestamps)
    video_configs = [
        {
            "filename": "images_1_002.avi",
            "ghost_probing_time": 5,
            "description": "Male child emerges from behind parked vehicle on left side",
            "timestamps": [4, 5, 6]  # before, during, after
        },
        {
            "filename": "images_1_003.avi", 
            "ghost_probing_time": 2,
            "description": "Male pedestrian emerges from behind parked black car on left side",
            "timestamps": [1, 2, 3]  # before, during, after
        },
        {
            "filename": "images_1_008.avi",
            "ghost_probing_time": 3,
            "description": "Male pedestrian emerges from behind parked white truck on left side", 
            "timestamps": [2, 3, 4]  # before, during, after
        }
    ]
    
    print("=" * 60)
    print("EXTRACTING GHOST PROBING FRAMES FOR MULTIMODAL FEW-SHOT LEARNING")
    print("=" * 60)
    
    total_extracted = 0
    
    for config in video_configs:
        print(f"\nProcessing {config['filename']}...")
        print(f"Ghost probing at {config['ghost_probing_time']}s: {config['description']}")
        
        video_path = video_dir / config["filename"]
        
        if not video_path.exists():
            print(f"ERROR: Video file not found: {video_path}")
            continue
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"ERROR: Could not open video {video_path}")
            continue
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"  FPS: {fps:.2f}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        
        if fps <= 0:
            print(f"  ERROR: Invalid FPS ({fps})")
            cap.release()
            continue
        
        video_name = config["filename"].replace(".avi", "")
        success_count = 0
        
        for timestamp in config["timestamps"]:
            if timestamp > duration:
                print(f"  WARNING: Timestamp {timestamp}s exceeds video duration {duration:.2f}s")
                continue
                
            # Calculate frame number
            frame_number = int(timestamp * fps)
            
            # Seek to the frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # Create filename with timestamp
                frame_filename = f"{video_name}_frame_at_{timestamp}s.jpg"
                frame_path = output_dir / frame_filename
                
                # Save frame with high quality
                success = cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                if success:
                    print(f"  ✓ Extracted frame at {timestamp}s -> {frame_filename}")
                    success_count += 1
                else:
                    print(f"  ✗ Failed to save frame at {timestamp}s")
            else:
                print(f"  ✗ Error: Could not extract frame at {timestamp}s")
        
        cap.release()
        
        if success_count > 0:
            total_extracted += success_count
            print(f"✓ Successfully extracted {success_count} frames from {config['filename']}")
        else:
            print(f"✗ Failed to extract frames from {config['filename']}")
    
    print(f"\n" + "=" * 60)
    print(f"EXTRACTION COMPLETE: {total_extracted} frames extracted")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # List extracted files
    print("\nExtracted files:")
    if output_dir.exists():
        for frame_file in sorted(output_dir.glob("*.jpg")):
            file_size = frame_file.stat().st_size
            print(f"  - {frame_file.name} ({file_size} bytes)")
    
    return total_extracted

if __name__ == "__main__":
    try:
        total_frames = extract_ghost_probing_frames()
        print(f"\n🎉 Extraction completed successfully! Total frames: {total_frames}")
    except Exception as e:
        print(f"\n❌ Extraction failed with error: {e}")
        import traceback
        traceback.print_exc()