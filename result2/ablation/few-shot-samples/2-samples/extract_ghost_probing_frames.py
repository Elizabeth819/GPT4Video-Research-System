#!/usr/bin/env python3
"""
Extract ghost probing image sequences for multimodal few-shot learning.
Extracts 3 frames (before-during-after) from specified videos at target timestamps.
"""

import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

def extract_frame_at_timestamp(video_path, timestamp, output_path):
    """Extract a single frame at specified timestamp."""
    try:
        # Use moviepy for precise frame extraction
        with VideoFileClip(video_path) as clip:
            if timestamp > clip.duration:
                print(f"Warning: Timestamp {timestamp}s exceeds video duration {clip.duration}s for {video_path}")
                return False
                
            # Extract frame at timestamp
            frame = clip.get_frame(timestamp)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Save frame
            success = cv2.imwrite(output_path, frame_bgr)
            if success:
                print(f"Extracted frame at {timestamp}s -> {output_path}")
                return True
            else:
                print(f"Failed to save frame to {output_path}")
                return False
                
    except Exception as e:
        print(f"Error extracting frame from {video_path} at {timestamp}s: {e}")
        return False

def main():
    # Define base paths
    base_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto"
    video_dir = os.path.join(base_dir, "DADA-2000-videos")
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define video specifications for ghost probing sequences
    video_specs = [
        {
            "video_file": "images_1_003.avi",
            "center_timestamp": 2.0,
            "sample_name": "sample1",
            "description": "Ghost probing at 2s - person emerging from behind obstruction"
        },
        {
            "video_file": "images_1_006.avi", 
            "center_timestamp": 6.0,
            "sample_name": "sample2",
            "description": "Ghost probing at 6s - person emerging from behind obstruction"
        },
        {
            "video_file": "images_1_008.avi",
            "center_timestamp": 2.0,  # User specified 2s instead of 3s from ground truth
            "sample_name": "sample3", 
            "description": "Ghost probing at 2s - person emerging from behind obstruction"
        }
    ]
    
    # Extract frames for each video
    results = []
    for spec in video_specs:
        video_path = os.path.join(video_dir, spec["video_file"])
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"ERROR: Video not found: {video_path}")
            results.append({"video": spec["video_file"], "status": "video_not_found"})
            continue
            
        print(f"\nProcessing {spec['video_file']} - {spec['description']}")
        
        # Define timestamps: before (-0.5s), during (center), after (+0.5s)
        timestamps = {
            "before": spec["center_timestamp"] - 0.5,
            "during": spec["center_timestamp"], 
            "after": spec["center_timestamp"] + 0.5
        }
        
        sample_results = {"video": spec["video_file"], "frames": {}}
        
        # Extract each frame
        for phase, timestamp in timestamps.items():
            output_filename = f"ghost_probing_{spec['sample_name']}_{phase}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            success = extract_frame_at_timestamp(video_path, timestamp, output_path)
            sample_results["frames"][phase] = {
                "timestamp": timestamp,
                "output_path": output_path,
                "success": success
            }
            
        results.append(sample_results)
    
    # Print summary
    print("\n" + "="*60)
    print("GHOST PROBING FRAME EXTRACTION SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\nVideo: {result['video']}")
        if "frames" in result:
            for phase, frame_info in result["frames"].items():
                status = "SUCCESS" if frame_info["success"] else "FAILED"
                print(f"  {phase.capitalize()} ({frame_info['timestamp']}s): {status}")
                if frame_info["success"]:
                    print(f"    -> {frame_info['output_path']}")
        else:
            print(f"  Status: {result['status']}")
    
    print(f"\nOutput directory: {output_dir}")
    print("Frame naming convention:")
    print("  ghost_probing_sample1_before.jpg  (images_1_003.avi at 1.5s)")
    print("  ghost_probing_sample1_during.jpg  (images_1_003.avi at 2.0s)")
    print("  ghost_probing_sample1_after.jpg   (images_1_003.avi at 2.5s)")
    print("  ghost_probing_sample2_before.jpg  (images_1_006.avi at 5.5s)")
    print("  ghost_probing_sample2_during.jpg  (images_1_006.avi at 6.0s)")
    print("  ghost_probing_sample2_after.jpg   (images_1_006.avi at 6.5s)")
    print("  ghost_probing_sample3_before.jpg  (images_1_008.avi at 1.5s)")
    print("  ghost_probing_sample3_during.jpg  (images_1_008.avi at 2.0s)")
    print("  ghost_probing_sample3_after.jpg   (images_1_008.avi at 2.5s)")

if __name__ == "__main__":
    main()