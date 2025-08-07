#!/usr/bin/env python3
"""
Frame extraction completed. This file serves as documentation and verification.
"""

# Execute the frame extraction directly
import cv2
import os
from pathlib import Path

# Base paths
base_dir = Path("/Users/wanmeng/repository/GPT4Video-cobra-auto")
video_dir = base_dir / "DADA-2000-videos"
output_dir = base_dir / "result2" / "ablation" / "few-shot-samples" / "2-samples" / "multimodal_fsl_frames"

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Video configurations
video_configs = [
    {
        "filename": "images_1_002.avi",
        "ghost_probing_time": 5,
        "description": "Male child emerges from behind parked vehicle on left side",
        "timestamps": [4, 5, 6]
    },
    {
        "filename": "images_1_003.avi", 
        "ghost_probing_time": 2,
        "description": "Male pedestrian emerges from behind parked black car on left side",
        "timestamps": [1, 2, 3]
    },
    {
        "filename": "images_1_008.avi",
        "ghost_probing_time": 3,
        "description": "Male pedestrian emerges from behind parked white truck on left side", 
        "timestamps": [2, 3, 4]
    }
]

print("Executing frame extraction...")

extraction_results = []

for config in video_configs:
    video_path = video_dir / config["filename"]
    video_name = config["filename"].replace(".avi", "")
    
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            for timestamp in config["timestamps"]:
                if timestamp <= duration and fps > 0:
                    frame_number = int(timestamp * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()
                    
                    if ret:
                        frame_filename = f"{video_name}_frame_at_{timestamp}s.jpg"
                        frame_path = output_dir / frame_filename
                        
                        if cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            extraction_results.append({
                                "video": config["filename"],
                                "timestamp": timestamp,
                                "ghost_probing_time": config["ghost_probing_time"],
                                "description": config["description"],
                                "filename": frame_filename,
                                "path": str(frame_path)
                            })
            
            cap.release()

print(f"Extraction completed. {len(extraction_results)} frames extracted.")

# Document the results
extraction_summary = {
    "total_frames": len(extraction_results),
    "videos_processed": len(video_configs),
    "output_directory": str(output_dir),
    "frames": extraction_results
}

print("\nExtraction Summary:")
print(f"Total frames extracted: {extraction_summary['total_frames']}")
print(f"Videos processed: {extraction_summary['videos_processed']}")
print(f"Output directory: {extraction_summary['output_directory']}")

print("\nExtracted frames:")
for result in extraction_results:
    timestamp_type = "DURING" if result["timestamp"] == result["ghost_probing_time"] else ("BEFORE" if result["timestamp"] < result["ghost_probing_time"] else "AFTER")
    print(f"  - {result['filename']} ({timestamp_type} ghost probing)")