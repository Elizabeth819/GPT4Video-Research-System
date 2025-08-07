#!/usr/bin/env python3
import os
import cv2
from moviepy.editor import VideoFileClip

# Video specifications
videos = [
    {"file": "images_1_003.avi", "center": 2.0, "name": "sample1"},
    {"file": "images_1_006.avi", "center": 6.0, "name": "sample2"}, 
    {"file": "images_1_008.avi", "center": 2.0, "name": "sample3"}
]

base_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto"
video_dir = f"{base_path}/DADA-2000-videos"
output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples"

print("Starting ghost probing frame extraction...")

for video in videos:
    video_path = f"{video_dir}/{video['file']}"
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        continue
        
    print(f"Processing {video['file']}...")
    
    try:
        with VideoFileClip(video_path) as clip:
            print(f"Video duration: {clip.duration}s")
            
            # Extract 3 frames: before, during, after
            timestamps = [
                (video['center'] - 0.5, "before"),
                (video['center'], "during"), 
                (video['center'] + 0.5, "after")
            ]
            
            for timestamp, phase in timestamps:
                if timestamp <= clip.duration and timestamp >= 0:
                    frame = clip.get_frame(timestamp)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    output_file = f"{output_dir}/ghost_probing_{video['name']}_{phase}.jpg"
                    cv2.imwrite(output_file, frame_bgr)
                    print(f"  {phase} ({timestamp}s) -> {output_file}")
                else:
                    print(f"  Skipping {phase} ({timestamp}s) - out of range")
                    
    except Exception as e:
        print(f"Error processing {video['file']}: {e}")

print("Extraction complete!")

# List generated files
print("\nGenerated files:")
for f in os.listdir(output_dir):
    if f.startswith("ghost_probing_") and f.endswith(".jpg"):
        print(f"  {f}")