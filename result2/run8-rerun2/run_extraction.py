#!/usr/bin/env python3
"""Direct execution of ghost probing frame extraction"""

import os
import sys

# Add project root to path
project_root = "/Users/wanmeng/repository/GPT4Video-cobra-auto"
sys.path.insert(0, project_root)

print("Starting Ghost Probing Frame Extraction...")

try:
    import cv2
    print("✅ OpenCV imported")
except ImportError:
    print("❌ OpenCV not available")
    sys.exit(1)

try:
    from moviepy.editor import VideoFileClip
    print("✅ MoviePy imported")
except ImportError:
    print("❌ MoviePy not available")  
    sys.exit(1)

# Execute extraction
def extract_frames():
    video_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos"
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/image-fewshot"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    videos = [
        ("images_1_003.avi", 2.0, "sample1"),
        ("images_1_006.avi", 6.0, "sample2"),
        ("images_1_008.avi", 2.0, "sample3")
    ]
    
    extracted = 0
    
    for video_file, event_time, sample_id in videos:
        video_path = os.path.join(video_dir, video_file)
        print(f"\nProcessing {video_file}...")
        
        if os.path.exists(video_path):
            try:
                clip = VideoFileClip(video_path)
                print(f"  Duration: {clip.duration:.2f}s")
                
                for delta, phase in [(-0.5, "before"), (0, "during"), (0.5, "after")]:
                    timestamp = event_time + delta
                    if 0 <= timestamp <= clip.duration:
                        frame = clip.get_frame(timestamp)
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        filename = f"ghost_probing_{sample_id}_{phase}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        
                        if cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            size = os.path.getsize(filepath)
                            print(f"  ✅ {phase}: {size:,} bytes")
                            extracted += 1
                
                clip.close()
            except Exception as e:
                print(f"  ❌ Error: {e}")
        else:
            print(f"  ❌ File not found")
    
    print(f"\nTotal extracted: {extracted} frames")
    return extracted > 0

if __name__ == "__main__":
    success = extract_frames()
    if success:
        print("✅ Extraction completed successfully!")
    else:
        print("❌ Extraction failed")
    sys.exit(0 if success else 1)