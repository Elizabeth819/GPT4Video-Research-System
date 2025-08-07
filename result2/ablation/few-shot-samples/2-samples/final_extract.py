#!/usr/bin/env python3
"""
Final ghost probing frame extraction script
This script extracts before/during/after frames from 3 videos for few-shot learning
"""

def main():
    import os
    import sys
    
    # Add project root to path
    project_root = "/Users/wanmeng/repository/GPT4Video-cobra-auto"
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print("GHOST PROBING FRAME EXTRACTION")
    print("=" * 50)
    
    # Import libraries
    print("Importing libraries...")
    try:
        import cv2
        from moviepy.editor import VideoFileClip
        import numpy as np
        print("✓ All libraries imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Define paths
    video_base = "/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos"
    output_base = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples"
    
    # Create output directory
    os.makedirs(output_base, exist_ok=True)
    
    # Video specifications
    videos = [
        {"name": "images_1_003.avi", "time": 2.0, "id": "sample1", "desc": "Ghost probing at 2s"},
        {"name": "images_1_006.avi", "time": 6.0, "id": "sample2", "desc": "Ghost probing at 6s"},
        {"name": "images_1_008.avi", "time": 2.0, "id": "sample3", "desc": "Ghost probing at 2s"}
    ]
    
    total_success = 0
    total_attempts = 0
    
    for video in videos:
        video_path = os.path.join(video_base, video["name"])
        
        print(f"\n{'='*30}")
        print(f"Processing: {video['name']}")
        print(f"Description: {video['desc']}")
        print(f"Path: {video_path}")
        
        if not os.path.exists(video_path):
            print("❌ Video file not found!")
            continue
        
        try:
            # Open video
            clip = VideoFileClip(video_path)
            duration = clip.duration
            print(f"✓ Video loaded (duration: {duration:.2f}s)")
            
            # Extract 3 frames: before, during, after
            center_time = video["time"]
            frames_to_extract = [
                (center_time - 0.5, "before"),
                (center_time, "during"),
                (center_time + 0.5, "after")
            ]
            
            for timestamp, phase in frames_to_extract:
                total_attempts += 1
                print(f"  Extracting {phase} frame at {timestamp:.1f}s...")
                
                if timestamp < 0 or timestamp > duration:
                    print(f"    ⚠️ Timestamp out of bounds (0-{duration:.2f}s)")
                    continue
                
                try:
                    # Get frame at timestamp
                    frame_rgb = clip.get_frame(timestamp)
                    
                    # Convert to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Generate filename
                    filename = f"ghost_probing_{video['id']}_{phase}.jpg"
                    filepath = os.path.join(output_base, filename)
                    
                    # Save frame
                    success = cv2.imwrite(filepath, frame_bgr)
                    
                    if success and os.path.exists(filepath):
                        file_size = os.path.getsize(filepath)
                        print(f"    ✓ Saved {filename} ({file_size:,} bytes)")
                        total_success += 1
                    else:
                        print(f"    ❌ Failed to save {filename}")
                
                except Exception as frame_err:
                    print(f"    ❌ Frame extraction error: {frame_err}")
            
            # Clean up video clip
            clip.close()
            
        except Exception as video_err:
            print(f"❌ Video processing error: {video_err}")
    
    # Final summary
    print(f"\n{'='*50}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*50}")
    print(f"Success rate: {total_success}/{total_attempts} frames")
    
    # List generated files
    output_files = []
    if os.path.exists(output_base):
        all_files = os.listdir(output_base)
        output_files = [f for f in all_files if f.startswith("ghost_probing_") and f.endswith(".jpg")]
        output_files.sort()
    
    if output_files:
        print(f"\nGenerated files ({len(output_files)}):")
        for filename in output_files:
            filepath = os.path.join(output_base, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"  {filename} ({size:,} bytes)")
        
        print(f"\nOutput directory: {output_base}")
        print("\nFrame sequence explanation:")
        print("Each sample contains 3 frames showing the ghost probing progression:")
        print("  • before: Normal scene with person hidden behind obstruction")
        print("  • during: Critical moment as person emerges from blind spot")
        print("  • after: Dangerous situation with person in vehicle's path")
        print("\nThese visual sequences demonstrate the complete ghost probing")
        print("pattern for multimodal few-shot learning in autonomous driving safety.")
        
        return True
    else:
        print("❌ No files were generated successfully")
        return False

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    print(f"\nScript completed with exit code: {exit_code}")
    exit(exit_code)