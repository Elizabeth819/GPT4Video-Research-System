#!/usr/bin/env python3
"""Manual ghost probing frame extraction"""

import os
import sys

def run_extraction():
    """Execute frame extraction with error handling"""
    
    # Set up environment
    project_root = "/Users/wanmeng/repository/GPT4Video-cobra-auto"
    sys.path.insert(0, project_root)
    
    print("GHOST PROBING FRAME EXTRACTION")
    print("=" * 40)
    
    # Import libraries with error handling
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not available")
        return []
    
    try:
        from moviepy.editor import VideoFileClip
        print("✓ MoviePy available")
    except ImportError:
        print("❌ MoviePy not available")
        return []
    
    # Paths
    video_dir = f"{project_root}/DADA-2000-videos"
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples"
    
    # Ensure output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Videos to process
    videos = [
        ("images_1_003.avi", 2.0, "sample1"),
        ("images_1_006.avi", 6.0, "sample2"), 
        ("images_1_008.avi", 2.0, "sample3")
    ]
    
    extracted_files = []
    
    for video_name, event_time, sample_id in videos:
        video_path = os.path.join(video_dir, video_name)
        print(f"\nProcessing {video_name}...")
        
        if not os.path.exists(video_path):
            print(f"  ❌ Not found: {video_path}")
            continue
            
        try:
            # Load video
            clip = VideoFileClip(video_path)
            print(f"  Duration: {clip.duration:.2f}s")
            
            # Extract 3 frames
            for offset, phase in [(-0.5, "before"), (0.0, "during"), (0.5, "after")]:
                timestamp = event_time + offset
                
                if 0 <= timestamp <= clip.duration:
                    try:
                        frame = clip.get_frame(timestamp)
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        filename = f"ghost_probing_{sample_id}_{phase}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        
                        if cv2.imwrite(filepath, frame_bgr):
                            size = os.path.getsize(filepath)
                            print(f"  ✓ {phase} ({timestamp:.1f}s): {filename} ({size:,} bytes)")
                            extracted_files.append((filename, filepath, size))
                        else:
                            print(f"  ❌ Failed to save {phase} frame")
                    except Exception as e:
                        print(f"  ❌ Frame error ({phase}): {e}")
                else:
                    print(f"  ⚠️ {phase} ({timestamp:.1f}s) out of bounds")
            
            clip.close()
            
        except Exception as e:
            print(f"  ❌ Video error: {e}")
    
    return extracted_files

# Execute extraction
print("Starting manual extraction...")
files = run_extraction()

print(f"\n{'='*40}")
print("RESULTS")
print(f"{'='*40}")
print(f"Files extracted: {len(files)}")

for filename, filepath, size in files:
    print(f"✓ {filename} ({size:,} bytes)")

if files:
    print(f"\nOutput directory:")
    print(f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples/")
    print(f"\nFrame naming convention:")
    print(f"  ghost_probing_sample1_before.jpg  (images_1_003.avi at 1.5s)")
    print(f"  ghost_probing_sample1_during.jpg  (images_1_003.avi at 2.0s)")
    print(f"  ghost_probing_sample1_after.jpg   (images_1_003.avi at 2.5s)")
    print(f"  ghost_probing_sample2_before.jpg  (images_1_006.avi at 5.5s)")
    print(f"  ghost_probing_sample2_during.jpg  (images_1_006.avi at 6.0s)")
    print(f"  ghost_probing_sample2_after.jpg   (images_1_006.avi at 6.5s)")
    print(f"  ghost_probing_sample3_before.jpg  (images_1_008.avi at 1.5s)")
    print(f"  ghost_probing_sample3_during.jpg  (images_1_008.avi at 2.0s)")
    print(f"  ghost_probing_sample3_after.jpg   (images_1_008.avi at 2.5s)")
    
    print(f"\nSequence explanation:")
    print(f"• BEFORE: Normal scene with person hidden behind obstruction")
    print(f"• DURING: Critical moment - person emerging from behind obstruction")
    print(f"• AFTER: Dangerous situation - person now in vehicle's path")
    print(f"\nThese visual sequences demonstrate the complete ghost probing")
    print(f"pattern for multimodal few-shot learning applications.")
else:
    print("❌ No files extracted successfully")

print("Extraction process complete.")