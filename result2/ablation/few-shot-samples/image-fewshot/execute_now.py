#!/usr/bin/env python3
"""
Direct execution of ghost probing frame extraction
This script attempts to extract 9 frames from 3 DADA-2000 videos
"""

import os
import sys

print("🎯 Ghost Probing Frame Extraction")
print("=" * 50)

# Test imports first
print("\n📦 Testing dependencies...")
dependencies_ok = True

try:
    import cv2
    print(f"✅ OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")
    dependencies_ok = False

try:
    from moviepy.editor import VideoFileClip
    print("✅ MoviePy imported")
except ImportError as e:
    print(f"❌ MoviePy import failed: {e}")
    dependencies_ok = False

if not dependencies_ok:
    print("\n❌ Missing dependencies. Cannot proceed.")
    sys.exit(1)

# Paths
video_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos"
output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/image-fewshot"

print(f"\n📂 Paths:")
print(f"  Video dir: {video_dir}")
print(f"  Output dir: {output_dir}")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Video configurations
videos_to_process = [
    ("images_1_003.avi", 2.0, "sample1", "Ghost probing at intersection"),
    ("images_1_006.avi", 6.0, "sample2", "Person emerges from building"),
    ("images_1_008.avi", 2.0, "sample3", "Blind spot emergence")
]

# Check video files exist
print(f"\n🔍 Checking video files:")
all_videos_exist = True
for video_file, _, _, desc in videos_to_process:
    video_path = os.path.join(video_dir, video_file)
    exists = os.path.exists(video_path)
    print(f"  {'✅' if exists else '❌'} {video_file}: {desc}")
    if exists:
        size = os.path.getsize(video_path)
        print(f"      Size: {size:,} bytes")
    else:
        all_videos_exist = False

if not all_videos_exist:
    print("\n❌ Not all video files found. Cannot proceed.")
    sys.exit(1)

print(f"\n{'='*50}")
print("🎬 FRAME EXTRACTION")
print("=" * 50)

extracted_count = 0
total_expected = 9
extracted_files = []

for video_file, event_time, sample_id, description in videos_to_process:
    video_path = os.path.join(video_dir, video_file)
    
    print(f"\n📹 Processing: {video_file}")
    print(f"   Description: {description}")
    print(f"   Event time: {event_time}s")
    
    try:
        # Load video
        with VideoFileClip(video_path) as clip:
            duration = clip.duration
            fps = clip.fps
            size = clip.size
            
            print(f"   ✅ Video loaded: {duration:.2f}s, {fps:.1f}fps, {size[0]}x{size[1]}px")
            
            # Extract 3 frames per video: before, during, after
            frame_specs = [
                (event_time - 0.5, "before", "Normal scene with person hidden"),
                (event_time, "during", "Critical moment - person emerging"), 
                (event_time + 0.5, "after", "Dangerous situation - person visible")
            ]
            
            for timestamp, phase, phase_desc in frame_specs:
                print(f"   📸 Extracting {phase} frame at {timestamp:.1f}s")
                print(f"      Context: {phase_desc}")
                
                # Check timestamp bounds
                if timestamp < 0:
                    print(f"      ⚠️ Timestamp before video start, using 0.0s")
                    timestamp = 0.0
                elif timestamp > duration:
                    print(f"      ⚠️ Timestamp after video end ({duration:.2f}s), using {duration-0.1:.1f}s")
                    timestamp = duration - 0.1
                
                try:
                    # Extract frame
                    frame_rgb = clip.get_frame(timestamp)
                    
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Save frame
                    filename = f"ghost_probing_{sample_id}_{phase}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Write with high quality
                    save_success = cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    if save_success and os.path.exists(filepath):
                        file_size = os.path.getsize(filepath)
                        print(f"      ✅ Saved: {filename} ({file_size:,} bytes)")
                        extracted_count += 1
                        extracted_files.append(filename)
                    else:
                        print(f"      ❌ Save failed for {phase} frame")
                        
                except Exception as frame_error:
                    print(f"      ❌ Frame extraction error: {frame_error}")
            
    except Exception as video_error:
        print(f"   ❌ Video processing error: {video_error}")

# Final results
print(f"\n{'='*50}")
print("📊 EXTRACTION RESULTS")
print("=" * 50)

print(f"Frames extracted: {extracted_count}/{total_expected}")
success_rate = (extracted_count / total_expected) * 100 if total_expected > 0 else 0
print(f"Success rate: {success_rate:.1f}%")

if extracted_count > 0:
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n📋 Extracted files:")
    
    # List all extracted files organized by sample
    for i in range(1, 4):
        sample_files = [f for f in extracted_files if f"sample{i}" in f]
        if sample_files:
            print(f"\n   Sample {i}:")
            for filename in sorted(sample_files):
                filepath = os.path.join(output_dir, filename)
                size = os.path.getsize(filepath)
                print(f"     {filename} ({size:,} bytes)")
    
    print(f"\n💡 Frame sequence explanation:")
    print(f"   • BEFORE: Normal driving scene with person hidden behind obstruction")
    print(f"   • DURING: Critical moment - person emerging from behind obstruction")
    print(f"   • AFTER: Dangerous situation - person now in vehicle's path")
    
    print(f"\n🔬 Applications:")
    print(f"   These temporal sequences demonstrate complete ghost probing patterns")
    print(f"   for multimodal few-shot learning, providing richer visual context")
    print(f"   compared to text-only descriptions.")

# Final status
if extracted_count == total_expected:
    print(f"\n✅ SUCCESS: All {total_expected} frames extracted successfully!")
    print(f"✅ Ready for multimodal few-shot learning experiments!")
    exit_code = 0
else:
    print(f"\n⚠️ PARTIAL SUCCESS: {extracted_count}/{total_expected} frames extracted")
    if extracted_count > 0:
        print(f"⚠️ Some frames available for limited multimodal experiments")
        exit_code = 1
    else:
        print(f"❌ No frames extracted - check error messages above")
        exit_code = 2

print(f"\nScript completed. Exit code: {exit_code}")
sys.exit(exit_code)