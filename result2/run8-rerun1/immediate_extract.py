#!/usr/bin/env python3

# Execute immediately - inline all dependencies
import os
import sys

# Execute ghost probing frame extraction
print("🎯 Starting Ghost Probing Frame Extraction")
print("=" * 60)

try:
    import cv2
    print(f"✅ OpenCV {cv2.__version__} loaded")
except ImportError as e:
    print(f"❌ OpenCV failed: {e}")
    exit(1)

try:
    from moviepy.editor import VideoFileClip
    print("✅ MoviePy loaded")
except ImportError as e:
    print(f"❌ MoviePy failed: {e}")
    exit(1)

# Paths
video_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos"
output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/image-fewshot"

print(f"Video directory: {video_dir}")
print(f"Output directory: {output_dir}")

os.makedirs(output_dir, exist_ok=True)

# Extraction targets - updated sample3 timing
videos = [
    ("images_1_003.avi", 2.0, "sample1", "Ghost probing at intersection"),
    ("images_1_006.avi", 9.0, "sample2", "Person emerges from building"),
    ("images_1_008.avi", 2.8, "sample3", "Blind spot emergence")
]

extracted = []

for video_file, event_time, sample_id, desc in videos:
    video_path = os.path.join(video_dir, video_file)
    
    print(f"\n📹 Processing: {video_file}")
    print(f"   Event: {desc} at {event_time}s")
    
    if not os.path.exists(video_path):
        print(f"   ❌ Not found: {video_path}")
        continue
    
    try:
        clip = VideoFileClip(video_path)
        print(f"   ✅ Loaded: {clip.duration:.2f}s, {clip.fps:.1f}fps, {clip.size[0]}x{clip.size[1]}")
        
        # Extract 3 frames - adjusted timing for each sample
        if sample_id == "sample1":
            frame_deltas = [
                (-0.8, "before", "Normal scene - person hidden"),
                (0, "during", "Critical moment - person emerging"),
                (0.5, "after", "Dangerous - person visible")
            ]
        elif sample_id == "sample2":
            frame_deltas = [
                (-1.0, "before", "Normal scene - person hidden"),    # 8s
                (0, "during", "Critical moment - person emerging"),  # 9s
                (1.0, "after", "Dangerous - person visible")        # 10s
            ]
        elif sample_id == "sample3":
            frame_deltas = [
                (-1.8, "before", "Normal scene - person hidden"),   # 1s
                (0, "during", "Critical moment - person emerging"), # 2.8s
                (0.2, "after", "Dangerous - person visible")       # 3s
            ]
        else:
            frame_deltas = [
                (-0.5, "before", "Normal scene - person hidden"),
                (0, "during", "Critical moment - person emerging"),
                (0.5, "after", "Dangerous - person visible")
            ]
        
        for delta, phase, desc in frame_deltas:
            timestamp = event_time + delta
            print(f"     Extracting {phase} at {timestamp:.1f}s: {desc}")
            
            if 0 <= timestamp <= clip.duration:
                try:
                    frame = clip.get_frame(timestamp)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    filename = f"ghost_probing_{sample_id}_{phase}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    
                    if cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        size = os.path.getsize(filepath)
                        print(f"       ✅ {filename} ({size:,} bytes)")
                        extracted.append(filename)
                    else:
                        print(f"       ❌ Save failed")
                except Exception as e:
                    print(f"       ❌ Error: {e}")
            else:
                print(f"       ⚠️ Out of bounds (0-{clip.duration:.2f}s)")
        
        clip.close()
        
    except Exception as e:
        print(f"   ❌ Video error: {e}")

# Results
print(f"\n{'='*60}")
print("RESULTS")
print("=" * 60)
print(f"Extracted: {len(extracted)}/9 frames")

if extracted:
    print(f"\n📁 Location: {output_dir}")
    print(f"\n📋 Files:")
    for filename in sorted(extracted):
        path = os.path.join(output_dir, filename)
        size = os.path.getsize(path)
        print(f"   {filename} ({size:,} bytes)")
    
    print(f"\n💡 Sequence pattern:")
    print(f"   🔵 BEFORE: Hidden state")
    print(f"   🟡 DURING: Emergence")
    print(f"   🔴 AFTER: Visible threat")
    
    if len(extracted) == 9:
        print(f"\n✅ SUCCESS: All 9 frames extracted!")
        print(f"✅ Ready for multimodal few-shot learning!")
    else:
        print(f"\n⚠️ PARTIAL: {len(extracted)}/9 frames extracted")
else:
    print(f"\n❌ FAILED: No frames extracted")

print(f"\nCompleted.")