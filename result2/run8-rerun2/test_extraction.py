#!/usr/bin/env python3
"""Test extraction with detailed diagnostics"""

import os
import sys

def test_dependencies():
    """Test if required libraries are available"""
    print("Testing dependencies...")
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
        opencv_ok = True
    except ImportError as e:
        print(f"❌ OpenCV: {e}")
        opencv_ok = False
    
    try:
        from moviepy.editor import VideoFileClip
        print("✅ MoviePy available")
        moviepy_ok = True
    except ImportError as e:
        print(f"❌ MoviePy: {e}")
        moviepy_ok = False
    
    return opencv_ok and moviepy_ok

def test_paths():
    """Test if required paths exist"""
    print("\nTesting paths...")
    
    video_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos"
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/image-fewshot"
    
    print(f"Video dir exists: {os.path.exists(video_dir)}")
    print(f"Output dir exists: {os.path.exists(output_dir)}")
    
    videos = ["images_1_003.avi", "images_1_006.avi", "images_1_008.avi"]
    for video in videos:
        path = os.path.join(video_dir, video)
        exists = os.path.exists(path)
        print(f"  {video}: {'✅' if exists else '❌'}")
        if exists:
            size = os.path.getsize(path)
            print(f"    Size: {size:,} bytes")
    
    return True

def run_extraction():
    """Run the actual frame extraction"""
    print("\nRunning extraction...")
    
    if not test_dependencies():
        print("❌ Dependencies missing")
        return False
    
    import cv2
    from moviepy.editor import VideoFileClip
    
    video_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos"
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/image-fewshot"
    
    os.makedirs(output_dir, exist_ok=True)
    
    extractions = [
        ("images_1_003.avi", 2.0, "sample1"),
        ("images_1_006.avi", 6.0, "sample2"),
        ("images_1_008.avi", 2.0, "sample3")
    ]
    
    total_extracted = 0
    
    for video_file, event_time, sample_id in extractions:
        video_path = os.path.join(video_dir, video_file)
        print(f"\n📹 {video_file} (event at {event_time}s)")
        
        if not os.path.exists(video_path):
            print("  ❌ File not found")
            continue
        
        try:
            clip = VideoFileClip(video_path)
            print(f"  ✅ Loaded: {clip.duration:.2f}s, {clip.fps}fps, {clip.size}")
            
            phases = [
                (event_time - 0.5, "before", "Hidden state"),
                (event_time, "during", "Emergence moment"),
                (event_time + 0.5, "after", "Visible threat")
            ]
            
            for timestamp, phase, desc in phases:
                if 0 <= timestamp <= clip.duration:
                    try:
                        frame = clip.get_frame(timestamp)
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        filename = f"ghost_probing_{sample_id}_{phase}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        
                        success = cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        if success and os.path.exists(filepath):
                            size = os.path.getsize(filepath)
                            print(f"    ✅ {phase} ({timestamp:.1f}s): {size:,} bytes - {desc}")
                            total_extracted += 1
                        else:
                            print(f"    ❌ {phase}: Save failed")
                    except Exception as e:
                        print(f"    ❌ {phase}: {e}")
                else:
                    print(f"    ⚠️ {phase} ({timestamp:.1f}s): Out of bounds (0-{clip.duration:.2f}s)")
            
            clip.close()
            
        except Exception as e:
            print(f"  ❌ Video error: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 RESULTS: {total_extracted}/9 frames extracted")
    
    if total_extracted > 0:
        print(f"\n📁 Files in {output_dir}:")
        for file in sorted(os.listdir(output_dir)):
            if file.endswith('.jpg') and 'ghost_probing' in file:
                path = os.path.join(output_dir, file)
                size = os.path.getsize(path)
                print(f"  {file} ({size:,} bytes)")
    
    return total_extracted == 9

if __name__ == "__main__":
    print("🎯 Ghost Probing Frame Extraction Test")
    print("=" * 50)
    
    test_paths()
    success = run_extraction()
    
    if success:
        print("\n✅ All 9 frames extracted successfully!")
        print("🔬 Ready for multimodal few-shot learning!")
    else:
        print("\n❌ Extraction incomplete")
    
    print(f"\nExit code: {0 if success else 1}")