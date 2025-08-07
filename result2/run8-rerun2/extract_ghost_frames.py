#!/usr/bin/env python3
"""
Ghost Probing Frame Extraction for Multimodal Few-Shot Learning
Extract frames from specific DADA-2000 videos at precise timestamps
"""

import os
import sys
import cv2
from moviepy.editor import VideoFileClip

def main():
    print("🎯 Ghost Probing Frame Extraction for Multimodal Few-Shot Learning")
    print("=" * 70)
    
    # Paths
    project_root = "/Users/wanmeng/repository/GPT4Video-cobra-auto"
    video_dir = os.path.join(project_root, "DADA-2000-videos")
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/image-fewshot"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Video configurations
    extractions = [
        ("images_1_003.avi", 2.0, "sample1", "Ghost probing at intersection"),
        ("images_1_006.avi", 6.0, "sample2", "Person emerges from building"),
        ("images_1_008.avi", 2.0, "sample3", "Blind spot emergence")
    ]
    
    extracted_files = []
    
    for video_file, event_time, sample_id, description in extractions:
        video_path = os.path.join(video_dir, video_file)
        print(f"\n📹 Processing: {video_file}")
        print(f"   Description: {description}")
        print(f"   Event time: {event_time}s")
        
        if not os.path.exists(video_path):
            print(f"   ❌ Video not found: {video_path}")
            continue
        
        try:
            with VideoFileClip(video_path) as clip:
                print(f"   ✅ Video loaded: {clip.duration:.2f}s duration")
                
                # Extract 3 frames: before, during, after
                for delta, phase in [(-0.5, "before"), (0, "during"), (0.5, "after")]:
                    timestamp = event_time + delta
                    
                    if 0 <= timestamp <= clip.duration:
                        frame = clip.get_frame(timestamp)
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        filename = f"ghost_probing_{sample_id}_{phase}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        
                        if cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            file_size = os.path.getsize(filepath)
                            print(f"   ✅ {phase}: {filename} ({file_size:,} bytes)")
                            extracted_files.append(filename)
                        else:
                            print(f"   ❌ Failed to save {phase} frame")
                    else:
                        print(f"   ⚠️ {phase} timestamp {timestamp:.1f}s out of bounds")
        
        except Exception as e:
            print(f"   ❌ Error processing {video_file}: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"🎉 Extraction Complete: {len(extracted_files)} frames extracted")
    
    if extracted_files:
        print(f"\n📁 Output directory: {output_dir}")
        print("\n📋 Extracted files:")
        
        # Group by sample
        for i in range(1, 4):
            sample_files = [f for f in extracted_files if f"sample{i}" in f]
            if sample_files:
                print(f"\n   Sample {i}:")
                for filename in sorted(sample_files):
                    filepath = os.path.join(output_dir, filename)
                    size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
                    print(f"     {filename} ({size:,} bytes)")
        
        print("\n💡 Frame sequence explanation:")
        print("   • BEFORE: Normal scene with person hidden")
        print("   • DURING: Critical moment - person emerging")
        print("   • AFTER: Dangerous situation - person in path")
        
        print("\n🔬 Ready for multimodal few-shot learning experiments!")
        return True
    else:
        print("❌ No frames extracted")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)