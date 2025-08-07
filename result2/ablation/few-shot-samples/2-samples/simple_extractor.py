#!/usr/bin/env python3
"""Simple ghost probing frame extractor"""

if __name__ == "__main__":
    import os
    import sys
    
    # Setup
    project_root = "/Users/wanmeng/repository/GPT4Video-cobra-auto"
    sys.path.insert(0, project_root)
    
    print("Simple Ghost Probing Frame Extractor")
    print("=" * 40)
    
    try:
        import cv2
        from moviepy.editor import VideoFileClip
        print("✅ Libraries imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        exit(1)
    
    # Paths
    video_dir = f"{project_root}/DADA-2000-videos"
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Videos to process
    videos = [
        ("images_1_003.avi", 2.0, "sample1", "Ghost probing at 2s"),
        ("images_1_006.avi", 6.0, "sample2", "Ghost probing at 6s"),
        ("images_1_008.avi", 2.0, "sample3", "Ghost probing at 2s")
    ]
    
    extracted = []
    
    for video_file, event_time, sample_id, desc in videos:
        video_path = os.path.join(video_dir, video_file)
        print(f"\nProcessing {video_file} - {desc}")
        
        if not os.path.exists(video_path):
            print(f"  ❌ Not found: {video_path}")
            continue
        
        try:
            clip = VideoFileClip(video_path)
            print(f"  ✅ Loaded: {clip.duration:.2f}s")
            
            # Extract 3 frames
            phases = [
                (event_time - 0.5, "before", "Normal scene with person hidden"),
                (event_time, "during", "Person emerging from obstruction"),
                (event_time + 0.5, "after", "Person in vehicle path")
            ]
            
            for timestamp, phase, description in phases:
                if 0 <= timestamp <= clip.duration:
                    frame = clip.get_frame(timestamp)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    filename = f"ghost_probing_{sample_id}_{phase}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    
                    if cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        size = os.path.getsize(filepath)
                        print(f"    ✅ {phase}: {filename} ({size:,} bytes)")
                        print(f"       {description}")
                        extracted.append(filename)
                    else:
                        print(f"    ❌ Failed: {phase}")
                else:
                    print(f"    ⚠️ {phase} ({timestamp}s) out of bounds")
            
            clip.close()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n{'='*40}")
    print(f"RESULTS: {len(extracted)} frames extracted")
    
    if extracted:
        print("\nExtracted files:")
        for filename in sorted(extracted):
            filepath = os.path.join(output_dir, filename)
            size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
            print(f"  {filename} ({size:,} bytes)")
        
        print(f"\nOutput directory: {output_dir}")
        print("\nFrame sequence explanation:")
        print("• BEFORE: Normal scene with person hidden behind obstruction")
        print("• DURING: Critical moment - person emerging from behind obstruction")
        print("• AFTER: Dangerous situation - person now in vehicle's path")
        print("\nThese sequences demonstrate the complete ghost probing pattern")
        print("for multimodal few-shot learning applications.")
    else:
        print("❌ No frames extracted")
    
    print("Extraction complete!")