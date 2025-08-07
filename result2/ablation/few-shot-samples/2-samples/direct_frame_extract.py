#!/usr/bin/env python3
"""
Direct frame extraction for ghost probing sequences
"""
import os
import sys

# Add the main project directory to Python path
project_root = "/Users/wanmeng/repository/GPT4Video-cobra-auto"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def extract_frames_direct():
    """Extract frames using direct library calls"""
    
    try:
        import cv2
        from moviepy.editor import VideoFileClip
        print("Successfully imported required libraries")
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    
    # Define paths and specifications
    video_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos"
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Video specifications for ghost probing extraction
    videos = [
        {"file": "images_1_003.avi", "time": 2.0, "id": "sample1"},
        {"file": "images_1_006.avi", "time": 6.0, "id": "sample2"}, 
        {"file": "images_1_008.avi", "time": 2.0, "id": "sample3"}
    ]
    
    print("="*50)
    print("GHOST PROBING FRAME EXTRACTION")
    print("="*50)
    
    extraction_results = []
    
    for video_spec in videos:
        video_path = os.path.join(video_dir, video_spec["file"])
        
        print(f"\nProcessing: {video_spec['file']}")
        print(f"Target time: {video_spec['time']}s")
        print(f"Video path: {video_path}")
        
        if not os.path.exists(video_path):
            print("❌ Video file not found!")
            extraction_results.append({
                "video": video_spec["file"],
                "status": "not_found",
                "frames": []
            })
            continue
        
        try:
            # Load video
            clip = VideoFileClip(video_path)
            print(f"✓ Video loaded - Duration: {clip.duration:.2f}s")
            
            # Define extraction timestamps
            center_time = video_spec["time"]
            timestamps = {
                "before": center_time - 0.5,
                "during": center_time,
                "after": center_time + 0.5
            }
            
            video_results = []
            
            for phase, timestamp in timestamps.items():
                print(f"  Extracting {phase} frame at {timestamp}s...")
                
                # Check timestamp bounds
                if timestamp < 0 or timestamp > clip.duration:
                    print(f"    ⚠️ Timestamp out of bounds")
                    video_results.append({
                        "phase": phase,
                        "timestamp": timestamp,
                        "status": "out_of_bounds"
                    })
                    continue
                
                try:
                    # Extract frame
                    frame = clip.get_frame(timestamp)
                    
                    # Convert RGB to BGR
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Save frame
                    output_filename = f"ghost_probing_{video_spec['id']}_{phase}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    success = cv2.imwrite(output_path, frame_bgr)
                    
                    if success and os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        print(f"    ✓ {output_filename} ({file_size:,} bytes)")
                        video_results.append({
                            "phase": phase,
                            "timestamp": timestamp,
                            "status": "success",
                            "file": output_path,
                            "size": file_size
                        })
                    else:
                        print(f"    ❌ Failed to save {output_filename}")
                        video_results.append({
                            "phase": phase,
                            "timestamp": timestamp,
                            "status": "save_failed"
                        })
                
                except Exception as frame_error:
                    print(f"    ❌ Frame extraction error: {frame_error}")
                    video_results.append({
                        "phase": phase,
                        "timestamp": timestamp,
                        "status": "extraction_error",
                        "error": str(frame_error)
                    })
            
            # Close video clip
            clip.close()
            
            extraction_results.append({
                "video": video_spec["file"],
                "status": "processed", 
                "frames": video_results
            })
            
        except Exception as video_error:
            print(f"❌ Video processing error: {video_error}")
            extraction_results.append({
                "video": video_spec["file"],
                "status": "video_error",
                "error": str(video_error),
                "frames": []
            })
    
    # Print final summary
    print("\n" + "="*50)
    print("EXTRACTION SUMMARY")
    print("="*50)
    
    total_expected = len(videos) * 3  # 3 frames per video
    total_extracted = 0
    
    for result in extraction_results:
        print(f"\n{result['video']}:")
        if result['status'] == 'processed':
            for frame in result['frames']:
                if frame['status'] == 'success':
                    total_extracted += 1
                    print(f"  ✓ {frame['phase']} ({frame['timestamp']}s)")
                else:
                    print(f"  ❌ {frame['phase']} ({frame['timestamp']}s) - {frame['status']}")
        else:
            print(f"  ❌ {result['status']}")
    
    print(f"\nResult: {total_extracted}/{total_expected} frames extracted")
    
    # List all generated files
    ghost_files = [f for f in os.listdir(output_dir) 
                   if f.startswith("ghost_probing_") and f.endswith(".jpg")]
    
    if ghost_files:
        print(f"\nGenerated files ({len(ghost_files)}):")
        for filename in sorted(ghost_files):
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"  {filename} ({size:,} bytes)")
    
    print(f"\nOutput directory: {output_dir}")
    print("\nFrame sequence explanation:")
    print("  before: Normal scene with obstruction hiding the person")
    print("  during: Critical moment - person emerging from behind obstruction")  
    print("  after:  Dangerous situation - person now in vehicle's path")
    print("\nThese sequences demonstrate the complete ghost probing pattern")
    print("for few-shot learning in autonomous driving safety analysis.")
    
    return total_extracted > 0

if __name__ == "__main__":
    success = extract_frames_direct()
    if success:
        print("\n✓ Ghost probing frame extraction completed successfully!")
    else:
        print("\n❌ Ghost probing frame extraction failed!")