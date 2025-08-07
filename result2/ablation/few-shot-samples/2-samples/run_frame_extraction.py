#!/usr/bin/env python3
"""
Ghost probing frame extraction using cv2 and moviepy
"""
import sys
import os
sys.path.append('/Users/wanmeng/repository/GPT4Video-cobra-auto')

# Add project root to Python path for imports
project_root = '/Users/wanmeng/repository/GPT4Video-cobra-auto'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Python path:", sys.path[:3])
print("Current working directory:", os.getcwd())

try:
    import cv2
    print("OpenCV version:", cv2.__version__)
except ImportError as e:
    print("OpenCV import error:", e)

try: 
    from moviepy.editor import VideoFileClip
    print("MoviePy imported successfully")
except ImportError as e:
    print("MoviePy import error:", e)

def extract_ghost_frames():
    """Extract ghost probing frames from specified videos"""
    
    # Define paths
    video_base = "/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos"
    output_base = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples"
    
    # Video specifications with timestamps
    video_specs = [
        {
            "filename": "images_1_003.avi",
            "center_time": 2.0,
            "sample_id": "sample1",
            "description": "Ghost probing - person emerging from behind obstruction at 2s"
        },
        {
            "filename": "images_1_006.avi", 
            "center_time": 6.0,
            "sample_id": "sample2",
            "description": "Ghost probing - person emerging from behind obstruction at 6s"
        },
        {
            "filename": "images_1_008.avi",
            "center_time": 2.0,
            "sample_id": "sample3", 
            "description": "Ghost probing - person emerging from behind obstruction at 2s"
        }
    ]
    
    results = []
    
    for spec in video_specs:
        video_path = os.path.join(video_base, spec["filename"])
        
        print(f"\n{'='*50}")
        print(f"Processing: {spec['filename']}")
        print(f"Description: {spec['description']}")
        print(f"Video path: {video_path}")
        print(f"Video exists: {os.path.exists(video_path)}")
        
        if not os.path.exists(video_path):
            print(f"ERROR: Video file not found - {video_path}")
            results.append({
                "video": spec["filename"],
                "status": "file_not_found",
                "frames": []
            })
            continue
        
        try:
            # Open video with MoviePy
            with VideoFileClip(video_path) as clip:
                print(f"Video duration: {clip.duration:.2f}s")
                print(f"Video FPS: {clip.fps}")
                print(f"Video size: {clip.size}")
                
                # Define frame extraction points
                frame_times = {
                    "before": spec["center_time"] - 0.5,
                    "during": spec["center_time"],
                    "after": spec["center_time"] + 0.5
                }
                
                sample_frames = []
                
                for phase, timestamp in frame_times.items():
                    print(f"\nExtracting {phase} frame at {timestamp}s...")
                    
                    if timestamp < 0 or timestamp > clip.duration:
                        print(f"  Timestamp {timestamp}s out of bounds (0-{clip.duration:.2f}s)")
                        sample_frames.append({
                            "phase": phase,
                            "timestamp": timestamp,
                            "status": "out_of_bounds",
                            "file": None
                        })
                        continue
                    
                    try:
                        # Extract frame at timestamp
                        frame_rgb = clip.get_frame(timestamp)
                        
                        # Convert RGB to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        
                        # Generate output filename
                        output_filename = f"ghost_probing_{spec['sample_id']}_{phase}.jpg"
                        output_path = os.path.join(output_base, output_filename)
                        
                        # Save frame
                        success = cv2.imwrite(output_path, frame_bgr)
                        
                        if success:
                            print(f"  SUCCESS: Frame saved to {output_filename}")
                            sample_frames.append({
                                "phase": phase,
                                "timestamp": timestamp,
                                "status": "success",
                                "file": output_path
                            })
                        else:
                            print(f"  ERROR: Failed to save frame to {output_path}")
                            sample_frames.append({
                                "phase": phase,
                                "timestamp": timestamp,
                                "status": "save_failed", 
                                "file": None
                            })
                            
                    except Exception as frame_error:
                        print(f"  ERROR: Frame extraction failed - {frame_error}")
                        sample_frames.append({
                            "phase": phase,
                            "timestamp": timestamp,
                            "status": "extraction_failed",
                            "file": None,
                            "error": str(frame_error)
                        })
                
                results.append({
                    "video": spec["filename"],
                    "status": "processed",
                    "frames": sample_frames
                })
                
        except Exception as video_error:
            print(f"ERROR: Failed to process video - {video_error}")
            results.append({
                "video": spec["filename"],
                "status": "processing_failed",
                "error": str(video_error),
                "frames": []
            })
    
    # Print final summary
    print(f"\n{'='*60}")
    print("GHOST PROBING FRAME EXTRACTION SUMMARY")
    print(f"{'='*60}")
    
    total_frames = 0
    successful_frames = 0
    
    for result in results:
        print(f"\nVideo: {result['video']}")
        print(f"Status: {result['status']}")
        
        if result['frames']:
            for frame in result['frames']:
                total_frames += 1
                phase_status = frame['status']
                timestamp = frame['timestamp']
                
                if phase_status == "success":
                    successful_frames += 1
                    print(f"  ✓ {frame['phase']} ({timestamp}s): {os.path.basename(frame['file'])}")
                else:
                    print(f"  ✗ {frame['phase']} ({timestamp}s): {phase_status}")
    
    print(f"\nResults: {successful_frames}/{total_frames} frames extracted successfully")
    print(f"Output directory: {output_base}")
    
    # List generated files
    generated_files = [f for f in os.listdir(output_base) 
                      if f.startswith("ghost_probing_") and f.endswith(".jpg")]
    
    if generated_files:
        print(f"\nGenerated files ({len(generated_files)}):")
        for filename in sorted(generated_files):
            filepath = os.path.join(output_base, filename)
            filesize = os.path.getsize(filepath) if os.path.exists(filepath) else 0
            print(f"  {filename} ({filesize:,} bytes)")
    else:
        print("\nNo files generated.")
    
    return results

if __name__ == "__main__":
    print("Starting ghost probing frame extraction...")
    print("="*60)
    
    results = extract_ghost_frames()
    
    print("\nFrame sequence analysis:")
    print("Each sample contains 3 frames showing the ghost probing process:")
    print("  before: Normal scene with obstruction (person hidden)")
    print("  during: Critical moment (person emerging from behind obstruction)")
    print("  after:  Dangerous situation (person in vehicle's path)")
    print("\nThese sequences demonstrate the complete ghost probing pattern")
    print("for multimodal few-shot learning applications.")