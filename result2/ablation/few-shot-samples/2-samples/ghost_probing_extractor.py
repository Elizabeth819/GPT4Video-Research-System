#!/usr/bin/env python3
"""
Ghost Probing Frame Extraction for Multimodal Few-Shot Learning

This script extracts critical frames from DADA-2000 videos showing ghost probing incidents.
Each sequence contains 3 frames: before, during, and after the ghost probing event.

Ghost probing refers to dangerous situations where a person suddenly emerges from behind
an obstruction (like a parked car or building) into the vehicle's path, creating a
"ghost" effect where they were previously invisible to the driver.

Video targets:
- images_1_003.avi at 2s: Person emerging from behind obstruction
- images_1_006.avi at 6s: Person emerging from behind obstruction  
- images_1_008.avi at 2s: Person emerging from behind obstruction

Frame timing:
- before: -0.5s from event (normal scene, person hidden)
- during: exact event time (person emerging)
- after: +0.5s from event (person in path)
"""

import os
import sys
import traceback

def extract_ghost_probing_frames():
    """Main extraction function"""
    
    # Setup paths
    project_root = "/Users/wanmeng/repository/GPT4Video-cobra-auto"
    video_dir = os.path.join(project_root, "DADA-2000-videos")
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples"
    
    # Add project to Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print("=" * 60)
    print("GHOST PROBING FRAME EXTRACTION FOR FEW-SHOT LEARNING")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Video directory: {video_dir}")
    print(f"Output directory: {output_dir}")
    
    # Verify directories
    if not os.path.exists(video_dir):
        print(f"❌ Video directory not found: {video_dir}")
        return False
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory ready: {output_dir}")
    
    # Import required libraries
    print("\nImporting libraries...")
    try:
        import cv2
        from moviepy.editor import VideoFileClip
        import numpy as np
        print("✓ OpenCV version:", cv2.__version__)
        print("✓ MoviePy imported successfully")
        print("✓ NumPy version:", np.__version__)
    except ImportError as e:
        print(f"❌ Library import failed: {e}")
        print("Please ensure dependencies are installed:")
        print("  pip install moviepy opencv-python numpy")
        return False
    
    # Define extraction specifications
    extraction_specs = [
        {
            "video_file": "images_1_003.avi",
            "event_time": 2.0,
            "sample_id": "sample1",
            "description": "Ghost probing incident at 2 seconds - person emerges from behind parked vehicle",
            "expected_pattern": "Person hidden behind obstruction suddenly appears in driving lane"
        },
        {
            "video_file": "images_1_006.avi", 
            "event_time": 6.0,
            "sample_id": "sample2",
            "description": "Ghost probing incident at 6 seconds - person emerges from behind building/structure",
            "expected_pattern": "Person concealed by infrastructure suddenly enters roadway"
        },
        {
            "video_file": "images_1_008.avi",
            "event_time": 2.0,
            "sample_id": "sample3", 
            "description": "Ghost probing incident at 2 seconds - person emerges from blind spot",
            "expected_pattern": "Person invisible due to occlusion suddenly becomes visible threat"
        }
    ]
    
    # Execute extraction for each video
    extraction_results = []
    total_frames_extracted = 0
    
    for spec in extraction_specs:
        print(f"\n{'='*40}")
        print(f"PROCESSING: {spec['video_file']}")
        print(f"{'='*40}")
        print(f"Event time: {spec['event_time']}s")
        print(f"Description: {spec['description']}")
        print(f"Expected pattern: {spec['expected_pattern']}")
        
        video_path = os.path.join(video_dir, spec["video_file"])
        print(f"Video path: {video_path}")
        print(f"Video exists: {os.path.exists(video_path)}")
        
        if not os.path.exists(video_path):
            print("❌ Video file not found - skipping")
            extraction_results.append({
                "video": spec["video_file"],
                "status": "file_not_found",
                "frames_extracted": 0,
                "frames": []
            })
            continue
        
        try:
            # Load video
            print("Loading video...")
            video_clip = VideoFileClip(video_path)
            
            video_info = {
                "duration": video_clip.duration,
                "fps": video_clip.fps,
                "size": video_clip.size
            }
            
            print(f"✓ Video loaded successfully")
            print(f"  Duration: {video_info['duration']:.2f}s")
            print(f"  FPS: {video_info['fps']}")
            print(f"  Resolution: {video_info['size']}")
            
            # Define frame extraction timestamps
            event_time = spec["event_time"]
            frame_timestamps = {
                "before": event_time - 0.5,  # Normal scene, person hidden
                "during": event_time,         # Person emerging 
                "after": event_time + 0.5     # Person in vehicle path
            }
            
            print(f"\nExtracting frames around {event_time}s:")
            
            sample_frames = []
            frames_this_video = 0
            
            for phase, timestamp in frame_timestamps.items():
                print(f"  {phase.capitalize()} frame at {timestamp:.1f}s...", end=" ")
                
                # Validate timestamp
                if timestamp < 0:
                    print("⚠️ Timestamp before video start")
                    sample_frames.append({
                        "phase": phase,
                        "timestamp": timestamp,
                        "status": "before_start",
                        "file": None
                    })
                    continue
                    
                if timestamp > video_info["duration"]:
                    print(f"⚠️ Timestamp after video end ({video_info['duration']:.2f}s)")
                    sample_frames.append({
                        "phase": phase,
                        "timestamp": timestamp,
                        "status": "after_end", 
                        "file": None
                    })
                    continue
                
                try:
                    # Extract frame at timestamp
                    frame_rgb = video_clip.get_frame(timestamp)
                    
                    # Convert RGB to BGR for OpenCV compatibility
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Generate output filename
                    output_filename = f"ghost_probing_{spec['sample_id']}_{phase}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Save frame with high quality
                    save_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
                    success = cv2.imwrite(output_path, frame_bgr, save_params)
                    
                    if success and os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        print(f"✓ Saved ({file_size:,} bytes)")
                        
                        sample_frames.append({
                            "phase": phase,
                            "timestamp": timestamp,
                            "status": "success",
                            "file": output_path,
                            "filename": output_filename,
                            "size_bytes": file_size
                        })
                        
                        frames_this_video += 1
                        total_frames_extracted += 1
                        
                    else:
                        print("❌ Save failed")
                        sample_frames.append({
                            "phase": phase,
                            "timestamp": timestamp,
                            "status": "save_failed",
                            "file": None
                        })
                
                except Exception as frame_error:
                    print(f"❌ Extraction error: {frame_error}")
                    sample_frames.append({
                        "phase": phase,
                        "timestamp": timestamp,
                        "status": "extraction_error",
                        "error": str(frame_error),
                        "file": None
                    })
            
            # Close video clip to free memory
            video_clip.close()
            
            extraction_results.append({
                "video": spec["video_file"],
                "status": "processed",
                "frames_extracted": frames_this_video,
                "video_info": video_info,
                "frames": sample_frames
            })
            
            print(f"✓ Completed {spec['video_file']} - {frames_this_video}/3 frames extracted")
            
        except Exception as video_error:
            print(f"❌ Video processing error: {video_error}")
            traceback.print_exc()
            
            extraction_results.append({
                "video": spec["video_file"],
                "status": "processing_error",
                "error": str(video_error),
                "frames_extracted": 0,
                "frames": []
            })
    
    # Generate comprehensive summary
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    
    total_videos = len(extraction_specs)
    successful_videos = len([r for r in extraction_results if r.get("frames_extracted", 0) > 0])
    expected_total_frames = total_videos * 3  # 3 frames per video
    
    print(f"Videos processed: {successful_videos}/{total_videos}")
    print(f"Frames extracted: {total_frames_extracted}/{expected_total_frames}")
    print(f"Success rate: {(total_frames_extracted/expected_total_frames)*100:.1f}%")
    
    # Detailed results
    for result in extraction_results:
        print(f"\n{result['video']}:")
        if result['status'] == 'processed':
            print(f"  Status: ✓ Processed ({result['frames_extracted']}/3 frames)")
            for frame in result['frames']:
                if frame['status'] == 'success':
                    print(f"    ✓ {frame['phase']}: {frame['filename']} ({frame['size_bytes']:,} bytes)")
                else:
                    print(f"    ❌ {frame['phase']}: {frame['status']}")
        else:
            print(f"  Status: ❌ {result['status']}")
    
    # List all generated files
    if os.path.exists(output_dir):
        ghost_files = [f for f in os.listdir(output_dir) 
                      if f.startswith("ghost_probing_") and f.endswith(".jpg")]
        ghost_files.sort()
        
        if ghost_files:
            print(f"\nGenerated files in {output_dir}:")
            for filename in ghost_files:
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    print(f"  {filename} ({size:,} bytes)")
    
    # Educational analysis
    print(f"\n{'='*60}")
    print("GHOST PROBING SEQUENCE ANALYSIS")
    print(f"{'='*60}")
    print("Each extracted sequence demonstrates the ghost probing phenomenon:")
    print()
    print("BEFORE frame (t-0.5s):")
    print("  • Shows normal driving scene")
    print("  • Person is hidden behind obstruction (car, building, etc.)")
    print("  • No visible threat to autonomous vehicle")
    print("  • Represents baseline 'safe' state")
    print()
    print("DURING frame (t=0s):")
    print("  • Critical moment of ghost probing event")
    print("  • Person emerges from behind obstruction")
    print("  • Transition from hidden to visible")
    print("  • Key frame for detection algorithms")
    print()
    print("AFTER frame (t+0.5s):")
    print("  • Person now in vehicle's path")
    print("  • Dangerous situation requiring immediate response")
    print("  • Shows full manifestation of ghost probing threat")
    print("  • Critical for safety assessment")
    print()
    print("MULTIMODAL FEW-SHOT LEARNING APPLICATIONS:")
    print("  • Visual pattern recognition for ghost probing detection")
    print("  • Training data for autonomous vehicle safety systems")
    print("  • Benchmark sequences for algorithm evaluation")
    print("  • Educational examples for safety research")
    
    success = total_frames_extracted > 0
    print(f"\n{'='*60}")
    if success:
        print("✓ GHOST PROBING FRAME EXTRACTION COMPLETED SUCCESSFULLY")
        print(f"✓ {total_frames_extracted} frames ready for few-shot learning")
    else:
        print("❌ GHOST PROBING FRAME EXTRACTION FAILED")
        print("❌ No frames were successfully extracted")
    print(f"{'='*60}")
    
    return success

if __name__ == "__main__":
    print("Starting Ghost Probing Frame Extraction...")
    success = extract_ghost_probing_frames()
    
    if success:
        print("\n🎯 SUCCESS: Ghost probing sequences ready for multimodal few-shot learning!")
    else:
        print("\n💥 FAILURE: Ghost probing extraction incomplete. Check error messages above.")
    
    exit(0 if success else 1)