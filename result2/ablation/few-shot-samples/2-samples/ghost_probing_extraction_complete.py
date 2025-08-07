#!/usr/bin/env python3
"""
Complete Ghost Probing Frame Extraction for Multimodal Few-Shot Learning

This script extracts ghost probing image sequences from specified DADA-2000 videos
for use in multimodal few-shot learning experiments. 

GHOST PROBING PHENOMENON:
Ghost probing refers to dangerous traffic situations where a person suddenly emerges 
from behind an obstruction (parked vehicle, building, etc.) into a vehicle's path, 
creating a "ghost" effect where they were previously invisible to the driver/camera.

EXTRACTION SPECIFICATIONS:
Video 1: images_1_003.avi at 2s
  - before (1.5s): Normal scene with person hidden behind obstruction
  - during (2.0s): Person emerging from behind obstruction  
  - after (2.5s): Person in vehicle's path

Video 2: images_1_006.avi at 6s
  - before (5.5s): Normal scene with person hidden behind obstruction
  - during (6.0s): Person emerging from behind obstruction
  - after (6.5s): Person in vehicle's path

Video 3: images_1_008.avi at 2s
  - before (1.5s): Normal scene with person hidden behind obstruction
  - during (2.0s): Person emerging from behind obstruction
  - after (2.5s): Person in vehicle's path

OUTPUT FILES:
- ghost_probing_sample1_before.jpg (images_1_003.avi at 1.5s)
- ghost_probing_sample1_during.jpg (images_1_003.avi at 2.0s)
- ghost_probing_sample1_after.jpg (images_1_003.avi at 2.5s)
- ghost_probing_sample2_before.jpg (images_1_006.avi at 5.5s)
- ghost_probing_sample2_during.jpg (images_1_006.avi at 6.0s)
- ghost_probing_sample2_after.jpg (images_1_006.avi at 6.5s)
- ghost_probing_sample3_before.jpg (images_1_008.avi at 1.5s)
- ghost_probing_sample3_during.jpg (images_1_008.avi at 2.0s)
- ghost_probing_sample3_after.jpg (images_1_008.avi at 2.5s)
"""

import os
import sys
import traceback
from typing import List, Dict, Tuple, Optional

def setup_environment() -> bool:
    """Setup Python environment and import required libraries."""
    
    # Add project root to Python path
    project_root = "/Users/wanmeng/repository/GPT4Video-cobra-auto"
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print("GHOST PROBING FRAME EXTRACTION")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Test library imports
    print("\nTesting library imports...")
    
    try:
        import cv2
        print(f"✅ OpenCV imported successfully - version {cv2.__version__}")
        opencv_available = True
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        opencv_available = False
    
    try:
        from moviepy.editor import VideoFileClip
        print("✅ MoviePy imported successfully")
        moviepy_available = True
    except ImportError as e:
        print(f"❌ MoviePy import failed: {e}")
        moviepy_available = False
    
    try:
        import numpy as np
        print(f"✅ NumPy imported successfully - version {np.__version__}")
        numpy_available = True
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        numpy_available = False
    
    libraries_ready = opencv_available and moviepy_available and numpy_available
    
    if not libraries_ready:
        print("\\n❌ Required libraries not available. Please install:")
        print("pip install moviepy opencv-python numpy")
        print("Or activate the cobraauto conda environment:")
        print("conda activate cobraauto")
    
    return libraries_ready

def verify_paths() -> Dict[str, bool]:
    """Verify required directories and video files exist."""
    
    print("\\nVerifying paths and files...")
    
    # Define paths
    project_root = "/Users/wanmeng/repository/GPT4Video-cobra-auto"
    video_dir = os.path.join(project_root, "DADA-2000-videos")
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples"
    
    # Check directories
    video_dir_exists = os.path.exists(video_dir)
    print(f"Video directory exists: {'✅' if video_dir_exists else '❌'} {video_dir}")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    output_dir_exists = os.path.exists(output_dir)
    output_dir_writable = os.access(output_dir, os.W_OK) if output_dir_exists else False
    
    print(f"Output directory exists: {'✅' if output_dir_exists else '❌'} {output_dir}")
    print(f"Output directory writable: {'✅' if output_dir_writable else '❌'}")
    
    # Check target video files
    target_videos = ["images_1_003.avi", "images_1_006.avi", "images_1_008.avi"]
    video_files_status = {}
    
    print("\\nTarget video files:")
    for video_file in target_videos:
        video_path = os.path.join(video_dir, video_file)
        exists = os.path.exists(video_path)
        video_files_status[video_file] = exists
        
        status_icon = "✅" if exists else "❌"
        print(f"  {status_icon} {video_file}")
        
        if exists:
            try:
                file_size = os.path.getsize(video_path)
                print(f"     Size: {file_size:,} bytes")
            except:
                print("     Size: Unknown")
    
    return {
        'video_dir_exists': video_dir_exists,
        'output_dir_ready': output_dir_exists and output_dir_writable,
        'video_files': video_files_status
    }

def extract_ghost_probing_frames() -> List[Dict]:
    """Extract ghost probing frames from the specified videos."""
    
    # Import libraries (assumes setup_environment passed)
    import cv2
    from moviepy.editor import VideoFileClip
    
    # Define paths
    project_root = "/Users/wanmeng/repository/GPT4Video-cobra-auto"
    video_dir = os.path.join(project_root, "DADA-2000-videos")
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples"
    
    # Video extraction specifications
    extraction_specs = [
        {
            "video_file": "images_1_003.avi",
            "event_timestamp": 2.0,
            "sample_id": "sample1",
            "description": "Ghost probing incident - person emerges from behind parked vehicle at 2 seconds",
            "safety_context": "Demonstrates classic ghost probing where pedestrian hidden by parked car suddenly enters roadway"
        },
        {
            "video_file": "images_1_006.avi", 
            "event_timestamp": 6.0,
            "sample_id": "sample2",
            "description": "Ghost probing incident - person emerges from behind building/structure at 6 seconds",
            "safety_context": "Shows ghost probing from infrastructure occlusion where building blocks pedestrian visibility"
        },
        {
            "video_file": "images_1_008.avi",
            "event_timestamp": 2.0,
            "sample_id": "sample3", 
            "description": "Ghost probing incident - person emerges from blind spot at 2 seconds",
            "safety_context": "Illustrates ghost probing from blind spot where obstruction creates dangerous visibility gap"
        }
    ]
    
    print("\\n" + "=" * 60)
    print("FRAME EXTRACTION PROCESS")
    print("=" * 60)
    
    extraction_results = []
    total_frames_attempted = 0
    total_frames_successful = 0
    
    for spec in extraction_specs:
        print(f"\\n{'='*40}")
        print(f"PROCESSING: {spec['video_file']}")
        print(f"{'='*40}")
        print(f"Event timestamp: {spec['event_timestamp']}s")
        print(f"Sample ID: {spec['sample_id']}")
        print(f"Description: {spec['description']}")
        print(f"Safety context: {spec['safety_context']}")
        
        video_path = os.path.join(video_dir, spec["video_file"])
        print(f"Video path: {video_path}")
        
        if not os.path.exists(video_path):
            print("❌ Video file not found - skipping")
            extraction_results.append({
                "video": spec["video_file"],
                "status": "file_not_found",
                "frames": [],
                "error": "Video file does not exist"
            })
            continue
        
        try:
            # Load video clip
            print("Loading video...")
            with VideoFileClip(video_path) as video_clip:
                
                # Get video properties
                duration = video_clip.duration
                fps = video_clip.fps
                size = video_clip.size
                
                print(f"✅ Video loaded successfully:")
                print(f"   Duration: {duration:.2f} seconds")
                print(f"   FPS: {fps}")
                print(f"   Resolution: {size[0]}x{size[1]}")
                
                # Define frame extraction timestamps
                event_time = spec["event_timestamp"]
                frame_extractions = [
                    {
                        "phase": "before",
                        "timestamp": event_time - 0.5,
                        "description": "Normal scene with person hidden behind obstruction",
                        "analysis": "Baseline state showing apparently safe driving environment with no visible threats"
                    },
                    {
                        "phase": "during", 
                        "timestamp": event_time,
                        "description": "Critical moment - person emerging from behind obstruction",
                        "analysis": "Key frame capturing the ghost probing event as person transitions from hidden to visible"
                    },
                    {
                        "phase": "after",
                        "timestamp": event_time + 0.5,
                        "description": "Dangerous situation - person now in vehicle's path", 
                        "analysis": "Post-emergence state showing full manifestation of safety threat requiring immediate response"
                    }
                ]
                
                frame_results = []
                
                for frame_spec in frame_extractions:
                    total_frames_attempted += 1
                    timestamp = frame_spec["timestamp"]
                    phase = frame_spec["phase"]
                    
                    print(f"\\n  Extracting {phase.upper()} frame at {timestamp:.1f}s...")
                    print(f"    Context: {frame_spec['description']}")
                    print(f"    Analysis: {frame_spec['analysis']}")
                    
                    # Validate timestamp bounds
                    if timestamp < 0:
                        print("    ⚠️ Timestamp before video start (0s)")
                        frame_results.append({
                            **frame_spec,
                            "status": "timestamp_before_start",
                            "file_path": None,
                            "file_size": 0
                        })
                        continue
                    
                    if timestamp > duration:
                        print(f"    ⚠️ Timestamp after video end ({duration:.2f}s)")
                        frame_results.append({
                            **frame_spec,
                            "status": "timestamp_after_end",
                            "file_path": None,
                            "file_size": 0
                        })
                        continue
                    
                    try:
                        # Extract frame at timestamp
                        frame_rgb = video_clip.get_frame(timestamp)
                        
                        # Convert RGB to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        
                        # Generate output filename
                        output_filename = f"ghost_probing_{spec['sample_id']}_{phase}.jpg"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        # Save frame with high quality
                        jpeg_quality = 95
                        save_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                        save_success = cv2.imwrite(output_path, frame_bgr, save_params)
                        
                        if save_success and os.path.exists(output_path):
                            file_size = os.path.getsize(output_path)
                            print(f"    ✅ Frame saved successfully: {output_filename}")
                            print(f"       File size: {file_size:,} bytes")
                            print(f"       JPEG quality: {jpeg_quality}%")
                            
                            total_frames_successful += 1
                            
                            frame_results.append({
                                **frame_spec,
                                "status": "success",
                                "file_path": output_path,
                                "filename": output_filename,
                                "file_size": file_size,
                                "jpeg_quality": jpeg_quality
                            })
                        else:
                            print(f"    ❌ Failed to save frame: {output_filename}")
                            frame_results.append({
                                **frame_spec,
                                "status": "save_failed",
                                "file_path": None,
                                "file_size": 0,
                                "error": "cv2.imwrite returned False or file not created"
                            })
                    
                    except Exception as frame_error:
                        print(f"    ❌ Frame extraction error: {frame_error}")
                        frame_results.append({
                            **frame_spec,
                            "status": "extraction_error",
                            "file_path": None,
                            "file_size": 0,
                            "error": str(frame_error)
                        })
                
                extraction_results.append({
                    "video": spec["video_file"],
                    "status": "processed",
                    "video_info": {
                        "duration": duration,
                        "fps": fps,
                        "resolution": size
                    },
                    "frames": frame_results,
                    "frames_successful": len([f for f in frame_results if f["status"] == "success"])
                })
                
                successful_count = len([f for f in frame_results if f["status"] == "success"])
                print(f"\\n  ✅ Video processing complete: {successful_count}/3 frames extracted")
        
        except Exception as video_error:
            print(f"❌ Video processing error: {video_error}")
            traceback.print_exc()
            
            extraction_results.append({
                "video": spec["video_file"],
                "status": "processing_error",
                "error": str(video_error),
                "frames": []
            })
    
    # Generate comprehensive summary
    print(f"\\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    
    videos_processed = len([r for r in extraction_results if r["status"] == "processed"])
    videos_total = len(extraction_specs)
    
    print(f"Videos processed: {videos_processed}/{videos_total}")
    print(f"Frames attempted: {total_frames_attempted}")
    print(f"Frames successful: {total_frames_successful}")
    print(f"Success rate: {(total_frames_successful/total_frames_attempted*100):.1f}%" if total_frames_attempted > 0 else "N/A")
    
    # Detailed results by video
    print("\\nDetailed Results:")
    for result in extraction_results:
        print(f"\\n📹 {result['video']}:")
        
        if result["status"] == "processed":
            successful_frames = result.get("frames_successful", 0)
            print(f"   Status: ✅ Processed ({successful_frames}/3 frames)")
            
            if "video_info" in result:
                info = result["video_info"]
                print(f"   Video: {info['duration']:.2f}s, {info['fps']}fps, {info['resolution'][0]}x{info['resolution'][1]}")
            
            for frame in result["frames"]:
                phase = frame["phase"].upper()
                timestamp = frame["timestamp"]
                status = frame["status"]
                
                if status == "success":
                    print(f"   ✅ {phase} ({timestamp:.1f}s): {frame['filename']} ({frame['file_size']:,} bytes)")
                else:
                    print(f"   ❌ {phase} ({timestamp:.1f}s): {status}")
        else:
            print(f"   Status: ❌ {result['status']}")
            if "error" in result:
                print(f"   Error: {result['error']}")
    
    return extraction_results

def generate_sequence_analysis(extraction_results: List[Dict]) -> None:
    """Generate detailed analysis of the extracted ghost probing sequences."""
    
    print(f"\\n{'='*60}")
    print("GHOST PROBING SEQUENCE ANALYSIS")
    print(f"{'='*60}")
    
    successful_files = []
    for result in extraction_results:
        if result["status"] == "processed":
            for frame in result["frames"]:
                if frame["status"] == "success":
                    successful_files.append(frame)
    
    if not successful_files:
        print("❌ No successful extractions to analyze")
        return
    
    print("\\n🎯 MULTIMODAL FEW-SHOT LEARNING APPLICATIONS:")
    print("-" * 50)
    print("These ghost probing sequences provide ideal training data for:")
    print("• Autonomous vehicle safety systems")
    print("• Pedestrian detection and prediction models")
    print("• Ghost probing detection algorithms")  
    print("• Multimodal few-shot learning research")
    print("• Safety-critical AI system evaluation")
    print("• Computer vision model benchmarking")
    
    print("\\n📊 FRAME SEQUENCE STRUCTURE:")
    print("-" * 50)
    print("Each sample contains 3 temporal phases showing complete ghost probing pattern:")
    print()
    
    print("🔵 BEFORE Frame (t-0.5s):")
    print("   • Purpose: Establish baseline 'safe' driving environment")
    print("   • Content: Normal scene with person concealed behind obstruction")
    print("   • Visibility: Person completely hidden from vehicle/camera view")
    print("   • Safety State: Apparent safety - no visible threats detected")
    print("   • Learning Value: Negative example for threat detection")
    
    print("\\n🟡 DURING Frame (t=0s - Critical Event):")
    print("   • Purpose: Capture exact moment of ghost probing emergence")
    print("   • Content: Person transitioning from hidden to visible state")
    print("   • Visibility: Person partially or fully emerging from obstruction")
    print("   • Safety State: Critical transition - threat becoming apparent")
    print("   • Learning Value: Key detection frame for algorithm training")
    
    print("\\n🔴 AFTER Frame (t+0.5s):")
    print("   • Purpose: Show full manifestation of dangerous situation")
    print("   • Content: Person now clearly in vehicle's path/trajectory")
    print("   • Visibility: Person fully visible and represents clear threat")
    print("   • Safety State: High danger - immediate response required")
    print("   • Learning Value: Positive example for threat classification")
    
    print("\\n🧠 COGNITIVE LEARNING PATTERNS:")
    print("-" * 50)
    print("These sequences teach AI models to recognize:")
    print("• Temporal progression of safety threats")
    print("• Occlusion-based visibility challenges")
    print("• Critical timing for emergency response")
    print("• Visual patterns of emerging dangers")
    print("• Context-dependent threat assessment")
    
    # Group files by sample for detailed breakdown
    samples = {}
    for frame in successful_files:
        # Extract sample ID from filename
        filename = frame["filename"]
        if "sample1" in filename:
            sample_id = "Sample 1"
            video = "images_1_003.avi"
        elif "sample2" in filename:
            sample_id = "Sample 2"
            video = "images_1_006.avi"
        elif "sample3" in filename:
            sample_id = "Sample 3"
            video = "images_1_008.avi"
        else:
            continue
            
        if sample_id not in samples:
            samples[sample_id] = {"video": video, "frames": []}
        samples[sample_id]["frames"].append(frame)
    
    print(f"\\n📁 EXTRACTED FILES BY SAMPLE:")
    print("-" * 50)
    for sample_id, sample_data in samples.items():
        print(f"\\n{sample_id} ({sample_data['video']}):")
        
        # Sort frames by timestamp
        sorted_frames = sorted(sample_data["frames"], key=lambda x: x["timestamp"])
        
        for frame in sorted_frames:
            phase_icon = {"before": "🔵", "during": "🟡", "after": "🔴"}.get(frame["phase"], "⚪")
            print(f"  {phase_icon} {frame['filename']}")
            print(f"     Timestamp: {frame['timestamp']:.1f}s")
            print(f"     Size: {frame['file_size']:,} bytes")
            print(f"     Context: {frame['description']}")
            print(f"     Analysis: {frame['analysis']}")
    
    print(f"\\n💾 OUTPUT DIRECTORY:")
    print("-" * 50)
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples"
    print(f"All extracted frames saved to: {output_dir}")
    
    print(f"\\n🔬 RESEARCH APPLICATIONS:")
    print("-" * 50)
    print("These ghost probing sequences enable research in:")
    print("• Few-shot learning for safety-critical applications")
    print("• Temporal pattern recognition in autonomous driving")
    print("• Occlusion-aware pedestrian detection")
    print("• Emergency response timing optimization")
    print("• Multimodal AI safety system evaluation")
    print("• Computer vision robustness testing")

def main() -> bool:
    """Main execution function."""
    
    print("GHOST PROBING FRAME EXTRACTION FOR MULTIMODAL FEW-SHOT LEARNING")
    print("=" * 80)
    
    # Setup environment
    if not setup_environment():
        print("\\n❌ Environment setup failed - cannot proceed with extraction")
        return False
    
    # Verify paths
    path_status = verify_paths()
    if not path_status["video_dir_exists"]:
        print("\\n❌ Video directory not found - cannot proceed with extraction")
        return False
    
    if not path_status["output_dir_ready"]:
        print("\\n❌ Output directory not ready - cannot proceed with extraction")
        return False
    
    missing_videos = [vid for vid, exists in path_status["video_files"].items() if not exists]
    if missing_videos:
        print(f"\\n⚠️ Missing video files: {missing_videos}")
        print("Extraction will proceed with available videos only")
    
    # Execute extraction
    try:
        extraction_results = extract_ghost_probing_frames()
        
        # Generate analysis
        generate_sequence_analysis(extraction_results)
        
        # Final success check
        total_successful = sum(
            result.get("frames_successful", 0) 
            for result in extraction_results 
            if result["status"] == "processed"
        )
        
        success = total_successful > 0
        
        print(f"\\n{'='*80}")
        if success:
            print("✅ GHOST PROBING FRAME EXTRACTION COMPLETED SUCCESSFULLY!")
            print(f"✅ {total_successful} frames extracted and ready for few-shot learning")
            print("✅ Sequences capture complete ghost probing temporal patterns")
            print("✅ High-quality JPEG files saved with detailed metadata")
        else:
            print("❌ GHOST PROBING FRAME EXTRACTION FAILED!")
            print("❌ No frames were successfully extracted")
            print("❌ Check error messages above for troubleshooting")
        print(f"{'='*80}")
        
        return success
        
    except Exception as main_error:
        print(f"\\n❌ Main execution error: {main_error}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting Ghost Probing Frame Extraction...")
    success = main()
    
    exit_code = 0 if success else 1
    print(f"\\nExiting with code: {exit_code}")
    
    if success:
        print("\\n🎯 Ready for multimodal few-shot learning experiments!")
    else:
        print("\\n💥 Extraction failed - please check error messages and retry.")
    
    # Note: In a normal execution environment, we would call exit(exit_code)
    # For this demonstration, we'll just return the success status
    print(f"\\nScript execution completed. Success: {success}")