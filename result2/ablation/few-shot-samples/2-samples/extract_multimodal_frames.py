#!/usr/bin/env python3
"""
Extract multimodal few-shot learning frames for ghost probing detection.
Creates a comprehensive set of before/during/after frames for the 3 key ghost probing videos.
"""

import cv2
import os
import json
from pathlib import Path
from datetime import datetime

def extract_multimodal_frames():
    """Extract frames for multimodal few-shot learning"""
    
    # Configuration
    base_dir = Path("/Users/wanmeng/repository/GPT4Video-cobra-auto")
    video_dir = base_dir / "DADA-2000-videos"
    output_dir = base_dir / "result2" / "ablation" / "few-shot-samples" / "2-samples" / "multimodal_frames"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ghost probing video configurations
    ghost_probing_configs = [
        {
            "video_id": "images_1_002",
            "filename": "images_1_002.avi",
            "ghost_probing_time": 5,
            "description": "Male child (5 years old) emerges from behind parked vehicle on left side",
            "analysis": "Run 8 analysis: True Positive - Child suddenly appears from blind spot",
            "timestamps": [4, 5, 6],  # before, during, after
            "frame_labels": ["before_emergence", "moment_of_emergence", "after_emergence"]
        },
        {
            "video_id": "images_1_003", 
            "filename": "images_1_003.avi",
            "ghost_probing_time": 2,
            "description": "Male pedestrian emerges from behind parked black car on left side",
            "analysis": "Run 8 analysis: True Positive - Adult pedestrian from behind black car",  
            "timestamps": [1, 2, 3],  # before, during, after
            "frame_labels": ["before_emergence", "moment_of_emergence", "after_emergence"]
        },
        {
            "video_id": "images_1_008",
            "filename": "images_1_008.avi", 
            "ghost_probing_time": 3,
            "description": "Male pedestrian wearing dark clothing emerges from behind parked white truck on left side",
            "analysis": "Run 8 analysis: True Positive - Dark-clothed pedestrian from behind white truck",
            "timestamps": [2, 3, 4],  # before, during, after  
            "frame_labels": ["before_emergence", "moment_of_emergence", "after_emergence"]
        }
    ]
    
    print("=" * 80)
    print("EXTRACTING MULTIMODAL FEW-SHOT LEARNING FRAMES")
    print("Target: Ghost Probing Detection")
    print("Source: DADA-2000 Dataset True Positive Examples")
    print("=" * 80)
    
    extraction_results = []
    total_extracted = 0
    
    for config in ghost_probing_configs:
        print(f"\n📹 Processing: {config['video_id']}")
        print(f"   Ghost probing at {config['ghost_probing_time']}s")
        print(f"   Description: {config['description']}")
        
        video_path = video_dir / config["filename"]
        
        if not video_path.exists():
            print(f"   ❌ ERROR: Video not found - {video_path}")
            continue
            
        # Open video with OpenCV
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"   ❌ ERROR: Could not open video")
            continue
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"   📊 Video properties: FPS={fps:.2f}, Frames={total_frames}, Duration={duration:.2f}s")
        
        if fps <= 0:
            print(f"   ❌ ERROR: Invalid FPS")
            cap.release()
            continue
            
        # Extract frames at specified timestamps
        video_results = []
        
        for i, (timestamp, label) in enumerate(zip(config["timestamps"], config["frame_labels"])):
            if timestamp > duration:
                print(f"   ⚠️  WARNING: Timestamp {timestamp}s exceeds duration {duration:.2f}s")
                continue
                
            # Calculate frame number and seek
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # Create descriptive filename
                filename = f"{config['video_id']}_{label}_{timestamp}s.jpg"
                frame_path = output_dir / filename
                
                # Save frame with high quality
                success = cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                if success:
                    file_size = frame_path.stat().st_size
                    print(f"   ✅ {label} ({timestamp}s) -> {filename} ({file_size} bytes)")
                    
                    video_results.append({
                        "timestamp": timestamp,
                        "label": label,
                        "filename": filename,
                        "path": str(frame_path),
                        "file_size": file_size,
                        "frame_number": frame_number
                    })
                    total_extracted += 1
                else:
                    print(f"   ❌ Failed to save {label} frame at {timestamp}s")
            else:
                print(f"   ❌ Failed to read {label} frame at {timestamp}s")
                
        cap.release()
        
        # Store results for this video
        if video_results:
            extraction_results.append({
                "video_id": config["video_id"],
                "filename": config["filename"], 
                "ghost_probing_time": config["ghost_probing_time"],
                "description": config["description"],
                "analysis": config["analysis"],
                "video_duration": duration,
                "video_fps": fps,
                "frames_extracted": len(video_results),
                "frames": video_results
            })
            
            print(f"   ✅ Successfully extracted {len(video_results)} frames from {config['video_id']}")
        else:
            print(f"   ❌ No frames extracted from {config['video_id']}")
    
    # Save extraction metadata
    metadata = {
        "extraction_timestamp": datetime.now().isoformat(),
        "purpose": "Multimodal Few-Shot Learning for Ghost Probing Detection", 
        "dataset": "DADA-2000",
        "total_videos_processed": len(ghost_probing_configs),
        "total_frames_extracted": total_extracted,
        "output_directory": str(output_dir),
        "extraction_results": extraction_results
    }
    
    metadata_path = output_dir / "extraction_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print final summary
    print(f"\n" + "=" * 80)
    print("EXTRACTION COMPLETED")
    print("=" * 80)
    print(f"📊 Total videos processed: {len(extraction_results)}")
    print(f"📊 Total frames extracted: {total_extracted}")
    print(f"📊 Output directory: {output_dir}")
    print(f"📊 Metadata saved: {metadata_path}")
    
    print(f"\n📋 EXTRACTED FRAMES SUMMARY:")
    for result in extraction_results:
        print(f"\n🎬 {result['video_id']} (Ghost probing at {result['ghost_probing_time']}s):")
        for frame in result['frames']:
            frame_type = "🔴 CRITICAL" if frame['label'] == "moment_of_emergence" else "📷 CONTEXT"
            print(f"   {frame_type} {frame['label']} ({frame['timestamp']}s) - {frame['filename']}")
    
    return total_extracted, extraction_results

if __name__ == "__main__":
    try:
        total_frames, results = extract_multimodal_frames()
        print(f"\n🎉 SUCCESS: Extracted {total_frames} frames for multimodal few-shot learning!")
        
        if total_frames > 0:
            print(f"\n💡 USAGE RECOMMENDATIONS:")
            print(f"   - Use 'moment_of_emergence' frames as PRIMARY examples in few-shot prompts")
            print(f"   - Use 'before_emergence' frames to show normal driving context") 
            print(f"   - Use 'after_emergence' frames to show the dangerous situation created")
            print(f"   - Compare against text-only few-shot learning for ablation study")
        
    except Exception as e:
        print(f"\n❌ EXTRACTION FAILED: {e}")
        import traceback
        traceback.print_exc()