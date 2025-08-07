#!/usr/bin/env python3
"""Direct execution of ghost probing frame extraction"""

# Execute the extraction directly using exec
extraction_code = '''
import os
import sys

# Setup
project_root = "/Users/wanmeng/repository/GPT4Video-cobra-auto"
sys.path.insert(0, project_root)

print("DIRECT GHOST PROBING FRAME EXTRACTION")
print("=" * 50)

# Import required libraries
try:
    import cv2
    from moviepy.editor import VideoFileClip
    print(f"✅ Libraries loaded: OpenCV {cv2.__version__}, MoviePy OK")
    libraries_ok = True
except ImportError as e:
    print(f"❌ Library import failed: {e}")
    libraries_ok = False

if libraries_ok:
    # Paths
    video_dir = os.path.join(project_root, "DADA-2000-videos")
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Video specifications
    video_specs = [
        {"file": "images_1_003.avi", "time": 2.0, "id": "sample1", "desc": "Ghost probing at 2s"},
        {"file": "images_1_006.avi", "time": 6.0, "id": "sample2", "desc": "Ghost probing at 6s"},
        {"file": "images_1_008.avi", "time": 2.0, "id": "sample3", "desc": "Ghost probing at 2s"}
    ]
    
    extracted_files = []
    
    for spec in video_specs:
        video_path = os.path.join(video_dir, spec["file"])
        print(f"\\nProcessing {spec['file']} ({spec['desc']})...")
        print(f"Path: {video_path}")
        print(f"Exists: {os.path.exists(video_path)}")
        
        if not os.path.exists(video_path):
            print("  ❌ Video file not found")
            continue
        
        try:
            # Load video
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
                fps = clip.fps
                size = clip.size
                
                print(f"  ✅ Video loaded: {duration:.2f}s, {fps}fps, {size}")
                
                # Extract frames
                event_time = spec["time"]
                frame_offsets = [
                    (-0.5, "before", "Normal scene with person hidden behind obstruction"),
                    (0.0, "during", "Critical moment - person emerging from behind obstruction"), 
                    (0.5, "after", "Dangerous situation - person in vehicle's path")
                ]
                
                for offset, phase, description in frame_offsets:
                    timestamp = event_time + offset
                    
                    print(f"    Extracting {phase} frame at {timestamp:.1f}s...")
                    print(f"      Context: {description}")
                    
                    if timestamp < 0 or timestamp > duration:
                        print(f"      ⚠️ Timestamp out of bounds (0-{duration:.2f}s)")
                        continue
                    
                    try:
                        # Get frame
                        frame_rgb = clip.get_frame(timestamp)
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        
                        # Save frame
                        filename = f"ghost_probing_{spec['id']}_{phase}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        
                        # High quality JPEG
                        save_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
                        success = cv2.imwrite(filepath, frame_bgr, save_params)
                        
                        if success and os.path.exists(filepath):
                            file_size = os.path.getsize(filepath)
                            print(f"      ✅ Saved: {filename} ({file_size:,} bytes)")
                            extracted_files.append({
                                'video': spec['file'],
                                'sample_id': spec['id'],
                                'phase': phase,
                                'timestamp': timestamp,
                                'filename': filename,
                                'filepath': filepath,
                                'size': file_size,
                                'description': description
                            })
                        else:
                            print(f"      ❌ Failed to save {filename}")
                    
                    except Exception as frame_err:
                        print(f"      ❌ Frame extraction error: {frame_err}")
        
        except Exception as video_err:
            print(f"  ❌ Video processing error: {video_err}")
    
    # Summary
    print(f"\\n{'='*50}")
    print("EXTRACTION RESULTS")
    print(f"{'='*50}")
    print(f"Total frames extracted: {len(extracted_files)}")
    print(f"Expected frames: {len(video_specs) * 3}")
    
    if extracted_files:
        print(f"\\nExtracted files:")
        for file_info in extracted_files:
            print(f"  ✅ {file_info['filename']} ({file_info['size']:,} bytes)")
            print(f"     {file_info['video']} at {file_info['timestamp']:.1f}s - {file_info['description']}")
        
        print(f"\\nOutput directory: {output_dir}")
        
        print(f"\\nGhost Probing Sequence Analysis:")
        print(f"Each sample demonstrates the complete ghost probing pattern:")
        print(f"• BEFORE: Shows normal driving scene with person hidden behind obstruction")
        print(f"• DURING: Captures the critical moment as person emerges from blind spot")
        print(f"• AFTER: Reveals dangerous situation with person now in vehicle's path")
        print(f"\\nThese visual sequences provide ideal training data for:")
        print(f"• Multimodal few-shot learning models")
        print(f"• Autonomous vehicle safety systems") 
        print(f"• Ghost probing detection algorithms")
        print(f"• Safety research and evaluation")
        
        # Group by sample for easy reference
        print(f"\\nFrames by sample:")
        for i in range(1, 4):
            sample_files = [f for f in extracted_files if f['sample_id'] == f'sample{i}']
            if sample_files:
                video_name = sample_files[0]['video']
                print(f"  Sample {i} ({video_name}):")
                for f in sorted(sample_files, key=lambda x: x['timestamp']):
                    print(f"    {f['filename']} - {f['description']}")
    else:
        print("❌ No frames were successfully extracted")

print("\\nDirect execution completed!")
'''

# Execute the extraction code
try:
    exec(extraction_code)
    print("✅ Extraction code executed successfully")
except Exception as e:
    print(f"❌ Execution error: {e}")
    import traceback
    traceback.print_exc()