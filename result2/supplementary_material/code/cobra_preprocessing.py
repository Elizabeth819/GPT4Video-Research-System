#!/usr/bin/env python3
"""
Cobra Video Preprocessing Pipeline
Implements video chunking, frame sampling, and audio extraction
as described in Section 3 of the AutoDrive-GPT paper.
"""

import cv2
import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple
import moviepy.editor as mp
from moviepy.audio.io.AudioFileClip import AudioFileClip
import numpy as np

class CobraPreprocessor:
    """
    Cobra video preprocessing pipeline for autonomous driving video analysis.
    
    Features:
    - Video chunking into temporal intervals
    - Frame sampling at specified FPS
    - Audio extraction and synchronization
    - Structured output for GPT-4o analysis
    """
    
    def __init__(self, 
                 interval_seconds: int = 10,
                 frames_per_interval: int = 10,
                 target_fps: float = 1.0,
                 temp_dir: str = None):
        """
        Initialize Cobra preprocessor.
        
        Args:
            interval_seconds: Duration of each video chunk in seconds
            frames_per_interval: Number of frames to extract per interval
            target_fps: Target frame rate for extraction (1 FPS = 1 frame per second)
            temp_dir: Temporary directory for intermediate files
        """
        self.interval_seconds = interval_seconds
        self.frames_per_interval = frames_per_interval
        self.target_fps = target_fps
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
        # Create temporary directories
        self.frames_dir = Path(self.temp_dir) / "frames" 
        self.audio_dir = Path(self.temp_dir) / "audio"
        self.frames_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
    
    def extract_video_info(self, video_path: str) -> Dict:
        """
        Extract basic video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video metadata
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration_seconds': duration,
            'total_intervals': int(np.ceil(duration / self.interval_seconds))
        }
    
    def extract_frames(self, video_path: str, video_id: str) -> List[Dict]:
        """
        Extract frames from video at specified intervals.
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for the video
            
        Returns:
            List of dictionaries containing frame information
        """
        video_info = self.extract_video_info(video_path)
        cap = cv2.VideoCapture(video_path)
        
        intervals_data = []
        
        for interval_idx in range(video_info['total_intervals']):
            start_time = interval_idx * self.interval_seconds
            end_time = min((interval_idx + 1) * self.interval_seconds, video_info['duration_seconds'])
            
            # Calculate frame positions within this interval
            frames_in_interval = []
            for frame_idx in range(self.frames_per_interval):
                # Evenly distribute frames across the interval
                time_offset = (frame_idx / max(1, self.frames_per_interval - 1)) * (end_time - start_time)
                frame_time = start_time + time_offset
                
                # Convert time to frame number
                frame_number = int(frame_time * video_info['fps'])
                
                if frame_number < video_info['frame_count']:
                    frames_in_interval.append((frame_idx, frame_number, frame_time))
            
            # Extract frames for this interval
            interval_frames = []
            for frame_idx, frame_number, frame_time in frames_in_interval:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    # Save frame
                    frame_filename = f"{video_id}_interval_{interval_idx:03d}_frame_{frame_idx:02d}.jpg"
                    frame_path = self.frames_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    
                    interval_frames.append({
                        'frame_index': frame_idx,
                        'frame_number': frame_number,
                        'timestamp': frame_time,
                        'frame_path': str(frame_path),
                        'frame_filename': frame_filename
                    })
            
            intervals_data.append({
                'interval_index': interval_idx,
                'start_time': start_time,
                'end_time': end_time,
                'frames': interval_frames
            })
        
        cap.release()
        return intervals_data
    
    def extract_audio(self, video_path: str, video_id: str) -> Dict:
        """
        Extract audio track from video.
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for the video
            
        Returns:
            Dictionary containing audio information
        """
        try:
            # Load video with moviepy
            video = mp.VideoFileClip(video_path)
            
            if video.audio is None:
                return {
                    'has_audio': False,
                    'audio_path': None,
                    'duration': 0
                }
            
            # Extract audio
            audio_filename = f"{video_id}_audio.wav"
            audio_path = self.audio_dir / audio_filename
            
            video.audio.write_audiofile(str(audio_path), verbose=False, logger=None)
            
            audio_info = {
                'has_audio': True,
                'audio_path': str(audio_path),
                'audio_filename': audio_filename,
                'duration': video.audio.duration,
                'fps': video.audio.fps if hasattr(video.audio, 'fps') else None
            }
            
            video.close()
            return audio_info
            
        except Exception as e:
            print(f"Audio extraction failed for {video_path}: {str(e)}")
            return {
                'has_audio': False,
                'audio_path': None,
                'error': str(e)
            }
    
    def process_video(self, video_path: str, video_id: str = None) -> Dict:
        """
        Complete video preprocessing pipeline.
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier (auto-generated if None)
            
        Returns:
            Dictionary containing all preprocessing results
        """
        if video_id is None:
            video_id = Path(video_path).stem
        
        print(f"Processing video: {video_id}")
        
        # Extract video information
        video_info = self.extract_video_info(video_path)
        print(f"Video duration: {video_info['duration_seconds']:.2f}s, "
              f"Resolution: {video_info['width']}x{video_info['height']}")
        
        # Extract frames
        print("Extracting frames...")
        intervals_data = self.extract_frames(video_path, video_id)
        
        # Extract audio
        print("Extracting audio...")
        audio_info = self.extract_audio(video_path, video_id)
        
        # Compile results
        processing_results = {
            'video_id': video_id,
            'video_path': video_path,
            'video_info': video_info,
            'audio_info': audio_info,
            'intervals': intervals_data,
            'preprocessing_config': {
                'interval_seconds': self.interval_seconds,
                'frames_per_interval': self.frames_per_interval,
                'target_fps': self.target_fps
            }
        }
        
        print(f"Completed processing: {len(intervals_data)} intervals, "
              f"{sum(len(interval['frames']) for interval in intervals_data)} frames")
        
        return processing_results
    
    def cleanup_temp_files(self, video_id: str = None):
        """
        Clean up temporary files.
        
        Args:
            video_id: If provided, only clean files for this video
        """
        import shutil
        
        if video_id:
            # Clean specific video files
            for file_path in self.frames_dir.glob(f"{video_id}*"):
                file_path.unlink()
            for file_path in self.audio_dir.glob(f"{video_id}*"):
                file_path.unlink()
        else:
            # Clean all temporary files
            if self.frames_dir.exists():
                shutil.rmtree(self.frames_dir)
            if self.audio_dir.exists():
                shutil.rmtree(self.audio_dir)
    
    def save_preprocessing_results(self, results: Dict, output_path: str):
        """
        Save preprocessing results to JSON file.
        
        Args:
            results: Processing results dictionary
            output_path: Path to save JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    """
    Example usage of Cobra preprocessor.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Cobra Video Preprocessing Pipeline')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--video-id', help='Video identifier (auto-generated if not provided)')
    parser.add_argument('--interval', type=int, default=10, help='Interval duration in seconds')
    parser.add_argument('--frames', type=int, default=10, help='Frames per interval')
    parser.add_argument('--fps', type=float, default=1.0, help='Target frame rate')
    parser.add_argument('--output', help='Output JSON file path')
    parser.add_argument('--no-cleanup', action='store_true', help='Keep temporary files')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = CobraPreprocessor(
        interval_seconds=args.interval,
        frames_per_interval=args.frames,
        target_fps=args.fps
    )
    
    # Process video
    try:
        results = preprocessor.process_video(args.video_path, args.video_id)
        
        # Save results if output path provided
        if args.output:
            preprocessor.save_preprocessing_results(results, args.output)
            print(f"Results saved to: {args.output}")
        
        # Cleanup unless specified otherwise
        if not args.no_cleanup:
            preprocessor.cleanup_temp_files(results['video_id'])
            print("Temporary files cleaned up")
        
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())