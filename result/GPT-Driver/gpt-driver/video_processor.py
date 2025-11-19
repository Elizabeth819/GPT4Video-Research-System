"""
Video processing utilities for GPT-Driver ghost probing detection.
Extracts frames from videos and converts them to base64 for API calls.
"""

import os
import cv2
import base64
import tempfile
import logging
from typing import List, Tuple, Optional
import numpy as np
from moviepy.editor import VideoFileClip

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles video frame extraction and processing for ghost probing detection."""
    
    def __init__(self, frame_interval: int = 10, max_frames: int = 10):
        """
        Initialize video processor.
        
        Args:
            frame_interval: Seconds between frame extractions
            max_frames: Maximum number of frames to extract per video
        """
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        self.temp_dir = None
        
    def extract_frames(self, video_path: str) -> List[str]:
        """
        Extract frames from video at specified intervals.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of base64 encoded frame images
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        frames = []
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            # Load video
            video = VideoFileClip(video_path)
            duration = video.duration
            
            logger.info(f"Processing video: {video_path}")
            logger.info(f"Video duration: {duration:.2f} seconds")
            
            # Calculate frame timestamps
            timestamps = []
            current_time = 0
            while current_time < duration and len(timestamps) < self.max_frames:
                timestamps.append(current_time)
                current_time += self.frame_interval
                
            # Extract frames
            for i, timestamp in enumerate(timestamps):
                try:
                    frame = video.get_frame(timestamp)
                    
                    # Convert to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Save frame temporarily
                    frame_path = os.path.join(self.temp_dir, f"frame_{i:03d}.jpg")
                    cv2.imwrite(frame_path, frame_bgr)
                    
                    # Convert to base64
                    base64_frame = self._image_to_base64(frame_path)
                    frames.append(base64_frame)
                    
                    logger.info(f"Extracted frame {i+1}/{len(timestamps)} at {timestamp:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error extracting frame at {timestamp}s: {e}")
                    continue
                    
            video.close()
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            raise
            
        finally:
            self._cleanup_temp_files()
            
        return frames
    
    def _image_to_base64(self, image_path: str) -> str:
        """
        Convert image file to base64 string.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    
    def _cleanup_temp_files(self):
        """Clean up temporary files and directories."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get basic video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video info
        """
        try:
            video = VideoFileClip(video_path)
            info = {
                'duration': video.duration,
                'fps': video.fps,
                'size': video.size,
                'filename': os.path.basename(video_path)
            }
            video.close()
            return info
        except Exception as e:
            logger.error(f"Error getting video info for {video_path}: {e}")
            return {}
    
    def validate_video(self, video_path: str) -> bool:
        """
        Validate if video file is readable and processable.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if video is valid, False otherwise
        """
        try:
            video = VideoFileClip(video_path)
            duration = video.duration
            video.close()
            
            if duration > 0:
                return True
            else:
                logger.warning(f"Video has zero duration: {video_path}")
                return False
                
        except Exception as e:
            logger.error(f"Video validation failed for {video_path}: {e}")
            return False


def process_dada_video(video_path: str, frame_interval: int = 10, max_frames: int = 10) -> Tuple[List[str], dict]:
    """
    Process DADA-2000 format video for ghost probing detection.
    
    Args:
        video_path: Path to DADA video file
        frame_interval: Seconds between frames
        max_frames: Maximum frames to extract
        
    Returns:
        Tuple of (base64_frames, video_info)
    """
    processor = VideoProcessor(frame_interval, max_frames)
    
    # Validate video first
    if not processor.validate_video(video_path):
        raise ValueError(f"Invalid video file: {video_path}")
    
    # Get video info
    video_info = processor.get_video_info(video_path)
    
    # Extract frames
    frames = processor.extract_frames(video_path)
    
    return frames, video_info


if __name__ == "__main__":
    # Test with a sample video
    test_video = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/images_1_001.avi"
    
    if os.path.exists(test_video):
        try:
            frames, info = process_dada_video(test_video)
            print(f"Successfully processed {info['filename']}")
            print(f"Duration: {info['duration']:.2f}s, FPS: {info['fps']:.2f}")
            print(f"Extracted {len(frames)} frames")
        except Exception as e:
            print(f"Test failed: {e}")
    else:
        print(f"Test video not found: {test_video}")