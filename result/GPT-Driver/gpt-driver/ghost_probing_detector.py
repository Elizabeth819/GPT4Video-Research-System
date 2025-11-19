"""
Ghost probing detection using GPT-4.1 balanced prompt.
Based on the optimized prompt from BALANCED_GPT41_PROMPT_FINAL.md
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import openai
from video_processor import VideoProcessor, process_dada_video
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GhostProbingDetector:
    """Ghost probing detection system using GPT-4.1 balanced prompt."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", max_retries: int = 3):
        """
        Initialize the ghost probing detector.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use (default: gpt-4o)
            max_retries: Maximum retry attempts for API calls
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        openai.api_key = api_key
        
        # Initialize video processor
        self.video_processor = VideoProcessor(frame_interval=10, max_frames=10)
        
        # Balanced prompt template based on BALANCED_GPT41_PROMPT_FINAL.md
        self.system_prompt = self._build_system_prompt()
        
    def _build_system_prompt(self) -> str:
        """Build the balanced ghost probing detection prompt."""
        return """You are VideoAnalyzerGPT analyzing a series of SEQUENTIAL images from a driving video to detect potential ghost probing incidents.

IMPORTANT: For ghost probing detection, consider THREE categories:

**1. HIGH-CONFIDENCE Ghost Probing (use "ghost probing" in key_actions)**:
- Object appears EXTREMELY close (within 1-2 vehicle lengths, <3 meters) 
- Appearance is SUDDEN and from blind spots (behind parked cars, buildings, corners)
- Occurs in HIGH-RISK environments: highways, rural roads, parking lots, uncontrolled intersections
- Requires IMMEDIATE emergency braking/swerving to avoid collision
- Movement is COMPLETELY UNPREDICTABLE and violates traffic expectations

**2. POTENTIAL Ghost Probing (use "potential ghost probing" in key_actions)**:
- Object appears suddenly but at moderate distance (3-5 meters)
- Sudden movement in environments where some unpredictability exists
- Requires emergency braking but collision risk is moderate
- Movement is unexpected but not completely impossible given the context

**3. NORMAL Traffic Situations (do NOT use "ghost probing")**:
- Pedestrians crossing at intersections, crosswalks, or traffic lights
- Vehicles making normal lane changes, turns, or merging with signals
- Cyclists following predictable paths in urban areas or bike lanes
- Any movement that is EXPECTED given the traffic environment and context

**Environment Context Guidelines**:
- INTERSECTION/CROSSWALK: Expect pedestrians and cyclists - use "emergency braking due to pedestrian crossing"
- HIGHWAY/RURAL: Higher chance of genuine ghost probing - be more sensitive
- PARKING LOT: Expect sudden vehicle movements - use "potential ghost probing" if very sudden
- URBAN STREET: Mixed - consider visibility and predictability

**Analysis Format**:
For each timestamp interval, provide:
1. **timestamp**: Time range (e.g., "0-10s")
2. **summary**: Brief scene description
3. **actions**: Current actions happening
4. **characters**: People present
5. **key_objects**: Important objects/vehicles
6. **key_actions**: Most significant actions (use ghost probing terms here if detected)
7. **next_action**: Predicted next action

**Critical**: Only use "ghost probing" or "potential ghost probing" in key_actions when criteria are strictly met. Use descriptive alternatives for normal traffic situations."""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_gpt_api(self, messages: List[Dict]) -> str:
        """
        Make API call to GPT with retry logic.
        
        Args:
            messages: List of message dictionaries for the API call
            
        Returns:
            Response content from GPT
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze a single video for ghost probing incidents.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Analyzing video: {video_path}")
        
        try:
            # Extract frames from video
            frames, video_info = process_dada_video(video_path)
            
            if not frames:
                raise ValueError("No frames extracted from video")
            
            # Prepare messages for API call
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add user message with frames
            user_content = [
                {
                    "type": "text",
                    "text": f"Analyze this driving video sequence ({video_info.get('filename', 'unknown')}) for ghost probing incidents. The video has {len(frames)} frames extracted at 10-second intervals."
                }
            ]
            
            # Add frames to content
            for i, frame in enumerate(frames):
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame}"
                    }
                })
            
            messages.append({"role": "user", "content": user_content})
            
            # Make API call
            response = self._call_gpt_api(messages)
            
            # Parse response
            result = {
                "video_path": video_path,
                "video_info": video_info,
                "analysis": response,
                "ghost_probing_detected": self._detect_ghost_probing(response),
                "timestamp": time.time()
            }
            
            logger.info(f"Analysis completed for {video_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing video {video_path}: {e}")
            return {
                "video_path": video_path,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _detect_ghost_probing(self, analysis_text: str) -> Dict[str, Any]:
        """
        Detect ghost probing mentions in the analysis text.
        
        Args:
            analysis_text: GPT analysis response
            
        Returns:
            Detection results
        """
        analysis_lower = analysis_text.lower()
        
        # Check for ghost probing indicators
        high_confidence = "ghost probing" in analysis_lower and "potential ghost probing" not in analysis_lower
        potential = "potential ghost probing" in analysis_lower
        
        detection_result = {
            "high_confidence_ghost_probing": high_confidence,
            "potential_ghost_probing": potential,
            "any_ghost_probing": high_confidence or potential,
            "confidence_level": "high" if high_confidence else ("potential" if potential else "none")
        }
        
        return detection_result
    
    def batch_analyze(self, video_folder: str, video_list: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Analyze multiple videos for ghost probing.
        
        Args:
            video_folder: Folder containing videos
            video_list: Optional list of specific video filenames to process
            
        Returns:
            List of analysis results
        """
        results = []
        
        # Get list of videos to process
        if video_list:
            videos = [os.path.join(video_folder, v) for v in video_list if os.path.exists(os.path.join(video_folder, v))]
        else:
            videos = [os.path.join(video_folder, f) for f in os.listdir(video_folder) 
                     if f.endswith(('.avi', '.mp4', '.mov'))]
        
        logger.info(f"Processing {len(videos)} videos from {video_folder}")
        
        for i, video_path in enumerate(videos, 1):
            logger.info(f"Processing video {i}/{len(videos)}: {os.path.basename(video_path)}")
            
            result = self.analyze_video(video_path)
            results.append(result)
            
            # Small delay to avoid rate limiting
            time.sleep(1)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """
        Save analysis results to JSON file.
        
        Args:
            results: List of analysis results
            output_path: Path to save results
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise


def main():
    """Test the ghost probing detector."""
    # Set up API key (should be set as environment variable)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    # Initialize detector
    detector = GhostProbingDetector(api_key)
    
    # Test with a single video
    test_video = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/images_1_001.avi"
    
    if os.path.exists(test_video):
        result = detector.analyze_video(test_video)
        print(f"Analysis result: {json.dumps(result, indent=2)}")
    else:
        logger.error(f"Test video not found: {test_video}")


if __name__ == "__main__":
    main()