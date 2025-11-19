"""
Batch processing script for ghost probing detection on DADA-100 videos.
Processes all 100 videos and generates evaluation reports.
"""

import os
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
from ghost_probing_detector import GhostProbingDetector
from evaluate_ghost_probing import GhostProbingEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_ghost_probing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchGhostProbingProcessor:
    """Batch processor for ghost probing detection on multiple videos."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", max_retries: int = 3):
        """
        Initialize batch processor.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use
            max_retries: Maximum retry attempts
        """
        self.detector = GhostProbingDetector(api_key, model, max_retries)
        self.results = []
        self.failed_videos = []
        self.start_time = None
        
    def get_dada_video_list(self, video_folder: str, limit: int = None) -> List[str]:
        """
        Get list of DADA-100 videos to process.
        
        Args:
            video_folder: Folder containing DADA videos
            limit: Optional limit on number of videos to process
            
        Returns:
            List of video file paths
        """
        videos = []
        
        if not os.path.exists(video_folder):
            raise FileNotFoundError(f"Video folder not found: {video_folder}")
        
        # Get all .avi files matching DADA naming pattern
        for filename in sorted(os.listdir(video_folder)):
            if filename.startswith('images_') and filename.endswith('.avi'):
                video_path = os.path.join(video_folder, filename)
                if os.path.exists(video_path):
                    videos.append(video_path)
                    
                    if limit and len(videos) >= limit:
                        break
        
        logger.info(f"Found {len(videos)} DADA videos to process")
        return videos
    
    def process_videos(self, video_folder: str, limit: int = None, 
                      start_from: int = 0, checkpoint_interval: int = 10) -> List[Dict[str, Any]]:
        """
        Process videos in batch.
        
        Args:
            video_folder: Folder containing videos
            limit: Optional limit on number of videos
            start_from: Index to start processing from (for resuming)
            checkpoint_interval: Save results every N videos
            
        Returns:
            List of processing results
        """
        self.start_time = time.time()
        videos = self.get_dada_video_list(video_folder, limit)
        
        if start_from > 0:
            videos = videos[start_from:]
            logger.info(f"Resuming from video {start_from}, processing {len(videos)} remaining videos")
        
        total_videos = len(videos)
        processed_count = 0
        
        for i, video_path in enumerate(videos, start_from + 1):
            logger.info(f"Processing video {i}/{start_from + total_videos}: {os.path.basename(video_path)}")
            
            try:
                result = self.detector.analyze_video(video_path)
                self.results.append(result)
                processed_count += 1
                
                # Log progress
                elapsed_time = time.time() - self.start_time
                avg_time_per_video = elapsed_time / processed_count
                remaining_videos = total_videos - processed_count
                eta_seconds = avg_time_per_video * remaining_videos
                eta_minutes = eta_seconds / 60
                
                logger.info(f"Processed {processed_count}/{total_videos} videos. "
                           f"ETA: {eta_minutes:.1f} minutes")
                
                # Save checkpoint
                if processed_count % checkpoint_interval == 0:
                    self._save_checkpoint(processed_count)
                
                # Small delay to avoid rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to process video {video_path}: {e}")
                self.failed_videos.append({
                    'video_path': video_path,
                    'error': str(e),
                    'timestamp': time.time()
                })
                continue
        
        # Final save
        self._save_checkpoint(processed_count, final=True)
        
        logger.info(f"Batch processing completed. Processed {processed_count} videos, "
                   f"{len(self.failed_videos)} failed")
        
        return self.results
    
    def _save_checkpoint(self, processed_count: int, final: bool = False):
        """Save intermediate results as checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint_data = {
            'timestamp': timestamp,
            'processed_count': processed_count,
            'total_results': len(self.results),
            'failed_videos': len(self.failed_videos),
            'results': self.results,
            'failed': self.failed_videos
        }
        
        suffix = 'final' if final else f'checkpoint_{processed_count}'
        checkpoint_file = f"ghost_probing_results_{suffix}_{timestamp}.json"
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            if final:
                logger.info(f"Final results saved to {checkpoint_file}")
            else:
                logger.info(f"Checkpoint saved: {checkpoint_file}")
                
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def evaluate_results(self, ground_truth_path: str) -> Dict[str, Any]:
        """
        Evaluate results against ground truth.
        
        Args:
            ground_truth_path: Path to ground truth CSV file
            
        Returns:
            Evaluation results
        """
        if not self.results:
            raise ValueError("No results to evaluate")
        
        evaluator = GhostProbingEvaluator(ground_truth_path)
        evaluation_results = evaluator.evaluate_results(self.results)
        
        return evaluation_results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate processing summary."""
        if not self.start_time:
            return {}
        
        total_time = time.time() - self.start_time
        processed_videos = len(self.results)
        failed_videos = len(self.failed_videos)
        
        # Count detection results
        high_confidence_detections = 0
        potential_detections = 0
        
        for result in self.results:
            if 'ghost_probing_detected' in result:
                detection = result['ghost_probing_detected']
                if detection.get('high_confidence_ghost_probing', False):
                    high_confidence_detections += 1
                elif detection.get('potential_ghost_probing', False):
                    potential_detections += 1
        
        summary = {
            'processing_summary': {
                'total_processing_time_minutes': total_time / 60,
                'average_time_per_video_seconds': total_time / max(processed_videos, 1),
                'videos_processed_successfully': processed_videos,
                'videos_failed': failed_videos,
                'success_rate': processed_videos / max(processed_videos + failed_videos, 1)
            },
            'detection_summary': {
                'high_confidence_ghost_probing': high_confidence_detections,
                'potential_ghost_probing': potential_detections,
                'no_ghost_probing': processed_videos - high_confidence_detections - potential_detections,
                'detection_rate': (high_confidence_detections + potential_detections) / max(processed_videos, 1)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return summary


def main():
    """Main batch processing function."""
    parser = argparse.ArgumentParser(description="Batch ghost probing detection on DADA-100 videos")
    parser.add_argument("--video-folder", default="/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos",
                       help="Folder containing DADA videos")
    parser.add_argument("--ground-truth", default="/Users/wanmeng/repository/GPT4Video-cobra-auto/result/groundtruth_labels.csv",
                       help="Path to ground truth labels CSV")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o", help="GPT model to use")
    parser.add_argument("--limit", type=int, help="Limit number of videos to process")
    parser.add_argument("--start-from", type=int, default=0, help="Start processing from video index")
    parser.add_argument("--checkpoint-interval", type=int, default=10, 
                       help="Save checkpoint every N videos")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after processing")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY environment variable")
        return
    
    # Validate paths
    if not os.path.exists(args.video_folder):
        logger.error(f"Video folder not found: {args.video_folder}")
        return
    
    if args.evaluate and not os.path.exists(args.ground_truth):
        logger.error(f"Ground truth file not found: {args.ground_truth}")
        return
    
    try:
        # Initialize processor
        processor = BatchGhostProbingProcessor(api_key, args.model)
        
        # Process videos
        logger.info("Starting batch processing...")
        results = processor.process_videos(
            args.video_folder, 
            limit=args.limit,
            start_from=args.start_from,
            checkpoint_interval=args.checkpoint_interval
        )
        
        # Generate summary
        summary = processor.generate_summary()
        print(f"\nProcessing Summary:")
        print(f"- Videos processed: {summary['processing_summary']['videos_processed_successfully']}")
        print(f"- Videos failed: {summary['processing_summary']['videos_failed']}")
        print(f"- Total time: {summary['processing_summary']['total_processing_time_minutes']:.1f} minutes")
        print(f"- High confidence detections: {summary['detection_summary']['high_confidence_ghost_probing']}")
        print(f"- Potential detections: {summary['detection_summary']['potential_ghost_probing']}")
        
        # Run evaluation if requested
        if args.evaluate and results:
            logger.info("Running evaluation...")
            evaluation_results = processor.evaluate_results(args.ground_truth)
            
            # Print evaluation summary
            metrics = evaluation_results['metrics']
            print(f"\nEvaluation Results:")
            print(f"- Accuracy: {metrics['accuracy']:.4f}")
            print(f"- Precision: {metrics['precision']:.4f}")
            print(f"- Recall: {metrics['recall']:.4f}")
            print(f"- F1 Score: {metrics['f1_score']:.4f}")
            
            # Save detailed evaluation
            evaluator = GhostProbingEvaluator(args.ground_truth)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            evaluator.generate_report(evaluation_results, f"evaluation_report_{timestamp}.txt")
            evaluator.save_detailed_results(evaluation_results, f"evaluation_detailed_{timestamp}.json")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise


if __name__ == "__main__":
    main()