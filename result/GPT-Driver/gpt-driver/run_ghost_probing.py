#!/usr/bin/env python3
"""
Main runner script for ghost probing detection on DADA-100 videos.
Provides simple interface for running detection and evaluation.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ghost_probing_detector import GhostProbingDetector
from evaluate_ghost_probing import GhostProbingEvaluator
from batch_ghost_probing import BatchGhostProbingProcessor

def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'ghost_probing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def run_single_video(args):
    """Run ghost probing detection on a single video."""
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY environment variable")
        return 1
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    print(f"Analyzing video: {args.video}")
    
    detector = GhostProbingDetector(api_key, args.model)
    result = detector.analyze_video(args.video)
    
    if 'error' in result:
        print(f"Error analyzing video: {result['error']}")
        return 1
    
    # Print results
    print(f"\nAnalysis Results:")
    print(f"Video: {result.get('video_path', 'Unknown')}")
    
    ghost_detection = result.get('ghost_probing_detected', {})
    if ghost_detection.get('any_ghost_probing', False):
        confidence = ghost_detection.get('confidence_level', 'unknown')
        print(f"🚨 Ghost Probing Detected (Confidence: {confidence})")
    else:
        print("✅ No Ghost Probing Detected")
    
    print(f"\nDetailed Analysis:")
    print(result.get('analysis', 'No analysis available'))
    
    # Save results if requested
    if args.output:
        detector.save_results([result], args.output)
        print(f"\nResults saved to: {args.output}")
    
    return 0

def run_batch_processing(args):
    """Run batch processing on multiple videos."""
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY environment variable")
        return 1
    
    if not os.path.exists(args.video_folder):
        print(f"Error: Video folder not found: {args.video_folder}")
        return 1
    
    print(f"Starting batch processing...")
    print(f"Video folder: {args.video_folder}")
    print(f"Model: {args.model}")
    if args.limit:
        print(f"Processing limit: {args.limit} videos")
    
    processor = BatchGhostProbingProcessor(api_key, args.model)
    
    try:
        results = processor.process_videos(
            args.video_folder,
            limit=args.limit,
            start_from=args.start_from,
            checkpoint_interval=args.checkpoint_interval
        )
        
        # Generate summary
        summary = processor.generate_summary()
        print(f"\n📊 Processing Summary:")
        print(f"   Successfully processed: {summary['processing_summary']['videos_processed_successfully']} videos")
        print(f"   Failed: {summary['processing_summary']['videos_failed']} videos")
        print(f"   Total time: {summary['processing_summary']['total_processing_time_minutes']:.1f} minutes")
        print(f"   High confidence detections: {summary['detection_summary']['high_confidence_ghost_probing']}")
        print(f"   Potential detections: {summary['detection_summary']['potential_ghost_probing']}")
        
        # Run evaluation if ground truth is provided
        if args.ground_truth and os.path.exists(args.ground_truth):
            print(f"\n🔍 Running evaluation against ground truth...")
            evaluation_results = processor.evaluate_results(args.ground_truth)
            
            metrics = evaluation_results['metrics']
            print(f"📈 Evaluation Results:")
            print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")
            print(f"   F1 Score: {metrics['f1_score']:.4f}")
            
            # Save detailed evaluation
            evaluator = GhostProbingEvaluator(args.ground_truth)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if args.output:
                base_name = os.path.splitext(args.output)[0]
                report_file = f"{base_name}_evaluation_report.txt"
                detailed_file = f"{base_name}_evaluation_detailed.json"
            else:
                report_file = f"evaluation_report_{timestamp}.txt"
                detailed_file = f"evaluation_detailed_{timestamp}.json"
            
            evaluator.generate_report(evaluation_results, report_file)
            evaluator.save_detailed_results(evaluation_results, detailed_file)
            
            print(f"\n📄 Evaluation reports saved:")
            print(f"   Summary report: {report_file}")
            print(f"   Detailed results: {detailed_file}")
        
        return 0
        
    except Exception as e:
        print(f"Error during batch processing: {e}")
        return 1

def run_evaluation_only(args):
    """Run evaluation on existing results."""
    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        return 1
    
    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth file not found: {args.ground_truth}")
        return 1
    
    print(f"Loading results from: {args.results}")
    
    try:
        import json
        with open(args.results, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract results list
        if isinstance(data, dict) and 'results' in data:
            results = data['results']
        elif isinstance(data, list):
            results = data
        else:
            print("Error: Invalid results file format")
            return 1
        
        print(f"Loaded {len(results)} results")
        
        # Run evaluation
        evaluator = GhostProbingEvaluator(args.ground_truth)
        evaluation_results = evaluator.evaluate_results(results)
        
        # Print summary
        metrics = evaluation_results['metrics']
        summary = evaluation_results['summary']
        
        print(f"\n📈 Evaluation Results:")
        print(f"   Total predictions: {summary['total_predictions']}")
        print(f"   Matched with ground truth: {summary['matched_videos']}")
        print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1 Score: {metrics['f1_score']:.4f}")
        
        # Save detailed evaluation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"evaluation_report_{timestamp}.txt"
        detailed_file = f"evaluation_detailed_{timestamp}.json"
        
        evaluator.generate_report(evaluation_results, report_file)
        evaluator.save_detailed_results(evaluation_results, detailed_file)
        
        print(f"\n📄 Reports saved:")
        print(f"   Summary: {report_file}")
        print(f"   Detailed: {detailed_file}")
        
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="Ghost Probing Detection for DADA Videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single video
  python run_ghost_probing.py single path/to/video.avi
  
  # Batch process all videos in folder
  python run_ghost_probing.py batch /path/to/DADA-100-videos
  
  # Batch process with evaluation
  python run_ghost_probing.py batch /path/to/videos --ground-truth labels.csv
  
  # Process limited number of videos
  python run_ghost_probing.py batch /path/to/videos --limit 10
  
  # Evaluate existing results
  python run_ghost_probing.py evaluate results.json labels.csv
        """
    )
    
    # Global arguments
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o", help="GPT model to use (default: gpt-4o)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Single video command
    single_parser = subparsers.add_parser("single", help="Analyze single video")
    single_parser.add_argument("video", help="Path to video file")
    single_parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    
    # Batch processing command
    batch_parser = subparsers.add_parser("batch", help="Batch process multiple videos")
    batch_parser.add_argument("video_folder", help="Folder containing DADA videos")
    batch_parser.add_argument("--ground-truth", help="Path to ground truth labels CSV")
    batch_parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    batch_parser.add_argument("--limit", type=int, help="Limit number of videos to process")
    batch_parser.add_argument("--start-from", type=int, default=0, help="Start from video index")
    batch_parser.add_argument("--checkpoint-interval", type=int, default=10, 
                             help="Save checkpoint every N videos")
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate existing results")
    eval_parser.add_argument("results", help="Path to results JSON file")
    eval_parser.add_argument("ground_truth", help="Path to ground truth labels CSV")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Run appropriate command
    if args.command == "single":
        return run_single_video(args)
    elif args.command == "batch":
        return run_batch_processing(args)
    elif args.command == "evaluate":
        return run_evaluation_only(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1

if __name__ == "__main__":
    sys.exit(main())