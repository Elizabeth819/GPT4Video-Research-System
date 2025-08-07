import cv2
import json
import os
from datetime import datetime

class VideoAnnotator:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.annotations = []
        self.segment_duration = 10  # 10 seconds per segment
        self.cap = None
        self.fps = None
        self.total_frames = None
        self.duration = None
        self.is_first_annotation = True  # New flag to track first annotation
        
        # Add option mappings
        self.key_actions_options = {
            "1": "ghost probing",
            "2": "cut-in",
            "3": "collision",
            "4": "overtaken left",
            "5": "overtake right"
        }
        
        self.speed_control_options = {
            "1": "accelerate",
            "2": "decelerate",
            "3": "rapid deceleration",
            "4": "slow steady driving",
            "5": "stop"
        }
        
        self.direction_control_options = {
            "1": "steer to circumvent",
            "2": "keep direction",
            "3": "turn left",
            "4": "turn right"
        }
        
        self.lane_control_options = {
            "1": "maintain current lane",
            "2": "change to left lane",
            "3": "change to right lane",
            "4": "slight left shift",
            "5": "slight right shift"
        }

    def initialize_video(self):
        """Initialize video capture"""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
            
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError("Cannot open video file")
                
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.total_frames / self.fps
            
            print(f"\nVideo Information:")
            print(f"Path: {self.video_path}")
            print(f"FPS: {self.fps}")
            print(f"Total Frames: {self.total_frames}")
            print(f"Duration: {self.duration:.2f} seconds\n")
            
        except Exception as e:
            raise Exception(f"Error initializing video: {str(e)}")

    def play_segment(self, start_frame, segment_frames):
        """Play video segment"""
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for _ in range(segment_frames):
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Add timestamp to frame
                timestamp = f"Time: {(start_frame + _) / self.fps:.2f}s"
                cv2.putText(frame, timestamp, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Video Annotation", frame)
                
                # Wait for key press, 'q' to quit
                key = cv2.waitKey(int(1000/self.fps)) & 0xFF
                if key == ord('q'):
                    return False
                    
            return True
            
        except Exception as e:
            print(f"Error playing video segment: {str(e)}")
            return False

    def adjust_timestamp(self, current_time):
        """Adjust timestamp to avoid overlapping"""
        if self.is_first_annotation:
            self.is_first_annotation = False
            return current_time
        else:
            # Round up to next second and add 1
            return (int(current_time) + 1)

    def get_selection_from_options(self, options, prompt, allow_multiple=False):
        """Helper function to handle numeric selection"""
        while True:
            print(f"\n{prompt}")
            for key, value in options.items():
                print(f"{key}: {value}")
            
            if allow_multiple:
                print("(For multiple selections, enter numbers separated by commas)")
            
            selection = input("Enter your selection: ").strip()
            
            if allow_multiple:
                try:
                    selections = [s.strip() for s in selection.split(',')]
                    selected_values = []
                    for s in selections:
                        if s in options:
                            selected_values.append(options[s])
                        else:
                            print(f"Invalid selection: {s}")
                            break
                    else:
                        return ', '.join(selected_values)
                    continue
                except Exception:
                    print("Invalid input format. Please try again.")
            else:
                if selection in options:
                    return options[selection]
                print("Invalid selection. Please try again.")

    def get_user_input(self, start_time):
        """Get user input for annotations"""
        adjusted_start_time = self.adjust_timestamp(start_time)
        adjusted_end_time = adjusted_start_time + self.segment_duration
        
        print(f"\n=== Current Segment (Start Time: {adjusted_start_time:.2f}s) ===")
        
        annotation = {
            "event_id": f"event_{len(self.annotations)+1:03d}",
            "video_id": os.path.splitext(os.path.basename(self.video_path))[0],
            "start_timestamp": f"{adjusted_start_time:.2f}s",
            "end_timestamp": f"{min(adjusted_end_time, self.duration):.2f}s",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            print("\nPlease enter the following information:")
            annotation["key_objects"] = input("Key Objects (e.g., a scooter on the right): ").strip()
            
            # Get selections using numeric options
            annotation["key_actions"] = self.get_selection_from_options(
                self.key_actions_options,
                "Select Key Actions:",
                allow_multiple=True
            )
            
            speed_control = self.get_selection_from_options(
                self.speed_control_options,
                "Select Speed Control:"
            )
            
            direction_control = self.get_selection_from_options(
                self.direction_control_options,
                "Select Direction Control:"
            )
            
            lane_control = self.get_selection_from_options(
                self.lane_control_options,
                "Select Lane Control:"
            )
            
            annotation["next_action"] = {
                "speed_control": speed_control,
                "direction_control": direction_control,
                "lane_control": lane_control
            }
            
            return annotation
            
        except Exception as e:
            print(f"Error getting user input: {str(e)}")
            return None

    def clean_annotations(self):
        """Remove all annotations with empty key_objects"""
        original_count = len(self.annotations)
        self.annotations = [
            annotation for annotation in self.annotations 
            if annotation["key_objects"].strip()
        ]
        removed_count = original_count - len(self.annotations)
        
        if removed_count > 0:
            print(f"\nRemoved {removed_count} empty annotations")
            print(f"Remaining annotations: {len(self.annotations)}")

    def save_annotations(self):
        """Save annotations to JSON file"""
        try:
            # Clean annotations before saving
            self.clean_annotations()
            
            if not self.annotations:
                print("\nWarning: No valid annotations to save!")
                return
                
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(self.annotations, f, indent=4, ensure_ascii=False)
            print(f"\nAnnotations saved to: {self.output_path}")
        except Exception as e:
            print(f"Error saving annotations: {str(e)}")

    def run(self):
        """Run annotation program"""
        try:
            self.initialize_video()
            
            cv2.namedWindow("Video Annotation", cv2.WINDOW_NORMAL)
            current_frame = 0
            segment_frames = int(self.segment_duration * self.fps)
            
            print("Instructions:")
            print("- Press 'q' to quit")
            print("- After each segment plays, enter annotation information as prompted")
            print("- Press Enter to continue to next segment\n")
            
            while current_frame < self.total_frames:
                # Check if there's enough frames left for a full segment
                remaining_frames = self.total_frames - current_frame
                if remaining_frames < segment_frames:
                    # For the last segment, only play remaining frames
                    segment_frames = remaining_frames
                
                # Play current segment
                if not self.play_segment(current_frame, segment_frames):
                    break
                
                # Get user input for annotations
                current_time = current_frame / self.fps
                annotation = self.get_user_input(current_time)
                
                if annotation:
                    self.annotations.append(annotation)
                    print("\nAnnotation added! Press Enter to continue to next segment...")
                    input()
                    # Update current_frame based on adjusted timestamp
                    next_start_time = float(annotation["end_timestamp"].rstrip('s'))
                    current_frame = int(next_start_time * self.fps)
                else:
                    current_frame += segment_frames
                
                # Check if we've reached the end of the video
                if current_frame >= self.total_frames:
                    print("\nReached the end of the video!")
                    break
            
            self.save_annotations()
            
        except Exception as e:
            print(f"Program error: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

def main():
    video_path = input("Enter video file path: ").strip()
    output_path = input("Enter output JSON file path (default: labelresult/cutin_001_label.json): ").strip()
    
    if not output_path:
        output_path = "labelresult/cutin_001_label.json"
    
    annotator = VideoAnnotator(video_path, output_path)
    annotator.run()

if __name__ == "__main__":
    main()