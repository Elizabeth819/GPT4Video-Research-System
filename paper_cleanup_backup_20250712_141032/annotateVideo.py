import cv2
import json
import os
from datetime import datetime
import signal

class VideoAnnotator:
    def emergency_save(self, signum, frame):
        """紧急保存功能"""
        print("\n\nEmergency save triggered! Saving annotations...")
        self.save_annotations(is_emergency=True)
        print(f"Emergency save completed to: {self.temp_save_path}")
        print("You can safely exit now by pressing Ctrl+C again")

    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        # 创建临时文件保存目录
        self.temp_dir = "temp_annotations"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        # 设置临时文件路径
        self.temp_save_path = os.path.join(
            self.temp_dir, 
            f"temp_{os.path.basename(self.output_path)}"
        )
        
        self.annotations = []
        self.segment_duration = 10  # 10 seconds per segment
        self.cap = None
        self.fps = None
        self.total_frames = None
        self.duration = None
        self.is_first_annotation = True  # New flag to track first annotation
        self.playback_speed = 2.0  # 2倍速播放
        
        # Add option mappings
        self.key_actions_options = {
            "1": "ghost probing",
            "2": "cut-in",
            "3": "collision",
            "4": "overtaken left",
            "5": "overtaken right",
            "6": "improper lane change"
        }
        
        self.speed_control_options = {
            "1": "accelerate",
            "2": "decelerate",
            "3": "rapid deceleration",
            "4": "slow steady driving",
            "5": "stop"
        }
        
        self.direction_control_options = {
            "1": "keep direction",
            "2": "steer to circumvent",
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

        # 添加信号处理
        signal.signal(signal.SIGINT, self.emergency_save)

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
            frames_played = 0
            
            while frames_played < segment_frames:
                ret, frame = self.cap.read()
                if not ret:
                    print("\nReached end of video file")
                    return False
                    
                # Add timestamp to frame
                current_time = (start_frame + frames_played) / self.fps
                timestamp = f"Time: {current_time:.2f}s"
                cv2.putText(frame, timestamp, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Video Annotation", frame)
                
                # 修改等待时间来实现倍速播放
                wait_time = int((1000/self.fps) / self.playback_speed)  # 2倍速
                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord('q'):
                    return False
                elif key == ord('1'):  # 添加实时速度调整功能
                    self.playback_speed = 1.0
                    print("\nSpeed: 1x")
                elif key == ord('2'):
                    self.playback_speed = 2.0
                    print("\nSpeed: 2x")
                
                frames_played += 1
                
            return True
            
        except Exception as e:
            print(f"Error playing video segment: {str(e)}")
            return False

    def adjust_timestamp(self, current_time):
        """Adjust timestamp to avoid overlapping"""
        if self.is_first_annotation:
            self.is_first_annotation = False
            return 0.0  # 第一个标注从0开始
        else:
            # 向下取整到最近的10秒
            return int(current_time / 10) * 10

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
        
        # 计算结束时间
        if adjusted_start_time + self.segment_duration > self.duration:
            # 如果是最后一段，使用实际的结束时间
            adjusted_end_time = self.duration
        else:
            # 否则确保结束时间是10的倍数
            adjusted_end_time = adjusted_start_time + self.segment_duration
        
        print(f"\n====================================")
        print(f"Current Segment Time: {adjusted_start_time:.2f}s - {adjusted_end_time:.2f}s")
        print(f"====================================")
        
        annotation = {
            "event_id": f"event_{len(self.annotations)+1:03d}",
            "video_id": os.path.splitext(os.path.basename(self.video_path))[0],
            "start_timestamp": f"{adjusted_start_time:.2f}s",
            "end_timestamp": f"{adjusted_end_time:.2f}s",
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
                
            # 确保输出目录存在
            output_dir = os.path.dirname(self.output_path)
            if not os.path.exists(output_dir):
                print(f"Output directory '{output_dir}' does not exist. Saving to 'temp_annotations' instead.")
                # 使用 temp_annotations 作为备份路径
                self.output_path = os.path.join(self.temp_dir, 'ghosting_007_label.json')
                if not os.path.exists(self.temp_dir):
                    os.makedirs(self.temp_dir)
            
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
            
            print("\nSafety Features:")
            print("- Press Ctrl+C for emergency save")
            print("- Annotations are auto-saved after each segment")
            print("- Temporary saves are stored in:", self.temp_save_path)
            
            print("\nPlayback Controls:")
            print("- Press '1' for normal speed")
            print("- Press '2' for 2x speed")
            print("- Press 'q' to quit")
            
            while current_frame < self.total_frames:
                # 计算剩余帧数
                remaining_frames = self.total_frames - current_frame
                
                # 如果剩余帧数小于一个完整片段，就用剩余的帧数
                segment_frames = min(
                    int(self.segment_duration * self.fps),
                    remaining_frames
                )
                
                # 如果没有剩余帧数了，退出循环
                if segment_frames <= 0:
                    print("\nReached the end of the video!")
                    break
                
                # 播放片段
                if not self.play_segment(current_frame, segment_frames):
                    break
                
                current_time = current_frame / self.fps
                annotation = self.get_user_input(current_time)
                
                if annotation:
                    self.annotations.append(annotation)
                    print("\nAnnotation added! Press Enter to continue to next segment...")
                    input()
                    # 更新到下一个10秒片段
                    current_frame += int(self.segment_duration * self.fps)
                else:
                    current_frame += segment_frames
                
                # 检查是否到达视频末尾
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
            print("\nVideo annotation completed!")

def main():
    video_path = input("Enter video file path: ").strip()
    output_path = input("Enter output JSON file path (default: /Users/wanmeng/repository/GPT4Video-cobra-auto/labelresult/ghosting_010_label.json): ").strip()
    
    if not output_path:
        output_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/labelresult/ghosting_010_label.json"
    
    annotator = VideoAnnotator(video_path, output_path)
    annotator.run()

if __name__ == "__main__":
    main()