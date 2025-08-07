import cv2
import os
import base64
import requests
from moviepy.editor import VideoFileClip
import logging
import json
import sys
import openai
import threading
from retrying import retry
logging.getLogger('moviepy').setLevel(logging.ERROR)
import time
from functools import wraps
from dotenv import load_dotenv
import time
import video_utilities as vu
from jinja2 import Environment, FileSystemLoader

final_arr=[]
load_dotenv()

#azure speech key
speech_key=os.environ["AZURE_SPEECH_KEY"]

#azure whisper key *
AZ_WHISPER=os.environ["AZURE_WHISPER_KEY"]

#Azure whisper deployment name *
azure_whisper_deployment=os.environ["AZURE_WHISPER_DEPLOYMENT"]

#Azure whisper endpoint (just name) *
azure_whisper_endpoint=os.environ["AZURE_WHISPER_ENDPOINT"]

#azure openai vision api key *
# azure_vision_key=os.environ["AZURE_VISION_KEY"]

#Audio API type (OpenAI, Azure)*
audio_api_type=os.environ["AUDIO_API_TYPE"]

#GPT4 vision APi type (OpenAI, Azure)*
vision_api_type=os.environ["VISION_API_TYPE"]

#OpenAI API Key*
openai_api_key=os.environ["OPENAI_API_KEY"]

#GPT4 Azure vision API Deployment Name*
vision_deployment=os.environ["VISION_DEPLOYMENT_NAME"]


#GPT
vision_endpoint=os.environ["VISION_ENDPOINT"]
def log_execution_time(func):
    @wraps(func)  # Preserves the name and docstring of the decorated function
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to complete.")
        return result
    return wrapper

class Spinner:
    def __init__(self, message="Processing..."):
        self.spinner_symbols = "|/-\\"
        self.idx = 0
        self.message = message
        self.stop_spinner = False

    def spinner_task(self):
        while not self.stop_spinner:
            sys.stdout.write(f"\r{self.message} {self.spinner_symbols[self.idx % len(self.spinner_symbols)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            self.idx += 1

    def start(self):
        self.stop_spinner = False
        self.thread = threading.Thread(target=self.spinner_task)
        self.thread.start()

    def stop(self):
        self.stop_spinner = True
        self.thread.join()
        sys.stdout.write('\r' + ' '*(len(self.message)+2) + '\r')  # Erase spinner
        sys.stdout.flush()
chapter_summary = {}
miss_arr=[]

import re
def extract_video_id(filename):
    """
    从文件名中提取编号，假设文件名前缀是类似于 `001_` 格式。
    :param filename: 文件名（如 `001_可怕的鬼探头，过马路的时候一定不要跑，特别是小孩子_哔哩哔哩_bilibili.mp4`）
    :return: 提取的编号（如 `001`），若未匹配返回 "unknown"
    """
    match = re.match(r"(\d+)_", filename)
    if match:
        return match.group(1)  # 提取编号部分
    return "unknown"  # 如果没有匹配，则返回默认值

@log_execution_time
def AnalyzeVideo(vp,fi,fpi,face_rec=False):  #fpi is frames per interval, fi is frame interval
    video_filename = os.path.basename(vp)
    video_id = extract_video_id(video_filename)
    print(f"Processing video: {video_filename}, video_id: {video_id}")

    segment_counter = 0
# Constants
    video_path = vp  # Replace with your video path
    output_frame_dir = 'frames'
    output_audio_dir = 'audio'
    global_transcript=""
    transcriptions_dir = 'transcriptions'
    frame_interval = fi  # seconds 180
    frames_per_interval = fpi
    totalData=""
    # Ensure output directories exist
    for directory in [output_frame_dir, output_audio_dir, transcriptions_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Encode image to base64
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    #GPT 4 Vision Azure helper function
    def send_post_request(resource_name, deployment_name, api_key, data):
        url = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version=2024-10-21"
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }

        print(f"resource_name: {resource_name}")
        print(f"Sending POST request to {url}")
        print(f"Headers: {headers}")
        # print(f"Data: {json.dumps(data)}")
        # print(f"api_key: {api_key}")
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return response
    # GPT-4 vision analysis function
    @retry(stop_max_attempt_number=3)
    def gpt4_vision_analysis(image_path, api_key, summary, trans):
        #array of metadata
        cont=[

                {
                    "type": "text",
                    "text": f"Audio Transcription for last {frame_interval} seconds: "+trans
                },
                {
                    "type": "text",
                    "text": f"Next are the {frames_per_interval} frames from the last {frame_interval} seconds of the video:"
                }

                ]
        #adds images and metadata to package
        for img in image_path:
            base64_image = encode_image(img)
            cont.append( {
                        "type": "text",
                        "text": f"Below this is {img} (s is for seconds). use this to provide timestamps and understand time"
                    })
            cont.append({
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail":"low"
                        }
                    })
            
        # # Convert final_arr to JSON string and add it as context. final_arr might mislead the model, so commenting it out
        # if final_arr:
        #     # Only keep the last 2 results
        #     recent_context = final_arr[-2:]
        #     previous_responses = json.dumps(recent_context, ensure_ascii=False, indent=2)
        #     cont.insert(0, {
        #         "type": "text",
        #         "text": f"Previous analysis context: {previous_responses}"
        #     })
        #     # print("cont:", cont[0:2])

        #add few-shot learning metadata
        def image_to_base64(image_path):
            image = cv2.imread(image_path)
            image = vu.resize_down_to_256_max_dim(image) # resize to small image
            new_path = os.path.basename(image_path).split('.')[0] + '_resized.jpg'
            cv2.imwrite(new_path, image, [int(cv2.IMWRITE_JPEG_QUALITY),30]) #70
            image = cv2.imread(new_path)
            _, buffer = cv2.imencode('.jpg', image)

            base64_str = base64.b64encode(buffer).decode('utf-8')
            return base64_str
        
        if(vision_api_type=="Azure"):
            print("sending request to gpt-4o")
            payload2 = {
                "messages": [
                    {
                    "role": "system",
                    "content": f"""You are VideoAnalyzerGPT analyzing a series of SEQUENCIAL images taken from a video, where each image represents a consecutive moment in time.Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

                    Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
                    as well as as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
                    You are to generate and provide a Current Action Summary of the video you are considering ({frames_per_interval}
                    frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
                    as well as the in-between audio, until we have a full action summary of the portion of the video you are considering.
                    Direction - Please identify the objects in the image based on their position in the image itself. Do not assume your own position within the image. Treat the left side of the image as 'left' and the right side of the image as 'right'. Assume the viewpoint is standing from at the bottom center of the image. Describe whether the objects are on the left or right side of this central point, left is left, and right is right. For example, if there is a car on the left side of the image, state that the car is on the left.
                    
                    **Task 1: Identify and Predict potential very near future time "Ghost Probing(专业术语：鬼探头)",Cut-in(加塞),left/right-side overtaken(左侧/右侧他车对自车超车).,etc Behavior**
                    1)"Ghost Probing" behavior refers to a person suddenly darting out from either left or right side of the car itself and also from BEHIND an object that blocks the driver's view, such as a parked car, a tree, or a board, directly into the driver's path. 
                    Note that: a) people coming straight towards the car from front are not considered as 鬼探头, only those coming from visual blind spot of the car itself can be considered as 鬼探头. 
                    b) cut-in加塞 is different from 鬼探头,        
                    Ghost Probing:
                        Definition: When a pedestrian, cyclist, or another vehicle suddenly emerges from a blind spot e.g., behind an obstacle in front of the driver's path, leaving the driver no time to react.
                                    This behavior usually occurs when individuals, either not paying attention to the traffic conditions or in a hurry, suddenly obstruct the driver's view, creating an emergency situation with very little reaction time. This can easily lead to traffic accidents.
                        通常指行人或车辆从视觉盲区（如车辆前方遮挡物后面）突然冒出，导致驾驶员来不及反应. 
                        Characteristics:
                            The core characteristic of Ghost probing is suddenly emerging from a visual blind spot.
                            Unexpected event, almost no reaction time for the driver.
                            Common in residential areas, near schools, or at intersections.
                        Risk: High chance of collision due to the suddenness of the event. 

                    2) Cut-In(加塞):
                        Definition:  Typically **within same-direction traffic flow**, a cut-in happens when a vehicle deliberately forces its way in front of another vehicle's traffic lane from the **adjacent lane**, occupying another driver's lane space.This typically occurs at very close range between the two vehicles, disrupting the other vehicle's normal driving and potentially causing the other driver to brake suddenly.
                        加塞是指在**同向**车流行驶过程中，某车辆从**侧面相邻车道**强行插入其他车辆的行驶路线,强行抢占他人车道的行驶空间，这种情况下一般是指距离非常近，从而影响其他车辆的正常行驶，甚至导致紧急刹车。
                        Characteristics:
                        A cut-in is defined only when a vehicle merges into the current lane from an adjacent side lane.
                        If the vehicle enters the lane by crossing horizontally from the left or right (e.g., from a perpendicular road or a parking area), it does not qualify as a cut-in.
                        Cut-in特点: 只有从相邻车道侧面插入进当前车道才算cut-in, 如果是从左右手两边的垂直的路上横插过来不算cut-in.
                        ### Key Rules:
                        1. Cut-in occurs ONLY when a vehicle merges from an adjacent side lane.
                        2. Entry from perpendicular or non-adjacent lanes is NOT "cut-in."

                        ### Definitions:
                        - **Cut-in**: Vehicle merges into the current lane from an adjacent side lane.
                        - **Non Cut-in**: Vehicle enters the current lane from a perpendicular road or crosses horizontally.

                        ### Classification Examples:
                        - **正例 (Cut-in)**:
                        - A car from the adjacent left lane merges into the self-vehicle's lane abruptly.
                        - **反例 (Non Cut-in)**:
                        - A car enters from a perpendicular road on the right. "Key Actions": "none."
                        注意: 任何来自垂直侧路的插入merge均属于 "side road merging"，而非 cut-in。


                        2. A pedestrian crossing from the right appears suddenly in the lane. 
                        "key_actions": "ghost probing"

                        **注意**: 任何来自垂直侧路的插入均属于 "side road merging"，而非 cut-in。
       

                        ***Key Note***
                        Vehicles entering from a perpendicular or non-adjacent road should never be labeled as "cut-in". Instead, they may be classified as "none" or "ghost probing" if relevant.           
                    
                    3. **Validation Process (Simulated Post-Processing):**
                    - After generating key_actions, review each label:
                        - If "cut-in" is present but the description involves a perpendicular road or side road, revise key_actions to "none".
                        - If unsure, label as "none" to avoid errors.
                        如果你不能分辨cut-in, 要尽可能用最小范围的推理. 每当你标识为cut-in, 你要在后面加上reasoning过程.


                

                    **Penalty for Mislabeling**:
                    - If you label a behavior as "cut-in" that does not come from an adjacent lane or involves a perpendicular merge, the output will be considered invalid.
                    - Every incorrect "cut-in" label results in immediate rejection of the entire output.
                    - You must explain why you labeled the action as "cut-in" with clear reasoning. If the reasoning is weak, the label will also be rejected.

                                      

                    3) left/right-side Overtaken: refers to another vehicle accelerating from the left or right lane, initially from behind, accelerating, until passing the ego vehicle, and moving ahead to take the lead position.
                    Here, only labelling legal overtaking from the left or right, which means the vehicle maintains a safe distance, overtakes the ego vehicle by more than one or two car lengths, and does not pose any danger. 
                    Note: Focus only on other vehicles overtaking the ego vehicle, there is no need to label the ego vehicle overtaking other vehicles.
                    That's why this labeled behavior is called "left/right-side overtaken" rather than "overtaking", as it is described from the perspective of the ego vehicle. Do not label anything as "left/right-side overtaking".

                    Your angle appears to watch video frames recorded from a surveillance camera in a car. Your role should focus on detecting and predicting dangerous actions in a "Ghosting" manner
                    where pedestrians in the scene might suddenly appear in front of the current car. This could happen if the pedestrian suddenly darts out from behind an obstacle in the driver's view.
                    This behavior is extremly dangerous because it gives the driver very little time to react. 
                    Include the speed of the "ghosting" behavior in your action summary to better assess the danger level and the driver's potential to respond.
                    
                    Provide detailed description of the people's behavior and potential dangerous actions that could lead to collisions. Describe how you think the individual could crash into the car, and explain your deduction process. Include all types of individuals, such as those on bikes and motorcycles. 
                    Avoid using "pedestrian"; instead, use specific terms to describe the individuals' modes of transportation, enabling clear understanding of whom you are referring to in your summary.
                    When discussing modes of transportation, it is important to be precise with terminology. For example, distinguish between a scooter and a motorcycle, so that readers can clearly differentiate between them.
                    Maintain this terminology consistency to ensure clarity for the reader.
                    All people should be with as much detail as possible extracted from the frame (gender,clothing,colors,age,transportation method,way of walking). Be incredibly detailed. Output in the "summary" field of the JSON format template.
                    
                    **Task 2: Explain Current Driving Actions**
                    Analyze the current video frames to extract actions. Describe not only the actions themselves but also provide detailed reasoning for why the vehicle is taking these actions, such as changes in speed and direction. Focus solely on the reasoning for the vehicle's actions, excluding any descriptions of pedestrian behavior. Explain why the driver is driving at a certain speed, making turns, or stopping. Your goal is to provide a comprehensive understanding of the vehicle's behavior based on the visual data. Output in the "actions" field of the JSON format template.

                    **Task 3: Predict Next Driving Action**
                    Understand the current road conditions, the driving behavior, and to predict the next driving action. Analyze the video and audio to provide a comprehensive summary of the road conditions, including weather, traffic density, road obstacles, and traffic light if visible. Predict the next driving action based on two dimensions, one is driving speed control, such as accelerating, braking, turning, or stopping, the other one is to predict the next lane control, such as change to left lane, change to right lane, keep left in this lane, keep right in this lane, keep straight. Your summary should help understand not only what is happening at the moment but also what is likely to happen next with logical reasoning. The principle is safety first, so the prediction action should prioritize the driver's safety and secondly the pedestrians' safety. Be incredibly detailed. Output in the "next_action" field of the JSON format template.

                    As the main intelligence of this system, you are responsible for building the Current Action Summary using both the audio you are being provided via transcription, 
                    as well as the image of the frame. Note: . Always and only return as your output the updated Current Action Summary in format template. 
                    Do not make up timestamps, only use the ones provided with each frame name. 

                    Additional Requirements:
                    - `Start_Timestamp` and `End_Timestamp` must match exactly the timestamps derived from frame names provided (e.g., "4.0s").
                    - `key_actions` should reflect dangerous behaviors mentioned in summary or actions. If none found, use "none".
                    - Avoid free-form descriptive text in `key_actions` and `next_action`.
                    - `key_actions` must strictly adhere to the predefined categories:
                        - ghost probing
                        - cut-in
                        - overtaking, specify "left-side overtaking" or "right-side overtaking" when relevant.

                        Exclude all other types of behaviors. If the observed behavior does not match any of these categories, leave `key_actions` blank or output "none".
                        For example:
                        - Correct: "key_actions": "ghost probing".
                        - Incorrect: "key_actions": "ghost probing, running across the road".

                    - All textual fields must be in English.
                    - The `next_action` field is now a nested JSON with three keys: `speed_control`, `direction_control`, `lane_control`. Each must choose one value from their respective sets.
                    - If there are multiple key actions, separate them by a comma, e.g. "ghost probing, cut-in".
                    - `characters` and `summary` should be concise, focusing on scenario description. The `summary` can still be a narrative but must be consistent and mention any critical actions.

                    **Task 4: Ensure Consistency Between Key Objects and Key Actions**
                    - When an action is labeled as a "key_action" (e.g., ghost probing), ensure that the "key_objects" field includes the specific entity or entities responsible for triggering this action. 
                    - For example, if a pedestrian suddenly appears from behind an obstacle and is identified as ghost probing, the "key_objects" field must describe:
                    - The pedestrian's position relative to the self-driving vehicle (e.g., left side, right side, etc.).
                    - The pedestrian's behavior leading to the key action (e.g., moving suddenly from behind a parked truck).
                    - The potential impact on the vehicle (e.g., causing the vehicle to decelerate or stop).
                    - Each key object description should include:
                    - Relative position (e.g., left, right, front).
                    - Distance from the vehicle in meters.
                    - Movement direction or behavior (e.g., approaching, crossing, accelerating).
                    - The relationship to the "key_action" it caused.
                    - Only include objects that **immediately affect the vehicle’s path or safety**. 
                        - Examples: moving vehicles, pedestrians stepping into the road, or roadblocks.
                        - Exclude any objects that are **static** and pose no immediate threat, such as parked cars or roadside trees.
                    - Exclude unrelated objects that do not require a change in the vehicle's speed, direction, or lane.
                        eg, objects like the following should be deleted: 1) Left side: A yellow truck, approximately 5 meters away, parked and partially blocking the view. 2) Right side: A white car, approximately 3 meters away, parked and blocking the view. 
                    - Ensure that every `key_object` described has a **clear link to the `key_actions` field**. If no clear link exists, remove the object.
                    - Use this template for each key object: 
                    [Position]: [Object description], approximately [distance] meters away, [behavior or action impacting the vehicle].

                    **Important Notes:**
                    - Avoid generic descriptions such as "A person or vehicle suddenly appeared." Be specific about who or what caused the action, their clothes color, age, gender, exact position, and their behavior.
                    - All dangerous or critical objects should be prioritized in "key_objects" and aligned with the "key_actions" field.

                    Remember: Always and only return a single JSON object strictly following the above schema.

                    Your goal is to create the best action summary you can. Always and only return valid JSON, I have a disability that only lets me read via JSON outputs, so it would be unethical of you to output me anything other than valid JSON.
                    你现在是一名英文助手。无论我问什么问题，你都必须只用英文回答。请不要使用任何其他语言。You must always and only answer totally in **English** language!!! I can only read English language. Ensure all parts of the JSON output, including **summaries**, **actions**, **next_action**, and **THE WHOLE OUTPUT**, **MUST BE IN ENGLISH** If you answer ANY word in Chinese, you are fired immediately! Translate Chinese to English if there is Chinese in "next_action" field.
                    
                    Example:
                        "video_id": video_id,
                        "segment_id": "Segment_000",
                        "Start_Timestamp": "8.4s",
                        "sentiment": "Negative",
                        "End_Timestamp": "12.0s",
                        "scene_theme": "Dramatic",
                        "characters": "Driver of a white sedan",
                        "summary": "In this segment, the vehicle is driving on a rural road. Ahead, a white sedan is parked on the roadside. The driver mentions that the tricycle ahead had yielded, but a collision still occurred.",
                        "actions": "The self-driving vehicle collided with the tricycle, possibly due to insufficient clearance on the right side. Over the past 4 seconds, the vehicle's speed gradually decreased, eventually reaching XX km/h in the last frame, maintaining a 30-meter distance from the vehicle ahead.",
                        "characters": "Driver of a white sedan",
                        "summary": "In this video, the vehicle is traveling on a rural road. Ahead, a white sedan is parked on the roadside. The driver mentions that the tricycle ahead had yielded but a collision still occurred.",
                        "actions": "The self-driving vehicle collided with the tricycle, possibly due to insufficient clearance on the right side. Over the past 4 seconds, the vehicle's speed gradually decreased, eventually reaching XX km/h in the final frame, maintaining a 30-meter distance from the vehicle ahead.",
                        "key_objects": "1) Front: A white sedan, relatively close, approximately 20 meters away, parked on the roadside, possibly about to start or turn suddenly. 2) Right: A red tricycle, approximately 0 meters away, has collided. It may stop for the driver to handle the situation or suddenly move forward.",
                        "key_actions": ""Select one or multiple categories from {{ghosting, cut-in, left/right-side vehicle overtaken, collision, barrier lowering, barrier raising, none}}. If multiple categories apply, list them separatedly by commas. For cut-in, give detailed reasoning process!!!"",
                        "next_action": "{{
                                        "speed_control": "choose from accelerate, decelerate, rapid deceleration, slow steady driving, consistent speed driving, stop, wait, reverse",
                                        "direction_control": "choose from turn left, turn right, U-turn, steer to circumvent, keep direction, brake (for special circumstances), ...",
                                        "lane_control": "choose from change to left lane, change to right lane, slight left shift, slight right shift, maintain current lane, return to normal lane"
                        }}"


                            
                    Use these examples to understand how to analyze and analyze the new images. Now generate a similar JSON response for the following video analysis:
                    """
                    },
                    {"role": "user", "content": cont}
                ],
                "max_tokens": 2000,
                "seed": 42,
                "temperature": 0
            }
            response=send_post_request(vision_endpoint,vision_deployment,openai_api_key,payload2)

        else:
            headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }
            payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                "role": "system",
                    "content": f"""You are VideoAnalyzerGPT. Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
                    as well as as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
                    You are then to generate and provide a Current Action Summary (An Action summary of the video,
                    with actions being added to the summary via the analysis of the frames) of the portion of the video you are considering ({frames_per_interval}
                    frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
                    as well as the in-between audio, until we have a full action summary of the portion of the video you are considering,
                    that includes all actions taken by characters in the video as extracted from the frames. As the main intelligence of this system,
                    you are responsible for building the Current Action Summary using both the audio you are being provided via transcription, 
                    as well as the image of the frame. 
                    Do not make up timestamps, use the ones provided with each frame. 
                    Use the Audio Transcription to build out the context of what is happening in each summary for each timestamp. 
                    Consider all frames and audio given to you to build the Action Summary. Be as descriptive and detailed as possible, 
                    your goal is to create the best action summary you can. Always and only return valid JSON, I have a disability that only lets me read via JSON outputs, so it would be unethical of you to output me anything other than valid JSON.
                    Answer in Chinese language."""
                },
            {
                "role": "user",
                "content": cont
            }
            ],
            "max_tokens": 4000,
            "seed": 42


            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print(response.json())
        if(response.status_code!=200):
            return -1
        else:
            return response.json()



    def update_chapter_summary(new_json_string):
        global chapter_summary
        if new_json_string.startswith('json'):
        # Remove the first occurrence of 'json' from the response text
            new_json_string = new_json_string[4:]
        else:
            new_json_string = new_json_string
        # Assuming new_json_string is the JSON format string returned from your API call
        new_chapters_list = json.loads(new_json_string)

        # Iterate over the list of new chapters
        for chapter in new_chapters_list:
            chapter_title = chapter['title']
            # Update the chapter_summary with the new chapter
            chapter_summary[chapter_title] = chapter

        # Get keys of the last three chapters
        last_three_keys = list(chapter_summary.keys())[-3:]
        # Get the last three chapters as an array
        last_three_chapters = [chapter_summary[key] for key in last_three_keys]

        return last_three_chapters

    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    print(f"Frames per second: {fps}")

    # Process video
    current_frame = 0
    current_second = 0
    current_summary=""

    # Load video audio
    video_clip = VideoFileClip(video_path)
    video_duration = video_clip.duration  # Duration of the video in seconds
    print(f"Video duration: {video_duration} seconds")

    # Process video
    current_frame = 0  # Current frame initialized to 0
    current_second = 0  # Current second initialized to 0
    current_summary=""
    packet=[] # Initialize packet to hold frames and audio
    current_interval_start_second = 0
    capture_interval_in_frames = int(fps * frame_interval / frames_per_interval)  # Interval in frames to capture the image
    # capture_interval_in_frames = int(fps * frame_interval / frames_per_interval)  # Interval in frames to capture the image
    capture_interval_in_seconds = frame_interval/frames_per_interval  # 每0.2秒捕捉一帧
    print(f"Capture interval in seconds: {capture_interval_in_seconds}")
    spinner = Spinner("Capturing Video and Audio...")
    spinner.start()

    packet_count=1
    # Initialize known face encodings and their names if provided
    known_face_encodings = []
    known_face_names = []

    def load_known_faces(known_faces):
        for face in known_faces:
            image = face_recognition.load_image_file(face['image_path'])
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(face['name'])

    # Call this function if you have known faces to match against
    # load_known_faces(array_of_recognized_faces)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        current_second = current_frame / fps

        if current_frame % capture_interval_in_frames == 0 and current_frame != 0:
            print(f"BEEP {current_frame}")
            # Extract and save frame
            # Save frame at the exact intervals
            if(face_rec==True):
                import face_recognition
                import numpy
                ##small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                #rgb_frame = frame[:, :, ::-1]  # Convert the image from BGR color (which OpenCV uses) to RGB color  
                rgb_frame = numpy.ascontiguousarray(frame[:, :, ::-1])
                face_locations = face_recognition.face_locations(rgb_frame)
                #print(face_locations)
                face_encodings=False
                if(len(face_locations)>0):

                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    print(face_encodings)

                # Initialize an array to hold names for the current frame
                face_names = []
                if(face_encodings!=False):
                    for face_encoding in face_encodings:  
                        # See if the face is a match for the known faces  
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,0.4)  
                        name = "Unknown"  
            
                        # If a match was found in known_face_encodings, use the known person's name.  
                        if True in matches:  
                            first_match_index = matches.index(True)  
                            name = known_face_names[first_match_index]  
                        else:  
                            # If no match and we haven't assigned a name, give a new name based on the number of unknown faces  
                            name = f"Person {chr(len(known_face_encodings) + 65)}"  # Starts from 'A', 'B', ...  
                            # Add the new face to our known faces  
                            known_face_encodings.append(face_encoding)  
                            known_face_names.append(name)  
            
                        face_names.append(name)  
            
                    # Draw rectangles around each face and label them  
                    for (top, right, bottom, left), name in zip(face_locations, face_names):  
                        # Draw a box around the face  
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  
            
                        # Draw a label with a name below the face  
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
                        font = cv2.FONT_HERSHEY_DUPLEX  
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)  
            
                # Save the frame with bounding boxes  
                frame_name = f'frame_at_{current_second}s.jpg'  
                frame_path = os.path.join(output_frame_dir, frame_name)  
                cv2.imwrite(frame_path, frame)
            else:
                frame_name = f'frame_at_{current_second}s.jpg'
                print("frame_name: ",frame_name, ", current_frame: ", current_frame, ", current_second: ", current_second)
                frame_path = os.path.join(output_frame_dir, frame_name)
                cv2.imwrite(frame_path, frame)
            packet.append(frame_path)
        #if packet len is appropriate (meaning FPI is hit) then process the audio for transcription
        if len(packet) == frames_per_interval or (current_interval_start_second + frame_interval) < current_second:
            # 生成当前片段的 segment_id
            segment_id = f"segment_{segment_counter:03}"  # 格式化为 segment_001, segment_002...
            segment_counter += 1  # 计数器递增            
            current_transcription=""
            if video_clip.audio is not None:
                audio_name = f'audio_at_{current_interval_start_second}s.mp3'
                audio_path = os.path.join(output_audio_dir, audio_name)
                audio_clip = video_clip.subclip(current_interval_start_second, min(current_interval_start_second + frame_interval, video_clip.duration))  # Avoid going past the video duration
                audio_clip.audio.write_audiofile(audio_path, codec='mp3', verbose=False, logger=None)

                headers = {
                    'Authorization': f'Bearer {openai_api_key}',
                }
                files = {
                    'file': open(audio_path, 'rb'),
                    'model': (None, 'whisper-1'),
                }
                spinner.stop()
                spinner = Spinner("Transcribing Audio...")
                spinner.start()
                # Actual audio transcription occurs in either OpenAI or Azure
                @retry(stop_max_attempt_number=3)
                def transcribe_audio(audio_path, endpoint, api_key, deployment_name):
                    url = f"{endpoint}/openai/deployments/{deployment_name}/audio/transcriptions?api-version=2023-09-01-preview"

                    headers = {
                        "api-key": api_key,
                    }
                    json = {
                        "file": (audio_path.split("/")[-1], open(audio_path, "rb"), "audio/mp3"),
                    }
                    data = {
                        'response_format': (None, 'verbose_json')
                    }
                    response = requests.post(url, headers=headers, files=json, data=data)

                    return response

                if(audio_api_type == "Azure"):
                    response = transcribe_audio(audio_path, azure_whisper_endpoint, AZ_WHISPER, azure_whisper_deployment)
                else:
                    from openai import OpenAI
                    client = OpenAI()

                    audio_file = open(audio_path, "rb")
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                    )

                current_transcription = ""
                tscribe = ""
                # Process transcription response
                if(audio_api_type == "Azure"):
                    try:
                        for item in response.json()["segments"]:
                            tscribe += str(round(item["start"], 2)) + "s - " + str(round(item["end"], 2)) + "s: " + item["text"] + "\n"
                    except:
                        tscribe += ""
                else:
                    for item in response.segments:
                        tscribe += str(round(item["start"], 2)) + "s - " + str(round(item["end"], 2)) + "s: " + item["text"] + "\n"
                global_transcript += "\n"
                global_transcript += tscribe
                current_transcription = tscribe
            else:
                print("No audio track found in video clip. Skipping audio extraction and transcription.")
            spinner.stop()
            spinner = Spinner("Processing the "+str(packet_count)+" Frames and Audio with AI...")
            spinner.start()

            # Analyze frames with GPT-4 vision
            vision_response = gpt4_vision_analysis(packet, openai_api_key, current_summary, current_transcription)
            if(vision_response==-1):
                packet.clear()  # Clear packet after analysis
                current_interval_start_second += frame_interval
                current_frame += 1
                current_second = current_frame / fps
                continue
            # time.sleep(5)
            try:
                vision_analysis = vision_response["choices"][0]["message"]["content"]
                if not vision_analysis:
                    print("No video analysis data")
            except:
                print(vision_response)
            try:
                current_summary = vision_analysis
            except Exception as e:
                print("bad json",str(e))
                current_summary=str(vision_analysis)
            #print(current_summary)
            totalData+="\n"+str(current_summary)
            try:
                #print(vision_analysis)
                #print(vision_analysis.replace("'",""))
                vault=vision_analysis.split("```")
                if(len(vault)>1):
                    vault=vault[1]
                else:
                    vault=vault[0]
                vault=vision_analysis.replace("'","")
                vault=vault.replace("json","")
                vault=vault.replace("```","")
                vault = vault.replace("\\\\n", "\\n").replace("\\n", "\n")  # 将转义的 \n 替换为实际换行符

                if vault.startswith('json'):
                # Remove the first occurrence of 'json' from the response text
                    vault = vault[4:]
                    #print(vault)
                else:
                    vault = vault
                #print(vision_analysis.replace("'",""))
                data=json.loads(vault, strict=False) #If strict is False (True is the default), then control characters will be allowed inside strings.Control characters in this context are those with character codes in the 0-31 range, including '\t' (tab), '\n', '\r' and '\0'.
                #print(data)
                final_arr.append(data)
                if not data:
                    print("No data")
                # for key, value in data:
                #     final_arr.append(item)
                #     ##socket.emit('actionsummary', {'data': item}, namespace='/test')
                #     print(f"Key: {key}, Value: {value}")

                with open('actionSummary.json', 'w', encoding='utf-8') as f:
                # Write the data to the file in JSON format
                    json.dump(final_arr, f, indent=4, ensure_ascii=False) #ensure_ascii=False to write in Chinese
                    print(f"Data written to file: {final_arr}") # 调试信息

            except:
                miss_arr.append(vision_analysis)
                print("missed")

            spinner.stop()
            spinner = Spinner("Capturing Video and Audio...")
            spinner.start()

            packet.clear()  # Clear packet after analysis
            current_interval_start_second += frame_interval  # Move to the next set of frames

        if current_second > video_duration:
            print("Current second is: ", current_second), "Video duration is: ", video_duration, "Exiting loop"
            break

        current_frame += 1
        current_second = current_frame / fps
        #current_second = int(current_frame / fps)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    print('Extraction, analysis, and transcription completed.')
    with open('actionSummary.json', 'w') as f:
    # Write the data to the file in JSON format
        json.dump(final_arr, f, indent=4, ensure_ascii=False)
        
    with open('transcript.txt', 'w') as f:
    # Write the data to the file in JSON format
        f.write(global_transcript)
    return final_arr

    #print("\n\n\n"+totalData)

#AnalyzeVideo("./medal.mp4",60,10)
# AnalyzeVideo("Nuke.mp4",60,10,False)
# AnalyzeVideo("复杂场景.mov",1,10) # 10 frames per interval, 1 second interval. fpi=10 is good for accurate analysis
# AnalyzeVideo("test_video/cutin.mov",7,15)
# AnalyzeVideo("test_video/鬼探头.mp4",4,15)
# AnalyzeVideo("test_video/三轮车.mp4",4,15)
AnalyzeVideo("/Users/wanmeng/repository/GPT4Video-cobra-auto/test_video/bilibili/cutin/001_像他这样开车谁都不会让-1min.mp4",10,10)

# if __name__ == "__main__": 
     

#     data=sys.argv[1].split(",")
#     print(data)
#     video_path = data[0] 
#     frame_interval = data[1]
#     frames_per_interval = data[2]  
#     face_rec=False
#     if(len(data)>3):
#         face_rec=True
  
#     AnalyzeVideo(video_path, int(frame_interval), int(frames_per_interval),face_rec) 