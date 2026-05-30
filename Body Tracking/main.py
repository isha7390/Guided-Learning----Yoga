import mediapipe as mp
import cv2 
import time
import numpy as np
import math

base_options = mp.tasks.BaseOptions
pose_landmarker = mp.tasks.vision.PoseLandmarker
pose_landmarker_result = mp.tasks.vision.PoseLandmarkerResult 
pose_landmarker_option = mp.tasks.vision.PoseLandmarkerOptions
vision_mode = mp.tasks.vision.RunningMode
annotated_frame = None
ref_cords ={}
scaled_ref_cords = {}
live_coords = {}
ref_raw = cv2.imread('pose_model.png')
body_points = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
SKELETON_CONNECTIONS = [
    (11, 12), # Shoulder to Shoulder
    (11, 13), (13, 15), # Left Arm (Shoulder -> Elbow -> Wrist)
    (12, 14), (14, 16), # Right Arm (Shoulder -> Elbow -> Wrist)
    (11, 23), (12, 24), # Shoulders to Hips
    (23, 24), # Hip to Hip
    (23, 25), (25, 27), # Left Leg (Hip -> Knee -> Ankle)
    (24, 26), (26, 28)  # Right Leg (Hip -> Knee -> Ankle)
]

def print_points(result: pose_landmarker_result, output_image: mp.Image, timestamp_ms: int):
    global annotated_frame 
    global ref_cords
    current_frame = np.copy(output_image.numpy_view())
    if result.pose_landmarks:
        user_landmark = result.pose_landmarks[0]
        
        # 2. Extract User Live Coordinates safely
        for idx in body_points:
            lm = user_landmark[idx]
            live_coords[idx] = (int(lm.x * 1280), int(lm.y * 720))
            
        # 3. Dynamically Scale the Reference Skeleton
        if ref_cords:
            # Anchor the shoulders
            scaled_ref_cords[11] = live_coords[11]
            scaled_ref_cords[12] = live_coords[12]
            
            # Define chains BEFORE looping
            bone_chains = [
                (11, 13), (13, 15),  # Left Arm
                (12, 14), (14, 16),  # Right Arm
                (11, 23), (12, 24),  # Torso 
                (23, 24),            # Hips 
                (23, 25), (25, 27),  # Left Leg 
                (24, 26), (26, 28)   # Right Leg 
            ]
            
            for parent, child in bone_chains:
                # Get the user's actual physical limb length
                user_bone_len = math.hypot(
                    live_coords[child][0] - live_coords[parent][0], 
                    live_coords[child][1] - live_coords[parent][1]
                )
                
                # Find the exact angle of the instructor's bone
                ref_dx = ref_cords[child][0] - ref_cords[parent][0]
                ref_dy = ref_cords[child][1] - ref_cords[parent][1]
                ref_angle = math.atan2(ref_dy, ref_dx)
                
                # Build the new blue joint using user's length + instructor's angle
                new_x = int(scaled_ref_cords[parent][0] + user_bone_len * math.cos(ref_angle))
                new_y = int(scaled_ref_cords[parent][1] + user_bone_len * math.sin(ref_angle))
                
                scaled_ref_cords[child] = (new_x, new_y)

        # 4. Draw the Blue Scaled Skeleton (using scaled_ref_cords, NOT ref_cords)
        if scaled_ref_cords:
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                if start_idx in scaled_ref_cords and end_idx in scaled_ref_cords:
                    cv2.line(current_frame, scaled_ref_cords[start_idx], scaled_ref_cords[end_idx], (255, 0, 0), 7)
            for idx in scaled_ref_cords:
                cv2.circle(current_frame, scaled_ref_cords[idx], 5, (255, 100, 0), -1)


        # 5. Draw the Red/Green User Skeleton safely
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if start_idx in live_coords and end_idx in live_coords:
                cv2.line(current_frame, live_coords[start_idx], live_coords[end_idx], (0, 0, 255), 7)

        for i in body_points:
            if i in live_coords:
                cv2.circle(current_frame, live_coords[i], 10, (0, 255, 0), -1)


        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if start_idx in live_coords and end_idx in live_coords:
                cv2.line(current_frame, live_coords[start_idx], live_coords[end_idx], (0, 0, 255), 3)

    threshold = 30 

    # A. First, figure out exactly which joints are matching the instructor
    matched_joints = []
    for i in body_points:
        if i in live_coords and i in scaled_ref_cords:
            live_x, live_y = live_coords[i]
            target_x, target_y = scaled_ref_cords[i]
            
            # If the joint is inside the square threshold, add it to our list
            if abs(live_x - target_x) <= threshold and abs(live_y - target_y) <= threshold:
                matched_joints.append(i)

    # B. Draw the bones (lines)
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if start_idx in live_coords and end_idx in live_coords:
            # If BOTH ends of the bone are in the correct spot, make the line Green
            if start_idx in matched_joints and end_idx in matched_joints:
                bone_color = (0, 255, 0)
            else:
                bone_color = (0, 0, 255) # Otherwise, Red
                
            cv2.line(current_frame, live_coords[start_idx], live_coords[end_idx], bone_color, 3)

    # C. Draw the joints (circles)
    for i in body_points:
        if i in live_coords:
            # If the joint is in our matched list, make it Green
            if i in matched_joints:
                joint_color = (0, 255, 0)
            else:
                joint_color = (0, 0, 255) # Otherwise, Red
                
            cv2.circle(current_frame, live_coords[i], 10, joint_color, -1)

    else: 
        cv2.putText(current_frame, "no body in frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    annotated_frame = current_frame

live_option = pose_landmarker_option(
    base_options = base_options(model_asset_path = 'pose_landmarker_full.task'),
    running_mode = vision_mode.LIVE_STREAM, 
    result_callback = print_points
)

static_option = pose_landmarker_option(
    base_options = base_options(model_asset_path = 'pose_landmarker_full.task'),
    running_mode = vision_mode.IMAGE, 
)

if ref_raw is not None:
    with pose_landmarker.create_from_options(static_option) as static_runner:
        mp_ref_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=ref_raw)
        static_result = static_runner.detect(mp_ref_image)
        
        if static_result.pose_landmarks:
            ref_landmarks = static_result.pose_landmarks[0]
            # Convert and map reference landmarks directly to our expected 1280x720 layout
            for idx in body_points:
                lm = ref_landmarks[idx]
                ref_cords[idx] = (int(lm.x * 1280), int(lm.y * 720))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

if not cap.isOpened():
    print("Index 1 failed, trying Index 2...")
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

cv2.namedWindow('MediaPipe Pose Detection')
cv2.setWindowProperty('MediaPipe Pose Detection', cv2.WND_PROP_TOPMOST, 1)
current_timestamp_ms = 0

with pose_landmarker.create_from_options(live_option) as landmarker:
     while cap.isOpened():
        success, frame = cap.read()  
        if not success:
            continue
            
        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        current_timestamp_ms += 1
        
        landmarker.detect_async(mp_image, current_timestamp_ms)
        
        # Display the processed frame containing the dots drawn by print_points
        if annotated_frame is not None:
            cv2.imshow('MediaPipe Pose Detection', annotated_frame)
        
        # Press Escape key to exit cleanly
        if cv2.waitKey(1) & 0xFF == 27: 
            break

cap.release()
cv2.destroyAllWindows()