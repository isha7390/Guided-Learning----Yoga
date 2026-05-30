import mediapipe as mp
import cv2 
import time
import numpy as np
import math
import tkinter as tk
import customtkinter as ctk
from PIL import Image

base_options = mp.tasks.BaseOptions
pose_landmarker = mp.tasks.vision.PoseLandmarker
pose_landmarker_result = mp.tasks.vision.PoseLandmarkerResult 
pose_landmarker_option = mp.tasks.vision.PoseLandmarkerOptions
vision_mode = mp.tasks.vision.RunningMode

#individual joints
body_points = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

#actual appendages 
SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)  
]

annotated_frame = None
ref_cords = {}
show_instructor = False  
current_timestamp_ms = 0

#drawing the lines/circles
def print_points(result: pose_landmarker_result, output_image: mp.Image, timestamp_ms: int):
    global annotated_frame, ref_cords, show_instructor
    
    current_frame = np.copy(output_image.numpy_view())
    
    if result.pose_landmarks:
        user_landmark = result.pose_landmarks[0]
        
        live_coords = {}
        for idx in body_points:
            lm = user_landmark[idx]
            live_coords[idx] = (int(lm.x * 1280), int(lm.y * 720))
            
        if show_instructor and ref_cords:
            scaled_ref_cords = {}
            
            scaled_ref_cords[11] = live_coords[11]
            scaled_ref_cords[12] = live_coords[12]
            
            bone_chains = [
                (11, 13), (13, 15), (12, 14), (14, 16), 
                (11, 23), (12, 24), (23, 24), 
                (23, 25), (25, 27), (24, 26), (26, 28)   
            ]
            
            for parent, child in bone_chains:
                user_bone_len = math.hypot(
                    live_coords[child][0] - live_coords[parent][0], 
                    live_coords[child][1] - live_coords[parent][1]
                )
                
                ref_dx = ref_cords[child][0] - ref_cords[parent][0]
                ref_dy = ref_cords[child][1] - ref_cords[parent][1]
                ref_angle = math.atan2(ref_dy, ref_dx)
                
                new_x = int(scaled_ref_cords[parent][0] + user_bone_len * math.cos(ref_angle))
                new_y = int(scaled_ref_cords[parent][1] + user_bone_len * math.sin(ref_angle))
                scaled_ref_cords[child] = (new_x, new_y)

            # Draw Instructor Skeleton
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                if start_idx in scaled_ref_cords and end_idx in scaled_ref_cords:
                    cv2.line(current_frame, scaled_ref_cords[start_idx], scaled_ref_cords[end_idx], (255, 0, 0), 2)
            for idx in scaled_ref_cords:
                cv2.circle(current_frame, scaled_ref_cords[idx], 5, (255, 100, 0), -1)

            # Grading Logic
            threshold = 30 
            matched_joints = []
            
            for i in body_points:
                if i in live_coords and i in scaled_ref_cords:
                    live_x, live_y = live_coords[i]
                    target_x, target_y = scaled_ref_cords[i]
                    if abs(live_x - target_x) <= threshold and abs(live_y - target_y) <= threshold:
                        matched_joints.append(i)

            # Color Coded Live Skeleton
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                if start_idx in live_coords and end_idx in live_coords:
                    bone_color = (0, 255, 0) if (start_idx in matched_joints and end_idx in matched_joints) else (0, 0, 255)
                    cv2.line(current_frame, live_coords[start_idx], live_coords[end_idx], bone_color, 3)

            for i in body_points:
                if i in live_coords:
                    joint_color = (0, 255, 0) if i in matched_joints else (0, 0, 255)
                    cv2.circle(current_frame, live_coords[i], 10, joint_color, -1)

        else:
            # Default Red Skeleton
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                if start_idx in live_coords and end_idx in live_coords:
                    cv2.line(current_frame, live_coords[start_idx], live_coords[end_idx], (0, 0, 255), 3)
            for i in body_points:
                if i in live_coords:
                    cv2.circle(current_frame, live_coords[i], 10, (0, 0, 255), -1)
    else: 
        cv2.putText(current_frame, "No body detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
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

#getting the coords for the refernce 
ref_raw = cv2.imread('pose_model.png')
if ref_raw is not None:
    with pose_landmarker.create_from_options(static_option) as static_runner:
        mp_ref_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=ref_raw)
        static_result = static_runner.detect(mp_ref_image)
        if static_result.pose_landmarks:
            ref_landmarks = static_result.pose_landmarks[0]
            for idx in body_points:
                lm = ref_landmarks[idx]
                ref_cords[idx] = (int(lm.x * 1280), int(lm.y * 720))
else:
    print("WARNING: 'pose_model.png' not found.")


ctk.set_appearance_mode("dark") 
ctk.set_default_color_theme("blue") 

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
if not cap.isOpened():
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

is_fullscreen = False
landmarker = pose_landmarker.create_from_options(live_option)

root = ctk.CTk()
root.title("AI Yoga Tracker")
root.geometry("1100x650")

# --- Left Menu Panel ---
left_frame = ctk.CTkFrame(root, width=250, corner_radius=0)
left_frame.pack(side="left", fill="y")

title_label = ctk.CTkLabel(left_frame, text="Yoga Menu", font=ctk.CTkFont(size=22, weight="bold"))
title_label.pack(pady=(30, 20))

def activate_pose():
    global show_instructor
    show_instructor = not show_instructor 
    
    if show_instructor:
        pose_button.configure(text="Stop Pose", fg_color="#e74c3c", hover_color="#c0392b")
    else:
        pose_button.configure(text="Pose 1: Tree Pose", fg_color="#2980b9", hover_color="#1f618d")

pose_button = ctk.CTkButton(left_frame, text="Pose 1: Tree Pose", font=ctk.CTkFont(size=14), 
                            command=activate_pose, height=40, fg_color="#2980b9", hover_color="#1f618d")
pose_button.pack(pady=10, padx=20, fill="x")

def toggle_fullscreen():
    global is_fullscreen
    is_fullscreen = not is_fullscreen
    root.attributes("-fullscreen", is_fullscreen)

fs_button = ctk.CTkButton(left_frame, text="Toggle Full Screen", font=ctk.CTkFont(size=12), 
                          fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), 
                          command=toggle_fullscreen, height=35)
fs_button.pack(side="bottom", pady=30, padx=20, fill="x")

# --- Right Video Panel ---
right_frame = ctk.CTkFrame(root, corner_radius=0, fg_color="black")
right_frame.pack(side="right", fill="both", expand=True)

video_label = ctk.CTkLabel(right_frame, text="")
video_label.pack(fill="both", expand=True)

# --- Video Rendering Loop ---
def update_frame():
    global current_timestamp_ms, annotated_frame
    
    success, frame = cap.read()
    if success:
        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        current_timestamp_ms += 1
        landmarker.detect_async(mp_image, current_timestamp_ms)
        
        if annotated_frame is not None:
            cv2_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image)
            
            window_width = right_frame.winfo_width()
            window_height = right_frame.winfo_height()
            
            if window_width > 50 and window_height > 50: 
                # CTkImage handles DPI and resizing automatically for sharp images
                ctk_image = ctk.CTkImage(light_image=pil_image, size=(window_width, window_height))
                video_label.configure(image=ctk_image)
                video_label.image = ctk_image # Anchor

    root.after(15, update_frame)

def on_closing():
    cap.release()
    landmarker.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Fire up the loops
update_frame()
root.mainloop()