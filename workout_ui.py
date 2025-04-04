import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import mediapipe as mp
import numpy as np
import threading
import os
import time

class PoseDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Exercise Pose Detection")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize variables
        self.template_video_path = tk.StringVar()
        self.is_running = False
        self.template_video = None
        self.user_video = None
        self.video_speed = 1.0
        self.is_paused = False
        self.frame_counter = 0
        self.stop_event = threading.Event()
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_template = None
        self.pose_user = None
        
        # Create UI
        self.create_widgets()
        
        # Window closing handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def initialize_pose_models(self):
        """Initialize MediaPipe pose models"""
        if hasattr(self, 'pose_template') and self.pose_template:
            self.pose_template.close()
        if hasattr(self, 'pose_user') and self.pose_user:
            self.pose_user.close()
            
        self.pose_template = self.mp_pose.Pose(
            static_image_mode=False, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.pose_user = self.mp_pose.Pose(
            static_image_mode=False, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg="#ffffff", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(main_frame, text="Exercise Pose Detection", 
                             font=("Helvetica", 24, "bold"), 
                             bg="#ffffff", fg="#34495e")
        title_label.pack(pady=(0, 20))
        
        # File selection section
        file_section = tk.LabelFrame(main_frame, text="Video Selection", 
                                   font=("Helvetica", 14, "bold"), 
                                   padx=15, pady=15, bg="#ecf0f1", relief=tk.RIDGE)
        file_section.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(file_section, text="Template Exercise Video:", 
                font=("Helvetica", 12), bg="#ecf0f1").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        file_frame = tk.Frame(file_section, bg="#ecf0f1")
        file_frame.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        entry = tk.Entry(file_frame, textvariable=self.template_video_path, width=50, 
                        font=("Helvetica", 11), state='readonly')
        entry.pack(side=tk.LEFT, padx=(0, 10))
        
        select_btn = ttk.Button(file_frame, text="Browse...", command=self.select_template_video)
        select_btn.pack(side=tk.LEFT)
        
        self.preview_label = tk.Label(file_section, text="No video selected", 
                                    font=("Helvetica", 11, "italic"), 
                                    bg="#ecf0f1", fg="#7f8c8d")
        self.preview_label.grid(row=2, column=0, sticky=tk.W, pady=5)
        
        # Controls section
        controls_section = tk.LabelFrame(main_frame, text="Controls", 
                                       font=("Helvetica", 14, "bold"), 
                                       padx=15, pady=15, bg="#ecf0f1", relief=tk.RIDGE)
        controls_section.pack(fill=tk.X, padx=10, pady=10)
        
        button_frame = tk.Frame(controls_section, bg="#ecf0f1")
        button_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(button_frame, text="▶ Start Detection", 
                                  command=self.start_detection, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        self.stop_btn = ttk.Button(button_frame, text="■ Stop Detection", 
                                 command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        
        self.pause_btn = ttk.Button(button_frame, text="❚❚ Pause", 
                                  command=self.pause_video, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=10)
        
        self.resume_btn = ttk.Button(button_frame, text="► Resume", 
                                   command=self.resume_video, state=tk.DISABLED)
        self.resume_btn.pack(side=tk.LEFT, padx=10)

        # Speed control
        speed_frame = tk.Frame(controls_section, bg="#ecf0f1")
        speed_frame.pack(pady=10, fill=tk.X)
        
        tk.Label(speed_frame, text="Video Speed:", font=("Helvetica", 12),
                bg="#ecf0f1").pack(side=tk.LEFT, padx=(0, 10))
        
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(speed_frame, from_=0.5, to=2.0, 
                              orient=tk.HORIZONTAL, variable=self.speed_var, 
                              command=self.update_speed, length=200)
        speed_scale.pack(side=tk.LEFT, padx=5)
        
        self.speed_label = tk.Label(speed_frame, text="1.0x", font=("Helvetica", 11), 
                                   bg="#ecf0f1", width=4)
        self.speed_label.pack(side=tk.LEFT, padx=5)

        # Result section
        result_section = tk.LabelFrame(main_frame, text="Results", 
                                     font=("Helvetica", 14, "bold"), 
                                     padx=15, pady=15, bg="#ecf0f1", relief=tk.RIDGE)
        result_section.pack(fill=tk.X, padx=10, pady=10)
        
        # Similarity meter
        meter_frame = tk.Frame(result_section, bg="#ecf0f1")
        meter_frame.pack(pady=10, fill=tk.X)
        
        tk.Label(meter_frame, text="Pose Similarity:", font=("Helvetica", 12), 
                bg="#ecf0f1").pack(side=tk.LEFT, padx=(0, 10))
        
        self.similarity_var = tk.DoubleVar(value=0)
        self.similarity_meter = ttk.Progressbar(meter_frame, variable=self.similarity_var, 
                                             orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.similarity_meter.pack(side=tk.LEFT, padx=5)
        
        self.similarity_label = tk.Label(meter_frame, text="0%", font=("Helvetica", 12, "bold"), 
                                       bg="#ecf0f1", width=6)
        self.similarity_label.pack(side=tk.LEFT, padx=5)

        # Skeleton visibility controls
        skeleton_frame = tk.Frame(controls_section, bg="#ecf0f1")
        skeleton_frame.pack(pady=10, fill=tk.X)

        self.show_skeleton_template = tk.BooleanVar(value=True)
        self.show_skeleton_user = tk.BooleanVar(value=True)

        template_skeleton_chk = ttk.Checkbutton(skeleton_frame, text="Show Template Skeleton", 
                                                variable=self.show_skeleton_template)
        template_skeleton_chk.pack(side=tk.LEFT, padx=10)

        user_skeleton_chk = ttk.Checkbutton(skeleton_frame, text="Show User Skeleton", 
                                            variable=self.show_skeleton_user)
        user_skeleton_chk.pack(side=tk.LEFT, padx=10)

        
        # Status message
        self.status_frame = tk.Frame(result_section, bg="#ecf0f1")
        self.status_frame.pack(pady=5, fill=tk.X)
        
        self.status_label = tk.Label(self.status_frame, text="Ready to start", 
                                   font=("Helvetica", 12), 
                                   bg="#ecf0f1", fg="#2c3e50")
        self.status_label.pack()

    def select_template_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.webm")])
        if file_path:
            self.template_video_path.set(file_path)
            file_name = os.path.basename(file_path)
            self.preview_label.config(text=f"Selected: {file_name}")
            self.start_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Video selected. Click 'Start Detection' to begin.")

    def start_detection(self):
        if not self.template_video_path.get():
            messagebox.showerror("Error", "Please select a template video first.")
            return
            
        if self.is_running:
            self.stop_detection()
            time.sleep(0.5)
            
        self.cleanup_resources()
        self.initialize_pose_models()
        
        self.status_label.config(text="Starting detection... Please wait")
        self.root.update()
        
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.NORMAL)
        self.resume_btn.config(state=tk.DISABLED)
        
        self.is_running = True
        self.is_paused = False
        self.stop_event.clear()
        
        self.processing_thread = threading.Thread(target=self.run_detection, daemon=True)
        self.processing_thread.start()
        
        self.status_label.config(text="Detection running")

    def stop_detection(self):
        if self.is_running:
            self.stop_event.set()
            self.status_label.config(text="Stopping detection...")
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
            
            self.is_running = False
            self.is_paused = False
            
            self.cleanup_resources()
            
            self.start_btn.config(state=tk.NORMAL if self.template_video_path.get() else tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.DISABLED)
            self.resume_btn.config(state=tk.DISABLED)
            
            self.similarity_var.set(0)
            self.similarity_label.config(text="0%")
            
            self.status_label.config(text="Detection stopped")

    def pause_video(self):
        if not self.is_running:
            return
            
        self.is_paused = True
        self.pause_btn.config(state=tk.DISABLED)
        self.resume_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Detection paused")

    def resume_video(self):
        if not self.is_running:
            return
            
        self.is_paused = False
        self.pause_btn.config(state=tk.NORMAL)
        self.resume_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Detection running")

    def update_speed(self, event=None):
        self.video_speed = self.speed_var.get()
        self.speed_label.config(text=f"{self.video_speed:.1f}x")

    def calculate_distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))


    # def compare_poses(self, pose1, pose2):
    #     if not pose1 or not pose2 or len(pose1) != len(pose2):
    #         return 0

    #     # Define important relative joint pairs (based on Mediapipe's landmark index)
    #     JOINT_RELATIONS = [
    #         (15, 13),  # Right wrist → right elbow
    #         (13, 11),  # Right elbow → right shoulder
    #         (11, 0),   # Right shoulder → nose
    #         (16, 14),  # Left wrist → left elbow
    #         (14, 12),  # Left elbow → left shoulder
    #         (12, 0),   # Left shoulder → nose
    #         (27, 25),  # Right ankle → right knee
    #         (25, 23),  # Right knee → right hip
    #         (28, 26),  # Left ankle → left knee
    #         (26, 24),  # Left knee → left hip
    #     ]

    #     total_distance = 0
    #     for a, b in JOINT_RELATIONS:
    #         if a >= len(pose1) or b >= len(pose1) or a >= len(pose2) or b >= len(pose2):
    #             continue  # skip if out of bounds
            
    #         vec1 = np.array(pose1[a]) - np.array(pose1[b])
    #         vec2 = np.array(pose2[a]) - np.array(pose2[b])
            
    #         distance = self.calculate_distance(vec1, vec2)
    #         print(a , " " , b , " : ", distance)
    #         print(vec1," ",vec2)
    #         total_distance += distance

    #     avg_distance = total_distance / len(JOINT_RELATIONS)
    #     similarity = 1 - min(avg_distance, 1.0)
    #     return max(0, min(similarity, 1)) * 100
    def normalize_pose(self, pose):
        if not pose:
            return None

        left_shoulder = np.array(pose[12])
        right_shoulder = np.array(pose[11])
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

        if shoulder_width == 0:
            return None

        return [(np.array(p) - right_shoulder) / shoulder_width for p in pose]
    
    def compare_poses(self, pose1, pose2):
        pose1 = self.normalize_pose(pose1)
        pose2 = self.normalize_pose(pose2)
        
        if not pose1 or not pose2 or len(pose1) != len(pose2):
            return 0

        JOINT_RELATIONS = [
            (15, 13), (13, 11), (11, 0),
            (16, 14), (14, 12), (12, 0),
            (27, 25), (25, 23), (28, 26), (26, 24)
        ]

        total_distance = 0
        for a, b in JOINT_RELATIONS:
            if a >= len(pose1) or b >= len(pose1) or a >= len(pose2) or b >= len(pose2):
                continue
            
            vec1 = np.array(pose1[a]) - np.array(pose1[b])
            vec2 = np.array(pose2[a]) - np.array(pose2[b])
            distance = np.linalg.norm(vec1 - vec2)
            total_distance += distance

        avg_distance = total_distance / len(JOINT_RELATIONS)
        similarity = 1 - min(avg_distance, 1.0)
        per = min(((max(0, min(similarity, 1)) * 100)/80) * 100,100)
        return per



    def update_similarity_display(self, similarity):
        self.similarity_var.set(similarity)
        self.similarity_label.config(text=f"{similarity:.1f}%")
        
        if similarity > 70:
            color = "#28a745"  # Green
        elif similarity > 40:
            color = "#ffc107"  # Yellow
        else:
            color = "#dc3545"  # Red
        
        self.similarity_label.config(fg=color)

    def run_detection(self):
        try:
            self.template_video = cv2.VideoCapture(self.template_video_path.get())
            if not self.template_video.isOpened():
                raise RuntimeError(f"Failed to open template video: {self.template_video_path.get()}")
                
            camera_indices = [0, 1, 2]
            for index in camera_indices:
                self.user_video = cv2.VideoCapture(index)
                if self.user_video.isOpened():
                    break
                self.user_video.release()
            else:
                raise RuntimeError("Failed to open webcam - please check camera connection")
            
            self.user_video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.user_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Get base FPS of template video
            base_fps = self.template_video.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if FPS not available
            frame_interval = 1.0 / base_fps  # Time per frame in seconds
            
            while self.is_running and not self.stop_event.is_set():
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # Calculate frame skipping based on video speed
                self.frame_counter += self.video_speed
                frames_to_skip = int(self.frame_counter)
                self.frame_counter -= frames_to_skip
                
                # Read template video frame with speed control
                ret1, frame1 = None, None
                for _ in range(max(1, frames_to_skip)):
                    ret1, frame1 = self.template_video.read()
                    if not ret1 or frame1 is None:
                        self.template_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret1, frame1 = self.template_video.read()
                
                # Read user video frame (always real-time)
                ret1, frame1 = self.user_video.read()
                ret2, frame2 = self.user_video.read()
                
                if not ret1 or frame1 is None or not ret2 or frame2 is None:
                    continue

                frame1 = cv2.flip(frame1, 1)
                frame2 = cv2.flip(frame2, 1)
                
                try:
                    if frame1.shape != frame2.shape:
                        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
                    
                    image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                    image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    
                    results1 = self.pose_template.process(image1)
                    results2 = self.pose_user.process(image2)
                    
                    pose1 = [[lm.x, lm.y] for lm in results1.pose_landmarks.landmark] if results1.pose_landmarks else []
                    pose2 = [[lm.x, lm.y] for lm in results2.pose_landmarks.landmark] if results2.pose_landmarks else []
                    
                    similarity = self.compare_poses(pose1, pose2)
                    
                    self.root.after(0, self.update_similarity_display, similarity)

                    if results1.pose_landmarks and self.show_skeleton_template.get():
                                self.mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                    if results2.pose_landmarks and self.show_skeleton_user.get():
                        self.mp_drawing.draw_landmarks(frame2, results2.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    
                    color = (0, 255, 0) if similarity > 70 else (0, 0, 255)
                    cv2.putText(frame2, f'Similarity: {similarity:.1f}%', (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    combined = np.hstack((frame1, frame2))
                    cv2.imshow('Pose Detection', combined)
                    
                    # Adjust wait time based on base FPS and video speed
                    wait_time = max(1, int((frame_interval * 1000) / self.video_speed))
                    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                        break
                
                except Exception as e:
                    print(f"Frame processing error: {str(e)}")
                    continue
        
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", f"Video initialization error: {str(e)}")
        
        finally:
            self.cleanup_resources()
            self.root.after(0, self.reset_ui)

    def cleanup_resources(self):
        """Clean up all resources"""
        try:
            if self.template_video is not None and self.template_video.isOpened():
                self.template_video.release()
            if self.user_video is not None and self.user_video.isOpened():
                self.user_video.release()
            if self.pose_template is not None:
                self.pose_template.close()
            if self.pose_user is not None:
                self.pose_user.close()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
        finally:
            self.template_video = None
            self.user_video = None
            self.pose_template = None
            self.pose_user = None

    def reset_ui(self):
        """Reset UI to initial state"""
        self.is_running = False
        self.is_paused = False
        self.similarity_var.set(0)
        self.similarity_label.config(text="0%", fg="#000000")
        self.start_btn.config(state=tk.NORMAL if self.template_video_path.get() else tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)
        self.resume_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Ready to start")

    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self.stop_detection()
        self.cleanup_resources()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = PoseDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()