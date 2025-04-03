import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import mediapipe as mp
import numpy as np
import threading
import os

class PoseDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Exercise Pose Detection")
        self.root.geometry("900x700")
        
        # MediaPipe initialization (Separate for template and user video)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_template = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose_user = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Variables
        self.template_video_path = tk.StringVar()
        self.is_running = False
        self.template_video = None
        self.user_video = None
        self.video_speed = 1.0
        self.is_paused = False

        # Create UI
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Template Exercise Video:", font=("Arial", 12, "bold")).pack(pady=(10, 0))
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=10)
        tk.Entry(file_frame, textvariable=self.template_video_path, width=50, state='readonly', font=("Arial", 10)).pack(side=tk.LEFT, padx=(0, 10))
        tk.Button(file_frame, text="Select Video", font=("Arial", 10), command=self.select_template_video).pack(side=tk.LEFT)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        self.start_btn = tk.Button(button_frame, text="Start Detection", font=("Arial", 10), command=self.start_detection, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=10)
        self.stop_btn = tk.Button(button_frame, text="Stop Detection", font=("Arial", 10), command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        self.pause_btn = tk.Button(button_frame, text="Pause", font=("Arial", 10), command=self.pause_video, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=10)
        self.resume_btn = tk.Button(button_frame, text="Resume", font=("Arial", 10), command=self.resume_video, state=tk.DISABLED)
        self.resume_btn.pack(side=tk.LEFT, padx=10)

        speed_frame = tk.Frame(self.root)
        speed_frame.pack(pady=10)
        tk.Label(speed_frame, text="Video Speed:", font=("Arial", 10)).pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=1.0)
        tk.Scale(speed_frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.speed_var, command=self.update_speed).pack(side=tk.LEFT)

        self.result_label = tk.Label(self.root, text="", font=("Arial", 12, "bold"), fg="blue")
        self.result_label.pack(pady=10)

    def select_template_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.webm")])
        if file_path:
            self.template_video_path.set(file_path)
            self.start_btn.config(state=tk.NORMAL)

    def start_detection(self):
        if not self.template_video_path.get():
            messagebox.showerror("Error", "Please select a template video first.")
            return
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.NORMAL)
        self.resume_btn.config(state=tk.DISABLED)
        self.is_running = True
        threading.Thread(target=self.run_detection, daemon=True).start()

    def stop_detection(self):
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)
        self.resume_btn.config(state=tk.DISABLED)
        self.result_label.config(text="Detection Stopped")
        
        if self.template_video:
            self.template_video.release()
            self.template_video = None  # เคลียร์ค่า
        
        if self.user_video:
            self.user_video.release()
            self.user_video = None  # เคลียร์ค่า
        
        cv2.destroyAllWindows()


    def pause_video(self):
        self.is_paused = True
        self.pause_btn.config(state=tk.DISABLED)
        self.resume_btn.config(state=tk.NORMAL)

    def resume_video(self):
        self.is_paused = False
        self.pause_btn.config(state=tk.NORMAL)
        self.resume_btn.config(state=tk.DISABLED)

    def update_speed(self, event=None):
        self.video_speed = self.speed_var.get()

    def calculate_distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def compare_poses(self, pose1, pose2):
        total_distance = 0
        num_points = len(pose1)
        for i in range(num_points):
            total_distance += self.calculate_distance(pose1[i], pose2[i])
        similarity = 1 - (total_distance / num_points)
        return max(0, min(similarity, 1)) * 100

    def run_detection(self):
        self.template_video = cv2.VideoCapture(self.template_video_path.get())
        self.user_video = cv2.VideoCapture(0)
        if not self.user_video.isOpened():
            messagebox.showerror("Error", "Cannot open webcam. Please check your camera and try again.")
            self.stop_detection()
            return
        while self.is_running:
            if self.is_paused:
                continue
            ret1, frame1 = self.template_video.read()
            ret2, frame2 = self.user_video.read()
            if not ret1:
                self.template_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            if not ret2:
                break
            frame2 = cv2.flip(frame2, 1)
            
            # ตรวจสอบว่าเฟรมไม่เป็น None และปรับขนาดให้เท่ากัน
            if frame1 is None or frame2 is None:
                continue
            
            # ปรับขนาด frame2 ให้เท่ากับ frame1
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

            image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            results1 = self.pose_template.process(image1)
            results2 = self.pose_user.process(image2)
            
            # การตรวจจับ pose
            pose1 = [[lm.x, lm.y] for lm in results1.pose_landmarks.landmark] if results1.pose_landmarks else []
            pose2 = [[lm.x, lm.y] for lm in results2.pose_landmarks.landmark] if results2.pose_landmarks else []
            
            # คำนวณความคล้ายคลึง
            similarity = self.compare_poses(pose1, pose2) if pose1 and pose2 else 0
            self.root.after(0, self.result_label.config, {"text": f"Pose Similarity: {similarity:.2f}%", "fg": "green" if similarity > 70 else "red"})
            
            # วาด landmark และแสดงผล
            self.mp_drawing.draw_landmarks(frame2, results2.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame2, f'Similarity: {similarity:.2f}%', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if similarity > 70 else (0, 0, 255), 2)

            # รวม frame1 และ frame2
            combined = np.hstack((frame1, frame2))
            
            # แสดงผล
            cv2.imshow('Pose Detection', combined)
            
            # ควบคุมความเร็วของวิดีโอ
            if cv2.waitKey(int(10 / self.video_speed)) & 0xFF == ord('q'):
                break
        self.stop_detection()


def main():
    root = tk.Tk()
    app = PoseDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
