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
        self.root.geometry("800x600")

        # MediaPipe initialization
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=False, 
                                      min_detection_confidence=0.5, 
                                      min_tracking_confidence=0.5)

        # Variables
        self.template_video_path = tk.StringVar()
        self.is_running = False
        self.template_video = None
        self.user_video = None

        # Create UI
        self.create_widgets()

    def create_widgets(self):
        # Template Video Selection
        tk.Label(self.root, text="Template Exercise Video:").pack(pady=(10,0))
        
        # Container for file selection
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=10)
        
        tk.Entry(file_frame, textvariable=self.template_video_path, width=50, state='readonly').pack(side=tk.LEFT, padx=(0,10))
        tk.Button(file_frame, text="Select Video", command=self.select_template_video).pack(side=tk.LEFT)

        # Buttons Frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        self.start_btn = tk.Button(button_frame, text="Start Detection", command=self.start_detection, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=10)

        self.stop_btn = tk.Button(button_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10)

        # Results Label
        self.result_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)

    def select_template_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.webm")]
        )
        if file_path:
            self.template_video_path.set(file_path)
            self.start_btn.config(state=tk.NORMAL)

    def start_detection(self):
        if not self.template_video_path.get():
            messagebox.showerror("Error", "Please select a template video first.")
            return

        # Disable start button, enable stop button
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Reset running flag
        self.is_running = True

        # Start detection in a separate thread
        threading.Thread(target=self.run_detection, daemon=True).start()

    def stop_detection(self):
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.result_label.config(text="Detection Stopped")

        # Release video captures
        if self.template_video:
            self.template_video.release()
        if self.user_video:
            self.user_video.release()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()

    def calculate_distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def compare_poses(self, pose1, pose2):
        total_distance = 0
        num_points = len(pose1)
        
        for i in range(num_points):
            total_distance += self.calculate_distance(pose1[i], pose2[i])
        
        similarity = 1 - (total_distance / num_points)
        return max(0, min(similarity, 1))  # Normalize to 0-1

    def run_detection(self):
        # Open template and user videos
        self.template_video = cv2.VideoCapture(self.template_video_path.get())
        self.user_video = cv2.VideoCapture(0)  # Open webcam

        while self.is_running:
            # Read frames
            ret1, frame1 = self.template_video.read()
            ret2, frame2 = self.user_video.read()

            # Reset template video if it ends
            if not ret1:
                self.template_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            if not ret2:
                break

            # Flip user video
            frame2 = cv2.flip(frame2, 1)

            # Convert to RGB
            image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            # Process poses
            results1 = self.pose.process(image1)
            results2 = self.pose.process(image2)

            # Compare poses if landmarks detected
            pose1 = []
            pose2 = []
            if results1.pose_landmarks and results2.pose_landmarks:
                for lm1, lm2 in zip(results1.pose_landmarks.landmark, results2.pose_landmarks.landmark):
                    pose1.append([lm1.x, lm1.y])
                    pose2.append([lm2.x, lm2.y])

                # Calculate similarity
                similarity = self.compare_poses(pose1, pose2) * 100

                # Update results label in main thread
                self.root.after(0, self.update_result_label, similarity)

                # Draw landmarks and similarity on user video
                self.mp_drawing.draw_landmarks(frame2, results2.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame2, f'Similarity: {similarity:.2f}%', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (0, 255, 0) if similarity > 70 else (0, 0, 255), 2)

            # Show videos
            cv2.imshow('Template Video', frame1)
            cv2.imshow('Your Pose (Webcam)', frame2)

            # Wait and check for quit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Clean up
        self.stop_detection()

    def update_result_label(self, similarity):
        color = "green" if similarity > 70 else "red"
        self.result_label.config(text=f"Pose Similarity: {similarity:.2f}%", fg=color)

def main():
    root = tk.Tk()
    app = PoseDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()