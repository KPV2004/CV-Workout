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
        
        # MediaPipe initialization
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_template = None
        self.pose_user = None
        self.initialize_pose_models()

        # Variables
        self.template_video_path = tk.StringVar()
        self.is_running = False
        self.template_video = None
        self.user_video = None
        self.video_speed = 1.0
        self.is_paused = False
        self.processing_thread = None
        self.stop_event = threading.Event()

        # Create UI with modern style
        self.create_widgets()
        
        # Protocol handler for window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def initialize_pose_models(self):
        """Initialize MediaPipe pose models"""
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
        # Main frame with padding
        main_frame = tk.Frame(self.root, bg="#ffffff", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title with better styling
        title_label = tk.Label(main_frame, text="Exercise Pose Detection", font=("Helvetica", 24, "bold"), 
                              bg="#ffffff", fg="#34495e")
        title_label.pack(pady=(0, 20))
        
        # File selection section with border and background
        file_section = tk.LabelFrame(main_frame, text="Video Selection", font=("Helvetica", 14, "bold"), 
                                    padx=15, pady=15, bg="#ecf0f1", relief=tk.RIDGE)
        file_section.pack(fill=tk.X, padx=10, pady=10)
        
        # File selection UI
        tk.Label(file_section, text="Template Exercise Video:", font=("Helvetica", 12), 
                bg="#ecf0f1").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        file_frame = tk.Frame(file_section, bg="#ecf0f1")
        file_frame.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        entry = tk.Entry(file_frame, textvariable=self.template_video_path, width=50, 
                        font=("Helvetica", 11), state='readonly')
        entry.pack(side=tk.LEFT, padx=(0, 10))
        
        select_btn = ttk.Button(file_frame, text="Browse...", command=self.select_template_video)
        select_btn.pack(side=tk.LEFT)
        
        # Preview label
        self.preview_label = tk.Label(file_section, text="No video selected", font=("Helvetica", 11, "italic"), 
                                     bg="#ecf0f1", fg="#7f8c8d")
        self.preview_label.grid(row=2, column=0, sticky=tk.W, pady=5)
        
        # Controls section
        controls_section = tk.LabelFrame(main_frame, text="Controls", font=("Helvetica", 14, "bold"), 
                                        padx=15, pady=15, bg="#ecf0f1", relief=tk.RIDGE)
        controls_section.pack(fill=tk.X, padx=10, pady=10)
        
        # Button frame with better styling
        button_frame = tk.Frame(controls_section, bg="#ecf0f1")
        button_frame.pack(pady=10)
        
        # Create styled buttons
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

        # Speed control with label
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
        result_section = tk.LabelFrame(main_frame, text="Results", font=("Helvetica", 14, "bold"), 
                                      padx=15, pady=15, bg="#ecf0f1", relief=tk.RIDGE)
        result_section.pack(fill=tk.X, padx=10, pady=10)
        
        # Similarity meter (progress bar)
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
        
        # Status message
        self.status_frame = tk.Frame(result_section, bg="#ecf0f1")
        self.status_frame.pack(pady=5, fill=tk.X)
        
        self.status_label = tk.Label(self.status_frame, text="Ready to start", font=("Helvetica", 12), 
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
            return
            
        self.status_label.config(text="Starting detection... Please wait")
        self.root.update()
        
        # ปรับสถานะปุ่ม
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.NORMAL)
        self.resume_btn.config(state=tk.DISABLED)
        
        # รีเซ็ตสถานะ
        self.is_running = True
        self.is_paused = False
        self.stop_event.clear()
        
        # สร้าง thread ใหม่
        self.processing_thread = threading.Thread(target=self.run_detection, daemon=True)
        self.processing_thread.start()
        
        self.status_label.config(text="Detection running")

    def stop_detection(self):
        if self.is_running:
            self.stop_event.set()
            self.status_label.config(text="Stopping detection...")
            
            # รอให้ thread จบการทำงาน
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
            
            self.is_running = False
            self.is_paused = False
            
            # ปรับสถานะปุ่ม
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.DISABLED)
            self.resume_btn.config(state=tk.DISABLED)
            
            # ปรับค่าความคล้ายกลับเป็น 0
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

    def compare_poses(self, pose1, pose2):
        if not pose1 or not pose2 or len(pose1) != len(pose2):
            return 0
            
        total_distance = 0
        num_points = len(pose1)
        for i in range(num_points):
            total_distance += self.calculate_distance(pose1[i], pose2[i])
        
        avg_distance = total_distance / num_points if num_points > 0 else 0
        similarity = 1 - min(avg_distance, 1.0)  # Normalize to 0-1
        return max(0, min(similarity, 1)) * 100

    def update_similarity_display(self, similarity):
        self.similarity_var.set(similarity)
        self.similarity_label.config(text=f"{similarity:.1f}%")
        
        # ปรับสีตามค่าความคล้าย
        if similarity > 70:
            color = "#28a745"  # เขียว
        elif similarity > 40:
            color = "#ffc107"  # เหลือง
        else:
            color = "#dc3545"  # แดง
        
        self.similarity_label.config(fg=color)

    def run_detection(self):
        try:
            # เปิดวิดีโอ
            self.template_video = cv2.VideoCapture(self.template_video_path.get())
            
            # พยายามเปิดกล้องเว็บแคมหลายครั้ง (ถ้าจำเป็น)
            max_attempts = 3
            for attempt in range(max_attempts):
                self.user_video = cv2.VideoCapture(0)
                if self.user_video.isOpened():
                    break
                time.sleep(1)  # รอ 1 วินาทีก่อนพยายามอีกครั้ง
            
            if not self.template_video.isOpened():
                self.root.after(0, messagebox.showerror, "Error", "Cannot open template video.")
                self.root.after(0, self.stop_detection)
                return
                
            if not self.user_video.isOpened():
                self.root.after(0, messagebox.showerror, "Error", "Cannot open webcam after multiple attempts. Please check your camera.")
                self.root.after(0, self.stop_detection)
                return
            
            # ลูปหลักของการประมวลผล
            while not self.stop_event.is_set():
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # อ่านเฟรมจากวิดีโอ
                ret1, frame1 = self.template_video.read()
                ret2, frame2 = self.user_video.read()
                
                # ตรวจสอบการอ่านวิดีโอ
                if not ret1:
                    self.template_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                    
                if not ret2 or frame2 is None:
                    self.root.after(0, messagebox.showwarning, "Warning", "Cannot read frame from webcam.")
                    break
                
                # พลิกภาพกล้อง
                frame2 = cv2.flip(frame2, 1)
                
                # ตรวจสอบเฟรมว่าถูกต้อง
                if frame1 is None:
                    continue
                
                try:
                    # ปรับขนาดเฟรมให้เท่ากัน (ถ้าจำเป็น)
                    if frame1.shape != frame2.shape:
                        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
                    
                    # แปลงสีและตรวจจับท่าทาง
                    image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                    image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    
                    results1 = self.pose_template.process(image1)
                    results2 = self.pose_user.process(image2)
                    
                    # ดึงจุด landmark
                    pose1 = []
                    pose2 = []
                    
                    if results1.pose_landmarks:
                        pose1 = [[lm.x, lm.y] for lm in results1.pose_landmarks.landmark]
                    
                    if results2.pose_landmarks:
                        pose2 = [[lm.x, lm.y] for lm in results2.pose_landmarks.landmark]
                    
                    # คำนวณความคล้าย
                    similarity = self.compare_poses(pose1, pose2)
                    
                    # อัพเดต UI จาก thread หลัก
                    self.root.after(0, self.update_similarity_display, similarity)
                    
                    # วาด landmark บนเฟรม
                    if results2.pose_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame2, 
                            results2.pose_landmarks, 
                            self.mp_pose.POSE_CONNECTIONS
                        )
                    
                    # เพิ่มข้อความบนเฟรม
                    color = (0, 255, 0) if similarity > 70 else (0, 0, 255)
                    cv2.putText(frame2, f'Similarity: {similarity:.1f}%', (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # รวมเฟรมและแสดงผล
                    combined = np.hstack((frame1, frame2))
                    cv2.imshow('Pose Detection', combined)
                    
                    # ควบคุมความเร็ว
                    wait_time = max(1, int(10 / self.video_speed))
                    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                        break
                
                except Exception as e:
                    self.root.after(0, messagebox.showerror, "Processing Error", f"Error processing frame: {str(e)}")
                    break
        
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", f"An unexpected error occurred: {str(e)}")
        
        finally:
            self.cleanup_resources()
            self.root.after(0, self.reset_ui)

    def cleanup_resources(self):
        """ทำความสะอาดและปล่อยทรัพยากรทั้งหมด"""
        try:
            if self.template_video and self.template_video.isOpened():
                self.template_video.release()
            
            if self.user_video and self.user_video.isOpened():
                self.user_video.release()
            
            if self.pose_template:
                self.pose_template.close()
            
            if self.pose_user:
                self.pose_user.close()
            
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
        
        finally:
            self.template_video = None
            self.user_video = None

    def reset_ui(self):
        """รีเซ็ต UI กลับสู่สถานะเริ่มต้น"""
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
        """จัดการเมื่อกดปิดหน้าต่าง"""
        if self.is_running:
            self.stop_detection()
            # รอให้ thread หยุดทำงาน
            time.sleep(0.5)
        
        # ทำความสะอาดทรัพยากรก่อนปิด
        self.cleanup_resources()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = PoseDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()