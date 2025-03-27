import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ฟังก์ชันคำนวณระยะทางระหว่าง 2 จุด (Euclidean Distance)
def calculate_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# ฟังก์ชันคำนวณความคล้ายของท่าทาง
def compare_poses(pose1, pose2):
    total_distance = 0
    num_points = len(pose1)
    
    for i in range(num_points):
        total_distance += calculate_distance(pose1[i], pose2[i])
    
    similarity = 1 - (total_distance / num_points)
    return max(0, min(similarity, 1))  # Normalize to 0-1

# ฟังก์ชันตรวจจับท่าแขนไขว้เป็นตัว X
def is_crossed_arms(landmarks, threshold=0.1):
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    left_wrist = np.array(landmarks[LEFT_WRIST])
    right_wrist = np.array(landmarks[RIGHT_WRIST])
    left_shoulder = np.array(landmarks[LEFT_SHOULDER])
    right_shoulder = np.array(landmarks[RIGHT_SHOULDER])

    # ตรวจสอบว่าข้อมือสองข้างอยู่ใกล้กัน
    wrist_distance = calculate_distance(left_wrist, right_wrist)

    # ตรวจสอบว่าข้อมืออยู่ในระดับใกล้กับช่วงหน้าอก
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    left_wrist_to_mid = calculate_distance(left_wrist, shoulder_midpoint)
    right_wrist_to_mid = calculate_distance(right_wrist, shoulder_midpoint)

    if wrist_distance < threshold and left_wrist_to_mid < 0.2 and right_wrist_to_mid < 0.2:
        return True
    return False

# โหลดวิดีโอต้นแบบ และเปิดกล้องสำหรับผู้ใช้
template_video = cv2.VideoCapture('test.webm')
user_video = cv2.VideoCapture(0)  # เปิดกล้องผู้ใช้

# เปิด MediaPipe Pose
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while template_video.isOpened() and user_video.isOpened():
        # อ่านเฟรมจากวิดีโอต้นแบบ
        ret1, frame1 = template_video.read()
        ret2, frame2 = user_video.read()

        # ถ้าวิดีโอต้นแบบจบ -> กลับไปเฟรมแรก
        if not ret1:
            template_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if not ret2:
            break

        # Flip กล้องผู้ใช้เพื่อให้เหมือนกระจก
        frame2 = cv2.flip(frame2, 1)

        # แปลงสีเป็น RGB
        image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # หาท่าทางด้วย MediaPipe
        results1 = pose.process(image1)
        results2 = pose.process(image2)

        # ดึง landmarks จากทั้ง 2 วิดีโอ
        pose1 = []
        pose2 = []
        if results1.pose_landmarks and results2.pose_landmarks:
            for lm1, lm2 in zip(results1.pose_landmarks.landmark, results2.pose_landmarks.landmark):
                pose1.append([lm1.x, lm1.y])
                pose2.append([lm2.x, lm2.y])

            # เปรียบเทียบความคล้าย
            similarity = compare_poses(pose1, pose2) * 100  # เปลี่ยนเป็น %

            # วาดผลลัพธ์บนวิดีโอผู้ใช้
            mp_drawing.draw_landmarks(frame2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame2, f'Similarity: {similarity:.2f}%', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if similarity > 70 else (0, 0, 255), 2)

        # แสดงวิดีโอต้นแบบและกล้องผู้ใช้
        cv2.imshow('Template Video', frame1)
        cv2.imshow('Your Pose (Webcam)', frame2)

        # กด 'q' เพื่อออก
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# ปิดวิดีโอและกล้อง
template_video.release()
user_video.release()
cv2.destroyAllWindows()
