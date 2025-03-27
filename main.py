import cv2
import mediapipe as mp
import numpy as np

# ตั้งค่าการใช้งาน MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# ตัวแปรสำหรับนับจำนวนสควอท
squat_count = 0
squat_position = False  # สถานะว่ากำลังอยู่ในท่าสควอทหรือไม่

def calculate_angle(a, b, c):
    """ คำนวณมุมระหว่างสามจุด (ใช้สำหรับวัดมุมที่หัวเข่า) """
    a = np.array(a)  # สะโพก
    b = np.array(b)  # หัวเข่า
    c = np.array(c)  # ข้อเท้า

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # พลิกภาพแนวนอนให้เหมือนกระจก
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # แปลงเป็น RGB
    results = pose.process(rgb_frame)  # ตรวจจับท่าทาง

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # ดึงค่าตำแหน่งของสะโพก, หัวเข่า และข้อเท้า (ขาขวา)
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        # คำนวณมุมของหัวเข่า
        knee_angle = calculate_angle(hip, knee, ankle)

        # เช็คว่าผู้ใช้กำลังอยู่ในท่าสควอทหรือไม่
        if knee_angle < 90:  # ถ้าหัวเข่าโค้งลงมาถึงระดับหนึ่ง ถือว่าเป็นท่าสควอท
            squat_position = True
        elif knee_angle > 150 and squat_position:  # ถ้ากลับมายืนตรง และเคยอยู่ในท่าสควอทมาก่อน
            squat_count += 1
            squat_position = False

        # แสดงมุมที่หัวเข่า
        cv2.putText(frame, f"Knee Angle: {int(knee_angle)}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # แสดงจำนวนครั้งของการสควอท
    cv2.putText(frame, f"Squat Count: {squat_count}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # วาดเส้นโครงร่างร่างกาย
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Squat Counter", frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
