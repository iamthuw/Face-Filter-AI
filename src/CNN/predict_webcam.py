import cv2
import dlib
import tensorflow as tf
import numpy as np
import argparse

# --- Các hằng số và thiết lập ---
MODEL_PATH = 'saved_model/facial_landmark_detector.h5'
IMAGE_SIZE = 128
NUM_LANDMARKS = 68

# --- Tải mô hình và công cụ phát hiện khuôn mặt ---
print("[INFO] Loading facial landmark predictor...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    detector = dlib.get_frontal_face_detector()
except Exception as e:
    print(f"[ERROR] Could not load model or detector: {e}")
    exit()

print("[INFO] Starting webcam stream...")
# Số 0 có nghĩa là sử dụng webcam mặc định của máy
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    exit()

# --- Vòng lặp xử lý real-time ---
while True:
    ret, frame = cap.read()
    
    # --- THAY THẾ BẰNG ĐOẠN KIỂM TRA MỚI ---
    # Kiểm tra xem có đọc được frame không VÀ frame có nội dung không
    if not ret or frame is None or frame.size == 0:
        print("Skipping empty frame...")
        continue # Bỏ qua frame này và thử lại với frame tiếp theo
    # ------------------------------------

    # Lật ảnh để có hiệu ứng gương soi
    frame = cv2.flip(frame, 1)
    # ... code còn lại giữ nguyên

    # Lật ảnh để có hiệu ứng gương soi
    frame = cv2.flip(frame, 1)
    
    # Chuyển ảnh sang màu xám để dlib phát hiện khuôn mặt nhanh hơn
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện tất cả các khuôn mặt trong khung hình
    faces = detector(gray, 1)

    # Lặp qua từng khuôn mặt phát hiện được
    for face in faces:
        # Lấy tọa độ bounding box
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        
        # Cắt vùng chứa khuôn mặt ra khỏi ảnh gốc
        # Thêm một khoảng đệm nhỏ để đảm bảo không bị mất thông tin
        pad = 20
        top, left = max(0, y - pad), max(0, x - pad)
        bottom, right = min(frame.shape[0], y + h + pad), min(frame.shape[1], x + w + pad)
        
        cropped_face = frame[top:bottom, left:right]

        if cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0:
            continue

        # Tiền xử lý ảnh khuôn mặt giống như khi huấn luyện
        resized_face = cv2.resize(cropped_face, (IMAGE_SIZE, IMAGE_SIZE))
        normalized_face = resized_face.astype('float32') / 255.0
        input_face = np.expand_dims(normalized_face, axis=0) # Thêm chiều batch

        # Đưa vào mô hình để dự đoán landmarks
        predicted_landmarks = model.predict(input_face, verbose=0)[0]
        
        # Reshape lại và chuyển về tọa độ gốc
        landmarks = predicted_landmarks.reshape((NUM_LANDMARKS, 2))
        landmarks = landmarks * np.array([right - left, bottom - top]) + np.array([left, top])
        landmarks = landmarks.astype(int)

        # Vẽ landmarks lên khung hình gốc
        for (lx, ly) in landmarks:
            cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)

    # Hiển thị khung hình kết quả ra màn hình
    cv2.imshow('Webcam Facial Landmark Detection', frame)

    # Đợi người dùng nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Dọn dẹp ---
print("[INFO] Cleaning up...")
cap.release()
cv2.destroyAllWindows()