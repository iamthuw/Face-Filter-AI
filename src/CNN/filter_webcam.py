"""
Face Filter AI - Webcam Realtime Filter Application

Mô tả:
    - Lấy video từ webcam.
    - Phát hiện khuôn mặt bằng Haar Cascade.
    - Dự đoán facial landmarks bằng mô hình deep learning.
    - Áp dụng các bộ lọc AR (glasses, mustache, pignose, blush).
    - Cho phép người dùng chọn filter bằng phím bấm.

Yêu cầu:
    - OpenCV
    - Tensorflow
    - Numpy
    - apply_filter.py với các hàm: draw_glasses, draw_mustache, draw_pignose, draw_blush
"""

import cv2
import numpy as np
import tensorflow as tf
import sys, os

# Thêm thư mục hiện tại vào sys.path để import apply_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_filter

# ================== CẤU HÌNH ==================

MODEL_PATH = "saved_model/facial_landmark_detector.h5"  # Đường dẫn mô hình landmark
IMAGE_SIZE = 128                                         # Kích thước input của model
NUM_LANDMARKS = 68                                       # Số lượng landmarks dự đoán

# ================== LOAD MODEL & DỮ LIỆU ==================

# Load model TensorFlow (không compile để tăng tốc load)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Load ảnh filter PNG có alpha channel (IMREAD_UNCHANGED để giữ alpha mask)
glasses = cv2.imread("filters/glasses.png", cv2.IMREAD_UNCHANGED)
mustache = cv2.imread("filters/mustache.png", cv2.IMREAD_UNCHANGED)
pignose = cv2.imread("filters/pignose.png", cv2.IMREAD_UNCHANGED)
blush = cv2.imread("filters/blush.png", cv2.IMREAD_UNCHANGED)

# Mở webcam mặc định (ID=0)
cap = cv2.VideoCapture(0)

# Haar Cascade frontal-face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 0:none, 1:glasses, 2:mustache, 3:pignose, 4:blush
current_filter = 0
print("Press keys 0:none,1:glasses,2:mustache,3:pignose,4:blush. Q to quit")


def predict_landmarks(face_img, bbox):
    """
    Dự đoán facial landmarks từ ảnh khuôn mặt đã crop.

    Parameters:
        face_img (np.ndarray):
            Ảnh khuôn mặt (crop từ frame gốc), dạng BGR.
        bbox (tuple):
            (x, y, w, h) - vị trí bounding box của mặt trong frame gốc.

    Returns:
        np.ndarray:
            Mảng (68, 2) chứa tọa độ landmarks trong ảnh gốc.
    """
    x, y, w, h = bbox

    # Resize ảnh mặt về IMAGE_SIZE x IMAGE_SIZE
    face_resized = cv2.resize(face_img, (IMAGE_SIZE, IMAGE_SIZE))

    # Chuẩn hóa [0,1] và thêm batch dimension => (1, 128, 128, 3)
    inp = np.expand_dims(face_resized / 255.0, axis=0)

    # Model output: (136,) => reshape thành (68, 2)
    pred = model.predict(inp, verbose=0)[0]
    landmarks = pred.reshape(NUM_LANDMARKS, 2)

    # Chuyển từ toạ độ normalized (0–1) sang tọa độ thật trong frame
    landmarks[:, 0] = landmarks[:, 0] * w + x
    landmarks[:, 1] = landmarks[:, 1] * h + y

    return landmarks


# ================== VÒNG LẶP CHÍNH ==================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển sang grayscale để tăng tốc detectMultiScale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Haar Cascade trả về list bounding boxes (x, y, w, h)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # Xử lý từng khuôn mặt tìm được
    for (x, y, w, h) in faces:
        # Crop phần khuôn mặt
        face = frame[y:y+h, x:x+w]

        # Dự đoán 68 landmarks
        landmarks = predict_landmarks(face, (x, y, w, h))

        # ================== ÁP DỤNG FILTER ==================
        if current_filter == 1:
            frame = apply_filter.draw_glasses(frame, landmarks, glasses)
        elif current_filter == 2:
            frame = apply_filter.draw_mustache(frame, landmarks, mustache)
        elif current_filter == 3:
            frame = apply_filter.draw_pignose(frame, landmarks, pignose)
        elif current_filter == 4:
            frame = apply_filter.draw_blush(frame, landmarks, blush)
        # current_filter == 0 => không áp dụng filter nào

        # ================== HIỂN THỊ DEBUG INFO ==================
        # Chỉ hiển thị landmarks + bounding box khi không bật filter
        if current_filter == 0:
            # Vẽ từng landmark (điểm xanh)
            for (lx, ly) in landmarks.astype(int):
                cv2.circle(frame, (lx, ly), 1, (0, 255, 0), -1)

            # Vẽ bounding box màu xanh dương
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Hiển thị frame ra cửa sổ
    cv2.imshow("Face Filter AI - Webcam", frame)

    # ================== NHẬN PHÍM TỪ BÀN PHÍM ==================
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        # Thoát chương trình
        break
    elif key in [ord(str(i)) for i in range(5)]:
        # Nhấn phím 0..4 để đổi filter
        current_filter = int(chr(key))

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
