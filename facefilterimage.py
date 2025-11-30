"""
Face Filter AI - Image Input for Multiple People (Interactive Filter, Save 5 Results)
Corrected landmarks mapping to fix filter position
"""

import cv2
import numpy as np
import tensorflow as tf
import sys, os

# Thêm thư mục hiện tại vào sys.path để import apply_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_filter

# ================== CẤU HÌNH ==================
MODEL_PATH = "src/CNN/saved_model/facial_landmark_detector.h5"
IMAGE_SIZE = 128
NUM_LANDMARKS = 68

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

glasses = cv2.imread("filters/glasses.png", cv2.IMREAD_UNCHANGED)
mustache = cv2.imread("filters/mustache.png", cv2.IMREAD_UNCHANGED)
pignose = cv2.imread("filters/pignose.png", cv2.IMREAD_UNCHANGED)
blush = cv2.imread("filters/blush.png", cv2.IMREAD_UNCHANGED)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Filter mapping
filter_dict = {0:"none", 1:"glasses", 2:"mustache", 3:"pignose", 4:"blush"}
current_filter = 0
print("Press keys 0:none,1:glasses,2:mustache,3:pignose,4:blush. Q to quit")

# ================== HÀM DỰ ĐOÁN LANDMARK ==================
def predict_landmarks(frame, bbox):
    x, y, w, h = bbox
    face_crop = frame[y:y+h, x:x+w]
    inp = np.expand_dims(cv2.resize(face_crop, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0, axis=0)
    pred = model.predict(inp, verbose=0)[0]
    landmarks = pred.reshape(NUM_LANDMARKS, 2)

    # Chuyển tọa độ normalized sang tọa độ gốc dựa trên bbox thực tế, KHÔNG padding
    landmarks[:, 0] = landmarks[:, 0] * w + x
    landmarks[:, 1] = landmarks[:, 1] * h + y
    return landmarks

# ================== LOAD ẢNH ==================
img_path = "k68a5.png"  # đổi theo ảnh của bạn
original_frame = cv2.imread(img_path)
if original_frame is None:
    print(f"Không tìm thấy file: {img_path}")
    sys.exit(1)

# Dự đoán landmarks và bounding boxes 1 lần
gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
landmarks_list = []
for (x, y, w, h) in faces:
    landmarks = predict_landmarks(original_frame, (x, y, w, h))
    landmarks_list.append((x, y, w, h, landmarks))

# ================== TỰ ĐỘNG LƯU 5 FILE ==================
for f_id in range(5):
    frame_copy = original_frame.copy()
    for (x, y, w, h, landmarks) in landmarks_list:
        if f_id == 1:
            frame_copy = apply_filter.draw_glasses(frame_copy, landmarks, glasses)
        elif f_id == 2:
            frame_copy = apply_filter.draw_mustache(frame_copy, landmarks, mustache)
        elif f_id == 3:
            frame_copy = apply_filter.draw_pignose(frame_copy, landmarks, pignose)
        elif f_id == 4:
            frame_copy = apply_filter.draw_blush(frame_copy, landmarks, blush)
        elif f_id == 0:
            # Debug: vẽ landmarks + bbox
            for (lx, ly) in landmarks.astype(int):
                cv2.circle(frame_copy, (lx, ly), 1, (0, 255, 0), -1)
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)

    out_filename = f"output_{f_id}_{filter_dict[f_id]}.jpg"
    cv2.imwrite(out_filename, frame_copy)
    print(f"Saved: {out_filename}")

# ================== VÒNG LẶP INTERACTIVE ==================
while True:
    frame = original_frame.copy()
    for (x, y, w, h, landmarks) in landmarks_list:
        if current_filter == 1:
            frame = apply_filter.draw_glasses(frame, landmarks, glasses)
        elif current_filter == 2:
            frame = apply_filter.draw_mustache(frame, landmarks, mustache)
        elif current_filter == 3:
            frame = apply_filter.draw_pignose(frame, landmarks, pignose)
        elif current_filter == 4:
            frame = apply_filter.draw_blush(frame, landmarks, blush)
        elif current_filter == 0:
            for (lx, ly) in landmarks.astype(int):
                cv2.circle(frame, (lx, ly), 1, (0, 255, 0), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Filter AI - Image", frame)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break
    elif key in [ord(str(i)) for i in range(5)]:
        current_filter = int(chr(key))
        print(f"Filter changed to {current_filter}")

cv2.destroyAllWindows()
