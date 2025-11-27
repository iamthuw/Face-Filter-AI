import cv2
import dlib
import tensorflow as tf
import numpy as np
import argparse

MODEL_PATH = 'saved_model/facial_landmark_detector.h5'
IMAGE_SIZE = 128
NUM_LANDMARKS = 68

print("[INFO] Loading facial landmark predictor...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    detector = dlib.get_frontal_face_detector()
except Exception as e:
    print(f"[ERROR] Could not load model or detector: {e}")
    exit()

print("[INFO] Starting webcam stream...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret or frame is None or frame.size == 0:
        print("Skipping empty frame...")
        continue 
    
    frame = cv2.flip(frame, 1)
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        
        pad = 20
        top, left = max(0, y - pad), max(0, x - pad)
        bottom, right = min(frame.shape[0], y + h + pad), min(frame.shape[1], x + w + pad)
        
        cropped_face = frame[top:bottom, left:right]

        if cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0:
            continue

        resized_face = cv2.resize(cropped_face, (IMAGE_SIZE, IMAGE_SIZE))
        normalized_face = resized_face.astype('float32') / 255.0
        input_face = np.expand_dims(normalized_face, axis=0) # Thêm chiều batch

        predicted_landmarks = model.predict(input_face, verbose=0)[0]
        
        landmarks = predicted_landmarks.reshape((NUM_LANDMARKS, 2))
        landmarks = landmarks * np.array([right - left, bottom - top]) + np.array([left, top])
        landmarks = landmarks.astype(int)

        for (lx, ly) in landmarks:
            cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)

    cv2.imshow('Webcam Facial Landmark Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("[INFO] Cleaning up...")
cap.release()
cv2.destroyAllWindows()