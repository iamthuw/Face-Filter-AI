# predict.py
import cv2
import numpy as np
import dlib
import tensorflow as tf
import argparse

# --- Các hằng số ---
MODEL_PATH = 'saved_model/facial_landmark_detector.h5'
IMAGE_SIZE = 128
NUM_LANDMARKS = 68

def predict_on_image(image_path):
    """
    Dự đoán và vẽ điểm mốc trên một ảnh.
    """
    # 1. Tải mô hình và bộ phát hiện khuôn mặt
    # Dòng mới
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    detector = dlib.get_frontal_face_detector()

    # 2. Đọc và xử lý ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
        
    display_image = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. Phát hiện khuôn mặt
    faces = detector(image_rgb, 1)
    if len(faces) == 0:
        print("No face detected.")
        return

    # 4. Với mỗi khuôn mặt, dự đoán điểm mốc
    for face in faces:
        left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
        width, height = right - left, bottom - top

        cropped_face = image_rgb[top:bottom, left:right]
        resized_face = cv2.resize(cropped_face, (IMAGE_SIZE, IMAGE_SIZE))
        
        input_image = np.expand_dims(resized_face.astype(np.float32) / 255.0, axis=0)
        
        # Dự đoán
        predicted_landmarks = model.predict(input_image)[0]
        
        # Chuyển đổi tọa độ về ảnh gốc
        landmarks = predicted_landmarks.reshape(NUM_LANDMARKS, 2)
        final_landmarks_x = (landmarks[:, 0] * width) + left
        final_landmarks_y = (landmarks[:, 1] * height) + top

        # Vẽ lên ảnh
        cv2.rectangle(display_image, (left, top), (right, bottom), (0, 255, 0), 2)
        for x, y in zip(final_landmarks_x, final_landmarks_y):
            cv2.circle(display_image, (int(x), int(y)), 2, (0, 0, 255), -1)

    # 5. Hiển thị kết quả
    cv2.imshow("Prediction", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict facial landmarks on an image.')
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    args = parser.parse_args()
    
    predict_on_image(args.image_path)