# predict_webcam.py (Phiên bản Đã Sửa Lỗi Tương Thích Tên Tham Số)
import cv2
import dlib
import tensorflow as tf
import numpy as np
import os 
from tensorflow.keras import layers 

# --- Các hằng số và thiết lập ---
MODEL_PATH = 'saved_model/facial_landmark_detector_vit.h5' 
IMAGE_SIZE = 128 
NUM_LANDMARKS = 68 
PATCH_SIZE = 16 
PROJECTION_DIM = 64 

class Patches(layers.Layer):
    """
    Custom Layer dùng để chia ảnh thành các patch con không chồng lấp,
    phục vụ cho kiến trúc Vision Transformer.

    Thuộc về Stage 1: Patch Embedding.

    Args:
        patch_size (int): Kích thước mỗi patch (patch_size x patch_size).

    Input:
        Tensor shape (batch_size, height, width, channels)

    Output:
        Tensor shape (batch_size, num_patches, patch_dim)
    """
    # Sửa: Đổi 'patch_size_val' thành 'patch_size'
    def __init__(self, patch_size, **kwargs): 
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def get_config(self):
        config = super(Patches, self).get_config()
        # Sửa: Lưu trữ dưới tên 'patch_size'
        config.update({"patch_size": self.patch_size}) 
        return config

class PatchEncoder(layers.Layer):
    """
    Custom Layer mã hóa các patch bằng Dense projection + Position embedding.

    Thuộc Stage 2: Linear Embedding + Positional Encoding.

    Args:
        num_patches (int): Tổng số lượng patch sau khi chia ảnh.
        projection_dim (int): Số chiều embedding (transformer dimension).

    Input:
        Tensor shape (batch_size, num_patches, patch_dim)

    Output:
        Tensor shape (batch_size, num_patches, projection_dim)
    """
    def __init__(self, num_patches, projection_dim, **kwargs): 
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        encoded_patches = self.projection(patch)
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded_patches += self.position_embedding(positions)
        return encoded_patches
    
    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        # Sửa: Lưu trữ dưới tên tham số đơn giản
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config
        
# --- END CUSTOM LAYER DEFINITIONS ---

def predict_webcam():
    """
    Chạy webcam real-time để phát hiện khuôn mặt bằng dlib và dự đoán 68 điểm Landmark
    sử dụng mô hình Vision Transformer đã huấn luyện.

    Quy trình:
    1. Load mô hình ViT với custom layers (Patches, PatchEncoder).
    2. Sử dụng dlib để phát hiện bounding box khuôn mặt.
    3. Cắt vùng mặt, resize và chuẩn hóa như lúc train.
    4. Đưa vào mô hình để dự đoán landmarks (tọa độ chuẩn hóa).
    5. Chuyển tọa độ normalized back → tọa độ thật ở ảnh gốc.
    6. Vẽ các landmark lên khung hình webcam.

    Returns:
        None
    """


    print("[INFO] Loading facial landmark predictor (ViT)...")
    
    CUSTOM_OBJECTS = {
        'Patches': Patches,
        'PatchEncoder': PatchEncoder
    }

    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
        detector = dlib.get_frontal_face_detector()
        print("[INFO] Model and dlib detector loaded successfully.")
    except Exception as e:
        print(f"[FATAL ERROR] Could not load model or detector: {e}")
        return

    # ... (Phần còn lại của logic webcam)
    print("[INFO] Starting webcam stream...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    # --- Vòng lặp xử lý real-time (giữ nguyên logic) ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            
            padding = int(max(w, h) * 0.1) 
            top, left = max(0, y - padding), max(0, x - padding)
            bottom, right = min(frame.shape[0], y + h + padding), min(frame.shape[1], x + w + padding)
            
            cropped_face = frame[top:bottom, left:right]

            if cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0:
                continue
                
            resized_face = cv2.resize(cropped_face, (IMAGE_SIZE, IMAGE_SIZE))
            normalized_face = resized_face.astype('float32') / 255.0
            input_face = np.expand_dims(normalized_face, axis=0)

            predicted_landmarks = model.predict(input_face, verbose=0)[0]
            
            landmarks = predicted_landmarks.reshape((NUM_LANDMARKS, 2))
            landmarks[:, 0] = landmarks[:, 0] * (right - left)
            landmarks[:, 1] = landmarks[:, 1] * (bottom - top)
            
            landmarks[:, 0] = landmarks[:, 0] + left
            landmarks[:, 1] = landmarks[:, 1] + top
            landmarks = landmarks.astype(int)

            for (lx, ly) in landmarks:
                cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)

        cv2.imshow('Webcam ViT Facial Landmark Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[INFO] Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    predict_webcam()