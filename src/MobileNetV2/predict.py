import tensorflow as tf
import numpy as np
import cv2
import os
import sys
from config import *

# --- 1. Cấu hình Tham số và Đường dẫn ---
MODEL_PATH = 'models/landmark_detector_inference.h5'
TEST_IMAGE_PATH = 'test_image_2.jpg' 
TARGET_SIZE = (INPUT_WIDTH, INPUT_HEIGHT) # (128, 128)

# --- 2. Hàm Tiền xử lý Ảnh (Đã sửa lỗi 1 kênh -> 3 kênh) ---
def preprocess_image(image_path, target_size):
    """
    Tải ảnh, resize và chuẩn hóa, đảm bảo ảnh đầu ra là 3 kênh (RGB).
    
    Lỗi cũ: Hàm này chuyển ảnh sang Grayscale (1 kênh), gây lỗi.
    Sửa lỗi: Loại bỏ bước chuyển sang Grayscale.
    """
    # Đọc ảnh (Mặc định: BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh tại đường dẫn: {image_path}")
        
    # CHUYỂN BGR SANG RGB (theo chuẩn huấn luyện của MobileNet)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize về kích thước đầu vào của mô hình (128x128)
    img = cv2.resize(img, target_size)
    
    # Chuẩn hóa về phạm vi 0-1
    img = img.astype('float32') / 255.0
    
    # Thêm chiều batch size
    # Output shape: (1, 128, 128, 3) -> KHẮC PHỤC LỖI
    img = np.expand_dims(img, axis=0)  
    
    return img

# --- 3. Hàm Hậu xử lý và Hiển thị ---
def visualize_landmarks(image_path, landmarks_pred):
    """
    Hiển thị ảnh gốc và vẽ các landmarks dự đoán.
    Giả định landmarks đã được chuẩn hóa trong phạm vi [0, 1].
    """
    original_img = cv2.imread(image_path)
    if original_img is None: return
        
    H, W, _ = original_img.shape
    
    # Giả định landmarks_pred có shape (1, 136) -> 68 cặp (x, y)
    landmarks = landmarks_pred[0].reshape(-1, 2)
    
    # Tái chuẩn hóa từ [0, 1] về tọa độ pixel ảnh gốc
    for (x_norm, y_norm) in landmarks:
        x_pixel = int(x_norm * W)
        y_pixel = int(y_norm * H)
        
        # Vẽ chấm tròn xanh lá
        cv2.circle(original_img, (x_pixel, y_pixel), 2, (0, 255, 0), -1) 
    
    # Hiển thị ảnh
    cv2.imshow("Landmark Prediction", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- 4. Logic Chính ---
def main():
    print("Bắt đầu tải mô hình...")
    try:
        # Tải mô hình. Cần thêm custom_objects vì mô hình được compile với metrics riêng.
        custom_objects = {'mse': tf.keras.metrics.MeanSquaredError, 'mae': tf.keras.metrics.MeanAbsoluteError}
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
        print("Tải mô hình thành công.")
    except Exception as e:
        print(f"Lỗi tải mô hình: {e}")
        return

    print(f"Đang xử lý ảnh: {TEST_IMAGE_PATH}")
    try:
        # Tiền xử lý ảnh
        input_tensor = preprocess_image(TEST_IMAGE_PATH, TARGET_SIZE)
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"Lỗi tiền xử lý ảnh: {e}")
        return

    # Dự đoán
    print("Bắt đầu dự đoán...")
    # Thêm verbose=0 để tránh in log rác khi predict
    predictions = model.predict(input_tensor, verbose=0)
    
    # Hiển thị kết quả
    print(f"Dự đoán thành công. Output shape: {predictions.shape}")
    visualize_landmarks(TEST_IMAGE_PATH, predictions)

if __name__ == "__main__":
    # Đặt test_image.jpg vào thư mục gốc của dự án trước khi chạy
    main()