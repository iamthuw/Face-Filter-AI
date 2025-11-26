# evaluate_model.py (Đã sửa lỗi để tải mô hình cũ mà không cần compile)
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import layers # Import layers để định nghĩa lớp tùy chỉnh

# Import các hằng số và hàm cần thiết
from preprocess import load_and_preprocess_data, NUM_LANDMARKS 

# --- Các hằng số và thiết lập ---
DATA_ROOT = 'ibug_300W_large_face_landmark_dataset'
TEST_XML_FILE = os.path.join(DATA_ROOT, 'labels_ibug_300W_test.xml') 
IMAGE_SIZE = 128
MODEL_LOAD_PATH = 'saved_model/facial_landmark_detector_vit.h5' 

# Hằng số ViT (Cần thiết để tái tạo các lớp tùy chỉnh)
PATCH_SIZE = 16 
PROJECTION_DIM = 64

# ====================================================================
# ĐỊNH NGHĨA LẠI CÁC LỚP CUSTOM LAYER CỦA VIT (Đã sửa tên tham số)
# ====================================================================
# Sửa lỗi tham số Patches/PatchEncoder để khớp với cấu hình đã lưu
# Chúng ta giả định mô hình được lưu với tên tham số đơn giản ('patch_size', 'num_patches')

class Patches(layers.Layer):
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
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
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
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config
        
# --- END CUSTOM LAYER DEFINITIONS ---

def evaluate_model():
    """
    Tải mô hình, đánh giá trên TOÀN BỘ tập kiểm tra TEST.xml và in ra các chỉ số hiệu suất.
    """
    print("--- 1. Loading Data for FINAL Evaluation ---")
    
    # ... (Phần tải dữ liệu)
    # Tải toàn bộ dữ liệu TEST (X_part1 = 80%, X_part2 = 20%)
    X_part1, X_part2, y_part1, y_part2 = load_and_preprocess_data(TEST_XML_FILE, DATA_ROOT, IMAGE_SIZE, test_split=0.2, augment=False)
    
    if len(X_part1) == 0 and len(X_part2) == 0:
        print("ERROR: No test samples loaded from TEST_XML_FILE. Cannot proceed.")
        return

    X_test_full = np.concatenate([X_part1, X_part2], axis=0)
    y_test_full = np.concatenate([y_part1, y_part2], axis=0)
    print(f"Loaded {len(X_test_full)} FINAL test samples (The entire TEST.xml data).")
    
    # --- 2. Loading Trained Model ---
    print("\n--- 2. Loading Trained Model ---")
    
    # Đăng ký các lớp đã được định nghĩa lại
    custom_objects = {
        'Patches': Patches,
        'PatchEncoder': PatchEncoder
    }
    
    try:
        # GIẢI PHÁP: Dùng compile=False để bỏ qua thông tin biên dịch lỗi
        model = tf.keras.models.load_model(MODEL_LOAD_PATH, custom_objects=custom_objects, compile=False)
        print(f"Model loaded successfully from {MODEL_LOAD_PATH}")
    except Exception as e:
        print(f"ERROR: Could not load model from {MODEL_LOAD_PATH}.")
        print(f"Chi tiết lỗi: {e}")
        print("HINT: Lỗi có thể do tên tham số trong lớp custom. Đã thử tải với compile=False.")
        return

    # --- 3. Dự đoán và Tính toán chỉ số ---
    print("\n--- 3. Making Predictions and Calculating Metrics ---")
    # Chúng ta vẫn có thể dùng mô hình để dự đoán ngay cả khi compile=False
    y_pred = model.predict(X_test_full, verbose=0) 

    # Làm phẳng mảng để tính toán chỉ số
    y_true_flat = y_test_full.flatten()
    y_pred_flat = y_pred.flatten()

    # Tính toán các chỉ số
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)

    # --- 4. In kết quả ---
    print("\n==============================================")
    print("           📊 FINAL EVALUATION RESULTS 📊")
    print(f"       (Tested on: {os.path.basename(TEST_XML_FILE)} - Full Set)")
    print("==============================================")
    print(f"Mean Squared Error (MSE):       {mse:.6f}")
    print(f"Mean Absolute Error (MAE):      {mae:.6f}")
    print(f"R-squared Score (R²):           {r2:.4f}")
    print("==============================================")


if __name__ == '__main__':
    evaluate_model()