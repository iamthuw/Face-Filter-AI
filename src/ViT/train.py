"""
train.py (Script Huấn luyện Mô hình ViT)
Script huấn luyện mô hình Vision Transformer (ViT) cho bài toán Landmark Detection.

Chức năng chính:
----------------
1. Tải và tiền xử lý dữ liệu landmark (ảnh & tọa độ) từ file XML.
2. Xây dựng mô hình ViT bằng hàm build_landmark_model().
3. Biên dịch mô hình với optimizer, loss function phù hợp (Adam, MSE).
4. Huấn luyện mô hình với EarlyStopping và ModelCheckpoint để tránh overfitting.
5. Lưu mô hình tốt nhất và trực quan hóa lịch sử huấn luyện (loss, val_loss).

Input:
------
- labels_ibug_300W_train.xml : File XML chứa đường dẫn ảnh và tọa độ landmark.
- Thư mục chứa dataset ảnh gốc.

Output:
-------
- File mô hình tốt nhất (.h5) lưu trong MODEL_SAVE_PATH.
- Biểu đồ training/validation loss .
"""

import os
import tensorflow as tf
from preprocess import load_and_preprocess_data, NUM_LANDMARKS
from model import build_landmark_model # Import hàm xây dựng mô hình ViT
from utils import plot_training_history # Import hàm vẽ đồ thị lịch sử

# --- Cấu hình GPU (nếu có) ---
# Tùy chỉnh để tránh lỗi GPU VRAM
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Giới hạn bộ nhớ GPU (ví dụ: 5GB)
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
        
        # Hoặc cho phép tăng trưởng bộ nhớ động (phổ biến hơn)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e) # Lỗi khởi tạo thiết bị ảo

# --- Các thiết lập chính cho quá trình huấn luyện ---
DATA_ROOT = 'ibug_300W_large_face_landmark_dataset' # Thư mục gốc chứa dữ liệu 300W
TRAIN_XML_FILE = os.path.join(DATA_ROOT, 'labels_ibug_300W_train.xml') # File XML dữ liệu huấn luyện
IMAGE_SIZE = 128 # Kích thước ảnh đầu vào cho mô hình (ví dụ: 128x128)
EPOCHS = 100 # Số lượng epoch huấn luyện (sử dụng EarlyStopping để dừng sớm)
BATCH_SIZE = 32 # Kích thước batch
MODEL_SAVE_PATH = 'saved_model/facial_landmark_detector_vit.h5' # Đường dẫn lưu mô hình tốt nhất

# --- Đảm bảo thư mục lưu mô hình tồn tại ---
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

def train_model():
    # 1. Tải và xử lý dữ liệu
    print("\n--- 1. Loading and Preprocessing Data ---")
    # load_and_preprocess_data sẽ chia 80% từ TRAIN_XML_FILE cho huấn luyện
    # và 20% cho validation (kiểm tra trong quá trình huấn luyện)
    X_train, X_val, y_train, y_val = load_and_preprocess_data(
        xml_file_path=TRAIN_XML_FILE, 
        data_root=DATA_ROOT, 
        image_size=IMAGE_SIZE,
        test_split=0.2, # Chia 20% cho validation
        augment=True # Kích hoạt tăng cường dữ liệu
    )
    print(f"Data loaded. Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    if len(X_train) == 0 or len(X_val) == 0:
        print("ERROR: Not enough data for training or validation. Check your XML file and image paths.")
        return

    # 2. Xây dựng mô hình Vision Transformer
    print("\n--- 2. Building Vision Transformer Model ---")
    model = build_landmark_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_landmarks=NUM_LANDMARKS)
    model.summary()

    # 3. Biên dịch mô hình
    print("\n--- 3. Compiling Model ---")
    # Sử dụng Adam optimizer và Mean Squared Error (MSE) làm hàm mất mát
    model.compile(optimizer='adam', loss='mse')

    # 4. Huấn luyện mô hình
    print("\n--- 4. Starting Training ---")
    # Các Callback:
    # - EarlyStopping: Dừng huấn luyện nếu validation loss không cải thiện trong 10 epoch liên tiếp.
    #                  restore_best_weights=True sẽ khôi phục trọng số tốt nhất.
    # - ModelCheckpoint: Lưu mô hình tốt nhất (dựa trên 'val_loss') vào đường dẫn đã chỉ định.
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'), # Tăng patience
            tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', verbose=1)
        ]
    )
    print("--- Training Finished ---")

    # 5. Lưu mô hình tốt nhất (ModelCheckpoint đã làm điều này, chỉ in thông báo)
    print(f"Best model saved to {MODEL_SAVE_PATH}")

    # 6. Trực quan hóa lịch sử huấn luyện
    print("\n--- 6. Plotting Training History ---")
    plot_training_history(history)
    print("Training history plot displayed.")

if __name__ == '__main__':
    train_model()