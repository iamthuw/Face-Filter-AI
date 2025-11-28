# train.py
"""
Module quản lý quy trình huấn luyện mô hình (Training Pipeline).

Mục đích:
    Script này điều phối toàn bộ quá trình từ chuẩn bị dữ liệu, xây dựng mô hình,
    huấn luyện (training), đến đánh giá và trực quan hóa kết quả.

Các bước chính:
    1. Xử lý dữ liệu thô thành XML chuẩn hóa.
    2. Tải dữ liệu ảnh và nhãn vào bộ nhớ (NumPy arrays).
    3. Khởi tạo và biên dịch mô hình MobileNetV2.
    4. Thiết lập các chiến lược callback (Lưu, Dừng sớm, Giảm Learning Rate).
    5. Thực hiện huấn luyện.
    6. Lưu mô hình rút gọn và vẽ biểu đồ đánh giá.
"""
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from config import *
from model_architecture import create_mobilenet_model
from preprocess import process_and_split_data, load_and_preprocess_data
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train():
    """
    Hàm chính thực thi quy trình huấn luyện mô hình nhận diện điểm mốc khuôn mặt.

    Quy trình chi tiết:
    1. Gọi `process_and_split_data` để đảm bảo dữ liệu XML được cập nhật mới nhất với các tham số cấu hình hiện tại.
    2. Tải dữ liệu huấn luyện và kiểm tra vào RAM thông qua `load_and_preprocess_data`.
    3. Khởi tạo kiến trúc MobileNetV2 tùy chỉnh và biên dịch với Adam optimizer.
    4. Định nghĩa các Callbacks:
        - ModelCheckpoint: Lưu lại trọng số tốt nhất.
        - EarlyStopping: Ngăn chặn overfitting.
        - ReduceLROnPlateau: Tối ưu hóa quá trình hội tụ.
    5. Chạy vòng lặp huấn luyện (model.fit).
    6. Lưu phiên bản mô hình nhẹ (Inference) để triển khai thực tế.
    7. Tính toán các chỉ số MSE, MAE, R2 trên tập Validation và vẽ biểu đồ theo dõi.

    Args:
        None (Sử dụng các biến toàn cục từ config.py).

    Returns:
        None
    """
    
    # 1. Tạo lại XML
    # Lý do: Đảm bảo file XML (chứa thông tin bounding box, padding) luôn đồng bộ
    # với các tham số INPUT_WIDTH, PADDING_RATIO mới nhất trong config/preprocess.
    print("--- STEP 1: Preparing Data ---")
    try:
        process_and_split_data(RAW_DATA_DIR, PREPROCESSED_DIR, INPUT_WIDTH, TEST_SPLIT)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Load dữ liệu
    # Tải ảnh và nhãn (landmark) vào bộ nhớ RAM dưới dạng mảng NumPy.
    # X: Ảnh đã chuẩn hóa [0, 1], y: Landmark đã chuẩn hóa [0, 1]
    print("--- STEP 2: Loading Data ---")
    X_train, y_train = load_and_preprocess_data(TRAIN_XML_PATH, RAW_DATA_DIR, INPUT_WIDTH)
    X_val, y_val = load_and_preprocess_data(TEST_XML_PATH, RAW_DATA_DIR, INPUT_WIDTH)
    print(f"Train set: {X_train.shape}, Val set: {X_val.shape}")

    # 3. Model
    # Khởi tạo mô hình với kiến trúc MobileNetV2 + Custom Regression Head
    model = create_mobilenet_model()
    
    # Sử dụng Mean Squared Error (MSE) làm hàm mất mát để tối ưu hóa khoảng cách tọa độ.
    # MAE được dùng làm metrics để dễ theo dõi sai số thực tế hơn.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='mse', metrics=['mae'])

    # 4. Callbacks
    # ModelCheckpoint: Chỉ lưu đè file model nếu val_loss giảm xuống thấp hơn kỷ lục cũ.
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    # EarlyStopping: Dừng train nếu sau 15 epoch mà model không tiến bộ, tránh lãng phí thời gian.
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    # ReduceLROnPlateau: Giảm Learning Rate đi một nửa (factor=0.5) nếu loss bị kẹt (không giảm) sau 5 epoch.
    # Giúp model thoát khỏi điểm cực tiểu cục bộ và tinh chỉnh kỹ hơn ở giai đoạn cuối.
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    # 5. Train (Gán kết quả vào biến history)
    print("--- STEP 3: Training ---")
    
    # --- KHỞI TẠO BIẾN HISTORY TẠI ĐÂY ---
    # Biến history lưu trữ giá trị loss và mae qua từng epoch để vẽ biểu đồ sau này.
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    # --- PHẦN ĐÁNH GIÁ & VẼ BIỂU ĐỒ ---
    if history is None:
        print("Lỗi: Quá trình huấn luyện không trả về lịch sử.")
        return

    print("\n--- STEP 4: Evaluation & Plotting ---")
    
    # Dự đoán trên tập Validation để tính toán các chỉ số độc lập
    y_pred = model.predict(X_val)
    
    # Tính toán chỉ số thống kê
    # R2 Score là chỉ số quan trọng xác định mức độ phù hợp của mô hình (càng gần 1 càng tốt)
    final_mse = mean_squared_error(y_val, y_pred)
    final_mae = mean_absolute_error(y_val, y_pred)
    final_r2 = r2_score(y_val, y_pred)
    
    print("\n" + "="*30)
    print("KẾT QUẢ ĐÁNH GIÁ (VALIDATION SET)")
    print("="*30)
    print(f"MSE: {final_mse:.6f}")
    print(f"MAE: {final_mae:.6f}")
    print(f"R2 Score: {final_r2:.4f}")
    print("="*30 + "\n")

    # Vẽ biểu đồ theo dõi quá trình huấn luyện
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Trích xuất dữ liệu từ history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs_range = range(1, len(loss) + 1)

    plt.figure(figsize=(14, 5))

    # Biểu đồ 1: Loss (MSE) - Dùng để xem model có hội tụ không
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label='Training Loss (MSE)')
    plt.plot(epochs_range, val_loss, label='Validation Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    # Biểu đồ 2: MAE - Dùng để xem sai số thực tế
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, mae, label='Training MAE')
    plt.plot(epochs_range, val_mae, label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    # Lưu biểu đồ ra file ảnh để báo cáo
    save_path = os.path.join(plots_dir, "training_results.png")
    plt.savefig(save_path)
    print(f"✅ Đã lưu biểu đồ tại: {save_path}")
    
    # Lưu bản mô hình nhẹ (Inference only)
    # include_optimizer=False: Bỏ qua trạng thái của Adam optimizer, giúp giảm dung lượng file (~1/3).
    # File này chỉ dùng để chạy (predict), không dùng để train tiếp.
    inference_model_path = os.path.join(MODEL_DIR, "landmark_detector_inference.h5")
    model.save(inference_model_path, include_optimizer=False)
    print(f"\n✅ Đã lưu bản mô hình nhẹ (Inference only) tại: {inference_model_path}")
    
    print("--- DONE ---")

if __name__ == '__main__':
    train()