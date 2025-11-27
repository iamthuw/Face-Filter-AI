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
    # 1. Tạo lại XML
    print("--- STEP 1: Preparing Data ---")
    try:
        process_and_split_data(RAW_DATA_DIR, PREPROCESSED_DIR, INPUT_WIDTH, TEST_SPLIT)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Load dữ liệu
    print("--- STEP 2: Loading Data ---")
    X_train, y_train = load_and_preprocess_data(TRAIN_XML_PATH, RAW_DATA_DIR, INPUT_WIDTH)
    X_val, y_val = load_and_preprocess_data(TEST_XML_PATH, RAW_DATA_DIR, INPUT_WIDTH)
    print(f"Train set: {X_train.shape}, Val set: {X_val.shape}")

    # 3. Model
    model = create_mobilenet_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='mse', metrics=['mae'])

    # 4. Callbacks
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    # 5. Train (Gán kết quả vào biến history)
    print("--- STEP 3: Training ---")
    
    # --- KHỞI TẠO BIẾN HISTORY TẠI ĐÂY ---
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
    
    # Dự đoán
    y_pred = model.predict(X_val)
    
    # Tính toán chỉ số
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

    # Vẽ biểu đồ
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs_range = range(1, len(loss) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label='Training Loss (MSE)')
    plt.plot(epochs_range, val_loss, label='Validation Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, mae, label='Training MAE')
    plt.plot(epochs_range, val_mae, label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(plots_dir, "training_results.png")
    plt.savefig(save_path)
    print(f"✅ Đã lưu biểu đồ tại: {save_path}")
    
    print("--- DONE ---")

if __name__ == '__main__':
    train()