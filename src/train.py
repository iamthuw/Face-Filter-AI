# train.py
import os
import tensorflow as tf
from preprocess import load_and_preprocess_data, NUM_LANDMARKS
from model import build_landmark_model
from utils import plot_training_history

# --- Các thiết lập chính ---
DATA_ROOT = 'ibug_300W_large_face_landmark_dataset'
TRAIN_XML_FILE = os.path.join(DATA_ROOT, 'labels_ibug_300W_train.xml')
IMAGE_SIZE = 128
EPOCHS = 50
BATCH_SIZE = 32
MODEL_SAVE_PATH = 'saved_model/facial_landmark_detector.keras'

# 1. Tải và xử lý dữ liệu
# Dòng mới
X_train, X_val, y_train, y_val = load_and_preprocess_data(TRAIN_XML_FILE, DATA_ROOT, IMAGE_SIZE)
print(f"Data loaded. Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

# 2. Xây dựng mô hình
model = build_landmark_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_landmarks=NUM_LANDMARKS)
model.summary()

# 3. Biên dịch mô hình
model.compile(optimizer='adam', loss='mse')

# 4. Huấn luyện mô hình
print("\n--- Starting Training ---")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
    ]
)
print("--- Training Finished ---")

# 5. Lưu mô hình (ModelCheckpoint đã làm việc này)
print(f"Best model saved to {MODEL_SAVE_PATH}")

# 6. Đánh giá và trực quan hóa
plot_training_history(history)