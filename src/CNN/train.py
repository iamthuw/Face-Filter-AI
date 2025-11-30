# train.py
import os
import tensorflow as tf
from preprocess import load_and_preprocess_data, NUM_LANDMARKS
from model import build_landmark_model
from utils import plot_training_history

# --- Các thiết lập chính ---
#Thư mục gốc chứa toàn bộ tập dữ liệu (ảnh + file XML)
DATA_ROOT = 'data/ibug_300W_large_face_landmark_dataset' 
# Đường dẫn tới file XML chứa tọa độ landmark và đường dẫn ảnh cho tập huấn luyện
TRAIN_XML_FILE = os.path.join(DATA_ROOT, 'labels_ibug_300W_train.xml')
# Kích thước ảnh đầu vào (ảnh sẽ được resize về kích thước vuông này)
IMAGE_SIZE = 128
# Số lượng chu kỳ huấn luyện đầy đủ
EPOCHS = 50
# Kích thước batch (số lượng mẫu được xử lý trước mỗi lần cập nhật trọng số)
#tập train đc chia ra làm n batch mỗi batch có 32 ảnh
BATCH_SIZE = 32
# Đường dẫn để lưu trữ mô hình tốt nhất
MODEL_SAVE_PATH = 'src/CNN/saved_model/facial_landmark_detector.h5'


# 1. Tải và xử lý dữ liệu
# Hàm này tải dữ liệu, phân tách thành tập huấn luyện/kiểm tra, và chuẩn hóa (scaling/resizing).
X_train, X_val, y_train, y_val = load_and_preprocess_data(TRAIN_XML_FILE, DATA_ROOT, IMAGE_SIZE)
print(f"Data loaded. Train samples: {len(X_train)}, Validation samples: {len(X_val)}")


# 2. Xây dựng mô hình
# Hàm build_landmark_model xây dựng kiến trúc CNN (Convolutional Neural Network)
model = build_landmark_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_landmarks=NUM_LANDMARKS)
model.summary()


# 3. Biên dịch mô hình
# Optimizer: 'adam' là một lựa chọn phổ biến cho tốc độ và hiệu quả.Optimizer = thuật toán cập nhật trọng số.Giống như người học: thấy mình sai → điều chỉnh → học tốt hơn.
# Loss: 'mse' (Mean Squared Error) được dùng vì đây là bài toán hồi quy (regressing) các tọa độ (x, y) liên tục.
# Hàm mất mát MSE đo lường độ sai lệch bình phương trung bình giữa tọa độ dự đoán và tọa độ thực tế.
model.compile(optimizer='adam', loss='mse')

# 4. Huấn luyện mô hình
print("\n--- Starting Training ---")
'''
Bước 1: Chia dữ liệu thành batch, mỗi batch gồm BATCH_SIZE mẫu.
Bước 2: Forward pass
Mỗi batch đi qua mô hình:
    Mô hình dự đoán đầu ra y_pred.
    Tính loss so với giá trị thực y_true.
Bước 3: Backward pass
    Keras tính gradient của loss theo các trọng số (dùng backpropagation).
    Optimizer (adam) cập nhật trọng số dựa trên gradient.
Bước 4: Kết thúc một epoch, toàn bộ dữ liệu đã được sử dụng để cập nhật trọng số nhiều lần (theo số batch).
Bước 5: Sau khi xong epoch, mô hình đánh giá trên X_val và không cập nhật trọng số. Tính val_loss để theo dõi hiệu suất trên dữ liệu chưa thấy.

'''
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        # 🧪 EarlyStopping: Dừng huấn luyện sớm nếu val_loss không cải thiện sau 10 epochs (patience=10),
        # nhằm ngăn chặn overfitting. restore_best_weights=True sẽ tải lại trọng số tốt nhất.
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        # 📝 ModelCheckpoint: Tự động lưu mô hình (trọng số và kiến trúc) chỉ khi val_loss đạt mức tốt nhất.
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
    ]
)
print("--- Training Finished ---")

# 5. Lưu mô hình (ModelCheckpoint đã làm việc này)
print(f"Best model saved to {MODEL_SAVE_PATH}")

# 6. Đánh giá và trực quan hóa
#vẽ biểu đồ loss trên tập huấn luyện và tập kiểm tra qua các epochs.
plot_training_history(history) 
