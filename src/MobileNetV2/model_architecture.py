# model_architecture.py
"""
Module định nghĩa kiến trúc mạng nơ-ron cho bài toán nhận diện điểm mốc khuôn mặt.

Module này sử dụng MobileNetV2 làm mạng xương sống (backbone) và thêm vào một
mạng đầu ra tùy chỉnh (custom head) để thực hiện hồi quy tọa độ landmark.
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten # <-- Thêm Flatten
from tensorflow.keras.models import Model
from config import INPUT_HEIGHT, INPUT_WIDTH, NUM_LANDMARKS
import os

# Đường dẫn đến file trọng số MobileNetV2 đã tải sẵn (để tránh lỗi tải online)
MOBILENET_WEIGHTS_PATH = os.path.join("models", "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")

def create_mobilenet_model(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), num_landmarks=NUM_LANDMARKS):
    """
    Khởi tạo và cấu hình mô hình MobileNetV2 cho bài toán hồi quy điểm mốc khuôn mặt.

    Hàm này thực hiện các bước sau:
    1. Tải kiến trúc MobileNetV2 (không có lớp phân loại top).
    2. Tải trọng số tiền huấn luyện từ ImageNet (nếu có file cục bộ).
    3. Áp dụng chiến lược Fine-tuning: Đóng băng các lớp đầu, mở khóa các lớp cuối.
    4. Thêm phần đầu (Head) tùy chỉnh với lớp Flatten để bảo toàn thông tin không gian.

    Args:
        input_shape (tuple): Kích thước ảnh đầu vào (chiều cao, chiều rộng, số kênh màu).
                             Mặc định lấy từ config.
        num_landmarks (int): Tổng số giá trị đầu ra cần dự đoán.
                             Với 68 điểm landmark (x, y), giá trị này là 136.

    Returns:
        tf.keras.Model: Đối tượng mô hình Keras đã được xây dựng (chưa biên dịch).
    """
    # --- 1. KHỞI TẠO BACKBONE ---
    # Sử dụng MobileNetV2 làm bộ trích xuất đặc trưng cơ sở.
    # include_top=False: Loại bỏ lớp phân loại gốc (1000 lớp ImageNet).
    # weights=None: Không tải trọng số tự động từ mạng (để tránh lỗi SSL/kết nối).
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False, 
        weights=None
    )
    
    # --- 2. TẢI TRỌNG SỐ (TRANSFER LEARNING) ---
    # Kiểm tra và tải trọng số từ file cục bộ nếu có.
    if os.path.exists(MOBILENET_WEIGHTS_PATH):
        print("Loading local weights...")
        # skip_mismatch=True: Bỏ qua lỗi nếu kích thước lớp không khớp (an toàn).
        base_model.load_weights(MOBILENET_WEIGHTS_PATH, by_name=True, skip_mismatch=True)
    
    # --- 3. CHIẾN LƯỢC FINE-TUNING ---
    # Mặc định cho phép toàn bộ mô hình được huấn luyện
    base_model.trainable = True
    # Đóng băng (Freeze) 100 lớp đầu tiên:
    # - Các lớp này học các đặc trưng cơ bản (cạnh, góc, màu) từ ImageNet.
    # - Việc đóng băng giúp giữ lại kiến thức này và giảm khối lượng tính toán.
    # - Chỉ các lớp từ 100 trở đi (đặc trưng cao cấp) mới được huấn luyện lại.
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # --- 4. XÂY DỰNG PHẦN ĐẦU (REGRESSION HEAD) ---
    x = base_model.output
    
    # Sử dụng Flatten thay vì GlobalAveragePooling2D:
    # - Flatten: Trải phẳng toàn bộ bản đồ đặc trưng (4x4x1280 -> 20480).
    #   Giữ nguyên thông tin vị trí không gian (spatial info), quan trọng cho định vị.
    # - GAP: Lấy trung bình, làm mất thông tin vị trí (chỉ tốt cho phân loại).    
    x = Flatten()(x) 
    # Lớp ẩn (Hidden Layer) với 512 nơ-ron để học các mối quan hệ phi tuyến tính
    x = Dense(512, activation='relu')(x)
    # Dropout 0.3: Tắt ngẫu nhiên 30% nơ-ron trong lúc train để chống Overfitting
    x = Dropout(0.3)(x) 
    # Lớp đầu ra (Output Layer):
    # - activation='linear': Trả về giá trị thực liên tục (tọa độ).
    # - Số lượng nơ-ron = num_landmarks (136).
    predictions = Dense(num_landmarks, activation='linear', name='landmarks')(x)
    # Tạo mô hình hoàn chỉnh
    return Model(inputs=base_model.input, outputs=predictions)