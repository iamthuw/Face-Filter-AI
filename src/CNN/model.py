# model.py
from tensorflow.keras import layers, models

def build_landmark_model(input_shape, num_landmarks):
    """
    Xây dựng kiến trúc mô hình Mạng Nơ-ron Tích chập (CNN) để thực hiện bài toán Hồi quy (Regression) tọa độ landmark khuôn mặt.

    Args:
        input_shape (tuple): Kích thước ảnh đầu vào (ví dụ: (128, 128, 3)).
        num_landmarks (int): Số lượng điểm mốc (landmarks) cần dự đoán (ví dụ: 68).

    Returns:
        tf.keras.Model: Mô hình CNN đã được xây dựng.
    """
    
    # 📝 Sử dụng models.Sequential vì các layer được xếp chồng lên nhau theo thứ tự tuần tự.
    model = models.Sequential([
        # --- Khối Tích chập 1 ---
        # Conv2D: 32 bộ lọc (filters) 3x3. input_shape chỉ cần thiết ở layer đầu tiên.
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        # MaxPooling2D: Giảm kích thước ảnh 2x2, giúp giảm số lượng tham số và tính toán, tăng khả năng khái quát.
        layers.MaxPooling2D((2, 2)),

        # --- Khối Tích chập 2 ---
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # --- Khối Tích chập 3 ---
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # --- Khối Tích chập 4 (Đặc trưng) ---
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        #  Flatten: Logic phức tạp: Chuyển đầu ra 3D (height, width, channels) của khối Conv thành 1D (vector)
        # để có thể đưa vào các layer Dense (fully connected layer).
        layers.Flatten(),
        
        # --- Khối Fully Connected ---
        layers.Dense(512, activation='relu'),
        # Dropout: Kỹ thuật điều chuẩn (regularization) bằng cách ngẫu nhiên loại bỏ 30% nơ-ron
        # trong quá trình huấn luyện để ngăn chặn overfitting.
        layers.Dropout(0.3),
        
        # Output Layer (Hồi quy):
        # Dense: Số lượng nơ-ron đầu ra bằng số landmark nhân 2 (vì mỗi landmark có tọa độ x và y).
        # Không dùng hàm kích hoạt (activation function) vì đây là bài toán hồi quy (dự đoán giá trị liên tục).
        layers.Dense(num_landmarks * 2) 
    ])
    return model