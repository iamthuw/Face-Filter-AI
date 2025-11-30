"""
config.py

Tập hợp các cấu hình (constants) dùng chung cho project nhận diện khuôn mặt + landmark.

Mục đích:
- Tập trung mọi tham số để dễ sửa, tái sử dụng.
- Có chú thích rõ để dễ hiểu ý nghĩa từng tham số.

Ghi chú:
- Sửa RAW_DATA_DIR thành thư mục chứa ảnh và file .pts/.xml của bạn.
- Các hằng số INPUT_WIDTH/HEIGHT/NUM_LANDMARKS phải khớp với dữ liệu và mô hình.
"""

import os

# --- ĐƯỜNG DẪN DỮ LIỆU ---
# HÃY SỬA ĐƯỜNG DẪN NÀY ĐẾN THƯ MỤC CHỨA ẢNH VÀ FILE .PTS CỦA BẠN
# Ví dụ: "dataset/300W" hoặc "/data/face_landmarks/raw"
RAW_DATA_DIR = "../../data/ibug_300W_large_face_landmark_dataset" 

# Thư mục chứa dữ liệu đã tiền xử lý (xml, csv, tfrecords,...)
PREPROCESSED_DIR = "../../data/ibug_300W_large_face_landmark_dataset/preprocessed"
TRAIN_XML_PATH = os.path.join(PREPROCESSED_DIR, 'train.xml')
TEST_XML_PATH = os.path.join(PREPROCESSED_DIR, 'test.xml')

# --- THAM SỐ ẢNH ---
# Kích thước đầu vào cho mạng (chiều rộng x chiều cao).
# Lưu ý: nếu dùng mạng pretrained, đảm bảo input_shape phù hợp (channels=3 RGB).
INPUT_WIDTH = 128
INPUT_HEIGHT = 128

# Số giá trị landmark trả về: 68 điểm x,y => 136 giá trị (x1,y1,x2,y2,...)
NUM_LANDMARKS = 136  

# --- THAM SỐ HUẤN LUYỆN ---
# Kích thước batch cho training / evaluation
BATCH_SIZE = 32

# Số epoch mặc định
NUM_EPOCHS = 130

# Hệ số học mặc định (learning rate)
LEARNING_RATE = 0.001 

# Tỷ lệ tách test/validation nếu bạn chia dữ liệu nội bộ
TEST_SPLIT = 0.2

# --- LƯU TRỮ MÔ HÌNH ---
MODEL_DIR = "models"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "landmark_detector.h5")
# Tạo thư mục model nếu chưa tồn tại
os.makedirs(MODEL_DIR, exist_ok=True)

# --- WEBCAM ---
# ID camera mặc định khi test realtime (0 là webcam mặc định)
WEBCAM_ID = 0