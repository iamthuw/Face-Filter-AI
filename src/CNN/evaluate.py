# evaluate.py
import os
import numpy as np
import time
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess_data

# --- Cấu hình Đường dẫn và Hằng số ---

DATA_ROOT = "ibug_300W_large_face_landmark_dataset"
# File XML chứa landmarks cho tập kiểm tra
TEST_XML_FILE = os.path.join(DATA_ROOT, "labels_ibug_300W_test.xml")
# Đường dẫn tới file mô hình đã huấn luyện tốt nhất
MODEL_PATH = "src/CNN/saved_model/facial_landmark_detector.h5"
# Kích thước ảnh đầu vào đã được chuẩn hóa khớp với kích thước huấn luyện
IMAGE_SIZE = 128 

def evaluate_on_test():
    """
    Tải mô hình đã lưu, tải tập dữ liệu kiểm tra (test set), thực hiện dự đoán
    và tính toán các chỉ số đánh giá quan trọng (MSE, MAE, R2 Score)
    cho bài toán hồi quy dự đoán landmarks.
    """

    # 1. Tải mô hình
    print(f"[INFO] Đang load model từ: {MODEL_PATH}...")
    try:
        # load_model: Tải kiến trúc và trọng số, compile=False vì không cần huấn luyện, chỉ cần dự đoán.
        model = load_model(MODEL_PATH, compile=False)
        print("[INFO] Load model thành công!")
    except Exception as e:
        print(f"[ERROR] Không tìm thấy model hoặc lỗi load: {e}. Vui lòng kiểm tra lại đường dẫn.")
        return

    # 2. Tải và xử lý dữ liệu kiểm tra
    print(f"[INFO] Đang load dữ liệu test từ: {TEST_XML_FILE}...")
    
    # 📝 load_and_preprocess_data: Tải ảnh (X) và tọa độ (y). Chỉ cần tập test, bỏ qua các giá trị trả về khác.
    X_test, _, y_test, _ = load_and_preprocess_data(TEST_XML_FILE, DATA_ROOT, IMAGE_SIZE,0)

    print(f"[INFO] Số lượng mẫu test: {len(X_test)}")

    # 3. Dự đoán (Predict)
    print("[INFO] Đang thực hiện dự đoán...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    prediction_time = end_time - start_time

    # 4. Tính toán các chỉ số
    # 📝 MSE (Mean Squared Error): Sai số bình phương trung bình. Ưu tiên phạt nặng sai số lớn.
    mse = mean_squared_error(y_test, y_pred)
    # 📝 MAE (Mean Absolute Error): Sai lệch tuyệt đối trung bình. Dễ diễn giải, ít nhạy cảm với outliers hơn MSE.
    mae = mean_absolute_error(y_test, y_pred)
    # 📝 R2 Score (Coefficient of Determination): Đo lường mức độ phù hợp của mô hình (tỷ lệ phương sai được giải thích). 1.0 là hoàn hảo.
    r2 = r2_score(y_test, y_pred)

    # 5. In kết quả
    print("\n" + "="*30)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST")
    print("="*30)
    print(f"MSE (Sai số bình phương trung bình): {mse:.4f}")
    print(f"MAE (Sai lệch trung bình - Pixels): {mae:.4f}")
    print(f"R2 Score (Độ chính xác mô hình):     {r2:.4f} ({r2*100:.2f}%)")
    print(f"Total prediction time:           {prediction_time:.4f} seconds")
    print("="*30)

    # Đánh giá sơ bộ bằng lời
    if r2 > 0.9:
        print("=> Mô hình hoạt động RẤT TỐT. Độ chính xác cao.")
    elif r2 > 0.7:
        print("=> Mô hình hoạt động KHÁ. Cần theo dõi để cải thiện thêm.")
    else:
        print("=> Mô hình cần cải thiện thêm. Có thể thử kiến trúc hoặc dữ liệu khác.")

if __name__ == "__main__":
    evaluate_on_test()
