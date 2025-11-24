import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess_data

DATA_ROOT = "ibug_300W_large_face_landmark_dataset"
TEST_XML_FILE = os.path.join(DATA_ROOT, "labels_ibug_300W_test.xml")
MODEL_PATH = "saved_model/facial_landmark_detector.h5"
IMAGE_SIZE = 128  

def evaluate_on_test():

    print(f"[INFO] Đang load model từ: {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("[INFO] Load model thành công!")
    except Exception as e:
        print(f"[ERROR] Không tìm thấy model hoặc lỗi load: {e}")
        return

    print(f"[INFO] Đang load dữ liệu test từ: {TEST_XML_FILE}...")
    
    
    X_test, _, y_test, _ = load_and_preprocess_data(TEST_XML_FILE, DATA_ROOT, IMAGE_SIZE)

    print(f"[INFO] Số lượng mẫu test: {len(X_test)}")

    # 3. Dự đoán (Predict)
    print("[INFO] Đang thực hiện dự đoán...")
    y_pred = model.predict(X_test)

    # 4. Xử lý dữ liệu trước khi tính toán (Flatten)
    # Sklearn cần dữ liệu dạng 2D (samples, features). 
    # Nếu y của bạn đang là dạng (Batch, 68, 2) thì cần duỗi ra thành (Batch, 136)
    if y_test.ndim > 2:
        y_test = y_test.reshape(y_test.shape[0], -1)
    if y_pred.ndim > 2:
        y_pred = y_pred.reshape(y_pred.shape[0], -1)

    # 5. Tính toán các chỉ số
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 6. In kết quả
    print("\n" + "="*30)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST")
    print("="*30)
    print(f"MSE (Sai số bình phương trung bình): {mse:.4f}")
    print(f"MAE (Sai lệch trung bình - Pixels): {mae:.4f}")
    print(f"R2 Score (Độ chính xác mô hình):     {r2:.4f} ({r2*100:.2f}%)")
    print("="*30)

    # Đánh giá sơ bộ bằng lời
    if r2 > 0.9:
        print("=> Mô hình hoạt động RẤT TỐT.")
    elif r2 > 0.7:
        print("=> Mô hình hoạt động KHÁ.")
    else:
        print("=> Mô hình cần cải thiện thêm.")

if __name__ == "__main__":
    evaluate_on_test()