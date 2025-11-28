# evaluate.py
"""
Module đánh giá hiệu suất mô hình (Model Evaluation).

Mục đích:
    Script này được sử dụng để kiểm thử độc lập mô hình đã huấn luyện trên tập dữ liệu kiểm tra (Test Set).
    Nó tải mô hình, thực hiện dự đoán và tính toán các chỉ số định lượng quan trọng để đánh giá độ chính xác.

Các chỉ số đánh giá:
    - MSE (Mean Squared Error): Sai số bình phương trung bình.
    - MAE (Mean Absolute Error): Sai số tuyệt đối trung bình.
    - R2 Score: Hệ số xác định (đo mức độ phù hợp của mô hình).

Yêu cầu:
    - File mô hình (.h5) phải tồn tại tại đường dẫn MODEL_SAVE_PATH.
    - File dữ liệu test (.xml) phải tồn tại tại TEST_XML_PATH.
"""
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import *
from preprocess import load_and_preprocess_data

def evaluate():
    """
    Hàm chính để thực hiện quy trình đánh giá.

    Quy trình thực hiện:
    1. Kiểm tra sự tồn tại của file mô hình.
    2. Tải và tiền xử lý dữ liệu kiểm tra (Test Set) từ file XML.
    3. Tải mô hình Keras đã lưu (xử lý các custom metrics).
    4. Thực hiện dự đoán trên toàn bộ tập test.
    5. Tính toán và in ra các chỉ số đánh giá (MSE, MAE, R2).

    Args:
        Không có tham số đầu vào trực tiếp. Các cấu hình đường dẫn được lấy từ file config.py.

    Returns:
        None: Kết quả được in trực tiếp ra màn hình console.
    """
    # 1. Kiểm tra file mô hình
    # Nếu chưa train xong hoặc file bị xóa, dừng chương trình để tránh lỗi crash
    if not os.path.exists(MODEL_SAVE_PATH): return
    
    # 2. Tải dữ liệu Test
    # Gọi hàm từ module preprocess để đọc XML, cắt ảnh, resize và chuẩn hóa
    # X_test: Mảng các ảnh đầu vào (N, 128, 128, 3)
    # y_test: Mảng các nhãn thực tế (N, 136)
    X_test, y_test = load_and_preprocess_data(TEST_XML_PATH, RAW_DATA_DIR, INPUT_WIDTH)
    
    # 3. Tải mô hình
    # Cần định nghĩa custom_objects vì model được compile với các metrics của tf.keras
    # Nếu không khai báo, Keras sẽ không hiểu 'mse' và 'mae' là hàm gì khi load
    custom_objects = {'mse': tf.keras.metrics.MeanSquaredError, 'mae': tf.keras.metrics.MeanAbsoluteError}
    # compile=False: Chỉ load trọng số và kiến trúc để dự đoán, không cần load optimizer (Adam)
    # Giúp load nhanh hơn và tránh lỗi phiên bản optimizer
    model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects=custom_objects, compile=False)
    
    # 4. Dự đoán (Inference)   
    y_pred = model.predict(X_test)
    
    # 5. Tính toán Metric
    # Sử dụng thư viện sklearn để tính toán các chỉ số thống kê
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MSE: {mse:.5f}")
    print(f"MAE: {mae:.5f}")
    print(f"R2 Score: {r2:.4f}")

if __name__ == '__main__':
    evaluate()