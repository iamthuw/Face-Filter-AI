# webcam_predict.py
"""
Module dự đoán và hiển thị điểm mốc khuôn mặt trên Webcam theo thời gian thực.

Mục đích:
    Sử dụng mô hình đã huấn luyện để nhận diện và vẽ 68 điểm landmark lên khuôn mặt
    được phát hiện từ webcam.

Các tính năng chính:
    1. Phát hiện khuôn mặt bằng Haar Cascade.
    2. Cắt và xử lý vùng khuôn mặt (ROI) theo chuẩn đầu vào của mô hình.
    3. Áp dụng bộ lọc làm mượt (Smoothing) để giảm rung lắc landmark.
    4. Giải chuẩn hóa tọa độ và hiển thị trực quan lên màn hình.
"""
import cv2
import numpy as np
import tensorflow as tf
from config import *

# ==========================================
# KHU VỰC TINH CHỈNH (TUNING PARAMETERS)
# ==========================================
# Các tham số này giúp điều chỉnh khung nhìn (Bounding Box) để khớp với
# cách mô hình đã được huấn luyện, đảm bảo landmark không bị lệch.

# 1. ĐỘ RỘNG/HẸP (SCALE)
# Tỷ lệ mở rộng khung hình (Padding).
# - Giá trị 0.18 tương đương mở rộng thêm 18% mỗi bên so với hộp Haar Cascade gốc.
# - Mục đích: Lấy thêm vùng trán, cằm và tai để mô hình có cái nhìn toàn cảnh.
PADDING_RATIO = 0.18

# 2. ĐỘ CAO/THẤP (SHIFT Y)
# Tỷ lệ dịch chuyển tâm hộp theo trục dọc.
# - Giá trị âm (-0.02) nghĩa là dịch tâm lên trên 2%.
# - Mục đích: Haar Cascade thường bắt khuôn mặt hơi thấp, việc dịch lên giúp lấy trọn vẹn cằm.
Y_OFFSET_RATIO = 0.05 # Dịch khung nhìn xuống 5% chiều cao để lấy thêm cằm

# ==========================================

class LandmarkSmoother:
    """
    Lớp xử lý làm mượt chuyển động của các điểm landmark.
    Sử dụng thuật toán Exponential Moving Average (EMA).
    """
    def __init__(self, alpha=0.6):
        """
        Khởi tạo bộ làm mượt.

        Args:
            alpha (float): Hệ số làm mượt (0 < alpha <= 1).
                           - alpha lớn (gần 1): Nhạy với chuyển động, ít trễ nhưng dễ rung.
                           - alpha nhỏ (gần 0): Rất mượt, ít rung nhưng có độ trễ (lag).
                           - 0.6 là mức cân bằng tốt cho webcam.
        """
        self.alpha = alpha
        self.prev = None
        
    
    def update(self, current):
        """
        Cập nhật vị trí mới dựa trên vị trí cũ và hệ số alpha.

        Args:
            current (np.array): Tọa độ landmark dự đoán từ mô hình hiện tại.

        Returns:
            np.array: Tọa độ landmark đã được làm mượt.
        """
        if self.prev is None:
            self.prev = current
        else:
            # Công thức EMA: New = alpha * Current + (1 - alpha) * Previous
            self.prev = self.alpha * current + (1 - self.alpha) * self.prev
        return self.prev

def main():
    """
    Hàm chính thực thi vòng lặp xử lý video từ webcam.
    """
    # 1. Tải mô hình
    # Ưu tiên tải file mô hình nhẹ (Inference only) để chạy nhanh hơn
    inference_model_path = os.path.join("models", "landmark_detector_inference.h5")
    
    # Kiểm tra file tồn tại
    if not os.path.exists(inference_model_path): 
        print(f"Chưa có file {inference_model_path}! Hãy chạy train.py trước.")
        # Fallback về file gốc nếu chưa có file nhẹ
        if os.path.exists(MODEL_SAVE_PATH):
            inference_model_path = MODEL_SAVE_PATH
            print("Đang dùng file gốc (nặng)...")
        else:
            return
    
    # Định nghĩa custom_objects để Keras hiểu các metrics khi load
    custom_objects = {'mse': tf.keras.metrics.MeanSquaredError, 'mae': tf.keras.metrics.MeanAbsoluteError}
    
    # compile=False: Không cần load optimizer vì ta chỉ dự đoán, không huấn luyện tiếp
    model = tf.keras.models.load_model(inference_model_path, custom_objects=custom_objects, compile=False)
    
    # 2. Khởi tạo Camera & Detector
    cap = cv2.VideoCapture(WEBCAM_ID)
    
    # Sử dụng mô hình Haar Cascade có sẵn của OpenCV để phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Khởi tạo bộ làm mượt
    smoother = LandmarkSmoother(alpha=0.5)

    print("--- Bấm 'q' để thoát ---")
    while True:
        # Đọc khung hình từ webcam
        ret, frame = cap.read()
        if not ret: break
        
        # Lật ảnh (Mirror) để thao tác tự nhiên hơn
        frame = cv2.flip(frame, 1)
        h_frame, w_frame = frame.shape[:2]
        
        # Chuyển sang ảnh xám để tăng tốc độ phát hiện khuôn mặt (Haar Cascade chỉ chạy trên ảnh xám)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện khuôn mặt
        # scaleFactor=1.3: Thu nhỏ ảnh 30% mỗi lần quét (nhanh hơn nhưng kém chính xác hơn 1.1)
        # minNeighbors=5: Yêu cầu ít nhất 5 vùng lân cận xác nhận là mặt (giảm nhiễu)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # -------------------------------------------
            # BƯỚC 1: TÍNH TOÁN HỘP CẮT (ROI Calculation)
            # -------------------------------------------

            # Làm vuông hộp giới hạn (Squaring)
            # Mô hình yêu cầu đầu vào vuông (128x128). Nếu cắt hình chữ nhật rồi resize, mặt sẽ bị méo.
            max_side = max(w, h)
            
            # Tính padding
            pad = int(max_side * PADDING_RATIO)
            
            # Kích thước cạnh hình vuông mới (bao gồm padding)
            square_side = max_side + 2 * pad
            
            # Tính tâm của hộp Haar Cascade
            center_x = x + w // 2
            # Áp dụng dịch chuyển dọc (Y Offset) để căn chỉnh lại tâm khuôn mặt
            shift_y = int(h * Y_OFFSET_RATIO) 
            center_y = y + h // 2 + shift_y
            
            # Tính tọa độ góc trên-trái của hộp cắt mới (căn giữa tâm)
            x1 = int(center_x - square_side // 2)
            y1 = int(center_y - square_side // 2)
            x2 = int(center_x + square_side // 2)
            y2 = int(center_y + square_side // 2)
            # -------------------------------------------
            # BƯỚC 2: XỬ LÝ CẮT ẢNH AN TOÀN (Safe Crop)
            # -------------------------------------------
            # Tạo một "canvas" màu đen kích thước chuẩn square_side.
            # Nếu hộp cắt bị tràn ra ngoài màn hình webcam, phần tràn sẽ là màu đen (thay vì gây lỗi crash).
            face_crop = np.zeros((square_side, square_side, 3), dtype=np.uint8)
            
            # Tính vùng giao nhau giữa hộp cắt và khung hình webcam
            src_x1 = max(0, x1); src_y1 = max(0, y1)
            src_x2 = min(w_frame, x2); src_y2 = min(h_frame, y2)
            # Tính vị trí tương ứng trên canvas đen
            dst_x1 = src_x1 - x1; dst_y1 = src_y1 - y1
            dst_x2 = dst_x1 + (src_x2 - src_x1); dst_y2 = dst_y1 + (src_y2 - src_y1)
            
            # Nếu vùng giao không hợp lệ (mặt ra khỏi màn hình hoàn toàn), bỏ qua
            if src_x2 <= src_x1 or src_y2 <= src_y1: continue
            # Copy phần ảnh mặt vào canvas
            face_crop[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]
            
           # -------------------------------------------
            # BƯỚC 3: DỰ ĐOÁN (Inference)
            # -------------------------------------------
            # Resize về 128x128 (đầu vào mô hình)
            inp = cv2.resize(face_crop, (INPUT_WIDTH, INPUT_HEIGHT))
             # Chuẩn hóa pixel về [0, 1]
            inp = inp.astype(np.float32) / 255.0
            # Thêm chiều batch: (128, 128, 3) -> (1, 128, 128, 3)
            inp = np.expand_dims(inp, axis=0)
            
            # Chạy mô hình
            # verbose=0 để không in log rác ra màn hình
            preds = model.predict(inp, verbose=0)[0]
            # Làm mượt kết quả dự đoán
            preds = smoother.update(preds)
            
            # -------------------------------------------
            # BƯỚC 4: VẼ KẾT QUẢ (Visualization)
            # -------------------------------------------
            landmarks = preds.reshape(-1, 2)
            for (lx, ly) in landmarks:
                # Giải chuẩn hóa (Denormalization):
                # Tọa độ thật = (Tỷ lệ 0-1 * Kích thước hộp) + Tọa độ góc hộp
                px = int(lx * square_side + x1)
                py = int(ly * square_side + y1)
                # Vẽ chấm tròn xanh lá
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)
                
            # Vẽ hộp đỏ bao quanh mặt (vùng nhìn thực tế của AI)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # Hiển thị khung hình
        cv2.imshow("Landmark Demo", frame)
        # Thoát nếu nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()