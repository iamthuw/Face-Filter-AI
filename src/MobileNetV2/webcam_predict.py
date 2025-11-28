# webcam_predict.py
import cv2
import numpy as np
import tensorflow as tf
from config import *

# ==========================================
# KHU VỰC TINH CHỈNH (TUNE TẠI ĐÂY)
# ==========================================

# 1. ĐỘ RỘNG/HẸP (SCALE)
# - Nếu landmark BÉ quá (co cụm) -> TĂNG số này lên (ví dụ 0.15, 0.2)
# - Nếu landmark TO quá (bay ra ngoài) -> GIẢM số này xuống (ví dụ 0.05, 0.0)
PADDING_RATIO = 0.18
# 2. ĐỘ CAO/THẤP (SHIFT Y)
# - Nếu landmark bị LỆCH LÊN TRÊN -> TĂNG số này (ví dụ 0.05, 0.1) để dịch khung nhìn xuống
# - Nếu landmark bị LỆCH XUỐNG DƯỚI -> GIẢM số này (ví dụ -0.05)
Y_OFFSET_RATIO = -0.02 # Dịch khung nhìn xuống 5% chiều cao để lấy thêm cằm

# ==========================================

class LandmarkSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev = None
    def update(self, current):
        if self.prev is None:
            self.prev = current
        else:
            self.prev = self.alpha * current + (1 - self.alpha) * self.prev
        return self.prev

def main():
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
    
    custom_objects = {'mse': tf.keras.metrics.MeanSquaredError, 'mae': tf.keras.metrics.MeanAbsoluteError}
    model = tf.keras.models.load_model(inference_model_path, custom_objects=custom_objects, compile=False)
    
    cap = cv2.VideoCapture(WEBCAM_ID)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smoother = LandmarkSmoother(alpha=0.5)

    print("--- Bấm 'q' để thoát ---")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h_frame, w_frame = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # -------------------------------------------
            # TÍNH TOÁN HỘP CẮT (Dựa trên tham số chỉnh)
            # -------------------------------------------
            
            # 1. Xác định cạnh lớn nhất để làm vuông
            max_side = max(w, h)
            
            # 2. Tính padding
            pad = int(max_side * PADDING_RATIO)
            
            # 3. Kích thước hộp vuông cuối cùng
            square_side = max_side + 2 * pad
            
            # 4. Tính tâm hộp (CÓ DỊCH CHUYỂN Y)
            center_x = x + w // 2
            # Dịch tâm xuống dưới một chút để lấy thêm cằm
            shift_y = int(h * Y_OFFSET_RATIO) 
            center_y = y + h // 2 + shift_y
            
            # 5. Tính tọa độ cắt
            x1 = int(center_x - square_side // 2)
            y1 = int(center_y - square_side // 2)
            x2 = int(center_x + square_side // 2)
            y2 = int(center_y + square_side // 2)
            
            # -------------------------------------------
            # XỬ LÝ CẮT ẢNH AN TOÀN (Chống lỗi tràn viền)
            # -------------------------------------------
            face_crop = np.zeros((square_side, square_side, 3), dtype=np.uint8)
            
            src_x1 = max(0, x1); src_y1 = max(0, y1)
            src_x2 = min(w_frame, x2); src_y2 = min(h_frame, y2)
            dst_x1 = src_x1 - x1; dst_y1 = src_y1 - y1
            dst_x2 = dst_x1 + (src_x2 - src_x1); dst_y2 = dst_y1 + (src_y2 - src_y1)
            
            if src_x2 <= src_x1 or src_y2 <= src_y1: continue
            face_crop[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]
            
            # -------------------------------------------
            # DỰ ĐOÁN & VẼ
            # -------------------------------------------
            inp = cv2.resize(face_crop, (INPUT_WIDTH, INPUT_HEIGHT))
            inp = inp.astype(np.float32) / 255.0
            inp = np.expand_dims(inp, axis=0)
            
            preds = model.predict(inp, verbose=0)[0]
            preds = smoother.update(preds)
            
            landmarks = preds.reshape(-1, 2)
            for (lx, ly) in landmarks:
                # Giải chuẩn hóa theo kích thước hộp vuông
                px = int(lx * square_side + x1)
                py = int(ly * square_side + y1)
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)
                
            # Vẽ hộp đỏ (Vùng nhìn thực tế của AI)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        cv2.imshow("Landmark Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()