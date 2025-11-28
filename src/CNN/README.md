Giới thiệu
Module này triển khai Convolutional Neural Network (CNN) dùng để:
Dự đoán tọa độ landmark trên ảnh khuôn mặt.
Mô hình được xây dựng bằng TensorFlow/Keras, dễ dàng huấn luyện, kiểm tra và triển khai.

2. Cấu trúc module
   ├── ibug_300W_large_face_landmark_dataset/ # 📂 Thư mục chứa dữ liệu đầu vào
   │
   ├── src/
   │ └── CNN/
   │ | ├── saved_model/ # 💾 Thư mục lưu trữ mô hình đã huấn luyện
   │ | │ └── facial_landmark_model.h5 # Ví dụ tệp mô hình Keras
   | | |
   │ | ├── model.py # Định nghĩa kiến trúc CNN
   │ | ├── train.py # Script Huấn luyện mô hình
   │ | ├── evaluate.py # Script Đánh giá mô hình (thường đi kèm)
   │ | ├── utils.py # Vẽ đồ thị lịch sử của hàm mất mát (loss) trong quá trình huấn luyện
   │ | └── requirements.txt # Thư viện cần thiết

3. Cài đặt
   Tải thư mục dữ liệu ảnh từ trên drive về+

# Cài đặt dependencies

pip install -r src/CNN/requirements.txt

4. Sử dụng
   4.1 Huấn luyện mô hình
   python train.py
