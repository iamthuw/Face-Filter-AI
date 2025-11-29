# Nhận dạng Điểm mốc Khuôn mặt (Facial Landmark Detection) bằng Vision Transformer (ViT)

## 1. Giới thiệu

Module này triển khai một **Vision Transformer (ViT)** để giải quyết bài toán **Dự đoán tọa độ điểm mốc (landmark)** trên ảnh khuôn mặt.

- **Mục tiêu:** Dự đoán chính xác các tọa độ $(x, y)$ của các điểm mốc quan trọng trên khuôn mặt.
- **Công nghệ:** Mô hình được xây dựng bằng **TensorFlow/Keras**, sử dụng kiến trúc **Transformer cho ảnh** để cải thiện hiệu quả so với CNN truyền thống.

---

## 2. Cấu trúc module

Dự án được tổ chức như sau:

```text
ViT/
├── ibug_300W_large_face_landmark_dataset/ # Thư mục chứa dữ liệu đầu vào (cần tải về)
├── saved_model/
│ └── facial_landmark_detector_vit.h5 # Mô hình ViT đã huấn luyện
│
├── evaluate_model.py # Script đánh giá mô hình
├── model.py # Định nghĩa kiến trúc Vision Transformer
├── predict_webcam.py # Dự đoán điểm mốc từ webcam
├── preprocess.py # Tiền xử lý dữ liệu
├── train.py # Script huấn luyện mô hình
├── utils.py # Hàm tiện ích: vẽ đồ thị, tính loss, metric
└── README.md # Tệp hướng dẫn này
```

---

## 3. Cài đặt (Installation)

### 3.1. Tải dữ liệu

Dự án có thể sử dụng **iBUG 300W Large Face Landmark Dataset** hoặc các bộ dữ liệu ảnh khuôn mặt khác.  
Đặt ảnh và nhãn vào thư mục phù hợp trước khi huấn luyện.

### 3.2. Cài đặt Dependencies

Cài đặt tất cả thư viện Python cần thiết, bao gồm TensorFlow, Keras, OpenCV, NumPy, Matplotlib:

```bash
pip install -r requirements.txt
```


# 4. Hướng dẫn sử dụng
### 4.1. Huấn luyện Mô hình

Chạy script train.py để huấn luyện mô hình ViT. Script sẽ thực hiện:

Tiền xử lý dữ liệu bằng preprocess.py.

Xây dựng mô hình ViT từ model.py.

Huấn luyện mô hình và lưu phiên bản tốt nhất vào saved_model/facial_landmark_detector_vit.h5.
python train.py

### 4.2. Đánh giá Mô hình

Sử dụng evaluate_model.py để đánh giá mô hình trên tập kiểm tra:
python evaluate_model.py
Kết quả sẽ hiển thị MSE, MAE, R2-square trên tập test
### 4.3. Dự đoán Từ Webcam

Chạy predict_webcam.py để dự đoán điểm mốc trên khuôn mặt trực tiếp từ webcam:

```bash
python predict_webcam.py
```


Ứng dụng sẽ hiển thị ảnh webcam với các điểm mốc được đánh dấu trực tiếp.
# 5. Tiện ích khác

File utils.py cung cấp các hàm:

Vẽ đồ thị lịch sử huấn luyện (loss/metric)

Tính toán các chỉ số đánh giá

Hiển thị ảnh với điểm mốc

# 6. Ghi chú

Mô hình sử dụng Vision Transformer nên cần GPU để huấn luyện hiệu quả.

Dataset có thể thay đổi, đảm bảo dữ liệu ảnh và nhãn khớp đúng định dạng.

Mọi script đều đã được thiết kế để chạy độc lập.
