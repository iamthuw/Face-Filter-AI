# Nhận dạng Điểm mốc Khuôn mặt (Facial Landmark Detection) bằng CNN

### 1. Giới thiệu

Module này triển khai một **Mạng nơ-ron Tích chập (CNN)** để giải quyết bài toán **Dự đoán tọa độ điểm mốc (landmark)** trên ảnh khuôn mặt.

* **Mục tiêu:** Dự đoán chính xác các tọa độ $(x, y)$ của các điểm mốc quan trọng trên khuôn mặt.
* **Công nghệ:** Mô hình được xây dựng bằng **TensorFlow/Keras**, dễ dàng huấn luyện, kiểm tra và triển khai.

---

### 2.  Cấu trúc module

Dự án được tổ chức như sau:

```text
.
├── ibug_300W_large_face_landmark_dataset/ # Thư mục chứa dữ liệu đầu vào (cần tải về)
│
├── src/
│   └── CNN/
│       ├── saved_model/              # Thư mục lưu trữ mô hình đã huấn luyện
│       │   └── facial_landmark_model.h5  # Ví dụ: tệp mô hình Keras đã lưu
│       │
│       ├── model.py                  # Định nghĩa kiến trúc CNN
│       ├── train.py                  # Script Huấn luyện mô hình
│       ├── evaluate.py               # Script Đánh giá mô hình
│       ├── utils.py                  # Các hàm tiện ích: Vẽ đồ thị lịch sử hàm mất mát (loss)
│       └── requirements.txt          # Thư viện cần thiết
```
### 3.  Cài đặt (Installation)

#### 3.1. Tải Dữ liệu

Dự án sử dụng bộ dữ liệu **iBUG 300W Large Face Landmark Dataset**. 

Vui lòng tải toàn bộ thư mục dữ liệu ảnh và tệp nhãn tương ứng về và đặt vào đường dẫn gốc của dự án theo cấu trúc trên
### 3.2. Cài đặt Dependencies

Sử dụng tệp `requirements.txt` để cài đặt tất cả các thư viện Python cần thiết cho dự án (bao gồm **TensorFlow**, **Keras**, **OpenCV**, v.v.).
```bash
pip install -r src/CNN/requirements.txt
```
### 4. 🏃 Hướng dẫn Sử dụng

#### 4.1. Huấn luyện Mô hình

Chạy script **`train.py`** để bắt đầu quá trình huấn luyện mô hình **CNN**. Script sẽ tự động:
* Sử dụng các hàm tiền xử lý từ `preprocess.py` để tải và xử lý dữ liệu từ thư mục `./ibug_300W_large_face_landmark_dataset/`.
* Xây dựng mô hình theo kiến trúc đã định nghĩa trong `model.py`.
* Bắt đầu quá trình huấn luyện và lưu mô hình tốt nhất vào thư mục `saved_model/`.

```bash
python src/CNN/train.py
```
#### 4.2. Đánh giá sai số
```bash
python src/CNN/evaluate.py
```
