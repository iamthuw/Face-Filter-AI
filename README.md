# 🤖 Báo cáo Bài tập nhóm Môn Trí tuệ Nhân tạo

# 🏷️ Tên nhóm: **Nhóm 10**

**📋 Thông tin:**

* **📚 Môn học:**  MAT3508 – Nhập môn Trí tuệ Nhân tạo  
* **📅 Học kỳ:** Học kỳ 1, Năm học 2025-2026
* **🏫 Trường:** VNU-HUS (Đại học Quốc gia Hà Nội – Trường Đại học Khoa học Tự nhiên)
* **📝 Tiêu đề:** Face-Filter-AI
* **📅 Ngày nộp:**  30/11/2025
* **📄 Báo cáo PDF:** https://drive.google.com/file/d/1aftZo73lW4IMU5jApfWPQFrtn_i7i3Vj/view?usp=share_link
* **🖥️ Slide thuyết trình:** [Liên kết tới slide thuyết trình]
* **📂 Kho lưu trữ:** https://github.com/iamthuw/Face-Filter-AI
- **Link data**: https://www.kaggle.com/datasets/toxicloser/ibug-300w-large-face-landmark-dataset


  
# 👥 Thành viên nhóm
| 👤 Họ và tên       | 🆔 Mã sinh viên | 🐙 GitHub         | 📊 Đóng góp        |
|--------------------|----------------|------------------|------------------|
| Phạm Thị Minh Thư | 23001562       | iamthuw          |Quản lý dự án, phát triển filter|
| Lê Thị Yến        | 23001963       | ltyen05          |Xây dựng mô hình ViT, kiểm thử|
| Chu Thị Mỹ Duyên  | 23001509       | chuthimyduyen    |Xử lý dữ liệu|
| Nguyễn Bảo Thạch  | 23001559       | NgThach          |Xây dựng mô hình MobileNetV2, kiểm thử|
| Nguyễn Tiến Lưỡng | 23001534       | NguyenTienLuong  |Xây dựng mô hình CNN, kiểm thử, phát triển mô hình chính|

# Cấu trúc thư mục
  ```text
  Face-Filter-AI/
  ├── data/               # Chứa dữ liệu (Dành cho việc huấn luyện, landmark, hoặc các tài nguyên khác).
  ├── src/                # Chứa mã nguồn chính được tổ chức theo các mô hình AI.
  │   ├── CNN/            # Mã nguồn và tài nguyên cho mô hình Convolutional Neural Network (CNN).
  │   ├── MobileNetV2/    # Mã nguồn và tài nguyên cho mô hình MobileNetV2.
  │   └── ViT/            # Mã nguồn và tài nguyên cho mô hình Vision Transformer (ViT).
  ├── .gitignore          # Chỉ định các file và thư mục mà Git bỏ qua.
  ├── README.md           # File tài liệu mô tả dự án và hướng dẫn sử dụng.
  ├── apply_filter.py
  ├── filter_webcam.py #File triển khai áp filter lên mặt
  ├── README.md   
  └── requirements.txt    # Danh sách các thư viện Python cần thiết (dependencies).
  ```

# Triển khai filter sử dụng webcam
Bấm số 1 chuyển sang filter kính, số 2 chuyển sang filter râu, số 3 chuyển sang filter mũi lợn, số 4 chuyển sang filter má hồng

Để có thể mở webcam và áp các filter lên khuôn mặt thực hiện lệnh sau 
```bash
python filter_webcam.py
```

# 🗒️ Tóm tắt
Dự án **Face Filter AI** sử dụng thị giác máy tính để phát hiện và phân tích khuôn mặt, từ đó áp dụng các hiệu ứng lên từng bộ phận trên khuôn mặt.  
Hệ thống cho phép gắn filter trực quan trên ảnh (có thể mở rộng sang video và camera real-time), mang lại trải nghiệm tự nhiên và sáng tạo cho người dùng. 

# 🎯 Bối cảnh
Các ứng dụng filter ngày càng phổ biến, gắn liền với nhu cầu sáng tạo và chia sẻ nội dung trên mạng xã hội.  
Nhóm chọn đề tài này để xây dựng một hệ thống AI mô phỏng filter, tập trung vào việc nhận diện khuôn mặt và áp dụng hiệu ứng phù hợp cho từng vùng, qua đó khám phá ứng dụng thực tế của thị giác máy tính.

# 🚀 Kế hoạch
- **Thu thập dữ liệu**: 
  - Thu thập các bộ dữ liệu khuôn mặt công khai.  
  - Đồng thời thu thập các bộ filter và virtual effects miễn phí từ những nguồn công khai để làm tài nguyên áp dụng.  

- **Tiền xử lý dữ liệu**: 
  - Chuẩn hóa kích thước ảnh, cân bằng ánh sáng.  
  - Trích xuất và gắn nhãn các vùng khuôn mặt bằng landmark detection để phục vụ cho việc áp dụng filter.  

- **Xây dựng mô hình AI**: 
  - Huấn luyện mô hình nhận diện và phân đoạn khuôn mặt.  
  - Tập trung xác định chính xác các vùng đặc trưng cần thiết để áp dụng filter.  

- **Áp dụng filter**: 
  - Tích hợp các filter/virtual effects đã thu thập.  
  - Xây dựng cơ chế tự động căn chỉnh để filter khớp với từng khuôn mặt.  

- **Đánh giá và tối ưu**: 
  - Kiểm tra kết quả về độ chính xác, tính thẩm mỹ và sự tự nhiên.  
  - Tinh chỉnh tham số để nâng cao chất lượng hiển thị.  

- **Tích hợp và thử nghiệm**: 
  - Kết hợp mô hình với giao diện ứng dụng.  
  - Chạy thử pipeline hoàn chỉnh để đánh giá tính ổn định và hiệu quả.  

- **Demo**: 
  - Phát triển ứng dụng mẫu (web/app) cho phép tải ảnh v
