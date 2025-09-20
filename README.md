# 🏷️ Tên nhóm
**Nhóm 10**

# 📝 Tên dự án
**Face Filter AI**

# 👥 Thành viên nhóm
| 👤 Họ và tên       | 🆔 Mã sinh viên | 🐙 GitHub         |
|--------------------|----------------|------------------|
| Phạm Thị Minh Thư | 23001562       | iamthuw          |
| Lê Thị Yến        | 23001963       | ltyen05          |
| Chu Thị Mỹ Duyên  | 23001509       | chuthimyduyen    |
| Nguyễn Bảo Thạch  | 23001559       | NgThach          |
| Nguyễn Tiến Lưỡng | 23001534       | NguyenTienLuong  |

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
