# utils.py
import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Vẽ đồ thị lịch sử của hàm mất mát (loss) trong quá trình huấn luyện
    trên cả tập huấn luyện (Train Loss) và tập kiểm tra/đánh giá (Validation Loss).
    Returns:
        None: Hàm hiển thị đồ thị trực tiếp bằng plt.show().
    """
    # 📝 Tạo một figure mới để đảm bảo kích thước đồ thị lớn và dễ nhìn.
    plt.figure(figsize=(10, 5)) 
    
    # Lấy giá trị loss từ tập huấn luyện
    plt.plot(history.history['loss'], label='Train Loss')
    # Lấy giá trị loss từ tập kiểm tra (validation)
    plt.plot(history.history['val_loss'], label='Validation Loss')
    
    # Đặt tiêu đề và nhãn cho đồ thị
    plt.title('Model Loss (MSE)')
    plt.xlabel('Epoch')
    # 📝 Loss (MSE): Đơn vị là Mean Squared Error, hàm mất mát được sử dụng trong train.py
    plt.ylabel('Loss (MSE)') 
    
    # Hiển thị chú giải (Legend) để phân biệt hai đường đồ thị
    plt.legend()
    # Thêm lưới (Grid) để dễ dàng đọc và so sánh các giá trị
    plt.grid(True)
    
    # 📝 Hiển thị đồ thị
    plt.show()