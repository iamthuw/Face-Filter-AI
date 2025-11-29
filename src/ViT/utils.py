# utils.py
import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Vẽ biểu đồ thể hiện sự thay đổi của loss và val_loss theo từng epoch
    trong quá trình huấn luyện mô hình.

    Args:
        history (tensorflow.python.keras.callbacks.History):
            Đối tượng History được trả về từ hàm model.fit(). 
            Chứa thông tin về loss và val_loss sau mỗi epoch.

    Returns:
        None: Hàm chỉ hiển thị biểu đồ, không trả về giá trị.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()