# utils.py
import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Vẽ đồ thị loss của quá trình huấn luyện.
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