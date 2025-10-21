import cv2
import numpy as np
import random

def random_flip(img, landmarks):
    """Lật ngang ảnh và điểm (50% xác suất)."""
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        w = img.shape[1]
        landmarks = [(w - x, y) for (x, y) in landmarks]
    return img, landmarks

def random_brightness(img):
    """Thay đổi độ sáng ngẫu nhiên."""
    if random.random() < 0.3:
        factor = 1.0 + (random.random() - 0.5) * 0.4
        img = np.clip(img * factor, 0, 255).astype(np.uint8)
    return img

def equalize_histogram(img):
    """
    Cân bằng histogram cho ảnh màu (theo kênh sáng Y).
    """
    # Đảm bảo ảnh kiểu uint8
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)

    # Nếu ảnh màu, chuyển sang YCrCb để cân bằng kênh sáng
    if len(img.shape) == 3 and img.shape[2] == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb = np.array(ycrcb, dtype=np.uint8)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        img_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        # Nếu ảnh xám
        img_eq = cv2.equalizeHist(img)

    return img_eq

def augment_image(img, landmarks):
    """Tổng hợp các augmentation."""
    img, landmarks = random_flip(img, landmarks)
    img = random_brightness(img)
    img = equalize_histogram(img)
    return img, landmarks
