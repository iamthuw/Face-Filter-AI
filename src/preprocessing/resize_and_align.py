import cv2
import numpy as np

def resize_image_and_landmarks(image_path, landmarks, size=(256, 256), padding=0.07):
    """
    Resize ảnh và scale landmarks chính xác, không bị lệch.
    - Giữ tỉ lệ khi resize, thêm viền nền xám nếu cần.
    - Mở rộng bounding box có padding (mặc định 7%).
    - Đảm bảo landmarks sau khi scale nằm hoàn toàn trong ảnh đầu ra.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    h, w = img.shape[:2]

    # Tính bounding box từ landmarks
    xs = np.array([x for x, y in landmarks])
    ys = np.array([y for x, y in landmarks])
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    w_box = x_max - x_min
    h_box = y_max - y_min

    # Mở rộng bounding box có padding
    x_min = int(max(x_min - w_box * padding, 0))
    y_min = int(max(y_min - h_box * padding, 0))
    x_max = int(min(x_max + w_box * padding, w))
    y_max = int(min(y_max + h_box * padding, h))

    # Crop ảnh theo box
    img_crop = img[y_min:y_max, x_min:x_max]
    crop_h, crop_w = img_crop.shape[:2]

    # Resize theo tỉ lệ, không làm méo mặt
    scale = min(size[0] / crop_w, size[1] / crop_h)
    new_w, new_h = int(crop_w * scale), int(crop_h * scale)
    img_resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Tạo khung ảnh đầu ra (nền xám)
    img_padded = np.full((size[1], size[0], 3), 128, dtype=np.uint8)

    # Căn giữa ảnh sau khi resize
    x_offset = (size[0] - new_w) // 2
    y_offset = (size[1] - new_h) // 2
    img_padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized

    # Cập nhật lại vị trí landmarks theo crop + scale + offset
    scaled_landmarks = []
    for (x, y) in landmarks:
        new_x = (x - x_min) * scale + x_offset
        new_y = (y - y_min) * scale + y_offset
        # Giữ điểm trong biên ảnh
        new_x = np.clip(new_x, 0, size[0] - 1)
        new_y = np.clip(new_y, 0, size[1] - 1)
        scaled_landmarks.append((new_x, new_y))

    return img_padded, scaled_landmarks
