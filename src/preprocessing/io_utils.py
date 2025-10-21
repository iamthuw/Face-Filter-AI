import os
import cv2

def save_image_and_landmarks(processed_dir, filename, img, landmarks):
    """Lưu ảnh và landmarks (.txt) sau khi xử lý."""
    img_dir = os.path.join(processed_dir, 'images')
    lm_dir = os.path.join(processed_dir, 'landmarks')

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lm_dir, exist_ok=True)

    img_path = os.path.join(img_dir, filename)
    lm_path = os.path.join(lm_dir, filename.replace('.jpg', '.txt'))

    cv2.imwrite(img_path, img)

    with open(lm_path, 'w', encoding='utf-8') as f:
        for (x, y) in landmarks:
            f.write(f"{x} {y}\n")

    return img_path
