# preprocess.py
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# --- Các hằng số và ánh xạ cho Data Augmentation ---
NUM_LANDMARKS = 68

# Cặp điểm mốc đối xứng cho việc lật ảnh
SYMMETRICAL_LANDMARKS = [
    (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9), # Jawline
    (17, 26), (18, 25), (19, 24), (20, 23), (21, 22), # Eyebrows
    (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46), # Eyes
    (31, 35), (32, 34), # Nose
    (48, 54), (49, 53), (50, 52), # Outer Mouth
    (55, 59), (56, 58), # Inner Mouth
    (60, 64), (61, 63), # Inner lip
]
# Tạo mảng ánh xạ để hoán đổi nhanh
FLIP_MAP = list(range(NUM_LANDMARKS))
for l, r in SYMMETRICAL_LANDMARKS:
    FLIP_MAP[l] = r
    FLIP_MAP[r] = l

def _parse_xml_annotation(xml_file_path):
    """
    Hàm này đọc file XML và trích xuất thông tin ảnh, bounding box, và landmarks.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    images_data = []
    
    for image_tag in root.find('images'):
        boxes = image_tag.findall('box')
        if not boxes: continue
        
        box = boxes[0] 
        
        landmarks = []
        for part_tag in box.findall('part'):
            x = float(part_tag.get('x'))
            y = float(part_tag.get('y'))
            landmarks.append([x, y])
            
        if len(landmarks) == NUM_LANDMARKS:
            images_data.append({
                'filename': image_tag.get('file'),
                'bbox': [int(box.get('top')), int(box.get('left')), int(box.get('width')), int(box.get('height'))],
                'landmarks': np.array(landmarks, dtype=np.float32)
            })
            
    return images_data # Quan trọng: Phải trả về danh sách dữ liệu

def _augment_data(image, landmarks):
    """
    Hàm này thực hiện tăng cường dữ liệu bằng cách lật ảnh.
    """
    augmented_images = [image]
    augmented_landmarks = [landmarks]

    # Ảnh lật ngang (Mirror)
    flipped_image = cv2.flip(image, 1)
    
    flipped_landmarks = landmarks.copy()
    img_width = image.shape[1]
    flipped_landmarks[:, 0] = (img_width - 1) - flipped_landmarks[:, 0]
    
    # Hoán đổi các cặp điểm đối xứng
    flipped_landmarks = flipped_landmarks[FLIP_MAP]

    augmented_images.append(flipped_image)
    augmented_landmarks.append(flipped_landmarks)

    return augmented_images, augmented_landmarks

# Dòng mới
def load_and_preprocess_data(xml_file_path, data_root, image_size):
    """
    Hàm chính để tải và xử lý toàn bộ dữ liệu.
    """
    print("Loading and preprocessing data...")
    all_data = _parse_xml_annotation(xml_file_path)
    if not all_data:
        raise ValueError(f"No data loaded from {xml_file_path}. Check the file path and content.")

    X_images = []
    y_landmarks = []
    

    for item in all_data:
        # Dòng mới (xóa dòng data_dir và sửa dòng image_path)
        image_path = os.path.join(data_root, item['filename'])
        image = cv2.imread(image_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        top, left, width, height = item['bbox']
        top, left = max(0, top), max(0, left)
        bottom, right = min(image.shape[0], top + height), min(image.shape[1], left + width)
        
        cropped_face = image[top:bottom, left:right]
        if cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0: continue
            
        resized_face = cv2.resize(cropped_face, (image_size, image_size))

        landmarks = item['landmarks'].copy()
        landmarks[:, 0] = (landmarks[:, 0] - left) / width * image_size
        landmarks[:, 1] = (landmarks[:, 1] - top) / height * image_size
        
        aug_imgs, aug_lms = _augment_data(resized_face, landmarks)
        
        for aug_img, aug_lm in zip(aug_imgs, aug_lms):
            X_images.append(aug_img)
            y_landmarks.append(aug_lm)

    # Chuẩn hóa
    X_images = np.array(X_images, dtype=np.float32) / 255.0
    y_landmarks = np.array(y_landmarks, dtype=np.float32) / image_size
    y_landmarks = y_landmarks.reshape(-1, NUM_LANDMARKS * 2)

    print(f"Total processed samples: {len(X_images)}")
    return train_test_split(X_images, y_landmarks, test_size=0.2, random_state=42)