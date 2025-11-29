# preprocess.py (Tiền xử lý và tăng cường dữ liệu)
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# --- Các hằng số ---
NUM_LANDMARKS = 68 # Số lượng điểm mốc khuôn mặt (ví dụ: 68 điểm của dlib)

# Cặp điểm mốc đối xứng cho việc lật ảnh
# Dựa trên thứ tự 68 điểm mốc của dlib
SYMMETRICAL_LANDMARKS = [
    (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9), # Jawline
    (17, 26), (18, 25), (19, 24), (20, 23), (21, 22), # Eyebrows
    (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46), # Eyes
    (31, 35), (32, 34), # Nose
    (48, 54), (49, 53), (50, 52), (51, 51), # Outer Mouth (51 là điểm trung tâm)
    (55, 59), (56, 58), # Inner Mouth (57 là điểm trung tâm)
    (60, 64), (61, 63), # Inner lip (62 là điểm trung tâm)
    (65, 67), # Inner lip
]
# Tạo mảng ánh xạ để hoán đổi nhanh các điểm khi lật
FLIP_MAP = list(range(NUM_LANDMARKS))
for l, r in SYMMETRICAL_LANDMARKS:
    FLIP_MAP[l] = r
    FLIP_MAP[r] = l

def _parse_xml_annotation(xml_file_path):
    """
    Hàm đọc file XML (dữ liệu 300W) và trích xuất thông tin ảnh, bounding box, và landmarks.
    
    Args:
        xml_file_path (str): Đường dẫn đến file XML chú thích.
        
    Returns:
        list: Danh sách các dictionary, mỗi dict chứa 'filename', 'bbox', 'landmarks'.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    images_data = []
    
    # Duyệt qua từng thẻ 'image' trong file XML
    for image_tag in root.find('images'):
        boxes = image_tag.findall('box')
        if not boxes: 
            continue # Bỏ qua ảnh không có bounding box
        
        # Lấy bounding box đầu tiên (thường chỉ có 1 khuôn mặt chính)
        box = boxes[0] 
        
        landmarks = []
        # Duyệt qua từng thẻ 'part' (điểm mốc) trong bounding box
        for part_tag in box.findall('part'):
            x = float(part_tag.get('x'))
            y = float(part_tag.get('y'))
            landmarks.append([x, y])
            
        # Chỉ thêm vào nếu đủ số lượng điểm mốc
        if len(landmarks) == NUM_LANDMARKS:
            images_data.append({
                'filename': image_tag.get('file'),
                'bbox': [int(box.get('top')), int(box.get('left')), int(box.get('width')), int(box.get('height'))],
                'landmarks': np.array(landmarks, dtype=np.float32)
            })
            
    return images_data

def _augment_data(image, landmarks):
    """
    Hàm tăng cường dữ liệu bằng cách lật ảnh ngang (mirroring).
    
    Args:
        image (np.array): Ảnh đã được cắt và resize.
        landmarks (np.array): Các điểm mốc tương ứng.
        
    Returns:
        tuple: (list_of_augmented_images, list_of_augmented_landmarks)
    """
    augmented_images = [image]
    augmented_landmarks = [landmarks]

    # Thực hiện lật ngang (Horizontal Flip)
    flipped_image = cv2.flip(image, 1) # Đối số 1 cho biết lật ngang
    
    flipped_landmarks = landmarks.copy()
    img_width = image.shape[1]
    # Điều chỉnh tọa độ x của điểm mốc sau khi lật
    flipped_landmarks[:, 0] = (img_width - 1) - flipped_landmarks[:, 0]
    
    # Hoán đổi vị trí các cặp điểm đối xứng
    flipped_landmarks = flipped_landmarks[FLIP_MAP]

    augmented_images.append(flipped_image)
    augmented_landmarks.append(flipped_landmarks)

    return augmented_images, augmented_landmarks

def load_and_preprocess_data(xml_file_path, data_root, image_size, test_split=0.2, augment=True):
    """
    Hàm chính để tải, tiền xử lý và tăng cường toàn bộ dữ liệu.

    Các bước thực hiện:
        1. Đọc thông tin ảnh và điểm landmark từ file XML.
        2. Crop khuôn mặt dựa trên bounding box.
        3. Resize ảnh về kích thước chuẩn.
        4. Chuyển đổi và chuẩn hóa tọa độ landmark.
        5. (Tuỳ chọn) Tăng cường dữ liệu bằng lật ảnh.
        6. Chia dữ liệu thành tập train và validation.
    
    Args:
        xml_file_path (str): Đường dẫn đến file XML chú thích (train.xml hoặc test.xml).
        data_root (str): Thư mục gốc chứa các thư mục con (afw, helen, ...) và ảnh.
        image_size (int): Kích thước cạnh của ảnh đầu ra (ví dụ: 128x128).
        test_split (float): Tỷ lệ dữ liệu dành cho tập validation (mặc định 0.2).
                            Nếu là 0, sẽ không chia validation (dùng cho tập test cuối cùng).
        augment (bool): Có thực hiện tăng cường dữ liệu (lật ảnh) hay không.
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val) nếu test_split > 0
               (X_data, _, y_data, _) nếu test_split == 0 (toàn bộ dữ liệu)
    """
    print("Loading and preprocessing data...")
    all_data = _parse_xml_annotation(xml_file_path)
    if not all_data:
        raise ValueError(f"No data loaded from {xml_file_path}. Check the file path and content.")

    X_images = []
    y_landmarks = []
    
    for item in all_data:
        image_path = os.path.join(data_root, item['filename'])
        image = cv2.imread(image_path)
        
        if image is None: 
            # print(f"Warning: Could not load image {image_path}") # Có thể bỏ comment để debug
            continue 
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Chuyển đổi BGR sang RGB
        
        # Lấy thông tin bounding box và tính toán vùng cắt
        top, left, width, height = item['bbox']
        
        # Đảm bảo bounding box nằm trong giới hạn ảnh và có kích thước hợp lệ
        # Thêm một chút padding để tránh cắt sát quá
        padding = int(max(width, height) * 0.1) # 10% padding
        top_padded = max(0, top - padding)
        left_padded = max(0, left - padding)
        bottom_padded = min(image.shape[0], top + height + padding)
        right_padded = min(image.shape[1], left + width + padding)

        cropped_face = image[top_padded:bottom_padded, left_padded:right_padded]
        
        if cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0: 
            continue # Bỏ qua ảnh không hợp lệ sau khi cắt

        # Resize khuôn mặt về kích thước chuẩn
        resized_face = cv2.resize(cropped_face, (image_size, image_size))

        # Điều chỉnh tọa độ điểm mốc cho ảnh đã cắt và resize
        landmarks = item['landmarks'].copy()
        
        # Chuyển tọa độ từ ảnh gốc sang tọa độ trên vùng cắt đệm
        landmarks[:, 0] = (landmarks[:, 0] - left_padded) 
        landmarks[:, 1] = (landmarks[:, 1] - top_padded) 
        
        # Chuẩn hóa tọa độ trên vùng cắt đệm về kích thước ảnh output
        # Tính toán lại width và height của vùng cắt đệm
        cropped_width = right_padded - left_padded
        cropped_height = bottom_padded - top_padded

        if cropped_width > 0 and cropped_height > 0: # Tránh chia cho 0
            landmarks[:, 0] = (landmarks[:, 0] / cropped_width) * image_size
            landmarks[:, 1] = (landmarks[:, 1] / cropped_height) * image_size
        else:
            continue # Nếu vùng cắt không hợp lệ, bỏ qua

        # Áp dụng tăng cường dữ liệu
        if augment:
            aug_imgs, aug_lms = _augment_data(resized_face, landmarks)
            X_images.extend(aug_imgs)
            y_landmarks.extend(aug_lms)
        else:
            X_images.append(resized_face)
            y_landmarks.append(landmarks)

    # Chuẩn hóa ảnh và tọa độ điểm mốc
    X_images = np.array(X_images, dtype=np.float32) / 255.0 # Chuẩn hóa pixel về [0, 1]
    y_landmarks = np.array(y_landmarks, dtype=np.float32) / image_size # Chuẩn hóa tọa độ về [0, 1]
    y_landmarks = y_landmarks.reshape(-1, NUM_LANDMARKS * 2) # Làm phẳng mảng tọa độ

    print(f"Total processed samples: {len(X_images)}")

    if test_split > 0:
        return train_test_split(X_images, y_landmarks, test_size=test_split, random_state=42)
    else:
        # Nếu test_split == 0, trả về toàn bộ dữ liệu mà không chia tách
        return X_images, None, y_landmarks, None # Trả về None cho X_val, y_val