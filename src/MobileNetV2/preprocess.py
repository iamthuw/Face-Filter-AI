# preprocess.py
"""
Module xử lý và chuẩn bị dữ liệu (Data Preprocessing).

Mục đích:
    Module này chịu trách nhiệm biến đổi dữ liệu thô (ảnh .jpg và tọa độ .pts) thành định dạng
    chuẩn hóa mà mô hình máy học có thể sử dụng được.

Chức năng chính:
    1. Đọc và phân tích cú pháp file .pts chứa tọa độ landmark.
    2. Tính toán Hộp giới hạn (Bounding Box) bao quanh khuôn mặt.
    3. Áp dụng kỹ thuật "Làm vuông" (Squaring) và "Mở rộng" (Padding) để tránh méo ảnh.
    4. Cắt ảnh (Crop) và Resize về kích thước chuẩn (128x128).
    5. Chuẩn hóa tọa độ landmark về hệ quy chiếu tương đối [0, 1].
    6. Tăng cường dữ liệu (Data Augmentation) bằng cách lật ảnh ngang.
    7. Lưu trữ dữ liệu đã xử lý vào file XML và tải lại khi huấn luyện.
"""
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from config import *

# --- CẤU HÌNH ---
# Tỷ lệ mở rộng vùng nhìn (Padding) 25% để bao gồm cả trán và cằm,
# giúp khắc phục lỗi landmark bị lệch trục dọc.
PADDING_RATIO = 0.25

# Khởi tạo Haar Cascade để phát hiện khuôn mặt.
# Việc sử dụng cùng một thuật toán detection cho cả train và test giúp đồng bộ hóa hệ quy chiếu.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cấu hình lật ảnh
NUM_PTS = 68
# Danh sách các cặp điểm đối xứng qua trục dọc khuôn mặt (ví dụ: mắt trái - mắt phải)
SYMMETRICAL_LANDMARKS = [
    (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
    (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
    (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
    (31, 35), (32, 34),
    (48, 54), (49, 53), (50, 52), (55, 59), (56, 58), (60, 64), (61, 63)
]
# Tạo mảng ánh xạ để hoán đổi vị trí các điểm khi lật ảnh
FLIP_MAP = list(range(NUM_PTS))
for l, r in SYMMETRICAL_LANDMARKS:
    FLIP_MAP[l] = r
    FLIP_MAP[r] = l

def _read_pts_file(pts_path):
    """
    Đọc file .pts chứa tọa độ landmark, xử lý bỏ qua header.

    Args:
        pts_path (str): Đường dẫn đến file .pts.

    Returns:
        np.array: Mảng numpy chứa 68 cặp tọa độ (x, y).
                  Trả về mảng rỗng nếu file không hợp lệ.
    """
    with open(pts_path, 'r') as f:
        lines = f.readlines()
    points = []
    start_reading = False
    for line in lines:
        line = line.strip()
        if line == '{': start_reading = True; continue # Bắt đầu đọc sau dấu {
        if line == '}': break # Kết thúc đọc khi gặp dấu }
        if start_reading and (line and (line[0].isdigit() or line.startswith('-'))):
            try:
                parts = line.split()
                if len(parts) >= 2: points.append([float(parts[0]), float(parts[1])])
            except ValueError: continue
    if len(points) == 68: return np.array(points, dtype=np.float32)
    return np.array([])

def _augment_flip(image, landmarks):
    """
    Thực hiện tăng cường dữ liệu bằng cách lật ảnh ngang (Horizontal Flip).

    Args:
        image (np.array): Ảnh gốc.
        landmarks (np.array): Tọa độ landmark tương ứng (đã chuẩn hóa).

    Returns:
        tuple: (Ảnh đã lật, Landmark đã lật và hoán đổi index).
    """
    img_flip = cv2.flip(image, 1)
    lm_flip = landmarks.copy()
    # Lật tọa độ X: x_new = 1.0 - x_old (do tọa độ đã chuẩn hóa 0-1)
    lm_flip[:, 0] = 1.0 - lm_flip[:, 0]
    # Hoán đổi thứ tự các điểm đối xứng (ví dụ: mắt trái thành mắt phải)
    lm_flip = lm_flip[FLIP_MAP]
    return img_flip, lm_flip

def process_and_split_data(raw_dir, output_dir, img_size=128, test_size=0.2):
    """
    Hàm chính để xử lý dữ liệu thô và tạo file XML huấn luyện/kiểm tra.

    Quy trình:
    1. Quét thư mục dữ liệu thô.
    2. Tính toán Bounding Box vuông vức bao quanh khuôn mặt với padding 25%.
    3. Chuẩn hóa tọa độ landmark theo Bounding Box này.
    4. Chia tập dữ liệu và lưu metadata vào file XML.

    Args:
        raw_dir (str): Đường dẫn thư mục chứa ảnh và .pts gốc.
        output_dir (str): Thư mục lưu file XML đầu ra.
        img_size (int): Kích thước ảnh chuẩn hóa (không dùng trực tiếp ở đây nhưng để tham khảo).
        test_size (float): Tỷ lệ chia tập kiểm tra (0.2 = 20%).
    """
    all_data = []
    print(f"--- XỬ LÝ DỮ LIỆU: HAAR CASCADE + PADDING {PADDING_RATIO*100}% + SAFE CROP ---")

    for root, _, files in os.walk(raw_dir):
        for fname in files:
            if not (fname.lower().endswith('.jpg') or fname.lower().endswith('.png')): continue
            
            img_path = os.path.join(root, fname)
            pts_path = os.path.splitext(img_path)[0] + '.pts'
            if not os.path.exists(pts_path): continue

            img = cv2.imread(img_path)
            if img is None: continue
            
            landmarks = _read_pts_file(pts_path)
            if len(landmarks) != 68: continue

            # 1. Tìm mặt bằng Haar Cascade
            # Sử dụng thuật toán này để đồng bộ hóa cách cắt ảnh với lúc chạy webcam
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0: continue
            
            # Lấy mặt lớn nhất nếu tìm thấy nhiều mặt
            bx, by, bw, bh = max(faces, key=lambda item: item[2] * item[3])

            # 2. Tính Box Vuông + Padding (Kỹ thuật Squaring)
            # Đảm bảo ảnh cắt ra là hình vuông để tránh bị méo khi resize về 128x128
            max_side = max(bw, bh)
            pad = int(max_side * PADDING_RATIO)
            square_side = max_side + 2 * pad
            
            # Tính tâm của hộp khuôn mặt
            center_x = bx + bw // 2
            center_y = by + bh // 2
            
            # Tọa độ hộp lý tưởng (có thể âm)
            x1 = int(center_x - square_side // 2)
            y1 = int(center_y - square_side // 2)
            
            # 3. Chuẩn hóa Landmark theo hộp lý tưởng này
            # Công thức: (Tọa độ thực - Góc trái hộp) / Kích thước hộp
            lm_norm = landmarks.copy()
            lm_norm[:, 0] = (lm_norm[:, 0] - x1) / square_side
            lm_norm[:, 1] = (lm_norm[:, 1] - y1) / square_side
            
            # Lọc dữ liệu nhiễu (nếu điểm bay quá xa khỏi hộp)
            if np.any(lm_norm < -0.1) or np.any(lm_norm > 1.1): continue

            all_data.append({
                'path': os.path.relpath(img_path, raw_dir),
                'bbox': (x1, y1, square_side, square_side),
                'landmarks': lm_norm.flatten()
            })

    if not all_data: raise ValueError("Không tìm thấy dữ liệu!")
    
    print(f"-> Đã chọn lọc {len(all_data)} mẫu.")
    train_set, test_set = train_test_split(all_data, test_size=test_size, random_state=42)

    def save_xml(data, path):
        """Hàm hỗ trợ lưu danh sách dữ liệu vào file XML."""
        root = ET.Element('dataset')
        images = ET.SubElement(root, 'images')
        for item in data:
            img_node = ET.SubElement(images, 'image', file=item['path'])
            x, y, w, h = item['bbox']
            # Lưu tọa độ hộp cắt (box vuông) để dùng khi load dữ liệu
            box = ET.SubElement(img_node, 'box', top=str(y), left=str(x), width=str(w), height=str(h))
            for i, val in enumerate(item['landmarks']):
                is_x = (i % 2 == 0)
                if is_x:
                    part = ET.SubElement(box, 'part', name=f"{i//2}", x=f"{val:.6f}")
                else:
                    part.set('y', f"{val:.6f}")
        ET.ElementTree(root).write(path)

    os.makedirs(output_dir, exist_ok=True)
    save_xml(train_set, os.path.join(output_dir, 'train.xml'))
    save_xml(test_set, os.path.join(output_dir, 'test.xml'))
    print("✅ Đã tạo file XML mới.")

def load_and_preprocess_data(xml_path, img_root, target_size):
    """
    Tải dữ liệu từ file XML, thực hiện cắt ảnh và chuẩn hóa pixel.

    Sử dụng kỹ thuật 'Safe Crop' (Canvas đen) để xử lý các hộp cắt tràn viền,
    đảm bảo tỉ lệ khuôn mặt không bị biến dạng.

    Args:
        xml_path (str): Đường dẫn file XML.
        img_root (str): Thư mục gốc chứa ảnh.
        target_size (int): Kích thước ảnh đầu ra (128).

    Returns:
        tuple: (Mảng ảnh X, Mảng nhãn y)
    """
    X_data, y_data = [], []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    print(f"--- Đang tải {xml_path} ---")
    
    for img_node in root.find('images'):
        rel_path = img_node.get('file')
        full_path = os.path.join(img_root, rel_path)
        if not os.path.exists(full_path): continue
        
        original_img = cv2.imread(full_path)
        if original_img is None: continue
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        box = img_node.find('box')
        if box is None: continue
        
        # Tọa độ hộp lý tưởng (có thể âm)
        x = int(float(box.get('left')))
        y = int(float(box.get('top')))
        w = int(float(box.get('width')))
        h = int(float(box.get('height')))
        
        # --- KỸ THUẬT SAFE CROP (CANVAS ĐEN) ---
        # 1. Tạo nền đen chuẩn kích thước w*h
        face_crop = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 2. Tính toán vùng giao nhau giữa hộp và ảnh gốc
        src_x1 = max(0, x)
        src_y1 = max(0, y)
        src_x2 = min(original_img.shape[1], x + w)
        src_y2 = min(original_img.shape[0], y + h)
        
        # 3. Tính toán vị trí dán lên nền đen
        dst_x1 = src_x1 - x
        dst_y1 = src_y1 - y
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        # Kiểm tra nếu vùng giao hợp lệ
        if src_x2 <= src_x1 or src_y2 <= src_y1: continue
        
        # 4. Copy ảnh vào nền đen
        face_crop[dst_y1:dst_y2, dst_x1:dst_x2] = original_img[src_y1:src_y2, src_x1:src_x2]
        
        # --- KẾT THÚC SAFE CROP ---
        
        # Resize về 128x128
        face_resized = cv2.resize(face_crop, (target_size, target_size))
        
        lms = []
        for part in box.findall('part'):
            lms.append(float(part.get('x')))
            lms.append(float(part.get('y')))
            
        if len(lms) != 136: continue
        lm_arr = np.array(lms, dtype=np.float32).reshape(-1, 2)
        
        # Thêm dữ liệu gốc vào mảng (Chuẩn hóa pixel 0-1)
        X_data.append(face_resized / 255.0)
        y_data.append(lm_arr.flatten())
        
        # Tăng cường dữ liệu (Lật ảnh) để tăng độ đa dạng
        flip_img, flip_lm = _augment_flip(face_resized, lm_arr)
        X_data.append(flip_img / 255.0)
        y_data.append(flip_lm.flatten())

    return np.array(X_data, dtype=np.float32), np.array(y_data, dtype=np.float32)