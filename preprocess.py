# preprocess.py
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from config import *

# --- CẤU HÌNH ---
PADDING_RATIO = 0.25  # Padding 25% để lấy trán và cằm

# Khởi tạo bộ phát hiện khuôn mặt (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cấu hình lật ảnh
NUM_PTS = 68
SYMMETRICAL_LANDMARKS = [
    (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
    (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
    (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
    (31, 35), (32, 34),
    (48, 54), (49, 53), (50, 52), (55, 59), (56, 58), (60, 64), (61, 63)
]
FLIP_MAP = list(range(NUM_PTS))
for l, r in SYMMETRICAL_LANDMARKS:
    FLIP_MAP[l] = r
    FLIP_MAP[r] = l

def _read_pts_file(pts_path):
    """Đọc file .pts, bỏ qua header, trả về tọa độ gốc."""
    with open(pts_path, 'r') as f:
        lines = f.readlines()
    points = []
    start_reading = False
    for line in lines:
        line = line.strip()
        if line == '{': start_reading = True; continue
        if line == '}': break
        if start_reading and (line and (line[0].isdigit() or line.startswith('-'))):
            try:
                parts = line.split()
                if len(parts) >= 2: points.append([float(parts[0]), float(parts[1])])
            except ValueError: continue
    if len(points) == 68: return np.array(points, dtype=np.float32)
    return np.array([])

def _augment_flip(image, landmarks):
    img_flip = cv2.flip(image, 1)
    lm_flip = landmarks.copy()
    lm_flip[:, 0] = 1.0 - lm_flip[:, 0]
    lm_flip = lm_flip[FLIP_MAP]
    return img_flip, lm_flip

def process_and_split_data(raw_dir, output_dir, img_size=128, test_size=0.2):
    all_data = []
    print(f"--- XỬ LÝ DỮ LIỆU: DÙNG HAAR CASCADE ĐỂ ĐỒNG BỘ HÓA ---")
    print(f"--- PADDING: {PADDING_RATIO*100}% | LÀM VUÔNG: CÓ ---")

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

            # --- 1. TÌM FACE BOX BẰNG HAAR CASCADE ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0: continue
            
            # Chọn mặt lớn nhất
            bx, by, bw, bh = max(faces, key=lambda item: item[2] * item[3])

            # --- 2. ÁP DỤNG PADDING VÀ LÀM VUÔNG ---
            max_side = max(bw, bh)
            pad = int(max_side * PADDING_RATIO)
            square_side = max_side + 2 * pad
            center_x = bx + bw // 2
            center_y = by + bh // 2
            
            x1 = int(center_x - square_side // 2)
            y1 = int(center_y - square_side // 2)

            # --- 3. CHUẨN HÓA LANDMARK ---
            lm_norm = landmarks.copy()
            lm_norm[:, 0] = (lm_norm[:, 0] - x1) / square_side
            lm_norm[:, 1] = (lm_norm[:, 1] - y1) / square_side
            
            # --- 4. LỌC DỮ LIỆU ---
            if np.any(lm_norm < 0.0) or np.any(lm_norm > 1.0): continue

            all_data.append({
                'path': os.path.relpath(img_path, raw_dir),
                'bbox': (x1, y1, square_side, square_side),
                'landmarks': lm_norm.flatten()
            })

    if not all_data: raise ValueError(f"Không tạo được dữ liệu nào hợp lệ từ {raw_dir}")
    
    print(f"-> Đã chọn lọc được {len(all_data)} mẫu dữ liệu chất lượng cao.")
    train_set, test_set = train_test_split(all_data, test_size=test_size, random_state=42)

    def save_xml(data, path):
        root = ET.Element('dataset')
        images = ET.SubElement(root, 'images')
        for item in data:
            img_node = ET.SubElement(images, 'image', file=item['path'])
            x, y, w, h = item['bbox']
            box = ET.SubElement(img_node, 'box', top=str(y), left=str(x), width=str(w), height=str(h))
            
            # --- ĐÃ SỬA LỖI TẠI ĐÂY ---
            # Tách rõ ràng if/else để gán biến 'part'
            for i, val in enumerate(item['landmarks']):
                is_x = (i % 2 == 0)
                if is_x:
                    # Tạo thẻ mới và gán vào biến part
                    part = ET.SubElement(box, 'part', name=f"{i//2}", x=f"{val:.6f}")
                else:
                    # Dùng biến part của vòng lặp trước để set y
                    part.set('y', f"{val:.6f}")
                    
        ET.ElementTree(root).write(path)

    os.makedirs(output_dir, exist_ok=True)
    save_xml(train_set, os.path.join(output_dir, 'train.xml'))
    save_xml(test_set, os.path.join(output_dir, 'test.xml'))
    print("✅ Đã tạo file XML mới (Đồng bộ Haar Cascade) thành công.")

def load_and_preprocess_data(xml_path, img_root, target_size):
    X_data, y_data = [], []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    print(f"--- Đang tải dữ liệu từ {xml_path} ---")
    
    for img_node in root.find('images'):
        rel_path = img_node.get('file')
        full_path = os.path.join(img_root, rel_path)
        if not os.path.exists(full_path): continue
        original_img = cv2.imread(full_path)
        if original_img is None: continue
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        box = img_node.find('box')
        if box is None: continue
        
        x = int(float(box.get('left'))); y = int(float(box.get('top')))
        w = int(float(box.get('width'))); h = int(float(box.get('height')))
        
        # Cắt ảnh an toàn
        face_crop = np.zeros((h, w, 3), dtype=np.uint8)
        src_x1 = max(0, x); src_y1 = max(0, y)
        src_x2 = min(original_img.shape[1], x + w); src_y2 = min(original_img.shape[0], y + h)
        dst_x1 = src_x1 - x; dst_y1 = src_y1 - y
        dst_x2 = dst_x1 + (src_x2 - src_x1); dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        if src_x2 <= src_x1 or src_y2 <= src_y1: continue
        face_crop[dst_y1:dst_y2, dst_x1:dst_x2] = original_img[src_y1:src_y2, src_x1:src_x2]
        
        face_resized = cv2.resize(face_crop, (target_size, target_size))
        
        lms = []
        for part in box.findall('part'):
            lms.append(float(part.get('x'))); lms.append(float(part.get('y')))
        if len(lms) != 136: continue
        lm_arr = np.array(lms, dtype=np.float32).reshape(-1, 2)
        
        X_data.append(face_resized / 255.0)
        y_data.append(lm_arr.flatten())
        flip_img, flip_lm = _augment_flip(face_resized, lm_arr)
        X_data.append(flip_img / 255.0)
        y_data.append(flip_lm.flatten())

    return np.array(X_data, dtype=np.float32), np.array(y_data, dtype=np.float32)