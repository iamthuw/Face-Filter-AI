import os
import glob
import cv2
from tqdm import tqdm

from preprocessing.read_pts import read_pts_file
from preprocessing.resize_and_align import resize_image_and_landmarks
from preprocessing.augment import augment_image
from preprocessing.enhance_image import enhance_image
from preprocessing.io_utils import save_image_and_landmarks
from preprocessing.split_dataset import split_dataset
from preprocessing.generate_xml import create_xml

def process_dataset(raw_dir='data/raw', processed_dir='data/processed', size=(256, 256)):
    pts_files = glob.glob(os.path.join(raw_dir, '**/*.pts'), recursive=True)
    print(f"🔍 Tìm thấy {len(pts_files)} file .pts. Bắt đầu xử lý...\n")
    all_data = []

    for pts_path in tqdm(
        pts_files,
        desc="Đang xử lý dataset",
        unit="file",
        ncols=80,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} files'
    ):
        img_path = pts_path.replace('.pts', '.jpg')
        if not os.path.exists(img_path):
            continue

        landmarks = read_pts_file(pts_path)
        if not landmarks:
            continue

        # Resize + scale landmarks (ảnh màu)
        img_resized, scaled_landmarks = resize_image_and_landmarks(img_path, landmarks, size)

        # Chuyển sang ảnh BGR (đề phòng ảnh đầu vào là grayscale)
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

        # Enhance ảnh + grayscale
        img_enhanced = enhance_image(img_resized)

        # Augmentation
        img_aug, scaled_landmarks = augment_image(img_enhanced, scaled_landmarks)

        # Lưu ảnh và landmarks
        filename = os.path.basename(img_path)
        img_saved_path = save_image_and_landmarks(processed_dir, filename, img_aug, scaled_landmarks)
        all_data.append({'file': img_saved_path, 'landmarks': scaled_landmarks})

    # Chia train/test
    train_data, test_data = split_dataset(all_data)
    create_xml(train_data, os.path.join(processed_dir, 'train.xml'))
    create_xml(test_data, os.path.join(processed_dir, 'test.xml'))

    print(f"\n✅ Hoàn tất! {len(train_data)} ảnh train, {len(test_data)} ảnh test.")
    print(f"📁 Dữ liệu lưu tại: {processed_dir}/images và {processed_dir}/landmarks")

if __name__ == '__main__':
    process_dataset()
