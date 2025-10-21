import os
import cv2

def visualize_processed_images(processed_dir='data/processed/images',
                               landmarks_dir='data/processed/landmarks',
                               num_samples=20):
    """Hiển thị một số ảnh đã xử lý kèm landmarks."""
    image_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.jpg')])
    if not image_files:
        print("❌ Không tìm thấy ảnh trong thư mục processed.")
        return

    sample_files = image_files[:num_samples]

    for filename in sample_files:
        img_path = os.path.join(processed_dir, filename)
        landmark_path = os.path.join(landmarks_dir, filename.replace('.jpg', '.txt'))

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Không đọc được ảnh: {filename}")
            continue

        landmarks = []
        if os.path.exists(landmark_path):
            with open(landmark_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        x, y = map(float, line.strip().split())
                        landmarks.append((x, y))
                    except ValueError:
                        continue

        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for (x, y) in landmarks:
            cv2.circle(img_color, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=-1)

        cv2.imshow(f"Ảnh: {filename}", img_color)
        key = cv2.waitKey(0)
        if key == 27:
            break
        cv2.destroyWindow(f"Ảnh: {filename}")

    cv2.destroyAllWindows()
    print(f"✅ Hoàn tất hiển thị {len(sample_files)} ảnh và landmarks.")

if __name__ == '__main__':
    visualize_processed_images()
