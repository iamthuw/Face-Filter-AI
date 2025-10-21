import cv2
import numpy as np
import os

def denoise_image(img):
    """Khử nhiễu bằng Non-local Means Denoising"""
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def equalize_lighting_gray(img, clip_limit=1.5):
    """Cân bằng sáng (grayscale) bằng CLAHE"""
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    img_eq = clahe.apply(img_gray)
    return img_eq

def gamma_correction(img, gamma=1.0):
    """Gamma correction để tránh quá sáng/tối"""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def enhance_image(img):
    """Enhance ảnh, trả về grayscale uint8"""
    img = denoise_image(img)
    img = equalize_lighting_gray(img, clip_limit=1.5)
    img = gamma_correction(img, gamma=1.1)
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def process_folder(input_dir, output_dir):
    """Xử lý toàn bộ ảnh trong folder"""
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]

    for i, filename in enumerate(files, 1):
        path = os.path.join(input_dir, filename)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Không đọc được ảnh: {filename}")
            continue

        enhanced = enhance_image(img)
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, enhanced)

        if i % 100 == 0 or i == len(files):
            print(f"✅ Đã xử lý {i}/{len(files)} ảnh")

if __name__ == "__main__":
    process_folder("data/processed/images", "data/processed/enhanced")
    print("🎉 Hoàn tất tăng chất lượng ảnh!")
