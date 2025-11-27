import os

# --- ĐƯỜNG DẪN DỮ LIỆU ---
RAW_DATA_DIR = "/Users/nguyenbaothach/Study/AI/MobileNet/data/raw" 

PREPROCESSED_DIR = "data/preprocessed"
TRAIN_XML_PATH = os.path.join(PREPROCESSED_DIR, 'train.xml')
TEST_XML_PATH = os.path.join(PREPROCESSED_DIR, 'test.xml')

# --- THAM SỐ ẢNH ---
INPUT_WIDTH = 128
INPUT_HEIGHT = 128
NUM_LANDMARKS = 136  # 68 điểm * 2

# --- THAM SỐ HUẤN LUYỆN ---
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001 # Khởi điểm, sẽ tự giảm
TEST_SPLIT = 0.2

# --- LƯU TRỮ MÔ HÌNH ---
MODEL_DIR = "models"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "landmark_detector.h5")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- WEBCAM ---
WEBCAM_ID = 0
