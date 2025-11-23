# train.py
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from config import *
from model_architecture import create_mobilenet_model
from preprocess import process_and_split_data, load_and_preprocess_data

def train():
    # 1. Tạo lại XML (BẮT BUỘC CHẠY LẠI ĐỂ CẬP NHẬT BOX VUÔNG)
    print("--- STEP 1: Preparing Data ---")
    try:
        process_and_split_data(RAW_DATA_DIR, PREPROCESSED_DIR, INPUT_WIDTH, TEST_SPLIT)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Load dữ liệu
    print("--- STEP 2: Loading Data ---")
    X_train, y_train = load_and_preprocess_data(TRAIN_XML_PATH, RAW_DATA_DIR, INPUT_WIDTH)
    X_val, y_val = load_and_preprocess_data(TEST_XML_PATH, RAW_DATA_DIR, INPUT_WIDTH)
    print(f"Train set: {X_train.shape}, Val set: {X_val.shape}")

    # 3. Model
    model = create_mobilenet_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='mse', metrics=['mae'])

    # 4. Callbacks
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    # Giảm LR nếu loss không giảm sau 5 epoch
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    # 5. Train
    print("--- STEP 3: Training ---")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    print("--- DONE ---")

if __name__ == '__main__':
    train()