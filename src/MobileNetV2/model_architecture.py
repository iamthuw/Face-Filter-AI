import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten # <-- Thêm Flatten
from tensorflow.keras.models import Model
from config import INPUT_HEIGHT, INPUT_WIDTH, NUM_LANDMARKS
import os

MOBILENET_WEIGHTS_PATH = os.path.join("models", "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")

def create_mobilenet_model(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), num_landmarks=NUM_LANDMARKS):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False, 
        weights=None
    )
    
    if os.path.exists(MOBILENET_WEIGHTS_PATH):
        print("Loading local weights...")
        base_model.load_weights(MOBILENET_WEIGHTS_PATH, by_name=True, skip_mismatch=True)
    
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # 4. Head (THAY ĐỔI QUAN TRỌNG)
    x = base_model.output
    
    # Thay GlobalAveragePooling bằng Flatten để giữ thông tin không gian
    x = Flatten()(x) 
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x) 
    
    predictions = Dense(num_landmarks, activation='linear', name='landmarks')(x)
    
    return Model(inputs=base_model.input, outputs=predictions)
