# model_architecture.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from config import INPUT_HEIGHT, INPUT_WIDTH, NUM_LANDMARKS
import os

MOBILENET_WEIGHTS_PATH = os.path.join("models", "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")

def create_mobilenet_model(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), num_landmarks=NUM_LANDMARKS):
    # 1. Backbone
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False, 
        weights=None
    )
    
    # 2. Load weights
    if os.path.exists(MOBILENET_WEIGHTS_PATH):
        print("Loading local weights...")
        base_model.load_weights(MOBILENET_WEIGHTS_PATH, by_name=True, skip_mismatch=True)
    else:
        print("WARNING: Weights not found! Training from scratch.")

    # 3. Fine-tuning: Mở khóa từ Block 8 trở đi
    base_model.trainable = True
    FINE_TUNE_AT = 100 
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    
    # 4. Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_landmarks, activation='linear', name='landmarks')(x)
    
    return Model(inputs=base_model.input, outputs=predictions)
