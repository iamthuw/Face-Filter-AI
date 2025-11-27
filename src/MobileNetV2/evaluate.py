import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import *
from preprocess import load_and_preprocess_data

def evaluate():
    if not os.path.exists(MODEL_SAVE_PATH): return
    
    X_test, y_test = load_and_preprocess_data(TEST_XML_PATH, RAW_DATA_DIR, INPUT_WIDTH)
    
    custom_objects = {'mse': tf.keras.metrics.MeanSquaredError, 'mae': tf.keras.metrics.MeanAbsoluteError}
    model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects=custom_objects, compile=False)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MSE: {mse:.5f}")
    print(f"MAE: {mae:.5f}")
    print(f"R2 Score: {r2:.4f}")

if __name__ == '__main__':
    evaluate()
