# model.py (Kiến trúc Vision Transformer cho Landmark Detection)
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Hằng số mô hình ViT ---
PATCH_SIZE = 16 
NUM_HEADS = 8
PROJECTION_DIM = 64
TRANSFORMER_LAYERS = 4
MLP_HIDDEN_DIM = 512 # Kích thước cho MLP Head trong Transformer block

# ====================================================================
# LỚP TÙY CHỈNH CHO VIỆC TÁI TẠO MÔ HÌNH (Patches, PatchEncoder)
# ====================================================================
# Các hàm này trả về định nghĩa lớp (Class), không phải thể hiện (Instance).
# Điều này rất quan trọng để Keras có thể lưu và tải mô hình đúng cách.

def _create_patches_class(image_size, patch_size):
    """
    Tạo và trả về lớp Patches (Class).
    Patches chia ảnh thành các mảng con (patches).
    """
    class Patches(layers.Layer):
        def __init__(self, patch_size_val, **kwargs): # Đổi tên biến để tránh trùng
            super(Patches, self).__init__(**kwargs)
            self.patch_size = patch_size_val

        def call(self, images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches
        
        def get_config(self):
            config = super(Patches, self).get_config()
            config.update({"patch_size": self.patch_size})
            return config

    return Patches # Trả về CLASS

def _create_patch_encoder_class(image_size, projection_dim, num_patches):
    """
    Tạo và trả về lớp PatchEncoder (Class).
    PatchEncoder nhúng các patch và thêm vị trí nhúng.
    """
    class PatchEncoder(layers.Layer):
        def __init__(self, num_patches_val, projection_dim_val, **kwargs): # Đổi tên biến
            super(PatchEncoder, self).__init__(**kwargs)
            self.num_patches = num_patches_val
            self.projection_dim = projection_dim_val # Lưu lại để get_config
            self.projection = layers.Dense(units=projection_dim_val)
            self.position_embedding = layers.Embedding(
                input_dim=num_patches_val, output_dim=projection_dim_val
            )

        def call(self, patch):
            encoded_patches = self.projection(patch)
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            encoded_patches += self.position_embedding(positions)
            return encoded_patches
        
        def get_config(self):
            config = super(PatchEncoder, self).get_config()
            config.update({
                "num_patches": self.num_patches,
                "projection_dim": self.projection_dim,
            })
            return config

    return PatchEncoder # Trả về CLASS

# ====================================================================
# HÀM CHÍNH ĐỂ XÂY DỰNG MÔ HÌNH VIT
# ====================================================================

def build_landmark_model(input_shape, num_landmarks):
    """
    Xây dựng kiến trúc mô hình Vision Transformer (ViT) cho landmark detection.
    
    Args:
        input_shape (tuple): Kích thước đầu vào của ảnh (ví dụ: (128, 128, 3)).
        num_landmarks (int): Số lượng điểm mốc cần dự đoán (ví dụ: 68).
        
    Returns:
        tf.keras.Model: Mô hình ViT đã được xây dựng.
    """
    image_size = input_shape[0]
    num_patches = (image_size // PATCH_SIZE) ** 2

    # Lấy các ĐỊNH NGHĨA LỚP (Classes)
    Patches_Class = _create_patches_class(image_size, PATCH_SIZE)
    PatchEncoder_Class = _create_patch_encoder_class(image_size, PROJECTION_DIM, num_patches)

    # Đầu vào của mô hình
    inputs = layers.Input(shape=input_shape)

    # 1. Patching & Embedding
    # Khởi tạo thể hiện của lớp Patches và gọi nó
    patches = Patches_Class(patch_size_val=PATCH_SIZE)(inputs) 
    # Khởi tạo thể hiện của lớp PatchEncoder và gọi nó
    encoded_patches = PatchEncoder_Class(num_patches_val=num_patches, projection_dim_val=PROJECTION_DIM)(patches) 
    
    # 2. Transformer Encoder Blocks
    # Tạo các khối Transformer Encoder
    for _ in range(TRANSFORMER_LAYERS):
        # Layer Normalization trước Self-Attention (Pre-Layer Norm)
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Multi-Head Self-Attention
        attention = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1)(x1, x1)
        # Kết nối còn lại (Residual Connection) và Dropout
        x2 = layers.Add()([attention, encoded_patches])

        # Layer Normalization trước MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP (Feed-Forward Network)
        x3 = layers.Dense(MLP_HIDDEN_DIM, activation=tf.nn.gelu)(x3)
        x3 = layers.Dense(PROJECTION_DIM)(x3)
        x3 = layers.Dropout(0.1)(x3)
        # Kết nối còn lại
        encoded_patches = layers.Add()([x3, x2])

    # 3. Pooling và Prediction Head
    # Sử dụng Layer Normalization cuối cùng
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Global Average Pooling trên các patch embeddings
    representation = layers.GlobalAveragePooling1D()(representation) 

    # Các lớp Dense cho đầu ra dự đoán điểm mốc
    x = layers.Dense(512, activation=tf.nn.gelu)(representation)
    x = layers.Dropout(0.3)(x)
    # Output: num_landmarks * 2 (ví dụ: 68 * 2 = 136 tọa độ)
    output = layers.Dense(num_landmarks * 2)(x) 

    # Tạo và trả về mô hình Keras
    model = models.Model(inputs=inputs, outputs=output)
    return model