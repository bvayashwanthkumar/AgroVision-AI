import numpy as np
import tensorflow as tf
from PIL import Image

from class_names import CLASS_NAMES


MODEL_PATH = "saved_model/best_model.keras"
IMAGE_SIZE = (224, 224)


print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded!")


# ==========================================================
# Get layers
# ==========================================================

base_model = model.get_layer(
    "efficientnetb0"
)

augmentation = model.get_layer(
    "sequential"
)

top_conv = base_model.get_layer(
    "top_conv"
)

top_bn = base_model.get_layer(
    "top_bn"
)

top_activation = base_model.get_layer(
    "top_activation"
)

global_pool = model.get_layer(
    "global_average_pooling2d"
)

dropout = model.get_layer(
    "dropout"
)

dense = model.get_layer(
    "dense"
)


# ==========================================================
# Load image
# ==========================================================

image_path = "test_images/potato1.jpg"

image = Image.open(
    image_path
).convert("RGB")

image = image.resize(
    IMAGE_SIZE
)

image_array = np.array(
    image,
    dtype=np.float32
)

image_tensor = tf.convert_to_tensor(
    image_array,
    dtype=tf.float32
)

image_tensor = tf.expand_dims(
    image_tensor,
    axis=0
)


# ==========================================================
# Prediction using ORIGINAL MODEL
# ==========================================================

original_prediction = model(
    image_tensor,
    training=False
)

original_index = int(
    tf.argmax(
        original_prediction[0]
    ).numpy()
)

original_confidence = float(
    original_prediction[
        0,
        original_index
    ].numpy()
) * 100


print("\n========================================")
print("ORIGINAL MODEL")
print("========================================")

print(
    "Class:",
    CLASS_NAMES[original_index]
)

print(
    "Confidence:",
    round(
        original_confidence,
        2
    ),
    "%"
)


# ==========================================================
# Apply exact outer Sequential layer
# ==========================================================

processed = augmentation(
    image_tensor,
    training=False
)


# ==========================================================
# EfficientNet feature extraction
# ==========================================================

feature_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=[
        top_conv.output,
        base_model.output
    ]
)


conv_output, efficientnet_output = feature_model(
    processed,
    training=False
)


# ==========================================================
# Recreate classification head
# ==========================================================

head_output = global_pool(
    efficientnet_output
)

head_output = dropout(
    head_output,
    training=False
)

head_output = dense(
    head_output
)


# ==========================================================
# Compare reconstructed prediction
# ==========================================================

reconstructed_index = int(
    tf.argmax(
        head_output[0]
    ).numpy()
)

reconstructed_confidence = float(
    head_output[
        0,
        reconstructed_index
    ].numpy()
) * 100


print("\n========================================")
print("RECONSTRUCTED MODEL")
print("========================================")

print(
    "Class:",
    CLASS_NAMES[reconstructed_index]
)

print(
    "Confidence:",
    round(
        reconstructed_confidence,
        2
    ),
    "%"
)


# ==========================================================
# Compare EfficientNet output
# ==========================================================

print("\n========================================")
print("COMPARISON")
print("========================================")

print(
    "Original prediction:",
    CLASS_NAMES[original_index]
)

print(
    "Reconstructed prediction:",
    CLASS_NAMES[reconstructed_index]
)

print(
    "Original confidence:",
    round(
        original_confidence,
        2
    ),
    "%"
)

print(
    "Reconstructed confidence:",
    round(
        reconstructed_confidence,
        2
    ),
    "%"
)


if original_index == reconstructed_index:

    print(
        "\n✅ PREDICTIONS MATCH"
    )

else:

    print(
        "\n❌ PREDICTIONS DO NOT MATCH"
    )