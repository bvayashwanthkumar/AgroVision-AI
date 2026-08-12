import tensorflow as tf
from config import *

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

# ===============================
# Training Dataset
# ===============================

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# ===============================
# Validation Dataset
# ===============================

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# ===============================
# Save Class Names BEFORE optimization
# ===============================

class_names = train_dataset.class_names

print("\nDataset Loaded Successfully\n")
print(f"Total Classes : {len(class_names)}")

# ===============================
# Optimize Dataset
# ===============================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = (
    train_dataset
    .cache()
    .shuffle(1000)
    .prefetch(buffer_size=AUTOTUNE)
)

validation_dataset = (
    validation_dataset
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)