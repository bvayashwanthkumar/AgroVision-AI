import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0


from config import *
from preprocess import train_dataset, validation_dataset, class_names

# =====================================================
# Create Output Directories
# =====================================================

os.makedirs("saved_model", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/history", exist_ok=True)

# =====================================================
# Number of Classes
# =====================================================

NUM_CLASSES = len(class_names)

print("=" * 60)
print("Building EfficientNetB0 Model...")
print("=" * 60)

# =====================================================
# Data Augmentation
# =====================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# =====================================================
# Base Model
# =====================================================

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# =====================================================
# Build Model
# =====================================================

inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = Model(inputs, outputs)

model.summary()

# =====================================================
# Compile Model
# =====================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================================================
# Callbacks
# =====================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "saved_model/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    verbose=1
)

csv_logger = tf.keras.callbacks.CSVLogger(
    "outputs/history/training_history.csv"
)

# =====================================================
# Train
# =====================================================

print("\nStarting Training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr,
        csv_logger
    ]
)

print("\nTraining Completed Successfully!")

# =====================================================
# Save Final Model
# =====================================================

model.save("saved_model/final_model.keras")

print("\nFinal Model Saved!")

# =====================================================
# Accuracy Graph
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.savefig("outputs/plots/accuracy.png")

plt.close()

# =====================================================
# Loss Graph
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.savefig("outputs/plots/loss.png")

plt.close()

print("\nGraphs Saved!")

print("\nProject Training Completed Successfully!")