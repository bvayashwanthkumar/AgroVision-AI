import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from class_names import CLASS_NAMES

# =====================================================
# Load Trained Model
# =====================================================

MODEL_PATH = "saved_model/best_model.keras"

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# =====================================================
# Image Size
# =====================================================

IMAGE_SIZE = (224, 224)

# =====================================================
# Prediction Function
# =====================================================

def predict_image(image_path):

    # Load Image
    img = image.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    # Convert Image to Array
    img_array = image.img_to_array(img)

    # Add Batch Dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)

    probabilities = predictions[0]

    predicted_index = np.argmax(probabilities)

    predicted_class = CLASS_NAMES[predicted_index]

    # Split Plant & Disease
    parts = predicted_class.split("___")

    plant = parts[0].replace("_", " ")

    if len(parts) > 1:
        disease = parts[1].replace("_", " ")
    else:
        disease = "Healthy"

    confidence = float(probabilities[predicted_index] * 100)

    # =====================================================
    # Top 3 Predictions
    # =====================================================

    top3_indices = np.argsort(probabilities)[-3:][::-1]

    top3 = []

    for idx in top3_indices:

        class_name = CLASS_NAMES[idx]

        class_parts = class_name.split("___")

        plant_name = class_parts[0].replace("_", " ")

        if len(class_parts) > 1:
            disease_name = class_parts[1].replace("_", " ")
        else:
            disease_name = "Healthy"

        top3.append({
            "plant": plant_name,
            "disease": disease_name,
            "confidence": round(float(probabilities[idx] * 100), 2)
        })

    return {
        "plant": plant,
        "disease": disease,
        "prediction": predicted_class,
        "confidence": round(confidence, 2),
        "top3": top3
    }


# =====================================================
# Test Prediction
# =====================================================

if __name__ == "__main__":

    image_path = input("Enter image path: ")

    result = predict_image(image_path)

    print("\n" + "=" * 50)
    print("Prediction Result")
    print("=" * 50)

    print(f"\nPlant      : {result['plant']}")
    print(f"Disease    : {result['disease']}")
    print(f"Confidence : {result['confidence']} %")

    print("\nTop 3 Predictions")
    print("-" * 50)

    for i, item in enumerate(result["top3"], start=1):

        print(
            f"{i}. {item['plant']} - {item['disease']} "
            f"({item['confidence']}%)"
        )