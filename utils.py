import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 128
CATEGORIES = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'PlantVillage', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
model = tf.keras.models.load_model('model/plant_disease_model.h5')

TREATMENTS = {
    'Tomato___Bacterial_spot': 'Use copper-based bactericides and avoid overhead watering.',
    'Potato___Late_blight': 'Use fungicide with active ingredients like chlorothalonil or mancozeb.',
    'Tomato___Healthy': 'Your plant is healthy! Maintain good conditions.'
}

def predict_disease(img_path):
    try:
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Could not read image at: {img_path}")
            return "Invalid Image", 0, 0, "No treatment available."

        # Resize and normalize
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_array = np.expand_dims(img_resized, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)
        print(f"✅ Raw prediction output: {prediction}")

        if prediction.shape[1] != len(CATEGORIES):
            print("⚠️ Prediction shape doesn't match categories!")
            return "Prediction Error", 0, 0, "No treatment available."

        index = np.argmax(prediction[0])
        confidence = round(prediction[0][index] * 100, 2)
        label = CATEGORIES[index]

        # Estimate severity
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (10, 50, 50), (30, 255, 255))
        severity_percent = round((np.sum(mask > 0) / mask.size) * 100, 2)

        # Get treatment
        treatment = TREATMENTS.get(label, 'No treatment available.')

        return label, confidence, severity_percent, treatment

    except Exception as e:
        print(f"❌ Exception during prediction: {e}")
        return "Error", 0, 0, "No treatment available."
