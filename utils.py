import cv2
import numpy as np
import tensorflow as tf
import os

IMG_SIZE = 224
# Categories will be determined by the model's output shape
model = tf.keras.models.load_model('model/plant_disease_model_with_unknown_class.h5')

# Get the number of classes from the model's output layer
NUM_CLASSES = model.layers[-1].units

# Initialize empty categories list - we'll populate it when we get the first prediction
CATEGORIES = []

TREATMENTS = {
    'Pepper__bell___Bacterial_spot': 'Apply copper-based bactericides and practice crop rotation. Remove and destroy infected plants. Avoid overhead watering.',
    'Pepper__bell___healthy': 'Your pepper plant is healthy! Continue regular maintenance and monitoring.',
    'Potato___Early_blight': 'Apply fungicides containing chlorothalonil or mancozeb. Practice crop rotation and remove infected leaves. Ensure proper spacing for air circulation.',
    'Potato___Late_blight': 'Apply fungicides with active ingredients like chlorothalonil or mancozeb. Remove and destroy infected plants. Avoid overhead watering.',
    'Potato___healthy': 'Your potato plant is healthy! Continue regular maintenance and monitoring.',
    'Tomato_Bacterial_spot': 'Apply copper-based bactericides. Remove infected leaves and avoid overhead watering. Practice crop rotation.',
    'Tomato_Early_blight': 'Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves and maintain proper plant spacing.',
    'Tomato_Late_blight': 'Apply fungicides with active ingredients like chlorothalonil or mancozeb. Remove and destroy infected plants. Avoid overhead watering.',
    'Tomato_Leaf_Mold': 'Apply fungicides containing chlorothalonil or mancozeb. Improve air circulation and reduce humidity. Remove infected leaves.',
    'Tomato_Septoria_leaf_spot': 'Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves and avoid overhead watering.',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Apply miticides or insecticidal soaps. Increase humidity and use predatory mites. Remove heavily infested leaves.',
    'Tomato__Target_Spot': 'Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves and maintain proper plant spacing.',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Remove and destroy infected plants. Control whitefly populations using insecticides or sticky traps. Use virus-resistant varieties.',
    'Tomato__Tomato_mosaic_virus': 'Remove and destroy infected plants. Control aphid populations. Use virus-resistant varieties and practice good sanitation.',
    'Tomato_healthy': 'Your tomato plant is healthy! Continue regular maintenance and monitoring.'
}

def calculate_severity(img, disease_type):
    """
    Calculate disease severity based on the type of disease and image analysis.
    Returns a severity percentage and a description of the severity level.
    """
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Define color ranges for different types of symptoms
    color_ranges = {
        'bacterial': {
            'yellow': [(20, 50, 50), (30, 255, 255)],
            'brown': [(10, 50, 50), (20, 255, 255)],
            'black': [(0, 0, 0), (180, 255, 30)]
        },
        'fungal': {
            'yellow': [(20, 50, 50), (30, 255, 255)],
            'brown': [(10, 50, 50), (20, 255, 255)],
            'white': [(0, 0, 200), (180, 30, 255)]
        },
        'viral': {
            'yellow': [(20, 50, 50), (30, 255, 255)],
            'mosaic': [(0, 0, 0), (180, 255, 255)]
        }
    }
    
    # Determine disease category
    if 'bacterial' in disease_type.lower():
        ranges = color_ranges['bacterial']
    elif 'mosaic' in disease_type.lower() or 'virus' in disease_type.lower():
        ranges = color_ranges['viral']
    else:
        ranges = color_ranges['fungal']
    
    # Calculate affected area for each color range
    total_affected = 0
    for color_name, (lower, upper) in ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        total_affected += np.sum(mask > 0)
    
    # Calculate severity percentage
    severity_percent = round((total_affected / (img.shape[0] * img.shape[1])) * 100, 2)
    
    # Determine severity level
    if severity_percent < 5:
        severity_level = "Very Low"
    elif severity_percent < 15:
        severity_level = "Low"
    elif severity_percent < 30:
        severity_level = "Moderate"
    elif severity_percent < 50:
        severity_level = "High"
    else:
        severity_level = "Very High"
    
    return severity_percent, severity_level

def predict_disease(img_path):
    try:
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Could not read image at: {img_path}")
            return "Invalid Image", 0, 0, "No treatment available."

        # Convert BGR to RGB (MobileNetV2 expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize and normalize
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_array = np.expand_dims(img_resized, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array, verbose=0)
        print(f"✅ Raw prediction output: {prediction}")

        # If categories haven't been populated yet, get them from the training data
        global CATEGORIES
        if not CATEGORIES:
            # Get the class names from the training data directory
            data_dir = '/Users/siddheshm/Downloads/PlantVillage'
            CATEGORIES = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
            print(f"📋 Loaded categories: {CATEGORIES}")

        if prediction.shape[1] != len(CATEGORIES):
            print(f"⚠️ Prediction shape ({prediction.shape[1]}) doesn't match categories ({len(CATEGORIES)})!")
            return "Prediction Error", 0, 0, "No treatment available."

        index = np.argmax(prediction[0])
        confidence = round(prediction[0][index] * 100, 2)
        label = CATEGORIES[index]

        # If the prediction is "unknown", return without severity and treatment
        if label.lower() == "unknown":
            return label, confidence, 0, "Unable to identify the disease. Please try with a clearer image of the affected leaf."

        # Calculate severity
        severity_percent, severity_level = calculate_severity(img, label)
        
        # Get treatment
        treatment = TREATMENTS.get(label, 'No treatment available.')
        
        # Add severity level to the treatment if not healthy
        if 'healthy' not in label.lower():
            treatment = f"Severity Level: {severity_level}\n\n{treatment}"

        return label, confidence, severity_percent, treatment

    except Exception as e:
        print(f"❌ Exception during prediction: {e}")
        return "Error", 0, 0, "No treatment available."
