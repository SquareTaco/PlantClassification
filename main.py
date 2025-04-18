#-----create kivy_venv:
#python -m venv kivy_venv

#-----activate venv:
#kivy_venv\Scripts\activate

#-----install tensorflow and some other things:
#pip install tensorflow opencv-python numpy matplotlib

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to the saved model (replace with your actual model path)
model_path = "C:/Users/ijtyl/OneDrive/Desktop/PlantClassification/PlantClassification.h5"

# Load the model from the saved file
model = tf.keras.models.load_model(model_path)

# Path to the image you want to predict (replace with your actual image path)
image_path = "C:/Users/ijtyl/OneDrive/Desktop/PlantClassification/vera.jpg"
# Define label_mapping (the classes from your dataset)
label_mapping = {
    'aloevera': 0, 'banana': 1, 'bilimbi': 2, 'cantaloupe': 3, 'cassava': 4, 'coconut': 5,
    'corn': 6, 'cucumber': 7, 'curcuma': 8, 'eggplant': 9, 'galangal': 10, 'ginger': 11,
    'guava': 12, 'kale': 13, 'longbeans': 14, 'mango': 15, 'melon': 16, 'orange': 17,
    'paddy': 18, 'papaya': 19, 'peper chili': 20, 'pineapple': 21, 'pomelo': 22, 'shallot': 23,
    'soybeans': 24, 'spinach': 25, 'sweet potatoes': 26, 'tobacco': 27, 'waterapple': 28, 'watermelon': 29
}

# Load and preprocess the image
image = cv2.imread(image_path)
image = cv2.resize(image, (128, 128))  # Adjust size to match model input
image = image.astype('float32') / 255.0  # Normalize the image
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Make a prediction
predicted_label = model.predict(image)
predicted_label_idx = np.argmax(predicted_label, axis=1)[0]  # Get the index of the predicted class

# Get the class name from the label mapping (ensure label_mapping is available)
predicted_class_name = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_label_idx)]

# Display the image and prediction result
image_display = cv2.imread(image_path)
image_display = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for display
plt.imshow(image_display)
plt.title(f"Predicted: {predicted_class_name}")
plt.axis('off')
plt.show()

# Print the prediction
print(f"Predicted Label (class name): {predicted_class_name}")