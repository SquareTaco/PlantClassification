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
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
import tkinter as tk
from tkinter import filedialog

model_path = "PlantClassification.h5"
image_path = filedialog.askopenfilename(title="Please select an image.")


def predict(path):
    #Load model
    model = tf.keras.models.load_model(model_path)

    label_mapping = {
        'aloevera': 0, 'banana': 1, 'bilimbi': 2, 'cantaloupe': 3, 'cassava': 4, 'coconut': 5,
        'corn': 6, 'cucumber': 7, 'curcuma': 8, 'eggplant': 9, 'galangal': 10, 'ginger': 11,
        'guava': 12, 'kale': 13, 'longbeans': 14, 'mango': 15, 'melon': 16, 'orange': 17,
        'paddy': 18, 'papaya': 19, 'peper chili': 20, 'pineapple': 21, 'pomelo': 22, 'shallot': 23,
        'soybeans': 24, 'spinach': 25, 'sweet potatoes': 26, 'tobacco': 27, 'waterapple': 28, 'watermelon': 29
    }

    # Load and preprocess the image
    image = cv2.imread(path)
    image = cv2.resize(image, (128, 128))  # Adjust size to match model input
    image = image.astype('float32') / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make a prediction
    predicted_label = model.predict(image)
    predicted_label_idx = np.argmax(predicted_label, axis=1)[0]  # Get the index of the predicted class

    # Get the class name from the label mapping (ensure label_mapping is available)
    predicted_class_name = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_label_idx)]

    # Print the prediction
    return predicted_class_name

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prediction = predict(image_path)
        self.layout = BoxLayout(orientation='vertical')
        self.predict_label = Label(text="[Prediction Here]", font_size=90)
        self.layout.add_widget(self.predict_label)
        self.predict_button = Button(text = "Make prediction", on_press=self.updatePredictLabel)
        self.layout.add_widget(self.predict_button)
        self.add_widget(self.layout)

    def updatePredictLabel(self, instance):
        self.predict_label.text = self.prediction

class PlantApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        return sm

#------Main Method------
def main():
    PlantApp().run()

if __name__ == "__main__":
    main()
