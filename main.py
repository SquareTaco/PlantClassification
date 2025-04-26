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
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
import tkinter as tk
from tkinter import filedialog
from kivy.graphics import Rectangle, Color
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.core.image import Image as CoreImage
from kivy.uix.label import Label
from kivy.properties import ListProperty, ColorProperty
import sqlite3

model_path = "PlantClassification.h5"

#Load model
model = tf.keras.models.load_model(model_path)

label_mapping = {
    'Aloevera': 0, 'Banana': 1, 'Bilimbi': 2, 'Cantaloupe': 3, 'Cassava': 4, 'Coconut': 5,
    'Corn': 6, 'Cucumber': 7, 'Curcuma': 8, 'Eggplant': 9, 'Galangal': 10, 'Ginger': 11,
    'Guava': 12, 'Kale': 13, 'Longbeans': 14, 'Mango': 15, 'Melon': 16, 'Orange': 17,
    'Paddy': 18, 'Papaya': 19, 'Peper Chili': 20, 'Pineapple': 21, 'Pomelo': 22, 'Shallot': 23,
    'Soybeans': 24, 'Spinach': 25, 'Sweet Potatoes': 26, 'Tobacco': 27, 'Waterapple': 28, 'Watermelon': 29
}

def predict(path):

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
        self.username = "Default"
        self.id = 1
        self.your_Account = user_account(self.username, self.id) #default account for now
        self.image_path = ""
        with self.canvas.before:
            self.bg_color = Color(1, 1, 1, 1)
            self.bg_rect = Rectangle(size_hint=self.size, pos=self.pos, texture= None)

        self.bind(size=self._update_bg)
        self.bind(pos=self._update_bg)

        self.layout = FloatLayout()
        self.sublayout = BoxLayout(orientation='horizontal', size_hint=(1, 0.2))

        self.predict_label = Label(text="[Prediction Here]", font_size=90, pos_hint={'center_x':0.5, 'center_y':0.8})
        self.layout.add_widget(self.predict_label)

        self.display_button = Button(
            text="Display Collected Plants",
            on_press=self.displayPlants,
            #size_hint = (None,None),
            #size=(200, 200),
            #pos_hint={'x': 0.0, 'bottom': 0.0}
        )
        self.sublayout.add_widget(self.display_button)

        self.image_button = Button(
            text="Select Image >", 
            on_press=self.switch_to_image,
            #size_hint = (None,None),
            #size=(200, 200),
            #pos_hint={'x': 0.8, 'bottom': 0.0}
        )
        self.sublayout.add_widget(self.image_button)

        self.layout.add_widget(self.sublayout)
        self.add_widget(self.layout)

    def _update_bg(self, instance, value):
        self.bg_rect.size = instance.size
        self.bg_rect.pos = instance.pos

    def displayPlants(self, instance):
        self.your_Account.displayPlants()
        pass

    def switch_to_image(self, instance):
        self.manager.current = 'image'


class ImageScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.default_texture = CoreImage("selectYourImage.png").texture
        self.background_texture = None

        with self.canvas.before:
            self.bg_rect = Rectangle(size=self.size,
                pos=self.pos, 
                texture=self.default_texture
            )

        self.bind(size=self._update_bg)
        self.bind(pos=self._update_bg)

        self.layout = FloatLayout(size_hint=(1,1))
        self.boxLayout = BoxLayout(orientation='horizontal')

        self.reSelect_button = Button(
            text = "Select Photo", 
            on_press=self.selectImage, 
            pos_hint={'x': 0.0, 'bottom': 0.1},
            size=(475, 150),
            size_hint=(None, None),
        )
        self.layout.add_widget(self.reSelect_button)

        self.error_label = ColorLabel(
            text="",
            pos_hint={'center_x': 0.5, 'center_y': 0.05},
            font_size=60
        )
        self.add_widget(self.error_label)

        self.select_button = Button(
            text = "Use photo",
            size_hint=(None, None),
            size=(475, 150),
            pos_hint={'x': 0.81, 'bottom': 0.1},
            on_press=self.switch_to_main
        )
        self.layout.add_widget(self.select_button)

        self.add_widget(self.boxLayout)
        self.add_widget(self.layout)

    def _update_bg(self, instance, value):
        self.bg_rect.size = instance.size
        self.bg_rect.pos = instance.pos

    def selectImage(self, instance):
        image_path = filedialog.askopenfilename(title="Please select an image.")

        if image_path:
            self.image_path = image_path
            try:
                img = CoreImage(self.image_path)
                self.background_texture = img.texture
                self.bg_rect.texture = self.background_texture
                self.error_label.text = ""
            except Exception as e:
                self.background_texture = None
                self.bg_rect.texture = None

    def switch_to_main(self, instance):
        if self.background_texture:
            self.main_screen = self.manager.get_screen('main')
            self.account = self.main_screen.your_Account
            self.main_screen.image_path = self.image_path
            self.main_screen.bg_rect.texture = self.bg_rect.texture
            prediction = predict(self.image_path)
            self.main_screen.predict_label.text = prediction
            self.account.collectPlant(prediction)
            self.manager.current = 'main'
        else:
            self.error_label.text = "Please select an image."


class ColorLabel(Label):
    text_color = ColorProperty([1, 0, 0, 1])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(text_color=self.update_color)
        self.update_color()

    def update_color(self, *args):
        self.color = self.text_color


class PlantFinderApp(App):
    def build(self):
        sm = ScreenManager()
        
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(ImageScreen(name='image'))
        return sm

class user_account():
    def __init__(self, username, id):
        self.id = int(id)
        self.username = str(username)
        conn = sqlite3.connect(self.username + "_account.db")
        database = conn.cursor()
        database.execute(
            """
            CREATE TABLE IF NOT EXISTS my_account (
                username TEXT,
                plant_name TEXT UNIQUE
            )
            """
        )
        conn.commit()
        conn.close()

    def collectPlant(self, plantName):
        c = sqlite3.connect(self.username + "_account.db")
        db = c.cursor()
        db.execute(
            "INSERT OR IGNORE INTO my_account (username, plant_name) VALUES (?, ?)", 
            (self.username, plantName))
        c.commit()
        c.close()

    def displayPlants(self):
        c = sqlite3.connect(self.username + "_account.db")
        db = c.cursor()
        db.execute("SELECT plant_name FROM my_account where username = ?", (self.username,))
        collectedPlants = db.fetchall()
        print(collectedPlants)

#------Main Method------
def main():
    PlantFinderApp().run()

if __name__ == "__main__":
    main()
