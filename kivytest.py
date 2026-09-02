from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import os
import re
from kivy.clock import Clock
from kivy.graphics.texture import Texture

@tf.keras.utils.register_keras_serializable()
class L1Dist(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

class SiameseApp(App):
    def build(self):
        # Load the trained Siamese model
        self.model = load_model('siamesemodel_1404.keras')


        # Create the main layout
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)

        # Create the title label
        title_label = Label(text="Facial Recognition using Siamese Neural Network", font_size=24, bold=True)

        # Create the image display widgets
        self.img1_display = Image(source='', size_hint_y=0.6)
        self.img2_display = Image(source='', size_hint_y=0.6)

        # Create the compare button
        self.compare_button = Button(text='Compare', on_press=self.compare_images, size_hint_y=0.1)

        # Create the result label
        self.result_label = Label(text='', font_size=20)

        # Create the file chooser
        self.file_chooser = FileChooserIconView(path=os.getcwd(), on_selection=self.handle_file_selection)

        # Add the widgets to the layout
        layout.add_widget(title_label)
        layout.add_widget(self.img2_display)
        layout.add_widget(self.img1_display)
        layout.add_widget(self.compare_button)
        layout.add_widget(self.result_label)
        layout.add_widget(self.file_chooser)

        return layout

    def handle_file_selection(self, selection):
        if selection:
            file_path = str(selection[0])
            # Process the selected file
            img2 = cv2.imread(file_path)
            img2 = self.preprocess(img2)
            texture = Texture.create(colorfmt='bgr')
            texture.blit_buffer(img2.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
            self.img2_display.texture = texture
            
    def compare_images(self, instance):
        # Load a verification image from the folder
        verification_folder = r'D:\clgproject\face recognition project\verificationkivyapp'
        verification_images = os.listdir(verification_folder)
        if verification_images:
            verification_image_path = os.path.join(verification_folder, verification_images[0])
            img1 = cv2.imread(verification_image_path)
        else:
            self.result_label.text = "No verification images found"
            return

        # Load the input image
        input_image_path = self.file_chooser.selection[0] if self.file_chooser.selection else None
        if input_image_path:
            img2 = cv2.imread(str(input_image_path))
        else:
            self.result_label.text = "Please select an input image"
            return

        # Preprocess the images
        img1 = self.preprocess(img1)
        img2 = self.preprocess(img2)

            # older
            verification_folder = r'D:\clgproject\face recognition project\verificationkivyapp'
            verification_images = os.listdir(verification_folder)
            if verification_images:
                verification_image_path = os.path.join(verification_folder, verification_images[0])
                img1 = cv2.imread(verification_image_path)
            else:
                self.result_label.text = "No verification images found"
                return

            # Load the input image
            input_image_path = self.file_chooser.selection[0] if self.file_chooser.selection else None
            if input_image_path:
                img2 = cv2.imread(str(input_image_path))
            else:
                self.result_label.text = "Please select an input image"
                return

            # Preprocess the images
            img1 = self.preprocess(img1)
            img2 = self.preprocess(img2)

            # Expand the dimensions to match the model's input shape
            img1 = np.expand_dims(img1, axis=0)
            img2 = np.expand_dims(img2, axis=0)

            # Make the prediction using the Siamese model
            output = self.model.predict([img1, img2])

            # Update the result label
            if output[0][0] > 0.5:
                self.result_label.text = "The images are similar"
            else:
                self.result_label.text = "The images are not similar"


    def preprocess(self, img):
        img = cv2.resize(img, (100, 100))  # Resize to the desired input size for the model
        img = img / 255.0
        return img

if __name__ == '__main__':
    SiameseApp().run()