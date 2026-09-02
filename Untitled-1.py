# %%
import cv2
import os
import uuid
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, losses, optimizers
from keras.utils import register_keras_serializable

# %% [markdown]
# Set GPU Memory Consumption Growth

# %%
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except RuntimeError as e:
    print(e)

# %% [markdown]
# Set up paths and create directories

# %%
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

os.makedirs(POS_PATH, exist_ok=True)
os.makedirs(NEG_PATH, exist_ok=True)
os.makedirs(ANC_PATH, exist_ok=True)

# %% [markdown]
# Collect Positives, Negative and Anchors

# %%
# Uncompress Tar GZ Labelled Faces in the Wild Dataset
!tar -xf lfw.tgz
# Move LFW Images to the following repository data/negative
for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        EX_PATH = os.path.join('lfw', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)

# %%

import uuid
# Establish a connection to the webcam
cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()
   
    # Cut down frame to 250x250px
    frame = frame[120:120+250,200:200+250, :]
    
    # Collect anchors 
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # Create the unique file path 
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out anchor image
        cv2.imwrite(imgname, frame)
    
    # Collect positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
        # Create the unique file path 
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out positive image
        cv2.imwrite(imgname, frame)
    
    # Show image back to screen
    cv2.imshow('Image Collection', frame)
    
    # Breaking gracefully
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
        
# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()

# %% [markdown]
# Load and Preprocess Images

# %%
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

# %% [markdown]
# Create Labelled Dataset

# %%
anchor = tf.data.Dataset.list_files(os.path.join(ANC_PATH, '*.jpg')).take(5000)
positive = tf.data.Dataset.list_files(os.path.join(POS_PATH, '*.jpg')).take(5000)
negative = tf.data.Dataset.list_files(os.path.join(NEG_PATH, '*.jpg')).take(5000)

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

# %% [markdown]
# Build Train and Test Partition

# %%
def preprocess_twin(input_img, validation_img, label):
    return preprocess(input_img), preprocess(validation_img), label

data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

train_data = data.take(round(len(data) * 0.7)).batch(8).prefetch(4)
test_data = data.skip(round(len(data) * 0.7)).take(round(len(data) * 0.3)).batch(8).prefetch(4)

# %% [markdown]
# Model Engineering

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

@tf.keras.utils.register_keras_serializable()
class L1Dist(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

def make_embedding():
    input_layer = layers.Input(shape=(100, 100, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='sigmoid')(x)
    return models.Model(inputs=input_layer, outputs=x)

def make_siamese_model():
    input_image = layers.Input(name='input_img', shape=(100, 100, 3))
    validation_image = layers.Input(name='validation_img', shape=(100, 100, 3))

    siamese_layer = L1Dist(name='distance')
    input_embedding_model = make_embedding()
    validation_embedding_model = make_embedding()
    input_embedding = input_embedding_model(input_image)
    validation_embedding = validation_embedding_model(validation_image)
    distances = siamese_layer(input_embedding, validation_embedding)
    classifier = layers.Dense(1, activation='sigmoid')(distances)

    return models.Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()

# %% [markdown]
# Training

# %%
os.makedirs('training_checkpoints', exist_ok=True)

# %%
import os

binary_cross_loss = keras.losses.BinaryCrossentropy()
opt = keras.optimizers.Adam(1e-4)

@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]
        yhat = siamese_model(X, training=True)
        loss = binary_cross_loss(y, yhat)

    grad = tape.gradient(loss, siamese_model.trainable_variables)
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    return loss

def train(data, EPOCHS):
    os.makedirs('training_checkpoints', exist_ok=True)  # Create the directory if it doesn't exist

    for epoch in range(1, EPOCHS + 1):
        print(f'\nEpoch {epoch}/{EPOCHS}')
        progbar = tf.keras.utils.Progbar(len(data))




        for idx, batch in enumerate(data):
            loss = train_step(batch)
            progbar.update(idx + 1, values=[('loss', loss)])

        if epoch % 10 == 0:
            checkpoint_path = os.path.join('training_checkpoints', f'ckpt-{epoch}.weights.h5')
            siamese_model.save_weights(checkpoint_path)

EPOCHS = 100
train(train_data, EPOCHS)

# %% [markdown]
# Evaluate Model

# %%
from sklearn.metrics import precision_score, recall_score

def evaluate(model, data):
    y_true = []
    y_pred = []

    for batch in data:
        test_input, test_val, y = batch
        predictions = model.predict([test_input, test_val])
        y_true.extend(y.numpy().tolist())
        y_pred.extend((predictions > 0.5).flatten().tolist())

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return precision, recall

precision, recall = evaluate(siamese_model, test_data)
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}')

# %% [markdown]
# Save Model

# %%
siamese_model.save('siamesemodel_1404.keras')

# %% [markdown]
# Opencv testing

# %%
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained Siamese model
model = load_model('siamesemodel_1404.keras')

# Function to preprocess the input images
def preprocess(img):
    img = cv2.resize(img, (100, 100))
    img = img / 255.0
    return img

# Load the two images you want to compare
img1 = cv2.imread('D:\\BCA_Finalyear_project\\data\\anchor\\0a0ca0a6-fa3b-11ee-a52d-d039571de50e.jpg')
img2 = cv2.imread('D:\\BCA_Finalyear_project\\data\\anchor\\0a0ca0a6-fa3b-11ee-a52d-d039571de50e.jpg')

# Preprocess the images
img1 = preprocess(img1)
img2 = preprocess(img2)

# Expand the dimensions to match the model's input shape
img1 = np.expand_dims(img1, axis=0)
img2 = np.expand_dims(img2, axis=0)

# Make the prediction using the Siamese model
output = model.predict([img1, img2])

# Check the output
if output[0][0] > 0.5:
    print("The images are similar")
else:
    print("The images are not similar")

# %% [markdown]
# Approach 2, using model.predict()

# %%
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained Siamese model
model = load_model('siamesemodel_1304.keras')

# Function to preprocess the input images
def preprocess(img):
    img = cv2.resize(img, (100, 100))
    img = img / 255.0
    return img

# Load the two images you want to compare
img1 = cv2.imread(r'D:\clgproject\face recognition project\data\positive\b53405a3-f7ec-11ee-ade2-d039571de50e.jpg')
img2 = cv2.imread('D:\\clgproject\\face recognition project\\data\\positive\\9b8ad719-f7ec-11ee-beaa-d039571de50e.jpg')

# Preprocess the images
img1 = preprocess(img1)
img2 = preprocess(img2)

# Convert images to TensorFlow tensors
img1_tensor = tf.convert_to_tensor(img1, dtype=tf.float32)
img2_tensor = tf.convert_to_tensor(img2, dtype=tf.float32)

# Expand the dimensions to match the model's input shape
img1_tensor = tf.expand_dims(img1_tensor, axis=0)
img2_tensor = tf.expand_dims(img2_tensor, axis=0)

# Make the prediction using the Siamese model
output = model.predict([img1_tensor, img2_tensor])

# Check the output
if output[0][0] > 0.5:
    print("The images are similar")
else:
    print("The images are not similar")


