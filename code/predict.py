import tensorflow as tf
import numpy as np
from models import YourModel, VGGModel
import hyperparameters as hp
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, ZeroPadding2D, BatchNormalization
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


model = VGGModel()
model.built = True
model.vgg16.load_weights("vgg16_imagenet.h5", by_name=True)
model.head.load_weights(
    "checkpoints/vgg_model/121421-151746/vgg.weights.e018-acc0.5242.h5", by_name=False)

image_path = "angry.png"

image = tf.keras.preprocessing.image.load_img(image_path)
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)
