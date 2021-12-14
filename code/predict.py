import tensorflow as tf
import numpy as np
from models import YourModel, VGGModel
import hyperparameters as hp
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, ZeroPadding2D, BatchNormalization
from keras.models import load_model
import os
import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

model = VGGModel()

print("Loading Models...")
# model.vgg16 = load_model("vgg.h5")
model.head = load_model("head.h5")

# print("Loading Weights...")
# model.vgg16.load_weights("vgg16_imagenet.h5", by_name=True)
# model.head.load_weights(
#     "checkpoints/vgg_model/121421-151746/vgg.weights.e018-acc0.5242.h5")

# model.compile(
#     optimizer=model.optimizer,
#     loss=model.loss_fn,
#     metrics=["sparse_categorical_accuracy"])


# image_path = "angry.png"

# print("Loading in Image...")
# image = tf.keras.preprocessing.image.load_img(image_path)
# input_arr = tf.keras.preprocessing.image.img_to_array(image)
# input_arr = np.array([input_arr])  # Convert single image to a batch.

# print("Predicting Image...")
# predictions = model.predict(input_arr)

# print("Finished!")
