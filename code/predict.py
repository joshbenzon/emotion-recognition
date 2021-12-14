import tensorflow as tf
import numpy as np
from models import YourModel, VGGModel
import hyperparameters as hp
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, ZeroPadding2D, BatchNormalization
from keras.models import load_model
import os
from tensorflow.keras.preprocessing import image
import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

print("Gettings Model...")
model = VGGModel()

print("Assigning Model Shape...")
model(tf.keras.Input(shape=(224, 224, 3)))

# print("Loading Models...")
# model.vgg16 = load_model("vgg.h5")
# model.head = load_model("head.h5")

print("Loading Weights...")
model.vgg16.load_weights("vgg16_imagenet.h5", by_name=True)
model.head.load_weights(
    "checkpoints/vgg_model/121421-151746/vgg.weights.e018-acc0.5242.h5")

print("Compiling Model...")
model.compile(
    optimizer=model.optimizer,
    loss=model.loss_fn,
    metrics=["sparse_categorical_accuracy"])


image_path = "angry.png"

print("Loading in Image...")
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
# img_preprocessed = preprocess_input(img_batch)

print("Predicting Image...")
prediction = model.predict(img_batch)

label_dict = {0: "Angry", 1: "Disgust", 2: "Fear",
              3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

prediction_list = list(prediction[0])

img_index = prediction_list.index(max(prediction_list))
print(label_dict[img_index])

print("Finished!")
print(prediction)
