import tensorflow as tf
import numpy as np
from models import VGGModel
import os
from tensorflow.keras.preprocessing import image
import numpy as np


def predict(image_path):
    """
    Predicts emotion of an image.
    """
    # Changes execution directory to the folder the file is in.
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    print("Creating Model...")
    model = VGGModel()
    model(tf.keras.Input(shape=(224, 224, 3)))

    print("Loading Weights...")
    model.vgg16.load_weights("vgg16_imagenet.h5", by_name=True)
    model.head.load_weights(
        "checkpoints/vgg_model/121421-151746/vgg.weights.e018-acc0.5242.h5")

    print("Compiling Model...")
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    print("Loading in Image...")
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    print("Predicting Image...")
    prediction = model.predict(img_batch)

    # Labels for each Emotion.
    label_dict = {0: "Angry", 1: "Disgust", 2: "Fear",
                  3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    prediction_list = list(prediction[0])
    img_index = prediction_list.index(max(prediction_list))

    print(str(label_dict[img_index]) + " with " +
          str(prediction_list[img_index]) + " accuracy.")

    print("Overall distribution:")
    print(prediction_list)

    print("Finished!")
    return label_dict[img_index]


predict("processed_face.jpg")
