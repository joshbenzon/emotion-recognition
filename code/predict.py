import tensorflow as tf
import numpy as np
from models import YourModel, VGGModel
import hyperparameters as hp
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, ZeroPadding2D, BatchNormalization


def predict_image(model, path):
    # model = VGGModel()
    # model = tf.keras.models.load_model("vgg16_imagenet.h5")
    
    #model.load_weights("vgg.weights.e024-acc0.4742.h5")
    # model.save_weights("\121321-205349\vgg.weights.e024-acc0.4742.h5")
    # print(model)

    img = tf.keras.preprocessing.image.load_img(path, color_mode="grayscale", grayscale = True, target_size=(48, 48))

    #predictions = model(img)

    print("loaded image in yay")

    label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
    img = np.array(img)
    print(img.shape)

    thing = tf.convert_to_tensor(img, dtype=tf.int8)
    img = tf.image.convert_image_dtype(thing, dtype=tf.float32)

    img = np.expand_dims(img, axis=0)
    print(img.shape)
    img = img.reshape(1, 48, 48, 1)
    # img = np.array(img)
    print("we reshaped")

    result = model.predict(img) #same as model(img) or model.call()

    result = list(result[0])

    img_index = result.index(max(result))
    print(label_dict[img_index], '~label~')

predict_image("angry.png")