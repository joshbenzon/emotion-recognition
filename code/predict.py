import tensorflow as tf
import numpy as np
from models import YourModel, VGGModel
import hyperparameters as hp
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, ZeroPadding2D, BatchNormalization


def create_model():
    model = tf.keras.Sequential()
    # model.build(input_shape)

    model.add(Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv1"))
    model.add(Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv2"))
    model.add(MaxPool2D(2, name="block1_pool"))
    model.add(Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv1"))
    model.add(Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv2"))
    model.add(MaxPool2D(2, name="block2_pool"))
    model.add(Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv1"))
    model.add(Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv2"))
    model.add(Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv3"))
    model.add(MaxPool2D(2, name="block3_pool"))
    model.add(Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv1"))
    model.add(Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv2"))
    model.add(Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv3"))
    model.add(MaxPool2D(2, name="block4_pool"))
    model.add(Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv1"))
    model.add(Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv2"))
    model.add(Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv3"))
    model.add(MaxPool2D(2, name="block5_pool"))
    model.add(Dropout(rate=0.3))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=7, activation='softmax'))

    model.save('new_model')


def predict_image(path):  # removed model
    # model = VGGModel()
    model = tf.keras.models.load_model('new_model')

    model.load_weights("vgg.weights.e024-acc0.4742.h5")

    img = tf.keras.preprocessing.image.load_img(path, color_mode="grayscale", grayscale=True, target_size=(48, 48))

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

    result = model.predict(img)  # same as model(img) or model.call()

    result = list(result[0])

    print(result)

    img_index = result.index(max(result))
    print(label_dict[img_index], '~label~')


print("hi")

create_model()
predict_image("angry.png")