import tensorflow as tf
import numpy as np
from models import YourModel, VGGModel
import hyperparameters as hp
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, ZeroPadding2D, BatchNormalization


def create_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv1"))
    model.add(Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"))
    model.add(MaxPool2D(2, name="block1_pool"))
    model.add(Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"))
    model.add(Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"))
    model.add(MaxPool2D(2, name="block2_pool"))
    model.add(Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv1"))
    model.add(Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"))
    model.add(Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"))
    model.add(MaxPool2D(2, name="block3_pool"))
    model.add(Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"))
    model.add(Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"))
    model.add(Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"))
    model.add(MaxPool2D(2, name="block4_pool"))
    model.add(Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1"))
    model.add(Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2"))
    model.add(Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3"))
    model.add(MaxPool2D(2, name="block5_pool"))
    
    return model

def predict_image(path):
    model = create_model()
    model.load_weights("\121321-205349\vgg.weights.e024-acc0.4742.h5")
    # model.save_weights("\121321-205349\vgg.weights.e024-acc0.4742.h5")
    # print(model)

    img = tf.keras.preprocessing.image.load_img(path, color_mode="grayscale", grayscale = True, target_size=(48, 48))

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

    result = model.predict(img)

    print("predick")

    result = list(result[0])

    print(result)

    img_index = result.index(max(result))
    print(label_dict[img_index])

    # model = VGGModel
    # data_dir = "\katie\Documents\Classes\cs1430\inside-out\data"
    # train_ds = tf.keras.utils.image_dataset_from_directory(
    #     data_dir,
    #     validation_split=0.2,
    #     subset="training",
    #     seed=123,
    #     image_size=hp.img_size,
    #     batch_size=hp.batch_size)

    # class_names = train_ds.class_names

    # url = path
    # path = tf.keras.utils.get_file(fname="~happy~", origin=url)

    # img = tf.keras.utils.load_img(path)

    # img_array = tf.keras.utils.img_to_array(img)
    # img_array = tf.expand_dims(img_array, 0)

    # predictions = model.predict(img_array)
    # score = tf.nn.softmax(predictions[0])

    # print("This image msot loikely belongs to {} with a {:.2f} percent confidence."
    #       .format(class_names[np.argmax(score)], 100 * np.max(score)))

predict_image("happy.jpg")