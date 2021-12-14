import tensorflow as tf
import numpy as np
from models import YourModel, VGGModel
import hyperparameters as hp

def predict_image(path):
    model = VGGModel()
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

predict_image("angry.png")