"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
from tensorflow.keras.layers import \
<<<<<<< HEAD
    Conv2D, MaxPool2D, Dropout, Flatten, Dense
=======
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, ZeroPadding2D, BatchNormalization
>>>>>>> b0890fd28073f952d31f8b8ebdbddd4711338ed9

import hyperparameters as hp


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()

        # TODO: Select an optimizer for your network (see the documentation
        #       for tf.keras.optimizers)

        self.optimizer = tf.keras.optimizers.RMSprop(
            hp.learning_rate, momentum=hp.momentum)

        # TODO: Build your own convolutional neural network, using Dropout at
        #       least once. The input image will be passed through each Keras
        #       layer in self.architecture sequentially. Refer to the imports
        #       to see what Keras layers you can use to build your network.
        #       Feel free to import other layers, but the layers already
        #       imported are enough for this assignment.
        #
        #       Remember: Your network must have under 15 million parameters!
        #       You will see a model summary when you run the program that
        #       displays the total number of parameters of your network.
        #
        #       Remember: Because this is a 15-scene classification task,
        #       the output dimension of the network must be 15. That is,
        #       passing a tensor of shape [batch_size, img_size, img_size, 1]
        #       into the network will produce an output of shape
        #       [batch_size, 15].
        #
        #       Note: Keras layers such as Conv2D and Dense give you the
        #             option of defining an activation function for the layer.
        #             For example, if you wanted ReLU activation on a Conv2D
        #             layer, you'd simply pass the string 'relu' to the
        #             activation parameter when instantiating the layer.
        #             While the choice of what activation functions you use
        #             is up to you, the final layer must use the softmax
        #             activation function so that the output of your network
        #             is a probability distribution.
        #
        #       Note: Flatten is a very useful layer. You shouldn't have to
        #             explicitly reshape any tensors anywhere in your network.

<<<<<<< HEAD
        self.architecture = [
            Conv2D(filters=108, kernel_size=(13, 13), padding='same',
                   strides=(4, 4), activation='relu'),
            MaxPool2D(pool_size=(5, 5), strides=(2, 2)),
            Conv2D(filters=256, kernel_size=(7, 7), padding='same',
                   strides=(1, 1), activation='relu'),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                   strides=(1, 1), activation='relu'),
            Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                   strides=(1, 1), activation='relu'),
            Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                   strides=(1, 1), activation='relu'),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            Dropout(rate=0.3),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dense(units=7, activation='softmax')
        ]
=======
        # Version #1
        # self.architecture = [
        #     # Block 1
        #     ZeroPadding2D((1, 1), input_shape=(48, 48, 1)),
        #     Conv2D(64, 3, 1, padding="valid",
        #            activation="relu", name="block1_conv1"),
        #     ZeroPadding2D((1, 1)),
        #     Conv2D(64, 3, 1, padding="valid",
        #            activation="relu", name="block1_conv2"),
        #     MaxPool2D(pool_size=(2, 2),
        #               strides=(2, 2), padding='valid'),
        #     # Block 2
        #     ZeroPadding2D((1, 1)),
        #     Conv2D(128, 3, 1, padding="valid",
        #            activation="relu", name="block2_conv1"),
        #     ZeroPadding2D((1, 1)),
        #     Conv2D(128, 3, 1, padding="valid",
        #            activation="relu", name="block2_conv2"),
        #     MaxPool2D(pool_size=(2, 2),
        #               strides=(2, 2), padding='valid'),
        #     # Block 3
        #     ZeroPadding2D((1, 1)),
        #     Conv2D(256, 3, 1, padding="valid",
        #            activation="relu", name="block3_conv1"),
        #     ZeroPadding2D((1, 1)),
        #     Conv2D(256, 3, 1, padding="valid",
        #            activation="relu", name="block3_conv2"),
        #     ZeroPadding2D((1, 1)),
        #     Conv2D(256, 3, 1, padding="valid",
        #            activation="relu", name="block3_conv3"),
        #     MaxPool2D(pool_size=(2, 2),
        #               strides=(2, 2), padding='valid'),
        #     # Block 4
        #     ZeroPadding2D((1, 1)),
    #     #     Conv2D(512, 3, 1, padding="valid",
    #     #            activation="relu", name="block4_conv1"),
    #     #     ZeroPadding2D((1, 1)),
    #     #     Conv2D(512, 3, 1, padding="valid",
    #     #            activation="relu", name="block4_conv2"),
    #     #     ZeroPadding2D((1, 1)),
    #     #     Conv2D(512, 3, 1, padding="valid",
    #     #            activation="relu", name="block4_conv3"),
    #     #     MaxPool2D(pool_size=(2, 2),
    #     #               strides=(2, 2), padding='valid'),
    #     #     # Block 5
    #     #     ZeroPadding2D((1, 1)),
    #     #     Conv2D(512, 3, 1, padding="same",
    #     #            activation="relu", name="block5_conv1"),
    #     #     ZeroPadding2D((1, 1)),
    #     #     Conv2D(512, 3, 1, padding="same",
    #     #            activation="relu", name="block5_conv2"),
    #     #     ZeroPadding2D((1, 1)),
    #     #     Conv2D(512, 3, 1, padding="same",
    #     #            activation="relu", name="block5_conv3"),
    #     #     MaxPool2D(pool_size=(2, 2),
    #     #               strides=(2, 2), padding='valid'),
    #     #
    #     #     # Head
    #     #     Flatten(),
    #     #     Dense(4096, activation='relu'),
    #     #     Dropout(0.5),
    #     #     Dense(4096, activation='relu'),
    #     #     Dropout(0.5),
    #     #     Dense(hp.num_classes, activation='softmax')
    #     # ]

    #     # Version 2
    #     c1 = Conv2D(32, 3, 1, padding="same", activation="relu", name="C1")
    #     c2 = Conv2D(64, 3, 1, padding="same", activation="relu", name="C2")
    #     m3 = MaxPool2D(2, name="M3")
    #     d4 = Dropout(0.25, name="D4")
    #     c5 = Conv2D(128, 3, 1, padding="same", activation="relu", name="C5")
    #     m6 = MaxPool2D(2, name="M6")
    #     c7 = Conv2D(128, 3, 1, padding="same", activation="relu", name="C7")
    #     m8 = MaxPool2D(2, name="M8")
    #     d9 = Dropout(0.25, name="D9")
    #     f10 = Flatten(name="F10")
    #     d11 = Dense(units=100, activation="relu", name="D11")
    #    #  d12 = Dropout(0.5, name="D12")
    #     d13 = Dense(units=7, activation="softmax", name="D13")

    #     self.architecture = [c1, c2, m3, d4, c5,
    #                          m6, c7, m8, d9, f10, d11, d13]

        self.architecture = [
        Conv2D(filters = 64,kernel_size = (3,3),padding = 'same',activation = 'relu',input_shape=(48, 48,1)),
        MaxPool2D(pool_size = 2,strides = 2),
        BatchNormalization(),

        Conv2D(filters = 128,kernel_size = (3,3),padding = 'same',activation = 'relu'),
        MaxPool2D(pool_size = 2,strides = 2),
        BatchNormalization(),
        Dropout(0.25),

        Conv2D(filters = 128,kernel_size = (3,3),padding = 'same',activation = 'relu'),
        MaxPool2D(pool_size = 2,strides = 2),
        BatchNormalization(),
        Dropout(0.25),

        Conv2D(filters = 256,kernel_size = (3,3),padding = 'same',activation = 'relu'),
        MaxPool2D(pool_size = 2,strides = 2),
        BatchNormalization(),

        Flatten(),
        Dense(units = 128,activation = 'relu',kernel_initializer='he_normal'),
        Dropout(0.25),
        Dense(units = 64,activation = 'relu',kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.25),
        Dense(units = 32,activation = 'relu',kernel_initializer='he_normal'),
        Dense(7,activation = 'softmax')
        ]

       
       
        

    
>>>>>>> b0890fd28073f952d31f8b8ebdbddd4711338ed9

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        # TODO: Select a loss function for your network (see the documentation
        #       for tf.keras.losses)

        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)


class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        # TODO: Select an optimizer for your network (see the documentation
        #       for tf.keras.optimizers)

        self.optimizer = tf.keras.optimizers.RMSprop(
            hp.learning_rate, momentum=hp.momentum)

        # Don't change the below:

        self.vgg16 = [
            # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]

        # TODO: Make all layers in self.vgg16 non-trainable. This will freeze the
        #       pretrained VGG16 weights into place so that only the classificaiton
        #       head is trained.

        for layer in self.vgg16:
            layer.trainable = False

        # TODO: Write a classification head for our 15-scene classification task.

        self.head = [
<<<<<<< HEAD
            Dropout(rate=0.3),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dense(units=7, activation='softmax')
=======
            Flatten(data_format=None), 
            Dropout(0.4), 
            Dense(units=100, activation='relu'), 
            Dense(units=15, activation='softmax')
>>>>>>> b0890fd28073f952d31f8b8ebdbddd4711338ed9
        ]
        

        # Don't change the below:
        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg16(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        # TODO: Select a loss function for your network (see the documentation
        #       for tf.keras.losses)

        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
