import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp


class Model(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(Model, self).__init__()

        self.optimizer = tf.keras.optimizers.RMSprop(
            hp.learning_rate, momentum=hp.momentum)

        # Block 1
        conv = Conv2D(64, 3, 1, padding="same",
                      activation="relu", name="block1_conv1")
        conv_two = Conv2D(64, 3, 1, padding="same",
                          activation="relu", name="block1_conv2")
        maxp = MaxPool2D(2, name="block1_pool")
        # Block 2
        conv_three = Conv2D(128, 3, 1, padding="same",
                            activation="relu", name="block2_conv1")
        conv_four = Conv2D(128, 3, 1, padding="same",
                           activation="relu", name="block2_conv2")
        maxp_two = MaxPool2D(2, name="block2_pool")

        # Block 3
        conv_five = Conv2D(256, 3, 1, padding="same",
                           activation="relu", name="block3_conv1")
        conv_six = Conv2D(256, 3, 1, padding="same",
                          activation="relu", name="block3_conv2")
        maxp_three = MaxPool2D(2, name="block3_pool")
        # Block 4
        conv_eight = Conv2D(512, 3, 1, padding="same",
                            activation="relu", name="block4_conv1")
        conv_nine = Conv2D(512, 3, 1, padding="same",
                           activation="relu", name="block4_conv2")
        maxp_four = MaxPool2D(2, name="block4_pool")
        # Block 5
        conv_eleven = Conv2D(512, 3, 1, padding="same",
                             activation="relu", name="block5_conv1")
        conv_twelve = Conv2D(512, 3, 1, padding="same",
                             activation="relu", name="block5_conv2")
        maxp_five = MaxPool2D(2, name="block5_pool")
        flat = tf.keras.layers.Flatten(data_format=None)
        drop = Dropout(0.4)
        dense = Dense(units=100, activation='relu')
        dense_1 = Dense(units=15, activation='softmax')

        self.architecture = [conv, conv_two, maxp,
                             conv_three, conv_four, maxp_two,
                             conv_five, conv_six, maxp_three,
                             conv_eight, conv_nine, maxp_four,
                             conv_eleven, conv_twelve, maxp_five, drop, flat, dense, dense_1]

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
