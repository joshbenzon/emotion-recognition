import os
import sys
import re
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import VGGModel
from preprocess import Datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(model, datasets, init_epoch):
    """ Trains model using dataset and hyperparameters. """
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    """ Tests model on testing dataset. """
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    """ Main Program. """
    data = '..'+os.sep+'data'+os.sep
    vgg_weight_path = 'vgg16_imagenet.h5'
    load_checkpoint = None
    evaluate = False

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    if load_checkpoint is not None:
        load_checkpoint = os.path.abspath(load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(load_checkpoint))

    # Finds path to dataset and vgg weights
    if os.path.exists(data):
        data = os.path.abspath(data)
    if os.path.exists(vgg_weight_path):
        vgg_weight_path = os.path.abspath(vgg_weight_path)

    os.chdir(sys.path[0])

    datasets = Datasets(data)

    model = VGGModel()
    checkpoint_path = "checkpoints" + os.sep + \
        "vgg_model" + os.sep + timestamp + os.sep
    model(tf.keras.Input(shape=(224, 224, 3)))

    model.vgg16.summary()
    model.head.summary()

    model.vgg16.load_weights(vgg_weight_path, by_name=True)

    if load_checkpoint is not None:
        model.head.load_weights(load_checkpoint, by_name=False)

    if not evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    if evaluate:
        test(model, datasets.test_data)
    else:
        train(model, datasets, init_epoch)


main()
