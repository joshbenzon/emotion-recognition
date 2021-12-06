#!/usr/bin/python
from datetime import datetime
import os
from model import Model
from preprocess import Datasets
import hyperparameters as hp
import tensorflow as tf


def train(model, datasets, init_epoch):
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    print("Starting CNN for Inside Out Group")
    time = datetime.now()
    init_epoch = 0

    relative_path = ".." + os.sep + "data" + os.sep
    absolute_path = os.path.abspath(relative_path)

    datasets = Datasets(absolute_path)

    model = Model()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))

    # Print summary of model
    model.summary()

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    train(model, datasets, init_epoch)


if __name__ == '__main__':
    main()
