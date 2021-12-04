#!/usr/bin/python
from datetime import datetime
import os
from model import Model
from preprocess import Datasets
import hyperparameters as hp


def train(model, datasets, checkpoint_path, logs_path, init_epoch):
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
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    relative_path = ".." + os.sep + "data" + os.sep
    absolute_path = os.path.abspath(relative_path)

    os.chdir(sys.path[0])

    dataset = Datasets(absolute_path)

    model = Model()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    checkpoint_path = "checkpoints" + os.sep + \
        "model" + os.sep + timestamp + os.sep
    logs_path = "logs" + os.sep + "model" + \
        os.sep + timestamp + os.sep

    # Print summary of model
    model.summary()


if __name__ == '__main__':
    main()
