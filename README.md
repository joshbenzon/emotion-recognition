# Inside Out Emotion Recognition

Inside Out is an emotion recognition project that uses deep learning to recognize facial expressions.

## Introduction

Inside Out uses deep learning to train a model to recognize basic emotions: happy, sad, angry, disgusted, fearful, neutral, and surprised. The project consists of a frontend GUI made using Tkinter, and a backend which can train the convolutional neural network and predict emotions of input images using tensorflow. Inside Out was built by Katie Baumgarten, Theo Fernandez, Grace Miller, and Josh Benzon for their final project in Computer Vision (CSCI1430) at Brown University.

## Model

The architecture we used to train the algorithm uses the pretrained model VGG16, along with a custom head. The head consists of the following layers:

1. Dropout(0.3)
2. Flatten()
3. Dense(128, 'relu')
4. Dense(7, 'softmax')

We also used specific preprocessing on our images in order to get our accuracy as high as possible. This consisted of the following arguments to the ImageDataGenerator function when reading in training images:

- brightness_range = [0.2, 1.2]
- width_shift_range = 0.1
- height_shift_range = 0.1
- horizontal_flip = True

Using VGG16 and our custom head, as well as the custom preprocessing, we were able to get our accuracy on the validation dataset up to 52.46%, which we consider a success.

## How to Use

In order to run any parts of the program, make sure that you are running Inside Out in a python3 virtual environment that has all of the necessary dependencies involved. This includes but is not limited to open-cv, tensorflow, os, tkinter, numpy, PIL, skimage, tensorboard, and random. Once the virtual environment is set up, you can train the model or run the GUI.

#### Training the Model

In order to train the model, run the command `python3 run.py` from the code directory. This will start training the model. The model's three best outputs will be saves in the `checkpoints/vgg_model` folder.

#### Using the GUI

In order to run the GUI, run the command `python3 gui.py` from the code directory. This will open a small window with two buttons: Take a Picture and Quit.

Clicking Take a Picture will open a video feed from your web camera if one is available. Pressing space bar or "p" will take a photo. The computer will then find your face in the image, detect your emotion, and then return a value in the GUI. You can repeat this process as many times as you like. If you open the camera accidentally, you can quit without taking a photo by clicking the "ESC" key.

Once you are done with the GUI, you can close it by pressing the Quit button. This will end the program.

## Troubleshooting

Here are the most common error we ran into while running this program.

#### Tensorflow Isn't Working

If you are on an m1 Mac, there are fixes but we've found them to be difficult to resolve. Using a different computer or cloud computing is probably your best option.

#### The GUI Didn't Take a Photo

This error occasionally occurs if it doesn't detect a face in the photo. Luckily, this error is easy to fix. Just take another photo! Make sure that you face centered in the web camera and fairly large. If possible, avoid having other faces in the background and make sure that your face is well lit. It should work the second time, but if you get this error again, just try to take another photo again.

#### Inside Out Predicted my Emotion Incorrectly

There are two possible causes for this error. The first is that it didn't find your face correctly in the photo that you took. You can check this by looking at `inside-out/code/processed_face.jpg`. If this is not your face, try taking the picture again using the GUI. The face detection is not always perfect and may detect another person face or even a random object in the background.

It is also the case that Inside Out is not always able to predict emotions correctly. From testing, we have found that it does best on happy, sad, neutral, surprised, and fearful. It struggles most and angry and disgusted. If it is struggling, be patient with it and know that it isn't perfect.
