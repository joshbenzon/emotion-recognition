"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import YourModel, VGGModel
from preprocess import Datasets
from skimage.transform import resize
from tensorboard_utils import \
    ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver

from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--task',
        required=True,
        choices=['1', '3'],
        help='''Which task of the assignment to run -
        training from scratch (1), or fine tuning VGG-16 (3).''')
    parser.add_argument(
        '--data',
        default='..'+os.sep+'data'+os.sep,
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-vgg',
        default='vgg16_imagenet.h5',
        help='''Path to pre-trained VGG-16 file (only applicable to
        task 3).''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    parser.add_argument(
        '--lime-image',
        default='test/Bedroom/image_0003.jpg',
        help='''Name of an image in the dataset to use for LIME evaluation.''')

    return parser.parse_args()


def LIME_explainer(model, path, preprocess_fn):
    """
    This function takes in a trained model and a path to an image and outputs 5
    visual explanations using the LIME model
    """

    def image_and_mask(title, positive_only=True, num_features=5,
                       hide_rest=True):
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=positive_only,
            num_features=num_features, hide_rest=hide_rest)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.title(title)
        plt.show()

    image = imread(path)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    image = preprocess_fn(image)
    image = resize(image, (hp.img_size, hp.img_size, 3))

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image.astype('double'), model.predict, top_labels=5, hide_color=0,
        num_samples=1000)

    # The top 5 superpixels that are most positive towards the class with the
    # rest of the image hidden
    image_and_mask("Top 5 superpixels", positive_only=True, num_features=5,
                   hide_rest=True)

    # The top 5 superpixels with the rest of the image present
    image_and_mask("Top 5 with the rest of the image present",
                   positive_only=True, num_features=5, hide_rest=False)

    # The 'pros and cons' (pros in green, cons in red)
    image_and_mask("Pros(green) and Cons(red)",
                   positive_only=False, num_features=10, hide_rest=False)

    # Select the same class explained on the figures above.
    ind = explanation.top_labels[0]
    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    plt.colorbar()
    plt.title("Map each explanation weight to the corresponding superpixel")
    plt.show()


def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """
    print("Checkpoint 1")
    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, ARGS.task, hp.max_num_weights)
    ]

    # Include confusion logger in callbacks if flag set
    if ARGS.confusion:
        callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

    print("Checkpoint 2")
    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )
    print("Checkpoint 3")


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    # If paths provided by program arguments are accurate, then this will
    # ensure they are used. If not, these directories/files will be
    # set relative to the directory of run.py
    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)
    if os.path.exists(ARGS.load_vgg):
        ARGS.load_vgg = os.path.abspath(ARGS.load_vgg)

    # Run script from location of run.py
    os.chdir(sys.path[0])

    datasets = Datasets(ARGS.data, ARGS.task)

    if ARGS.task == '1':
        model = YourModel()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
        checkpoint_path = "checkpoints" + os.sep + \
            "your_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "your_model" + \
            os.sep + timestamp + os.sep

        # Print summary of model
        model.summary()
    else:
        model = VGGModel()
        checkpoint_path = "checkpoints" + os.sep + \
            "vgg_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "vgg_model" + \
            os.sep + timestamp + os.sep
        model(tf.keras.Input(shape=(224, 224, 3)))

        # Print summaries for both parts of the model
        model.vgg16.summary()
        model.head.summary()

        # Load base of VGG model
        model.vgg16.load_weights(ARGS.load_vgg, by_name=True)

    # Load checkpoints
    if ARGS.load_checkpoint is not None:
        if ARGS.task == '1':
            model.load_weights(ARGS.load_checkpoint, by_name=False)
        else:
            model.head.load_weights(ARGS.load_checkpoint, by_name=False)

    # Make checkpoint directory if needed
    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    if ARGS.evaluate:
        test(model, datasets.test_data)

        # TODO: change the image path to be the image of your choice by changing
        # the lime-image flag when calling run.py to investigate
        # i.e. python run.py --evaluate --lime-image test/Bedroom/image_003.jpg
        path = ARGS.data + os.sep + ARGS.lime_image
        LIME_explainer(model, path, datasets.preprocess_fn)
    else:
        train(model, datasets, checkpoint_path, logs_path, init_epoch)


# Make arguments global
ARGS = parse_args()

#main()


import cv2 #opencv-based functions
import time
import math
import numpy as np
from scipy import ndimage
from skimage import io
from skimage import img_as_float, img_as_ubyte
from skimage.color import rgb2gray


class live_FFT2():
    """
    This function shows the live Fourier transform of a continuous stream of 
    images captured from an attached camera.
    """

    wn = "FD"
    use_camera = True
    im = 0
    imJack = 0
    phaseOffset = 0
    rollOffset = 0
    # Variable for animating basis reconstruction
    amplitudeCutoffRadius = 1
    amplitudeCutoffDirection = 1
    # Variables for animated basis demo
    magnitude = 2
    orientation = 0

    def __init__(self, **kwargs):

        # Camera device
        # If you have more than one camera, you can access them by cv2.VideoCapture(1), etc.
        self.vc = cv2.VideoCapture(0)
        if not self.vc.isOpened():
            print( "No camera found or error opening camera; using a static image instead." )
            self.use_camera = False

        if self.use_camera == False:
            # No camera!
            self.im = rgb2gray(img_as_float(io.imread('YuanningHuCrop.png'))) # One of our intrepid TAs (Yuanning was one of our HTAs for Spring 2019)
        else:
            # We found a camera!
            # Requested camera size. This will be cropped square later on, e.g., 240 x 240
            ret = self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            ret = self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Set the size of the output window
        cv2.namedWindow(self.wn, 0)

        # Load the Jack image for comparison
        self.imJack = rgb2gray(img_as_float(io.imread('JacksonGibbonsCrop.png'))) # Another one of our intrepid TAs (Jack was one of our TAs for Spring 2017)

        # Main loop
        while True:
            a = time.perf_counter()
            self.camimage_ft()
            print('framerate = {} fps \r'.format(1. / (time.perf_counter() - a)))
    
    
        if self.use_camera:
            # Stop camera
            self.vc.release()


    def camimage_ft(self):
        
        if self.use_camera:
            # Read image
            rval, im = self.vc.read()
            # Convert to grayscale and crop to square
            # (not necessary as rectangular is fine; just easier for didactic reasons)
            im = img_as_float(rgb2gray(im))
            # Note: some cameras across the class are returning different image sizes
            # on first read and later on. So, let's just recompute the crop constantly.
            
            if im.shape[1] > im.shape[0]:
                cropx = int((im.shape[1]-im.shape[0])/2)
                cropy = 0
            elif im.shape[0] > im.shape[1]:
                cropx = 0
                cropy = int((im.shape[0]-im.shape[1])/2)

            self.im = im[cropy:im.shape[0]-cropy, cropx:im.shape[1]-cropx]

        # Set size
        width = self.im.shape[1]
        height = self.im.shape[0]
        cv2.resizeWindow(self.wn, width*2, height*2)


        #                
        # Students: Concentrate here.
        # This code reads an image from your webcam. If you have no webcam, e.g.,
        # a department machine, then it will use a picture of an intrepid TA.
        #
        # Output image visualization:
        # Top left: input image
        # Bottom left: amplitude image of Fourier decomposition
        # Bottom right: phase image of Fourier decomposition
        # Top right: reconstruction of image from Fourier domain
        #
        # Let's start by peforming the 2D fast Fourier decomposition operation
        imFFT = np.fft.fft2( self.im )
        
        # Then creating our amplitude and phase images
        amplitude = np.sqrt( np.power( imFFT.real, 2 ) + np.power( imFFT.imag, 2 ) )
        phase = np.arctan2( imFFT.imag, imFFT.real )
        
        # We will reconstruct the image from this decomposition later on (far below at line 260); have a look now.

        ###########################################################
        # # Just the central dot
        #amplitude = np.zeros( self.im.shape )
        # # a = np.fft.fftshift(amplitude)
        
        # # NOTE: [0,0] here is the 0-th frequency component around which all other frequencies oscillate
        #amplitude[0,0] = 40000
        
        # # amplitude = np.fft.fftshift(a)


        # # #########################################################
        # # Part 0: Scanning the basis and looking at the reconstructed image for each frequency independently
        # # To see the effect, uncomment this block, read throug the comments and code, and then execute the program.
        
        # # Let's begin by zeroing out the amplitude and phase.
        # amplitude = np.zeros( self.im.shape )
        # phase = np.zeros( self.im.shape )

        # # Next, let's only set one basis sine wave to have any amplitude - just like the 'white dot on black background' images in lecture
        # # Let's animate how it looks as we move radially through the frequency space
        # self.orientation += math.pi / 30.0
        # if self.orientation > math.pi * 2:
        #     self.orientation = 0
        #     self.magnitude += 2
        # if self.magnitude >= 50: # could go to width/2 for v. high frequencies
        #     self.magnitude = 2

        # cx = math.floor(width/2)
        # cy = math.floor(height/2)
        # xd = self.magnitude*math.cos(self.orientation)
        # yd = self.magnitude*math.sin(self.orientation)
        # a = np.fft.fftshift(amplitude)
        # # This is where we set the pixel corresponding to the basis frequency to be 'lit'
        # a[int(cy+yd), int(cx+xd)] = self.im.shape[0]*self.im.shape[1] / 2.0
        # amplitude = np.fft.fftshift(a)

        # # Note the reconstructed image (top right) as we light up different basis frequencies.

        ########################################################
        # Part 1: Reconstructing from different numbers of basis frequencies
        
        # What if we set some frequency amplitudes to zero, but vary
        # over time which ones we set?

        # Make a circular mask over the amplitude image
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X-(width/2))**2 + (Y-(height/2))**2)
        # Suppress amplitudes less than cutoff radius
        mask = dist_from_center >= self.amplitudeCutoffRadius
        a = np.fft.fftshift(amplitude)
        a[mask] = 0
        amplitude = np.fft.fftshift(a)

        # Slowly undulate the cutoff radius back and forth
        # If radius is small and direction is decreasing, then flip the direction!
        if self.amplitudeCutoffRadius <= 1 and self.amplitudeCutoffDirection < 0:
            self.amplitudeCutoffDirection *= -1
        # If radius is large and direction is increasing, then flip the direction!
        if self.amplitudeCutoffRadius > width/3 and self.amplitudeCutoffDirection > 0:
            self.amplitudeCutoffDirection *= -1
        
        self.amplitudeCutoffRadius += self.amplitudeCutoffDirection


        # ########################################################
        # # Part 2: Replacing amplitude / phase with that of another image
        
        # imJack = cv2.resize( self.imJack, self.im.shape )
        # imJackFFT = np.fft.fft2( imJack )
        # amplitudeJack = np.sqrt( np.power( imJackFFT.real, 2 ) + np.power( imJackFFT.imag, 2 ) )
        # phaseJack = np.arctan2( imJackFFT.imag, imJackFFT.real )
        
        # # Comment in either or both of these
        # #amplitude = amplitudeJack
        # #phase = phaseJack


        #########################################################
        # Part 3: Replacing amplitude / phase with that of a noisy image
        
        # Generate some noise
        self.uniform_noise = np.random.uniform( 0, 1, self.im.shape )
        imNoiseFFT = np.fft.fft2( self.uniform_noise )
        amplitudeNoise = np.sqrt( np.power( imNoiseFFT.real, 2 ) + np.power( imNoiseFFT.imag, 2 ) )
        phaseNoise = np.arctan2( imNoiseFFT.imag, imNoiseFFT.real )
        
        # Comment in either or both of these
        #amplitude = amplitudeNoise
        #phase = phaseNoise


        #########################################################
        ## Part 4: Understanding amplitude and phase
        #
        # Play with the images. What can you discover?
        
        # Zero out phase?
        # phase = np.zeros( self.im.shape ) # + 0.5 * phase

        # Flip direction?
        # phase = -phase

        # # Rotate phase values?
        # self.phaseOffset += 0.05
        # phase = np.arctan2( imFFT.imag, imFFT.real ) + self.phaseOffset
        # # Always place within -pi to pi
        # phase += np.pi
        # phase %= 2*np.pi
        # phase -= np.pi

        # Rotate whole image? Together? Individually?
        #phase = np.rot90( phase )
        #amplitude = np.rot90( amplitude )
        
        # Are these manipulations meaningful?
        # What other manipulations might we perform?


        #########################################################
        ## Reconstruct the original image
        # I need to build a new real+imaginary number from the amplitude / phase
        # This is going from polar coordinates to Cartesian coordinates in the complex number space
        recReal = np.cos( phase ) * amplitude
        recImag = np.sin( phase ) * amplitude
        rec = recReal + 1j*recImag
        # Now inverse Fourier transform
        newImage = np.fft.ifft2( rec ).real
        
        # Image output
        amplitude[amplitude == 0] = np.finfo(float).eps # prevent any log(0) errors
        outputTop = np.concatenate((self.im,newImage),axis = 1)
        outputBottom = np.concatenate((np.log(np.fft.fftshift(amplitude)) / 10, np.fft.fftshift(phase)),axis = 1)
        output = np.clip(np.concatenate((outputTop,outputBottom),axis = 0),0,1)
        # NOTE: One student's software crashed at this line without casting to uint8,
        # but this operation via img_as_ubyte is _slow_. Add this back in if you code crashes.
        #cv2.imshow(self.wn, output)
        #cv2.imshow(self.wn, img_as_ubyte(output))
        cv2.imshow(self.wn, (output*255).astype(np.uint8)) # faster alternative
        
        cv2.waitKey(1)

        return


if __name__ == '__main__':
    live_FFT2()