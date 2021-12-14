import cv2
import os
from PIL import Image
import hyperparameters as hp
import numpy as np
from skimage.transform import rescale, resize

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def takePicture():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        cv2.imshow("Take A Photo with an Emotion", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('p') or key == ord(' '):
            cv2.imwrite("image.jpg", frame)
            break

    cap.release()
    cv2.destroyAllWindows()


def findFace():
    # Get user supplied values
    imagePath = "image.jpg"
    cascPath = "haarcascade_frontalface_default.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    x, y, w, h = faces[0]

    formatImage(x, y, w, h)


def formatImage(x, y, w, h):
    image = Image.open("image.jpg").convert("L")
    image = image.crop((x, y, x+w, y+h))
    image.thumbnail((48, 48), Image.ANTIALIAS)
    image.save('processed_image.jpg', quality=95)


def detectEmotion():
    takePicture()
    findFace()
    # predict emotion using tf, return emotion
    emotion = "Neutral"
    return emotion
