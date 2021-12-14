import cv2
import os
from PIL import Image
import hyperparameters as hp


def findFace():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

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
    image = Image.open("image.jpg").convert("L")
    image = image.crop((x, y, x+w, y+h))
    image.thumbnail((48, 48), Image.ANTIALIAS)
    image.save('image.jpg', quality=95)


findFace()
