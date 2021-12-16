import cv2
import os
from PIL import Image
import hyperparameters as hp
import predict

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def takePicture(image_name):
    """
    This function takes a picture using the OpenCV Library.
    """
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        cv2.imshow("Take A Photo with an Emotion", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('p') or key == ord(' '):
            cv2.imwrite(image_name, frame)
            break

    cap.release()
    cv2.destroyAllWindows()


def findFace(image_name):
    """
    Finds the users face in the picture that they take.
    """
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    image = cv2.imread(image_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    x, y, w, h = faces[0]
    formatImage(x, y, w, h, image_name)


def formatImage(x, y, w, h, image_name):
    """
    Converts the image to be a square photo of the person's face, like the training data.
    """
    image = Image.open(image_name).convert("L")
    image = image.crop((x, y, x+w, y+h))
    image.thumbnail((hp.img_size, hp.img_size))
    image.save('processed_' + image_name, quality=100, subsampling=0)


def detectEmotion(image_name):
    """
    Takes a picture of the user, finds their face, and then finds assigns them an emotion and returns the emotion.
    """
    takePicture(image_name)
    findFace(image_name)
    emotion = predict("processed_" + image_name)
    # emotion = "N/A"
    return emotion
