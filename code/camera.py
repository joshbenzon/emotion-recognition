import cv2
import os
from PIL import Image
import hyperparameters as hp
import numpy as np
from skimage.transform import rescale, resize


def takePicture():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('p') or key == ord(' '):
            cv2.imwrite("image.jpg", frame)
            break

    cap.release()
    cv2.destroyAllWindows()

    # img = Image.open("image.jpg").convert("L")

    # img = resize(img, (hp.img_size, hp.img_size))

    # img.thumbnail((48, 48), Image.ANTIALIAS)

    # img = np.array(img)
    # img = resize(img, (48, 48))
    # img = Image.fromarray(img)
    # img.save("image.jpg")


takePicture()
