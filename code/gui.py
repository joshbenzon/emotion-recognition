from camera import detectEmotion
from tkinter import *
from PIL import ImageTk, Image
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


class GUI:
    def __init__(self, master):
        self.master = master
        master.title("Inside Out Emotion Recognition")

        self.label = Label(master, text="Take a Picture of Your Face!")
        self.label.pack()

        self.camera_button = Button(
            master, text="Take Picture", command=self.getEmotion)
        self.camera_button.pack()

        self.close_button = Button(master, text="Quit", command=master.quit)
        self.close_button.pack()

        self.output = Text(root, height=1,
                           width=13,
                           bg="light blue")
        self.output.pack()

    def getEmotion(self):
        self.output.delete("1.0", END)
        emotion = detectEmotion()
        self.output.insert(END, emotion)


if os.path.exists("image.jpg"):
    os.remove("image.jpg")
    os.remove("processed_image.jpg")

root = Tk()

gui = GUI(root)
root.mainloop()
