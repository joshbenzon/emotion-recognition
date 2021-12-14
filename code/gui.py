from camera import detectEmotion
from tkinter import *
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

image_name = "face.jpg"


class GUI:
    """
    TKinter GUI for Inside Out Project.
    """

    def __init__(self, master):
        """
        Initializes the GUI, a take picture button, and a quit button.
        """
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
        """
        Takes a picture of the user, finds their emotion, and then returns it to them in a text box.
        """
        self.output.delete("1.0", END)
        emotion = detectEmotion(image_name)
        self.output.insert(END, emotion)


if os.path.exists(image_name):
    os.remove(image_name)
    os.remove("processed_" + image_name)

root = Tk()

gui = GUI(root)
root.mainloop()
