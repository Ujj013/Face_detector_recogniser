import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset'

recogniser = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):

    imagePaths = [os.path.join(path, f) for in os.listdir(path)]
    #it should be for f in ... maybe
    #it joins one or more path components intelligently. So basically an array of paths is created.
    #Each path is of type "path/f" where path is defined above and f is in os.listdir(path)
    #it returns a string which represents the concatenated path components. 
    #os.listdir() method in python is used to get the list of all files and directories in the 
    #specified directory.
    #Basically this code stores the paths of all files in the dataset folder in the imagePaths variable 
    facesamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        #Returns a converted copy of this image
        # Syntax: Image.convert(mode=None, matrix=None, dither=None, palette=0, colors=256)
        # for grayscale the mode is 'L'
        img_numpy = np.array(PIL_img, 'uint8')#converting the image into a numpy array

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        #os.path.split splits the path into head and tail and returns a list containing these.
        #the tail is basically the filename in the path and is the last object in the returned list
        #-1 is used to access this last string. now this string is split into multiple strings by 
        #a delimmiter '.' and then the first element of this list is accessed.

        faces = detector.detectMultiScale(img_numpy)
        #default scalefactor and minneighbours for this are 1.1 and 3.

        for (x, y, w, h) in faces:
            facesamples.append(img.numpy[y:y+h, x:x+w])
            ids.append(id)

        return facesamples.ids

print("\n [INFO] Training faces.....")

faces, ids = getImagesAndLabels(path)
recogniser.train(faces, np.array(ids))

recognizer.write('trainer/trainer.yml')

print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))