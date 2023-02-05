import cv2
import numpy as np
import os

recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.read('trainer/trainer.yml') #it reads our already trained model!
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath) #this is basically the face detector


font = cv2.FONT_HERSHEY_TRIPLEX

id = 0

names = [0, "Ujjawal sir", 2, 3, 4, 5]
#put in names here corresponding to the ids you give to the faces!
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
#same as before. starting the camera and setting its frame size.

minW = 0.1*cam.get(3) # == 64
minH = 0.1*cam.get(4) # == 48

while True:
    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
        id, confidence = recogniser.predict(gray[y:y+h, x:x+w])
        #returns ths id based on the face scanned and its confidence level.
        
        if(confidence < 100):
            id = names[id]
            confidence = "{0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "{0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x+5, y+5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 255), 1)
        #puts id and the confidence level in the image
    cv2.imshow('Camera', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27 :
        break
    #same as before
print("\n [INFO] Exiting the Program")
cam.release()
cv2.destoryAllWindows()




