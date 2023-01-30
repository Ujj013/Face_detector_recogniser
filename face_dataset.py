import cv2
import os #for os permissions

cam = cv2.VideoCapture(0) #to capture a video using our camera

#we will set the frame size used by our camera
cam.set(3, 640) #for width - 3 is the propid
cam.set(4, 480) #for height - 4 is the propid
#therefore 640x480 is set as the frame size

#now cam stores the video
#now we'll detect and extract the face out of this video
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#hence opencv will use the harrcascade file to detect the face
#face_detector is an object like in ml

#let's store an id for the face as well. this will be entered by the user
face_id = input('\n Enter user id:')

#now some unnecessary print messages
print("[INFO] Initializing Face Capture.")

#this model will give around 60% accuracy and to increase accuracy we build a bigger dataset!
count = 0
while(True):
    ret, img = cam.read() #cam.read() Grabs, decodes and returns the next video frame.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #we convert our image into grayscale as all processing is done on gray images
    faces = face_detector.detectMultiScale(gray_img, 1.3, 5) #1.3, 5 are standard arguments
    #It takes 3 common arguments — the input image, scaleFactor, and minNeighbours.
    #and will return a rectangle with coordinates(x,y,w,h) around the detected face.