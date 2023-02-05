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
    
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.rectangle(image, start_point, end_point, color, thickness)
        #color: It is the color of border line of rectangle to be drawn. For BGR, we pass a tuple. eg: (255, 0, 0) for blue color.
        #thickness: It is the thickness of the rectangle border line in px. Thickness of -1 px will fill the rectangle shape by the specified color.
        #Return Value: It returns an image.
        #basically rectangle banaya hai img pe where the face is
        count+=1
    
        cv2.imwrite("dataset/User."+str(face_id) + '.' + str(count) + ".jpg", gray_img[y:y+h, x:x+w])
        #It returns true if image is saved successfully.
        #saved the gray image(only the face part - rectangle)
        cv2.imshow('image', img) #viewing the image with rectangle drawn in (0, 255, 0) - green!
    

    k = cv2.waitKey(100) & 0xff
    #0xFF is a hexadecimal constant which is 11111111 in binary. By using bitwise AND (&) with this constant, it leaves only the last 8 bits of the original (in this case, whatever cv2.waitKey(0) is).
    #basically the code says to either wait for 100ms or wait until specified key is pressed
    if k == 27:
        break
    elif count>=30:
        break
#basically 30 photos ke baad break. idk what the significance of 27 was as waitkey mp returns the ascii value of the key pressed but the guy said it means to wait for 27 seconds idk.

print("\n [INFO] Exiting Program")
cam.release() #basically turns the camera off
cv2.destroyAllWindows() 



#this code is used to detect the faces initially! And also to create a dataset. Now we need to train our model using the dataset.


    