import cv2
import numpy as np
import matplotlib.pyplot  as plt
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
image = cv2.imread("google.jpg")

fix_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


plt.imshow(fix_image)
faces= face_cascade.detectMultiScale(fix_image,1.3,2)

def detect_face(fix_image):
    face_rects=face_cascade.detectMultiScale(fix_image)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(fix_image, (x,y), (x+w,y+h), (255,0,0),10)
    return fix_image
result = detect_face(fix_image)
plt.imshow(result)
