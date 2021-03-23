import cv2
import numpy as np
import matplotlib.pyplot  as plt
full_body=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

cap = cv2.VideoCapture("People.mp4")

while cap.isOpened():
    ret,frame = cap.read()
    bodies = full_body.detectMultiScale(frame,1.2,3)
    if ret == True:
        for (x,y,w,h) in bodies:
            cv2.rectangle(frame, (x,y),(x+w,y+h), (255,0,0),10)
            cv2.imshow("Pedestrians",frame)
        if cv2.waitKey(1) == 27:
            break
    else : 
        break
cap.release()
cv2.destroyAllWindows()
