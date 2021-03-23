import cv2
import numpy as np
import matplotlib.pyplot  as plt
car_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
image = cv2.imread("car.jpg")
def display(img):
    fig=plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)
display(image)

def detect_car(image):
    plate_imge= image.copy()
    plates= car_cascade.detectMultiScale(plate_imge,1.1,1)
    for (x,y,w,h) in plates:
        cv2.rectangle(plate_imge, (x,y), (x+w,y+h), (255,0,0),7)
    return plate_imge
result = detect_car(image)
display(result)
