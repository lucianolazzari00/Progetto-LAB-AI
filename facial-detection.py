import cv2
import sys
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

try:
    img_input = sys.argv[1]
except IndexError:
    print("Error! missing one argument \n -usage: facial-detection.py [filename]")
    exit(0)

absolute_path = os.path.join(os.getcwd(), 'assets', 'img_test', img_input);
img = cv2.imread(absolute_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
)
for (x, y, w, h) in faces:
    if 1: #face is a child
        blurred = cv2.blur(img[y:y+h,x:x+w],(20,20))
        img[y:y+h,x:x+w] = blurred
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)


cv2.imshow('bl_img', img)
cv2.waitKey(4000)
