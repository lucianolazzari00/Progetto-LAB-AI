import cv2
import torch
import sys
import os
from AdultChild_classifier import ComplexCNN, predict
import torchvision.transforms as transforms
from PIL import Image


try:
    img_input = sys.argv[1]
except IndexError:
    print("Error! missing one argument \n -usage: facial-detection.py [filename]")
    exit(0)

absolute_path = os.path.join(os.getcwd(), 'assets', 'img_test', img_input);
img = cv2.imread(absolute_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

##=====import the model=======

# Load state dict from the disk (make sure it is the same name as above)
state_dict = torch.load("adch_model2.0.tar")
#state_dict = torch.load("adch_model.tar", map_location=torch.device('cpu')) ### TODO cambia la cpu

# Create a new model and load the state
trained_model = ComplexCNN()
trained_model.load_state_dict(state_dict)
trained_model = trained_model.to(device)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
)
for (x, y, w, h) in faces:
    cropped_face = img[y:y+h,x:x+w]
    prediction = predict(trained_model,cropped_face)
    print(prediction)
    if prediction == 1: #face is a child
        blurred = cv2.blur(img[y:y+h,x:x+w],(20,20))
        img[y:y+h,x:x+w] = blurred
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)


cv2.imshow('bl_img', img)
cv2.waitKey(4000)

