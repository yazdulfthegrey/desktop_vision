import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import pyautogui as pyt
import cv2
from flask import Flask

model = RFDETRBase()
currentlyseen = []

app =Flask(__name__)

@app.route('/')
def home():
    return 'Don\'t mind me, just watching your desktop :) '
    

@app.route('/getseen')
def getseen():
    seenobjects = kickstartmodel()
    seenthings = []
    for object in seenobjects:
        positionarr = []
        for p in object['position']:
            positionarr.append(str(p))
        seenthing = {
            "object": str(object['object']),
            "position": positionarr
        }
        seenthings.append(seenthing)
    return {"seenthings":seenthings}



def kickstartmodel():
    image = pyt.screenshot()
    detections = model.predict(image, threshold=0.5)
    num = 0

    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    #sv.plot_image(annotated_image)
    currentlyseen = []
    for detect in detections:
        print(str(detect[0]) + ' ' + str(labels[num])) #this gets data of each label and the person
        posarr = []
        arrtouse = detect[0]
        arrtouse = arrtouse[num*4:(num*4)+4]
        for d in arrtouse:
            posarr.append(d)
        seenobject = {
            'position':detect[0],
            'object': labels[num]
        }
        print(seenobject)
        currentlyseen.append(seenobject)
        num = num + 1
    return currentlyseen

if __name__ == '__main__':
    app.run(port=1984)