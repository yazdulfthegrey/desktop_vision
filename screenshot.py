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
    seenobjects = kickstartmodel()
    sentences = []
    for object in seenobjects:
        #print(object['position'])
        sentences.append('I can see ' + str(object['object']) + ' at position:' + str(object['position']))
        print('I can see ' + str(object['object']) + ' at position:' + str(object['position']))
    return sentences
    

@app.route('/getseen')
def getseen():
    return "I can currently see" + currentlyseen



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
    sv.plot_image(annotated_image)
    currentlyseen = []
    for detect in detections:
        print(str(detect[0]) + ' ' + str(labels[num])) #this gets data of each label and the person
        seenobject = {
            'position': detect[0],
            'object': labels[num]
        }
        currentlyseen.append(seenobject)
        num = num + 1
    return currentlyseen

if __name__ == '__main__':
    app.run()