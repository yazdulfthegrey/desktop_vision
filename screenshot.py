import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import pyautogui as pyt
from flask import Flask
import time
from inference import get_model


model = RFDETRBase()
clothemodel = get_model(model_id="clothing-detection-s4ioc/6", api_key='yNSAr9QG1hHBxEIMXJTu')
firearmmodel = get_model(model_id="gun_detection-coyoc/1", api_key='yNSAr9QG1hHBxEIMXJTu')
tracker = sv.ByteTrack()
currentlyseen = []

app =Flask(__name__)

@app.route('/')
def home():
    kickstartmodel()
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
            "position": positionarr,
            "clothesworn": object['clothesworn'],
            "armed": object['armed']
        }
        seenthings.append(seenthing)
    return {"seenthings":seenthings}

def scanforclothes(img):
    clothesworn = []

    results = clothemodel.infer(img)[0]
    detections = sv.Detections.from_inference(results)
    for detect in detections:
        #print(detect[5]['class_name'])
        clothesworn.append(detect[5]['class_name'])

    # create supervision annotators
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(scene=img, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # display the image
    #sv.plot_image(annotated_image)
    return clothesworn

def scanforfirearm(img):
    firearmspotted = False

    results = firearmmodel.infer(img)[0]
    detections = sv.Detections.from_inference(results)
    for detect in detections:
        if (detect[5]['class_name'] == 'Gun'):
            firearmspotted = True

    # create supervision annotators
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(scene=img, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # display the image
    #sv.plot_image(annotated_image)
    return firearmspotted



def kickstartmodel():
    image = pyt.screenshot()
    detections = model.predict(image, threshold=0.5)
    detections = tracker.update_with_detections(detections)
    num = 0

    labels = [
        f"#{tracker_id} {COCO_CLASSES[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    sv.plot_image(annotated_image)
    currentlyseen = []
    for detect in detections:
        #print(str(detect[0]) + ' ' + str(labels[num])) #this gets data of each label and the person
        posarr = []
        arrtouse = detect[0]
        arrtouse = arrtouse[num*4:(num*4)+4]
        subimage = image.copy()
        subimage = subimage.crop(detect[0])
        clothesworn = scanforclothes(subimage)
        firearm = scanforfirearm(subimage)
        for d in arrtouse:
            posarr.append(d)
        seenobject = {
            'position':detect[0],
            'object': labels[num],
            'clothesworn':clothesworn,
            'armed': firearm
        }
        #print(seenobject)
        currentlyseen.append(seenobject)
        num = num + 1
        seenobjects = currentlyseen

    return currentlyseen
        #time.sleep(0.01)
        #kickstartmodel()
        

if __name__ == '__main__':
    app.run(port=1984)