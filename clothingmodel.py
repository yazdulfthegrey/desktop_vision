from io import BytesIO
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import pyautogui as pyt
from flask import Flask
import time
from inference import get_model


model = get_model(model_id="clothing-detection-s4ioc/6", api_key='yNSAr9QG1hHBxEIMXJTu')
#model = RFDETRBase()
tracker = sv.ByteTrack()
currentlyseen = []

time.sleep(5)

image = pyt.screenshot()
results = model.infer(image)[0]
detections = sv.Detections.from_inference(results)
num = 0

# create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)



# app =Flask(__name__)

# @app.route('/')
# def home():
#     kickstartmodel()
#     return 'Don\'t mind me, just watching your desktop :) '
    

# @app.route('/getseen')
# def getseen():
#     seenobjects = kickstartmodel()
#     seenthings = []
#     for object in seenobjects:
#         positionarr = []
#         for p in object['position']:
#             positionarr.append(str(p))
#         seenthing = {
#             "object": str(object['object']),
#             "position": positionarr
#         }
#         seenthings.append(seenthing)
#     return {"seenthings":seenthings}



# def kickstartmodel():
#     image = pyt.screenshot()
#     detections = model.predict(image, threshold=0.5)
#     detections = tracker.update_with_detections(detections)
#     num = 0

#     labels = [
#         f"#{tracker_id} {COCO_CLASSES[class_id]}"
#         for class_id, tracker_id
#         in zip(detections.class_id, detections.tracker_id)
#     ]

#     annotated_image = image.copy()
#     annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
#     annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
#     #sv.plot_image(annotated_image)
#     currentlyseen = []
#     for detect in detections:
#         #print(str(detect[0]) + ' ' + str(labels[num])) #this gets data of each label and the person
#         posarr = []
#         arrtouse = detect[0]
#         arrtouse = arrtouse[num*4:(num*4)+4]
#         for d in arrtouse:
#             posarr.append(d)
#         seenobject = {
#             'position':detect[0],
#             'object': labels[num]
#         }
#         print(seenobject)
#         currentlyseen.append(seenobject)
#         num = num + 1
#         seenobjects = currentlyseen

#     return currentlyseen
#         #time.sleep(0.01)
#         #kickstartmodel()
        

# if __name__ == '__main__':
#     app.run(port=1984)