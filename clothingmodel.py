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