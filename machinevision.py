import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRBase()

def callback(frame, index):
    detections = model.predict(frame, threshold=0.5)
    num = 0
        
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)
    for detect in detections:
        #print(labels[num])
        print(str(detect[0]) + ' ' + str(labels[num])) #this gets data of each label and the person
        num = num + 1
    return annotated_frame

sv.process_video(
    source_path="./vid.mp4",
    target_path="./newvideo.mp4",
    callback=callback
)