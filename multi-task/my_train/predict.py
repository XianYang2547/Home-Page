import sys

sys.path.insert(0, "/home/xianyang/Desktop/YOLOv8-multi-task-main")

from ultralytics import YOLO

number = 11  # input how many tasks in your work
# model = YOLO('runs/multi/yolopm11/weights/best.pt')  # Validate the model
# model.export(format='onnx')
#
#
model = YOLO('runs/multi/TRAIN_943/weights/best.pt')
# model.export(format='engine')
model.predict(source=r"/media/xianyang/8806782B06781C7E/Users/MSI/Desktop/test", imgsz=(384,672), device=[0],
              name='predict', save=True, show=False,conf=0.5, iou=0.5, show_labels=True)
