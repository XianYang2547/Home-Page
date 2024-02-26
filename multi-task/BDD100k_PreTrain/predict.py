import sys

sys.path.insert(0, "/home/xianyang/Desktop/YOLOv8-multi-task-main")

from ultralytics import YOLO

model = YOLO('runs/multi/TRAIN_Pro36+32+32/weights/best.pt')
# model.export(format='onnx')
model.predict(source=r"test.jpg", imgsz=(384, 672), device=[0],
              name='v4_daytime', save=True, show=False, conf=0.5, iou=0.5, show_labels=True)
