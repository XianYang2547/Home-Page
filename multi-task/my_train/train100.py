import sys

sys.path.insert(0, "/home/xianyang/Desktop/YOLOv8-multi-task")

from ultralytics import YOLO

model = YOLO('/home/xianyang/Desktop/YOLOv8-multi-task/BDD100k_PreTrain/runs/multi/TRAIN_Pro36+32+32/weights/best.pt',
             task='multi')

model.train(data='data100.yaml', batch=24, epochs=100, imgsz=640,
            device=[0], name='TRAIN_100', val=True, task='multi', classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            combine_class=[], single_cls=False)
