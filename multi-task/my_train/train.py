import sys

sys.path.insert(0, "/home/xianyang/Desktop/YOLOv8-multi-task")

from ultralytics import YOLO

model = YOLO('/media/xianyang/0F778EE14F116528/YOLOv8-multi-task/my_train/runs/multi/TRAIN_943_no_pred/weights/last.pt',
             task='multi')

model.train(data='mydata.yaml', batch=24, epochs=100, imgsz=640,
            device=[0], name='TRAIN_943_no_pred', val=True, task='multi', classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            combine_class=[], single_cls=False)
