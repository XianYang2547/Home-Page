# import sys
# import os
#
# custom_package_path = "/home/xianyang/Desktop/Project/Train_Folder/ultralytics"
# sys.path.insert(0, custom_package_path)
# from Train_Folder.ultralytics import YOLO


from ultralytics import YOLO
mo = YOLO('/home/xianyang/Desktop/YFZH_Road_Assets/Train_Folder/runs/segment/mysegalllane2/weights/best.pt')
mo.export(format='onnx', imgsz=640, simplify=True, device=0, half=False)

# mo1 = YOLO('other/剑南大道city/cityseg_new.pt')
# mo1.export(format='onnx', opset=11, imgsz=640, simplify=True, device=0, half=False)
