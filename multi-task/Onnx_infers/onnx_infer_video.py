# -*- coding: utf-8 -*-
# @Time    : 2024/1/29 8:53
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : onnx_infer_video.py
# ------❤❤❤------ #

import time
import onnxruntime as ort
from my_utils import *
import cv2

video_path = './image_video_file/test.mp4'
cap = cv2.VideoCapture(video_path)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# output_path = './runs/test.mp4'
# fourcc = cv2.VideoWriter.fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))

label_dict = {0: 'Straight arrow', 1: 'Turn left', 2: 'Turn right', 3: 'Straight and left', 4: 'Straight and right',
              5: 'Turn left and back', 6: 'Rhomboid', 7: 'Zebra crossing', 8: 'Traffic light', 9: 'Traffic sign',
              10: 'Drivable area', 11: 'Lane line'}

ort_session = ort.InferenceSession('./weights/best.onnx',providers=['CUDAExecutionProvider','CPUExecutionProvider'])
available_providers = ort.get_available_providers()
if 'CUDAExecutionProvider' in available_providers:
    print('ONNX Runtime GPU support is available.')
else:
    print('ONNX Runtime GPU support is not available.')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 预处理
    image = img_preprocess(frame, size=(384, 672))
    # 推理
    start_time = time.time()
    preds = ort_session.run(None, {'images': image})
    end_time = time.time()
    print(f"推理: {(end_time - start_time)*1000:.2f} ms")
    # 后处理
    start_time = time.time()
    results = postprocess(preds, image, frame,conf_thres=0.5,iou_thres=0.5)
    end_time = time.time()
    print(f"后处理: {(end_time - start_time)*1000:.2f} ms")
    # 显示
    start_time = time.time()
    image_result ,lane_seg= show(results, frame, label_dict)
    end_time = time.time()
    print(f"显示: {(end_time - start_time)*1000:.2f} ms")

    cv2.imshow('res', image_result)
    cv2.imshow('lane',lane_seg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # out.write(image_result)