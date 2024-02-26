# -*- coding: utf-8 -*-
# @Time    : 2024/1/25 下午3:00
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : onnx_infer.py
# ------❤❤❤------ #

import time
import onnxruntime as ort
from my_utils import *

# label_dict = {0: 'Straight arrow', 1: 'Turn left', 2: 'Turn right', 3: 'Straight and left', 4: 'Straight and right',
#               5: 'Turn left and back', 6: 'Rhomboid', 7: 'Zebra crossing', 8: 'Traffic light', 9: 'Traffic sign',
#               10: 'Drivable area', 11: 'Lane line'}
label_dict = {0:'lta', 1:'straight line', 2:'zebra crossing', 3:'left turn',
              4:'sr turn', 5:'sl turn', 6:'right turn', 7:'Rhomboid', 8:'turn around',
              9:'line'}

ort_session = ort.InferenceSession('./weights/best.onnx')

org_im = cv2.imread('./image_video_file/test.jpg')

# 预处理
start_time = time.time()
image = img_preprocess(org_im, size=(384, 672))
end_time = time.time()
print(f"预处理: {(end_time - start_time)*1000:.2f} ms")

# 推理
start_time = time.time()
preds = ort_session.run(None, {'images': image})
end_time = time.time()
print(f"推理: {(end_time - start_time)*1000:.2f} ms")

# 后处理
start_time = time.time()
results = postprocess(preds, image, org_im,conf_thres=0.25,iou_thres=0.45)
end_time = time.time()
print(f"后处理: {(end_time - start_time)*1000:.2f} ms")

# 显示
start_time = time.time()
image_result,_= show(results, org_im, label_dict)
end_time = time.time()
print(f"显示: {(end_time - start_time)*1000:.2f} ms")

# cv2.imshow('res', image_result)
# cv2.imshow('seg', seg_mask)
# cv2.waitKey()
cv2.imwrite('./runs/image_result.jpg', image_result)
