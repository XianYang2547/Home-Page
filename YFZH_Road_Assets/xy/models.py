# -*- coding: utf-8 -*-
# @Time    : 2024/5/20 5:20
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : models.py
# ------❤❤❤------ #

import logging
import os
import time
from collections import Counter

from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

import colorlog
import cupy as cp
import cv2
import numpy as np
import onnxruntime as ort
import tensorrt as trt
from cuda import cudart


os.environ['CUDA_MODULE_LOADING'] = 'LAZY'


class My_dection():
    def __init__(self):
        self.Det_Models = None
        self.Seg_Models = None
        self.conf_threshold = 0.45
        self.iou_threshold = 0.45
        self.label_dict = None
        self.classes = None
        self.color_palette = None
        self.SUF = ('.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm')
        self.logger = My_LoggerConfig().get_logger()

    def __call__(self, image):
        pre_start_time = time.time()
        im, ratio, pad_w, pad_h = self.preprocess(image, [640, 640])
        pre_end_time = time.time()
        # 推理
        inf_start_time = time.time()
        pred0 = self.Det_Models(im)
        pred1 = self.Seg_Models(im)
        inf_end_time = time.time()
        # 后处理
        post_start_time = time.time()
        results = self.postprocess(pred0, pred1, im, image, ratio, pad_w, pad_h, self.conf_threshold,
                                   self.iou_threshold)
        post_end_time = time.time()
        # 统计推理时间用于打印
        self.message = [(pre_end_time - pre_start_time) * 1000, (inf_end_time - inf_start_time) * 1000,
                        (post_end_time - post_start_time) * 1000]
        return results

    def warm_up(self, n):
        for i in range(n):
            dummy_input = np.random.rand(1, 3, 640, 640).astype(self.Det_Models.dtypes)
            self.Det_Models(dummy_input)
            self.Seg_Models(dummy_input)
        print("Model Warmup " + f"{n}" + " times")

    def preprocess(self, image, img_size):
        "预处理"
        img, ratio, (dw, dh) = self.letterbox(image, img_size, stride=32, auto=False)
        # 使用 CuPy 进行图像处理
        img_cp = cp.asarray(img)  # 将 NumPy 数组转换为 CuPy 数组
        img_cp = img_cp[:, :, ::-1].transpose(2, 0, 1)  # 转换色彩空间并转置维度
        img_cp = cp.ascontiguousarray(img_cp)  # 转换为连续数组
        img_cp = img_cp.astype(cp.float16 if self.Det_Models.dtypes == np.float16 else cp.float32)  # 使用 CuPy 进行类型转换
        img_cp /= 255.0  # 标准化像素值
        img = cp.asnumpy(img_cp)  # 将 CuPy 数组转换回 NumPy 数组
        return img[None], ratio, dw, dh  # 返回预处理后的图像和相关参数

    def letterbox(self, im, new_shape, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        "等比例缩放 "
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                            multi_label=False, labels=(), max_det=300, nc=0, max_time_img=0.05, max_nms=30000, max_wh=7680, ):
        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        if isinstance(prediction,
                      (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].max(1) > conf_thres  # candidates
        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        time_limit = 0.5 + max_time_img * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        output = [np.zeros((0, 6 + nm))] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x.transpose(1, 0)[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = np.zeros((len(lb), nc + nm + 5))
                v[:, :4] = lb[:, 1:5]  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x = np.concatenate((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls) ndarray不好直接split
            # box, cls, mask = np.split(x,[4, nc, nm], 1)
            box = self.xywh2xyxy(x[:, :4])
            cls = x[:, 4:mi]
            mask = x[:, mi:]

            if multi_label:
                i, j = (cls > conf_thres).nonzero(as_tuple=False).T
                x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf = np.max(x[:, 4:mi], 1).reshape(box.shape[:1][0], 1)
                j = np.argmax(x[:, 4:mi], 1).reshape(box.shape[:1][0], 1)
                x = np.concatenate((box, conf, j, mask), 1)[conf.reshape(box.shape[:1][0]) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == np.array(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            x = x[x[:, 4].argsort(axis=0)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = self.nms_boxes(boxes, scores)  # NMS
            i = i[:max_det]  # limit detections
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = np.multiply(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]

        return output

    def xywh2xyxy(self, x):
        "中心点转换"
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    def nms_boxes(self, boxes, scores):
        "NMS"
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        areas = w * h
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= 0.45)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def box_iou(self, box1, box2, eps=1e-7):
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (np.min(a2, b2) - np.max(a1, b1)).clamp(0).prod(2)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    def postprocess(self, pred0, pred1, im0, image, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        pass

    def scale_mask(self, masks, im0_shape, ratio_pad=None):
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]),
                           interpolation=cv2.INTER_LINEAR)  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

    def postprocess_det(self, preds, img, org_img, conf_thres, iou_thres):
        pred0 = self.non_max_suppression(preds, conf_thres, iou_thres)
        for i, pred in enumerate(pred0):
            pred[:, :4] = self.scale_boxes(img.shape[2:], pred[:, :4], org_img.shape)
            return pred[:, :6]

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        self.clip_boxes(boxes, img0_shape)
        return boxes

    def clip_boxes(self, boxes, shape):
        # Clip boxes (xyxy) to image shape (height, width)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

    def my_show(self, det, seg, im, zed_show=False):
        s = 'Result: '
        if len(det) != 0:
            if zed_show:  # ZED风格
                overlay = im.copy()
                for i in det[0]:
                    x1, y1, x2, y2, score, class_id = i
                    self.drawZED(im, [x1, y1, x2, y2], score, class_id, overlay)
                cv2.addWeighted(im, 0.7, overlay, 0.3, 0.0, im)
            else:  # YOLO风格  耗时比zed快一半
                for i in det[0]:
                    x1, y1, x2, y2, score, class_id = i
                    self.drawYOLO(im, [x1, y1, x2, y2], score, class_id)
            for count, element in [(j, self.label_dict[i]) for i, j in
                                   Counter([int(result[-1]) for result in det[0]]).items()]:
                s += f"{element}*{count} "

        if len(seg[0]) != 0:
            bboxes, segments = seg[0], seg[1]
            for count, element in [(j, self.classes[i]) for i, j in
                                   Counter([int(result[-1]) for result in bboxes[0]]).items()]:
                s += f"{element}*{count} "
            im_canvas = im.copy()
            for (*box, conf, cls_), segment in zip(bboxes[0], segments[0]):
                cv2.polylines(im, np.int32([segment]), True,
                              self.color_palette[int(cls_) + len(self.label_dict)], 2)
                # cv2.rectangle(im, [int(num) for num in box][:2], [int(num) for num in box][2:],
                #               self.color_palette[int(cls_) + len(self.label_dict)], 2)
                cv2.putText(im, f'{self.classes[cls_]}:{conf:.3f}',
                            (int(box[0] + (box[2] - box[0]) / 2), int(box[1] + (box[3] - box[1]) / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            self.color_palette[int(cls_) + len(self.label_dict)], thickness=2)
                cv2.fillPoly(im_canvas, np.int32([segment]),
                             self.color_palette[int(cls_) + len(self.label_dict)])

            im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)
        message = f"🚀🚀🚀 Total:{(self.message[0] + self.message[1] + self.message[2]):6.2f}ms," \
                  f" Pre:{self.message[0]:6.2f}ms," \
                  f" Infer:{self.message[1]:6.2f}ms," \
                  f" Post:{self.message[2]:6.2f}ms - | \033[95m{s}\033[0m"
        self.logger.info(message)
        return im

    def my_show_track(self, det, seg, im):
        s = 'Result: '
        if len(det) != 0:
            for i in det[0]:
                if np.all(i == 0):  # 过滤掉全0的
                    continue
                x1, y1, x2, y2, score, class_id, track_id = i
                self.drawYOLO(im, [x1, y1, x2, y2], score, class_id, track_id)
            for count, element in [(j, self.label_dict[i]) for i, j in
                                   Counter([int(result[-2]) for result in det[0]]).items()]:  # -2才是类别 -1是id
                s += f"{element}*{count} "

        if len(seg[0]) != 0:
            bboxes, segments = seg[0], seg[1]
            for count, element in [(j, self.classes[i]) for i, j in
                                   Counter([int(result[-1]) for result in bboxes[0]]).items()]:
                s += f"{element}*{count} "
            im_canvas = im.copy()
            for (*box, conf, track_id, cls_), segment in zip(bboxes[0], segments[0]):
                if int(cls_) != 0:
                    cv2.polylines(im, np.int32([segment]), True,
                                  self.color_palette[int(cls_) + len(self.label_dict)], 2)
                    # cv2.rectangle(im, [int(num) for num in box][:2], [int(num) for num in box][2:],
                    #               self.color_palette[int(cls_) + len(self.label_dict)], 2)
                    # cv2.putText(im, f'{self.classes[cls_]}:{conf:.3f} ID: {int(track_id)}',
                    #             ([int(num) for num in box][0], [int(num) for num in box][1] - 2),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    #             self.color_palette[int(cls_) + len(self.label_dict)], thickness=2)
                    cv2.putText(im, f'{self.classes[cls_]}:{conf:.3f} ID: {int(track_id)}',
                                (int(box[0] + (box[2] - box[0]) / 2), int(box[1] + (box[3] - box[1]) / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                self.color_palette[int(cls_) + len(self.label_dict)], thickness=2)
                    cv2.fillPoly(im_canvas, np.int32([segment]),
                                 self.color_palette[int(cls_) + len(self.label_dict)])
                if int(cls_) == 0:  # 车道线
                    cv2.putText(im, f'{self.classes[cls_]}:{conf:.3f} ID: {int(track_id)}',
                                (int(box[0] + (box[2] - box[0]) / 2), int(box[1] + (box[3] - box[1]) / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)
                    # cv2.polylines(im, np.int32([segment]), True, (255, 0, 255), 2)
                    # cv2.fillPoly(im_canvas, np.int32([segment]),
                    #              self.color_palette[int(cls_) + len(self.label_dict)])


            im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)
        message = f"🚀🚀🚀 Total:{(self.message[0] + self.message[1] + self.message[2]):6.2f}ms," \
                  f" Pre:{self.message[0]:6.2f}ms," \
                  f" Infer:{self.message[1]:6.2f}ms," \
                  f" Post:{self.message[2]:6.2f}ms - | \033[95m{s}\033[0m"
        self.logger.info(message)

        return im

    def my_show_yellow_color(self, det, seg, im, svo=False, zed_show=False):
        s = 'Result: '
        if len(det) != 0:
            if zed_show:  # ZED风格
                overlay = im.copy()
                for i in det[0]:
                    x1, y1, x2, y2, score, class_id = i
                    self.drawZED(im, [x1, y1, x2, y2], score, class_id, overlay)
                cv2.addWeighted(im, 0.7, overlay, 0.3, 0.0, im)
            else:  # YOLO风格
                for i in det[0]:
                    x1, y1, x2, y2, score, class_id = i
                    self.drawYOLO(im, [x1, y1, x2, y2], score, class_id)
            for count, element in [(j, self.label_dict[i]) for i, j in
                                   Counter([int(result[-1]) for result in det[0]]).items()]:
                s += f"{element}*{count} "

        if len(seg[0]) != 0:
            bboxes, segments = seg[0], seg[1]
            for count, element in [(j, self.classes[i]) for i, j in
                                   Counter([int(result[-1]) for result in bboxes[0]]).items()]:
                s += f"{element}*{count} "
            im_canvas = im.copy()
            for (*box, conf, cls_), segment in zip(bboxes[0], segments[0]):
                if svo:
                    points = np.int32(segment)
                    mask = np.zeros(im.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [points], 255)
                    # 虚实
                    img = cv2.bitwise_and(im, im, mask=mask)
                    # 将图像转换为二维数组，每个像素点为一个三维（RGB）向量
                    pixels = img.reshape((-1, 3))
                    pixels = np.float32(pixels)
                    # 使用K-means聚类来分割颜色
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                    k = 3  # 聚类的类数，假设有3种不同颜色的块
                    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    # 将中心点转换为整数（RGB颜色值）
                    centers = np.uint8(centers)
                    # 根据标签重建量化后的图像
                    quantized_img = centers[labels.flatten()]
                    quantized_img = quantized_img.reshape(img.shape)
                    # 将每一个聚类结果转化为独立的色块掩码
                    masks = [cv2.inRange(quantized_img, centers[i], centers[i]) for i in range(k)]

                    # 计算每个 mask 中非零像素的数量（即色块面积）
                    mask_areas = [np.count_nonzero(mask) for mask in masks]
                    # 图像总面积
                    total_area = img.shape[0] * img.shape[1]
                    # 设置一个面积上限比例，比如大于 80% 视为背景
                    background_threshold = 0.8 * total_area
                    # 设置最小面积阈值，比如小于 500 像素的色块不保留
                    min_area_threshold = 800
                    # 过滤掉背景和面积小于 500 的 mask
                    filtered_masks = [mask for mask, area in zip(masks, mask_areas) if
                                      background_threshold > area > min_area_threshold]
                    # 实线的np.count_nonzero前景比背景大

                    # 统计每个聚类中的色块数量，并过滤较小面积的色块
                    # 对每个色块掩码进行轮廓检测
                    block_counts = []
                    for i, mask in enumerate(filtered_masks):
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        block_counts.append(len(contours))
                        # 显示每个聚类后的色块
                        color_img = cv2.bitwise_and(img, img, mask=mask)
                        # cv2.imshow(f'Color Block {i + 1}', color_img)

                    # 判断是否为虚线
                    for i, count in enumerate(block_counts):
                        print(f'Color Block {i + 1} has {count} contours.')
                    if max(block_counts) > 1:
                        linetype='dashed line'
                    else:
                        linetype='solid line'

                    # 颜色
                    mean_color = cv2.mean(im, mask=mask)[:3]
                    mean_bgr = np.array(mean_color, dtype=np.uint8).reshape(1, 1, 3)
                    mean_hsv = cv2.cvtColor(mean_bgr, cv2.COLOR_BGR2HSV).flatten()
                    hue, saturation, value = mean_hsv  # 色调 饱和度 亮度
                    is_yellow = (15 <= hue <= 45) and (saturation > 25) and (value > 70)

                    cv2.polylines(im, np.int32([segment]), True,
                                  self.color_palette[int(cls_) + len(self.label_dict)], 2)

                    cv2.putText(im, f'{self.classes[cls_]}:{conf:.3f} {is_yellow} {linetype}',
                                (int(box[0] + (box[2] - box[0]) / 2), int(box[1] + (box[3] - box[1]) / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                self.color_palette[int(cls_) + len(self.label_dict)], thickness=2)

                else:
                    cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)
                    cv2.fillPoly(im_canvas, np.int32([segment]),
                                 self.color_palette[int(cls_) + len(self.label_dict)])  # fixme segment 为空好像报错
            im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)
        message = f"🚀🚀🚀Pre:{self.message[0]:6.2f}ms, Infer:{self.message[1]:6.2f}ms, Post:{self.message[2]:6.2f}ms - | \033[95m{s}\033[0m"
        self.logger.info(message)

        return im

    def drawYOLO(self, im0, box, score, class_id, track_id=None):
        "画yolo框"
        color = self.color_palette[int(class_id)]
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(im0, p1, p2, color, 2)
        label = f'{self.label_dict[class_id]}: {score:.2f} ID: {int(track_id)}' if track_id else f'{self.label_dict[class_id]}: {score:.2f}'
        w0, h0 = cv2.getTextSize(label, 0, 0.5, 1)[0]
        outside = p1[1] - h0 >= 3
        p2 = p1[0] + w0, p1[1] - h0 - 3 if outside else p1[1] + h0 + 3
        cv2.rectangle(im0, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(im0, label, (p1[0], p1[1] - 2 if outside else p1[1] + h0 + 2), 0, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)

    def drawZED(self, im, box, score, class_id, overlay):
        "画zed框"
        color = self.color_palette[int(class_id)]
        top_left_corner = [box[0], box[1]]
        top_right_corner = [box[2], box[1]]
        bottom_right_corner = [box[2], box[3]]
        bottom_left_corner = [box[0], box[3]]
        # Creation of the 2 horizontal lines
        cv2.line(im, (int(top_left_corner[0]), int(top_left_corner[1])),
                 (int(top_right_corner[0]), int(top_right_corner[1])), color, 3)
        cv2.line(im, (int(bottom_left_corner[0]), int(bottom_left_corner[1])),
                 (int(bottom_right_corner[0]), int(bottom_right_corner[1])), color, 3)
        # Creation of 2 vertical lines
        self.draw_vertical_line(im, bottom_left_corner, top_left_corner, color, 3)
        self.draw_vertical_line(im, bottom_right_corner, top_right_corner, color, 3)
        roi_height = int(top_right_corner[0] - top_left_corner[0])
        roi_width = int(bottom_left_corner[1] - top_left_corner[1])
        overlay_roi = overlay[int(top_left_corner[1]):int(top_left_corner[1] + roi_width)
        , int(top_left_corner[0]):int(top_left_corner[0] + roi_height)]

        overlay_roi[:, :] = color
        p1, p2 = (int(top_left_corner[0]), int(top_left_corner[1])), (
            int(bottom_right_corner[0]), int(bottom_right_corner[1]))

        confidence = str('{:.2f}'.format(score))
        label = f"{self.label_dict[class_id]} {confidence}"
        w0, h0 = cv2.getTextSize(label, 0, 0.5, 1)[0]
        outside = p1[1] - h0 >= 3
        p2 = p1[0] + w0 + w0 // 4, p1[1] - h0 - 3 if outside else p1[1] + h0 + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)

        cv2.putText(im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h0 + 2),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def draw_vertical_line(self, left_display, start_pt, end_pt, clr, thickness):
        "画垂直线"
        n_steps = 7
        pt1 = [((n_steps - 1) * start_pt[0] + end_pt[0]) / n_steps, ((n_steps - 1) * start_pt[1] + end_pt[1]) / n_steps]
        pt4 = [(start_pt[0] + (n_steps - 1) * end_pt[0]) / n_steps, (start_pt[1] + (n_steps - 1) * end_pt[1]) / n_steps]

        cv2.line(left_display, (int(start_pt[0]), int(start_pt[1])), (int(pt1[0]), int(pt1[1])), clr, thickness)
        cv2.line(left_display, (int(pt4[0]), int(pt4[1])), (int(end_pt[0]), int(end_pt[1])), clr, thickness)

    # for masks2segments
    def draw_axis(self, img, p, q, color, scale):
        angle = np.arctan2(p[1] - q[1], p[0] - q[0])  # 计算角度
        hypotenuse = np.sqrt((p[1] - q[1]) ** 2 + (p[0] - q[0]) ** 2)  # 计算斜边长度

        # 放大箭头长度
        q = (int(p[0] - scale * hypotenuse * np.cos(angle)),
             int(p[1] - scale * hypotenuse * np.sin(angle)))

        # 画主轴线
        cv2.line(img, p, q, color, 1, cv2.LINE_AA)

        # 画箭头
        p1 = (int(q[0] + 9 * np.cos(angle + np.pi / 4)),
              int(q[1] + 9 * np.sin(angle + np.pi / 4)))
        cv2.line(img, q, p1, color, 1, cv2.LINE_AA)

        p2 = (int(q[0] + 9 * np.cos(angle - np.pi / 4)),
              int(q[1] + 9 * np.sin(angle - np.pi / 4)))
        cv2.line(img, q, p2, color, 1, cv2.LINE_AA)

    def get_orientation(self, pts, img=None):
        # 确保轮廓点是二维浮点型数组
        data_pts = np.array(pts, dtype=np.float64).reshape(-1, 2)

        # 计算轮廓的质心
        mean = np.mean(data_pts, axis=0)
        data_pts -= mean

        # PCA 主成分分析
        _, eigenvectors, _ = cv2.PCACompute2(data_pts, mean=None)

        # 获取质心
        # cntr = (int(mean[0]), int(mean[1]))

        # if img is not None:
        #     # 绘制质心
        #     cv2.circle(img, cntr, 3, (255, 0, 255), 2)
        #     # 绘制主轴方向
        #     p1 = (int(cntr[0] + 0.02 * eigenvectors[0, 0] * 150),
        #           int(cntr[1] + 0.02 * eigenvectors[0, 1] * 150))
        #     draw_axis(img, cntr, p1, (0, 255, 0), 1)

        return eigenvectors[0]  # 返回主轴方向的向量

    def find_adjacent_min_distances(self, contours):
        min_distances = []
        for i in range(len(contours) - 1):
            # 计算相邻轮廓之间的最短距离
            dists = cdist(contours[i].reshape(-1, 2), contours[i + 1].reshape(-1, 2))
            min_distance = dists.min()  # 获取最短距离
            min_distances.append(min_distance)

        return min_distances

    def calculate_angle(self, vec1, vec2):
        """计算轮廓的角度差异"""
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        cos_theta = dot_product / (magnitude1 * magnitude2)
        # 夹角计算，弧度转为角度
        angle = np.arccos(cos_theta)
        return np.degrees(angle)

    def alpha_shape(self, points, alpha):
        """
        计算点集的Alpha Shape（凹包络）。
        :param points: 点的集合 (n, 2)。
        :param alpha: Alpha 参数，决定凹包络的紧致程度。
        :return: 包络边界的点对集合。
        """
        if len(points) < 4:
            # 如果只有三个点，直接返回凸包
            return Delaunay(points).convex_hull

        tri = Delaunay(points)
        edges = set()
        for ia, ib, ic in tri.simplices:
            a = np.linalg.norm(points[ia] - points[ib])
            b = np.linalg.norm(points[ib] - points[ic])
            c = np.linalg.norm(points[ic] - points[ia])
            s = (a + b + c) / 2.0
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            circum_r = a * b * c / (4.0 * area)
            if circum_r < 1.0 / alpha:
                edges.add(tuple(sorted([ia, ib])))
                edges.add(tuple(sorted([ib, ic])))
                edges.add(tuple(sorted([ic, ia])))
        return np.array(list(edges))

class Build_TRT_model():
    def __init__(self, weight):
        with open(weight, "rb") as f:
            engineString = f.read()
        logger = trt.Logger(trt.Logger.WARNING)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
        self.tensorname = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.context = engine.create_execution_context()
        self.nIO = engine.num_io_tensors
        self.nInput = [engine.get_tensor_mode(self.tensorname[i]) for i in range(self.nIO)].count(
            trt.TensorIOMode.INPUT)  # 1个输入
        self.dtypes = trt.nptype(engine.get_tensor_dtype(self.tensorname[0]))

    def __call__(self, image):
        # 准备输入数据
        bufferH = [image]
        # 为输出数据分配内存 2个输出
        for i in range(self.nInput, self.nIO):
            bufferH.append(np.empty(self.context.get_tensor_shape(self.tensorname[i]), self.dtypes))
        # 为GPU分配内存
        bufferD = []
        for i in range(self.nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
        # 将数据从主机端复制到设备端
        for i in range(self.nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        for i in range(self.nIO):
            self.context.set_tensor_address(self.tensorname[i], int(bufferD[i]))
        self.context.execute_async_v3(0)
        # 将数据从设备端复制到主机端
        for i in range(self.nInput, self.nIO):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        for buf in bufferD:
            cudart.cudaFree(buf)  # 释放,不然要炸
        return bufferH[1:]


class Build_Ort_model():
    def __init__(self, weight):
        self.session = ort.InferenceSession(weight, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        if ort.get_device() == 'GPU' else ['CPUExecutionProvider'])
        print(ort.get_device())
        self.dtypes = np.half if self.session.get_inputs()[0].type == 'tensor(float16)' else np.single

    def __call__(self, image):
        res = self.session.run(None, {self.session.get_inputs()[0].name: image})
        return res


class My_LoggerConfig:
    def __init__(self, name='xy', level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._configure_handler()

    def _configure_handler(self):
        # 创建一个处理器，用于将日志输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建一个彩色格式化器
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s - %(name)s | - %(message_log_color)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={
                'message': {
                    'DEBUG': 'white',
                    'INFO': 'blue',
                    'WARNING': 'purple',
                    'ERROR': 'bg_red',
                    'CRITICAL': 'bg_red',
                }
            }
        )

        # 将格式化器添加到处理器
        console_handler.setFormatter(formatter)

        # 将处理器添加到记录器
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
