# -*- coding: utf-8 -*-
# @Time    : 2024/1/25 下午4:03
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : my_utils.py
# ------❤❤❤------ #


import cv2
import numpy as np


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
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
    return im


def img_preprocess(im, size):
    if im.shape[2] == 4:
        im = im[:, :, :3]
    img = letterbox(im, size)
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32)
    img /= 255
    return img[None]


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def nms_boxes(boxes, scores):
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


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (np.min(a2, b2) - np.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300, nc=0, max_time_img=0.05, max_nms=30000, max_wh=7680, ):
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
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
        box = xywh2xyxy(x[:, :4])
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
        i = nms_boxes(boxes, scores)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = np.multiply(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]

    return output


def postprocess(preds, image, org_im, conf_thres, iou_thres):
    results = []
    for i, pred in enumerate(preds):
        if i == 0:
            det_res = postprocess_det(pred, image, org_im, conf_thres, iou_thres)
            results.append(det_res)
        else:
            seg_res = postprocess_seg(pred, org_im, i)
            results.append(seg_res)
    return results


def postprocess_det(preds, img, org_img, conf_thres, iou_thres):
    det_res = []
    pred0 = non_max_suppression(preds, conf_thres, iou_thres)
    for i, pred in enumerate(pred0[0]):
        pred[:4] = scale_boxes(img.shape[2:], pred[:4], org_img.shape)
        det_res.append(pred)
    return det_res


def postprocess_seg(pred, org_im, i):
    pred = np.argmax(pred, axis=1)[0]
    color_area = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    if i == 1:
        color_area[pred == 1] = [0, 255, 255]
    if i ==2:
        color_area[pred == 1] = [0, 255, 0]
    color_seg = cv2.resize(color_area, (org_im.shape[1], org_im.shape[0]), interpolation=cv2.INTER_LINEAR)
    return color_seg


def Draw(det, im0, label_dict, color=(175, 50, 255)):
    for i in det:
        # p1, p2, conf, classes_num = (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), round(i[4], 3), int(i[5])
        cv2.rectangle(im0, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), color, 2, cv2.LINE_AA)
        cv2.putText(im0, f"{label_dict[int(i[5])]} {round(i[4] * 100, 1)}%", (int(i[0]), int(i[1])), 0, thickness=2,
                    fontScale=1, lineType=cv2.LINE_AA, color=(175, 50, 255))
    return im0


def show(results, im0, label_dict):
    # 解析results
    # det, road_seg, lane_seg = results[0], results[1], results[2]
    det, lane_seg = results[0], results[1]# 去掉分割
    # 先渲染
    # im0[:, :, :3][np.any(road_seg != [0, 0, 0], axis=-1)] = 0.5 * im0[:, :, :3][
    #     np.any(road_seg != [0, 0, 0], axis=-1)] + 0.4 * road_seg[
    #                                                             np.any(road_seg != [0, 0, 0], axis=-1)]
    im0[:, :, :3][np.any(lane_seg != [0, 0, 0], axis=-1)] = 0.5 * im0[:, :, :3][
        np.any(lane_seg != [0, 0, 0], axis=-1)] + 0.4 * lane_seg[
                                                                np.any(lane_seg != [0, 0, 0], axis=-1)]
    # 将框花在原图上
    im0 = Draw(det, im0, label_dict)

    return im0, lane_seg


########################################################################################################################
import pyzed.sl as sl

def detections_to_custom_box(detections, im0):
    output = []
    for det in detections:
        if len(det):
            xyxy, conf, cls = det[0:4], det[4], det[5]
            xywh = xyxy2xywh(xyxy)
            obj = sl.CustomBoxObjectData()  # 创建自定义检测框对象
            obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)  # 转换为 ZED SDK 所需的格式(a, b, c, d)
            obj.label = cls
            obj.probability = conf
            obj.is_grounded = False  # 表示该自定义检测框不是站立物体
            output.append(obj)
    return output


def xyxy2xywh(x):
    # x is a list
    y = np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2abcd(xywh, im_shape):
    # xywh是个list
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5 * xywh[2]) * im_shape[1]
    x_max = (xywh[0] + 0.5 * xywh[2]) * im_shape[1]
    y_min = (xywh[1] - 0.5 * xywh[3]) * im_shape[0]
    y_max = (xywh[1] + 0.5 * xywh[3]) * im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max
    return output


def process_zed_Seg(road_seg, lane_seg, image_net):
    image_net = image_net[:, :, :3]
    seg_mask = np.zeros((image_net.shape[0], image_net.shape[1], 3), dtype=np.uint8)
    seg_mask[road_seg == 1] = [54, 217, 143]
    seg_mask[lane_seg == 1] = [175, 50, 255]
    # 先渲染
    color_road = np.stack([road_seg * 0, road_seg * 255, road_seg * 0], axis=-1)
    color_lane = np.stack([lane_seg * 255, lane_seg * 0, lane_seg * 0], axis=-1)
    alpha = 0.5
    image_net[np.any(color_road != [0, 0, 0], axis=-1)] = (1 - alpha) * image_net[
        np.any(color_road != [0, 0, 0], axis=-1)] + alpha * color_road[
                                                              np.any(color_road != [0, 0, 0], axis=-1)]
    image_net[np.any(color_lane != [0, 0, 0], axis=-1)] = (1 - alpha) * image_net[
        np.any(color_lane != [0, 0, 0], axis=-1)] + alpha * color_lane[
                                                              np.any(color_lane != [0, 0, 0], axis=-1)]
    return image_net
