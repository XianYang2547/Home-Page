# -*- coding: utf-8 -*-
# @Time    : 2024/5/20 5:20
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : My_infer_Fast.py
# ------❤❤❤------ #

import argparse
import os
import yaml
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cupy as cp
import cv2
import numpy as np

from xy import *


class Dection_Faster(My_dection):
    def __init__(self, det_weights, seg_weights):
        super().__init__()
        with open('./xy/xyz_class.yaml', 'r', encoding='utf-8') as file:
            myclass = yaml.safe_load(file)
        # ---------------------------------------------------------------------------------------------------
        if opt.use_model == "city":
            self.label_dict = {int(key): value for key, value in myclass['city_det'].items()}
            self.classes = {int(key): value for key, value in myclass['city_seg'].items()}

        if opt.use_model == "gaosu":
            self.label_dict = {int(key): value for key, value in myclass['gaosu_det'].items()}
            self.classes = {int(key): value for key, value in myclass['gaosu_seg'].items()}

        self.color_palette = np.random.uniform(50, 255, size=(len(self.label_dict) + len(self.classes), 3))
        # ---------------------------------------------------------------------------------------------------
        if os.path.splitext(opt.det_weights)[1] and os.path.splitext(opt.seg_weights)[1] == '.plan':
            self.Det_Models = Build_TRT_model(str(Path(det_weights).resolve()))
            self.Seg_Models = Build_TRT_model(str(Path(seg_weights).resolve()))
            assert self.Det_Models.dtypes == self.Seg_Models.dtypes, "Det_Models.dtypes should be == Seg_Models.dtypes"
            self.warm_up(15)
        elif os.path.splitext(opt.det_weights)[1] and os.path.splitext(opt.seg_weights)[1] == '.onnx':
            self.Det_Models = Build_Ort_model(str(Path(det_weights).resolve()))
            self.Seg_Models = Build_Ort_model(str(Path(seg_weights).resolve()))
            self.warm_up(15)
        self.conf_threshold = opt.conf_threshold
        self.iou_threshold = opt.iou_threshold

    def postprocess(self, pred0, pred1, im0, image, ratio, dw, dh, conf_threshold, iou_threshold, nm=32):
        det, seg = [], [[], []]
        # for det
        det = [self.postprocess_det(pred, im0, image, conf_threshold, iou_threshold) for pred in pred0]
        # for seg  根据维度来给x, protos赋值
        masks = None
        if len(pred1) != 0:
            if pred1[0].ndim == 4 and pred1[1].ndim == 3:
                x, protos = pred1[1], pred1[0]
            elif pred1[1].ndim == 4 and pred1[0].ndim == 3:
                x, protos = pred1[0], pred1[1]
            r = self.non_max_suppression(x, conf_threshold, iou_threshold, nc=len(self.classes))[0]
            if len(r) == 0:
                return det, seg

            shape = im0.shape[2:]
            bboxes, conf, labels, maskconf = np.split(r, [4, 5, 6], 1)

            proto_gpu = cp.asarray(np.squeeze(protos, axis=0).reshape(32, -1))
            maskconf_gpu = cp.asarray(maskconf)
            masks_gpu = self.sigmoid(maskconf_gpu @ proto_gpu).reshape(-1, 160, 160)
            masks = self.crop_mask(masks_gpu, bboxes / 4.0).transpose([1, 2, 0])
            masks = cv2.resize(cp.asnumpy(masks), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            # 确保masks有第三个维度
            if len(masks.shape) == 2:
                masks = masks[:, :, np.newaxis]
            masks = masks.transpose(2, 0, 1)
            # 转换为二值掩码
            m = masks > 0.5
            # 找到轮廓（由cv2.findContours的参数，有时候会找不到区域很小的轮廓，返回 np.zeros((0, 2))）造成结果如下：5个目标5个框，4个轮廓
            segments = self.masks2segments(m, dw, dh, ratio)
            # for i, arr in enumerate(segments):
            #     if arr.size == 0:
            #         print(f"数组 {i} 是空的，执行其他操作。")
            # 去掉空轮廓和对应的框、置信度、标签
            valid_indices = [i for i, seg in enumerate(segments) if seg.size > 0]
            bboxes = bboxes[valid_indices]
            conf = conf[valid_indices]
            labels = labels[valid_indices]
            segment = [segments[i] for i in valid_indices]
            # 更新seg
            seg[0].append(np.concatenate(((bboxes - (dw, dh, dw, dh)) / ratio[0], conf, labels), axis=1))
            seg[1].append(segment)
        return det, seg, masks

    def masks2segments0(self, masks, dw, dh, ratio):
        def get_point(x):
            c = cv2.findContours(x.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            if c:
                if len(c)<3:
                    c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
                else:
                    c=np.concatenate([x.reshape(-1, 2) for x in c])
                # c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))
            return (c.astype('float32') - (dw, dh)) / ratio

        with ThreadPoolExecutor() as executor:
            segments = list(executor.map(get_point, masks))

        return segments
    def masks2segments(self, masks, dw, dh, ratio):
        segments=[]
        distance_threshold=30
        for mask in masks:
            contours = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            contours = [contour for contour in contours if contour.shape[0] >= 10]
            if contours:
                if len(contours)==1:
                    c = np.array(contours[np.array([len(x) for x in contours]).argmax()]).reshape(-1, 2)
                    segments.append((c.astype('float32') - (dw, dh)) / ratio)
                    continue
                # 轮廓朝向
                directions = []
                for contour in contours:
                    direction = self.get_orientation(contour, mask)
                    directions.append(direction)
                # 计算方向一致性，假设第一个轮廓为基准
                base_direction = directions[0]
                consistent_count = 0
                for i in range(1, len(directions)):
                    angle_diff = self.calculate_angle(base_direction, directions[i])
                    if angle_diff <= 30:
                        consistent_count += 1
                consistency_ratio = consistent_count / (len(directions) - 1) if len(directions) > 1 else 1.0
                consistent = consistency_ratio >= 0.5
                # 轮廓面积
                areas = sorted([cv2.contourArea(c) for c in contours])
                max_area = max(areas)
                if len(areas)>4:
                    min_area = areas[3]
                elif len(areas)>3:
                    min_area = areas[2]
                elif len(areas)>2:
                    min_area = areas[1]

                else:
                    min_area = min(areas) if min(areas)!=0 else 1
                areaT = (max_area / min_area) < 10

                disT = False
                adjacent_distances=self.find_adjacent_min_distances(contours)
                if adjacent_distances:
                    satisfied_count = sum(1 for dist in adjacent_distances if dist <= distance_threshold)
                    total_distances = len(adjacent_distances)
                    # 计算满足条件的比例
                    satisfied_ratio = satisfied_count / total_distances
                    if satisfied_ratio >= 0.5:
                        disT=True
                    else:
                        disT=False
                if consistent and areaT and disT:
                    # 合并所有轮廓
                    c = np.concatenate([contour.reshape(-1, 2) for contour in contours])
                else:
                    # 选择面积最大的轮廓
                    c = np.array(contours[np.array([len(x) for x in contours]).argmax()]).reshape(-1, 2)
            else:
                # 没有找到轮廓时返回空数组
                c = np.zeros((0, 2))
            segments.append((c.astype('float32') - (dw, dh)) / ratio)
        return segments
    def crop_mask(self, masks, boxes):
        n, h, w = masks.shape
        boxes = cp.asarray(boxes)
        x1, y1, x2, y2 = cp.split(boxes[:, :, None], 4, 1)
        r = cp.arange(w, dtype=x1.dtype)[None, None, :]
        c = cp.arange(h, dtype=x1.dtype)[None, :, None]

        mask = (r >= x1) & (r < x2) & (c >= y1) & (c < y2)
        return masks * mask.astype(masks.dtype)


def main(opt):
    Model = Dection_Faster(opt.det_weights, opt.seg_weights)
    SUF1 = ('.jpeg', '.jpg', '.png', '.webp')
    SUF2 = ('.mp4', '.avi')
    SUF3 = ('.svo', '.svo2')
    if opt.path != 'camera':
        "use local file"
        if isinstance(opt.path, str):
            opt.path = Path(opt.path)

        assert opt.path.exists()
        # 创建输出文件路径
        if not os.path.exists(opt.base_directory):
            os.makedirs(opt.base_directory)

        if opt.path.is_dir():  # image dir not include video
            images = [i.absolute() for i in opt.path.iterdir() if i.suffix in SUF1]
            detect_image_or_imgdir(opt, images, Model, draw_bezier=False)
        else:  # 传入文件
            if opt.path.suffix in SUF1:  # image
                images = [opt.path.absolute()]
                detect_image_or_imgdir(opt, images, Model, draw_bezier=True,saveMask=True)
            elif opt.path.suffix in SUF2:  # video
                detect_video(opt, Model)

            elif opt.path.suffix in SUF3:  # svo
                detect_svo_track(opt, Model, f"rtsp://{opt.ip_add[0]}:8554/test", use_RTSP=False)

            else:
                print('Input Err')
    else:
        "use camera"
        detect_svo_track_socket_rstp(opt, Model, '172.16.20.68', 9001, f"rtsp://{opt.ip_add[0]}:8554/zed")


def make_parser():
    # model config
    TYPE, scales, version= 'gaosu', 's', 'v8'
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_model', type=str, default=TYPE)
    parser.add_argument('--det_weights', type=str, default=f"./model_files/{version}/linuxtrt/{TYPE}/{scales}/{TYPE}det.plan")
    parser.add_argument('--seg_weights', type=str, default=f'./model_files/{version}/linuxtrt/{TYPE}/{scales}/{TYPE}seg.plan')
    parser.add_argument('--iou_threshold', type=str, default=0.45)
    parser.add_argument('--conf_threshold', type=str, default=0.45)
    # get ip for rtsp
    parser.add_argument('--ip_add', type=str, default=get_ip_addresses())
    # input output path
    parser.add_argument('--path', type=str, default=f"./test_data/test.svo" if TYPE == 'city' else f"./test_data/438.jpg")
    # parser.add_argument('--path', type=str, default='camera')
    # parser.add_argument('--path', type=str, default='/home/xianyang/Desktop/chengya1.svo')
    # parser.add_argument('--path', type=str, default='2.jpg')
    parser.add_argument('--base_directory', type=str, default='./output_folder')
    # save config
    parser.add_argument('--save_orin_svo', type=str, default=True)
    parser.add_argument('--save_video', type=str, default=True)
    parser.add_argument('--save_video_ffmpeg', type=str, default=True)
    parser.add_argument('--save_img', type=str, default=False)
    parser.add_argument('--save_mask',default=False)
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.8, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    return parser

if __name__ == '__main__':
    opt = make_parser().parse_args()
    main(opt)
