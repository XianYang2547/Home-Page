# -*- coding: utf-8 -*-
# @Time    : 2024/5/20 5:20
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : My_infer.py
# ------❤❤❤------ #

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cupy as cp
import cv2
import numpy as np
import yaml

from xy import *


class Dection(My_dection):
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

        if opt.use_model == "city_new":
            self.label_dict = {int(key): value for key, value in myclass['city_det_new'].items()}
            self.classes = {int(key): value for key, value in myclass['city_seg_new'].items()}
        self.color_palette = np.random.uniform(50, 255, size=(len(self.label_dict) + len(self.classes), 3))
        # ---------------------------------------------------------------------------------------------------
        if os.path.splitext(opt.det_weights)[1] and os.path.splitext(opt.seg_weights)[1] == '.plan':
            self.Det_Models = Build_TRT_model(str(Path(det_weights).resolve()))
            self.Seg_Models = Build_TRT_model(str(Path(seg_weights).resolve()))
            self.warm_up(15)
        elif os.path.splitext(opt.det_weights)[1] and os.path.splitext(opt.seg_weights)[1] == '.onnx':
            self.Det_Models = Build_Ort_model(str(Path(det_weights).resolve()))
            self.Seg_Models = Build_Ort_model(str(Path(seg_weights).resolve()))
            self.warm_up(15)
        self.conf_threshold = opt.conf_threshold
        self.iou_threshold = opt.iou_threshold

    def postprocess(self, pred0, pred1, im0, image, ratio, dw, dh, conf_threshold, iou_threshold, nm=32):
        post = time.time()
        det, seg = [], [[], []]
        # for det
        det = [self.postprocess_det(pred, im0, image, conf_threshold, iou_threshold) for pred in pred0]
        # for seg 根据维度来给x, protos赋值
        masks = None
        if len(pred1) != 0:
            if pred1[0].ndim == 4 and pred1[1].ndim == 3:
                x, protos = pred1[1], pred1[0]
            elif pred1[1].ndim == 4 and pred1[0].ndim == 3:
                x, protos = pred1[0], pred1[1]
            x = self.non_max_suppression(x, conf_threshold, iou_threshold, nc=len(self.classes))[0]  # x1y1x2y2
            x = self.convert_to_center_width_height(x)
            if len(x) > 0:
                x[..., [0, 1]] -= x[..., [2, 3]] / 2
                x[..., [2, 3]] += x[..., [0, 1]]
                x[..., :4] -= [dw, dh, dw, dh]
                x[..., :4] /= min(ratio)
                x[..., [0, 2]] = np.clip(x[:, [0, 2]], 0, image.shape[1])
                x[..., [1, 3]] = np.clip(x[:, [1, 3]], 0, image.shape[0])
                # 使用CuPy进行掩码处理
                protos_gpu = cp.asarray(protos[0])
                masks = self.process_mask(protos_gpu, x[:, 6:], x[:, :4], image.shape)
                masks = cp.asnumpy(masks)
                # 这一版的mask的点很多，处理慢，但效果好
                segments = self.masks2segments(masks)  # 处理找不到区域很小的轮廓情况,对齐mask和框
                valid_indices = [i for i, seg in enumerate(segments) if seg.size > 0]
                segment = [segments[i] for i in valid_indices]
                for i in range(len(seg)):  # i=0时添加x[..., :6]i=1时添加segments
                    seg[i].append([x[..., :6][valid_indices], segment][i])

        # print(f"post: {(time.time() - post) * 1000:6.2f}ms")
        return det, seg, masks

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        c, mh, mw = protos.shape
        # 确保输入为 CuPy 数组
        protos = cp.asarray(protos)
        masks_in = cp.asarray(masks_in)
        # 矩阵乘法和重塑操作使用 CuPy
        masks = cp.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = cp.ascontiguousarray(masks)

        # 将掩码从 P3 形状重新缩放到原始输入图像形状
        masks = self.scale_mask(masks, im0_shape)

        masks = cp.einsum('HWN -> NHW', masks)  # HWN -> NHW

        masks = self.crop_mask(masks, bboxes)
        return cp.asnumpy(cp.greater(masks, 0.5))

    def masks2segments(self, masks):
        '''
        output_dir = 'mask_images'
        os.makedirs(output_dir, exist_ok=True)
        for i in range(len(masks)):
            # 将布尔值转换为 uint8 类型
            # True 转换为 255 (白色), False 转换为 0 (黑色)
            mask_image = (masks[i] * 255).astype(np.uint8)
            # 保存图像
            filename = os.path.join(output_dir, f'mask_slice_{i + 1}.png')
            cv2.imwrite(filename, mask_image)
            print(f'Saved: {filename}')
        '''

        def get_point(index, mask):
            distance_threshold = 30
            contours = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            contours = [contour for contour in contours if contour.shape[0] >= 10]
            if contours:
                if len(contours) == 1:
                    c = np.array(contours[np.array([len(x) for x in contours]).argmax()]).reshape(-1, 2)
                    return c.astype('float32')
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
                # # 轮廓面积
                # areas = sorted([cv2.contourArea(c) for c in contours])
                # max_area = max(areas)
                # if len(areas)>4:
                #     min_area = areas[3]
                # elif len(areas)>3:
                #     min_area = areas[2]
                # elif len(areas)>2:
                #     min_area = areas[1]
                #
                # else:
                #     min_area = min(areas) if min(areas)!=0 else 1
                # areaT = (max_area / min_area) < 10

                disT = False
                adjacent_distances = self.find_adjacent_min_distances(contours)
                if adjacent_distances:
                    satisfied_count = sum(1 for dist in adjacent_distances if dist <= distance_threshold)
                    total_distances = len(adjacent_distances)
                    # 计算满足条件的比例
                    satisfied_ratio = satisfied_count / total_distances
                    if satisfied_ratio >= 0.5:
                        disT = True
                    else:
                        disT = False
                if consistent and disT:
                    mask_shape = mask.astype('uint8')
                    convex_hull_image = np.zeros(mask_shape.shape, dtype=np.uint8)
                    # 合并所有轮廓
                    all_points = np.concatenate([contour.reshape(-1, 2) for contour in contours])
                    alpha = 0.008  # 调整alpha参数，值越小包络越贴近形状
                    edges = self.alpha_shape(all_points, alpha)
                    # 使用掩膜而不是彩色图像绘制
                    for edge in edges:
                        pt1 = tuple(all_points[edge[0]])
                        pt2 = tuple(all_points[edge[1]])
                        cv2.line(convex_hull_image, pt1, pt2, 255, 1)  # 使用255绘制白色线条
                    # 找到新的轮廓
                    new_contours, _ = cv2.findContours(convex_hull_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # 选择最大的轮廓
                    if new_contours:
                        c = np.array(max(new_contours, key=cv2.contourArea)).reshape(-1, 2)
                    masks[index] = convex_hull_image

                else:
                    # 选择面积最大的轮廓
                    c = np.array(contours[np.array([len(x) for x in contours]).argmax()]).reshape(-1, 2)
            else:
                # 没有找到轮廓时返回空数组
                c = np.zeros((0, 2))
            return c.astype('float32')

        with ThreadPoolExecutor() as executor:
            segments = list(executor.map(lambda pair: get_point(*pair), enumerate(masks)))
        return segments

    # debug
    def masks2segments1(self, masks):
        segments = []
        distance_threshold = 30
        # for mask in masks:
        for index, mask in enumerate(masks):
            contours = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            contours = [contour for contour in contours if contour.shape[0] >= 10]
            if contours:
                if len(contours) == 1:
                    c = np.array(contours[np.array([len(x) for x in contours]).argmax()]).reshape(-1, 2)
                    segments.append((c.astype('float32')))
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
                # areas = sorted([cv2.contourArea(c) for c in contours])
                # max_area = max(areas)
                # if len(areas)>4:
                #     min_area = areas[3]
                # elif len(areas)>3:
                #     min_area = areas[2]
                # elif len(areas)>2:
                #     min_area = areas[1]
                #
                # else:
                #     min_area = min(areas) if min(areas)!=0 else 1
                # areaT = (max_area / min_area) < 10

                disT = False
                adjacent_distances = self.find_adjacent_min_distances(contours)
                if adjacent_distances:
                    satisfied_count = sum(1 for dist in adjacent_distances if dist <= distance_threshold)
                    total_distances = len(adjacent_distances)
                    # 计算满足条件的比例
                    satisfied_ratio = satisfied_count / total_distances
                    if satisfied_ratio >= 0.5:
                        disT = True
                    else:
                        disT = False
                # if consistent and areaT and disT:
                if consistent and disT:
                    # 合并所有轮廓
                    # c = np.concatenate([contour.reshape(-1, 2) for contour in contours])
                    # 创建一个二值掩膜用于绘制
                    mask_shape = mask.astype('uint8')
                    convex_hull_image = np.zeros(mask_shape.shape, dtype=np.uint8)
                    # 合并所有轮廓
                    all_points = np.concatenate([contour.reshape(-1, 2) for contour in contours])
                    alpha = 0.008  # 调整alpha参数，值越小包络越贴近形状
                    edges = self.alpha_shape(all_points, alpha)
                    # 使用掩膜而不是彩色图像绘制
                    for edge in edges:
                        pt1 = tuple(all_points[edge[0]])
                        pt2 = tuple(all_points[edge[1]])
                        cv2.line(convex_hull_image, pt1, pt2, 255, 1)  # 使用255绘制白色线条
                    # 找到新的轮廓
                    new_contours, _ = cv2.findContours(convex_hull_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # 选择最大的轮廓
                    if new_contours:
                        c = np.array(max(new_contours, key=cv2.contourArea)).reshape(-1, 2)
                    masks[index] = convex_hull_image
                else:
                    # 选择面积最大的轮廓
                    c = np.array(contours[np.array([len(x) for x in contours]).argmax()]).reshape(-1, 2)
            else:
                # 没有找到轮廓时返回空数组
                c = np.zeros((0, 2))
            segments.append((c.astype('float32')))
        return segments

    def scale_mask(self, masks, im0_shape, ratio_pad=None):
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]
        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        # Slicing and resizing
        masks = masks[top:bottom, left:right]
        masks = cp.asnumpy(masks)  # Convert to numpy for OpenCV
        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]),
                           interpolation=cv2.INTER_LINEAR)  # INTER_CUBIC would be better
        # Ensure masks has third dimension
        if len(masks.shape) == 2:
            masks = masks[:, :, None]

        return cp.asarray(masks)  # Convert back to CuPy array

    def crop_mask(self, masks, boxes):
        n, h, w = masks.shape
        # 使用 CuPy 数组
        boxes = cp.asarray(boxes)
        x1, y1, x2, y2 = cp.split(boxes[:, :, None], 4, 1)
        r = cp.arange(w, dtype=x1.dtype)[None, None, :]
        c = cp.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def convert_to_center_width_height(self, x):
        # 获取形状
        n, _ = x.shape

        # 提取 x1, y1, x2, y2（假设在前四列）
        x1 = x[:, 0]
        y1 = x[:, 1]
        x2 = x[:, 2]
        y2 = x[:, 3]

        # 计算中心点和宽高
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # 创建新的数组，保持其他数据不变
        new_x = np.empty_like(x)

        # 填充新的值
        new_x[:, 0] = cx
        new_x[:, 1] = cy
        new_x[:, 2] = w
        new_x[:, 3] = h
        new_x[:, 4:] = x[:, 4:]  # 将其他内容复制过去

        return new_x


def main(opt):
    Model = Dection(opt.det_weights, opt.seg_weights)
    SUF1 = ('.jpeg', '.jpg', '.png', '.webp')
    SUF2 = ('.mp4', '.avi')
    SUF3 = ('.svo', '.svo2')
    if opt.path != 'camera':
        if isinstance(opt.path, str):
            opt.path = Path(opt.path)

        assert opt.path.exists()
        # 创建输出文件路径
        if not os.path.exists(opt.base_directory):
            os.makedirs(opt.base_directory)

        if opt.path.is_dir():  # image dir not include video
            images = [i.absolute() for i in opt.path.iterdir() if i.suffix in SUF1]
            # detect_image_or_imgdir(opt, images, Model, draw_bezier=False)
            detect_image_or_imgdir_json(opt, images, Model)
        else:  # 传入文件
            if opt.path.suffix in SUF1:  # image
                images = [opt.path.absolute()]
                detect_image_or_imgdir(opt, images, Model, draw_bezier=True, saveMask=True)

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
    TYPE, scales, version = 'gaosu', 's', 'v8'
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_model', type=str, default=TYPE)
    parser.add_argument('--det_weights', type=str,
                        default=f"./model_files/{version}/linuxtrt/{TYPE}/{scales}/{TYPE}det.plan")
    parser.add_argument('--seg_weights', type=str,
                        default=f'./model_files/{version}/linuxtrt/{TYPE}/{scales}/{TYPE}seg.plan')

    parser.add_argument('--iou_threshold', type=str, default=0.5)
    parser.add_argument('--conf_threshold', type=str, default=0.5)
    # get ip for rtsp
    parser.add_argument('--ip_add', type=str, default=get_ip_addresses())
    # input output path
    parser.add_argument('--path', type=str,
                        default=f"./test_data/test.svo" if TYPE == 'city' else f"./test_data/438.jpg")
    # parser.add_argument('--path', type=str, default='camera')
    # parser.add_argument('--path', type=str, default='/home/xianyang/Desktop/chengya1.svo')
    parser.add_argument('--base_directory', type=str, default='./output_folder')
    # save config
    parser.add_argument('--save_orin_svo', type=str, default=True)
    parser.add_argument('--save_video', type=str, default=True)
    parser.add_argument('--save_video_ffmpeg', type=str, default=True)
    parser.add_argument('--save_img', type=str, default=False)
    parser.add_argument('--save_mask', default=False)

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    return parser


if __name__ == '__main__':
    opt = make_parser().parse_args()
    main(opt)
