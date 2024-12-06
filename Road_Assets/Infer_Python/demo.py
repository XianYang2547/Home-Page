# -*- coding: utf-8 -*-
# @Time    : 2024/5/20 5:20
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : My_infer.py
# ------❤❤❤------ #

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cupy as cp
import cv2
import numpy as np
import yaml

from xy import *
from xy.funcs import detect_image_or_imgdir, detect_video, get_ip_addresses


class Dection(My_detection):
    def __init__(self, opt):
        super().__init__()
        with open('./xy/xyz_class.yaml', 'r', encoding='utf-8') as file:
            myconfig = yaml.safe_load(file)
        # 类别分类
        self.lane = {int(key): value for key, value in myconfig['classes']['lane'].items()}  # 车道线
        self.seg = {int(key): value for key, value in myconfig['classes']['seg'].items()}  # 护栏 隔音带  水泥墙  绿化带  路缘石
        self.obj = {int(key): value for key, value in myconfig['classes']['obj'].items()}  # 画框显示
        self.other = {int(key): value for key, value in myconfig['classes']['other'].items()}  # 路口黄网线 导流区 待行区 防抛网 隔离挡板
        self.lane_seg_other = {**self.lane, **self.seg, **self.other}  # 在画图显示中用到的
        self.classes = {**self.lane, **self.seg, **self.obj, **self.other}  # total
        # 颜色板
        palette = myconfig['palette']
        self.color_palette = palette[0][:len(self.classes)]

        if os.path.splitext(opt.model)[1] == '.plan':
            self.Models = Build_TRT_model(str(Path(opt.model).resolve()))
            self.warm_up(15)
        elif os.path.splitext(opt.model)[1] == '.onnx':
            self.Models = Build_Ort_model(str(Path(opt.model).resolve()))
            self.warm_up(15)
        self.conf_threshold = opt.conf_threshold
        self.iou_threshold = opt.iou_threshold

    def postprocess(self, pred, im0, image, ratio, dw, dh, conf_threshold, iou_threshold, nm=32):
        seg = [[], []]
        masks = None
        if len(pred) != 0:
            if pred[0].ndim == 4 and pred[1].ndim == 3:
                x, protos = pred[1], pred[0]
            elif pred[1].ndim == 4 and pred[0].ndim == 3:
                x, protos = pred[0], pred[1]
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
        return seg, masks

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
                    if satisfied_ratio > 0.5:
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
                    if satisfied_ratio > 0.5:
                        disT = True
                    else:
                        disT = False
                if consistent and disT:
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


def make_parser():
    # model config
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=f"{os.path.abspath('../models/best.plan')}")
    parser.add_argument('--iou_threshold', type=float, default=0.4)
    parser.add_argument('--conf_threshold', type=float, default=0.4)
    # input output path
    parser.add_argument('--path', type=str, default=f"{os.path.abspath('../assets/test.jpg')}")
    parser.add_argument('--base_directory', type=str, default=f"{os.path.abspath('../output')}")
    # rtsp
    parser.add_argument('--rtsp', type=str, default=False)
    parser.add_argument('--url', type=str, default=get_ip_addresses())
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6)
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    # save config
    parser.add_argument('--save_orin_svo', type=str, default=True)
    parser.add_argument('--save_img', type=str, default=False)
    parser.add_argument('--save_mask', default=False)
    return parser


def main():
    opt = make_parser().parse_args()
    Model = Dection(opt)
    SUF1 = ('.jpeg', '.jpg', '.png', '.webp')
    SUF2 = ('.mp4', '.avi')
    SUF3 = ('.svo', '.svo2')
    use_camera = False
    use_zed_camera = False
    if isinstance(opt.path, str):
        opt.path = Path(opt.path)
        if 'camera' == opt.path.name:
            use_camera = True
        elif 'zed_camera' == opt.path.name:
            use_zed_camera = True
        else:
            assert opt.path.exists()

    # 创建输出文件路径
    if not os.path.exists(opt.base_directory):
        os.makedirs(opt.base_directory)

    # 判断输入类型
    if opt.path.suffix in SUF1:  # image
        images = [opt.path.absolute()]
        detect_image_or_imgdir(opt, images, Model, saveMask=True)

    elif opt.path.suffix in SUF2 or use_camera:  # video or camera
        if use_camera:
            opt.path = 'camera'
        detect_video(opt, Model, f"rtsp://{opt.url[0]}:8554/test")

    elif opt.path.suffix in SUF3 or use_zed_camera:  # svo or camera
        from xy.detect_svo import detect_svo_track
        if use_zed_camera:
            opt.path = 'zed_camera'
        detect_svo_track(opt, Model, f"rtsp://{opt.url[0]}:8554/test")

    elif opt.path.is_dir():  # dir [images/videos/svo/svo2]
        # images
        images = [i.absolute() for i in opt.path.iterdir() if i.suffix in SUF1]
        if images:
            detect_image_or_imgdir(opt, images, Model, saveMask=True)
        # videos
        videos = [i.absolute() for i in opt.path.iterdir() if i.suffix in SUF2]
        if videos:
            for video_path in videos:
                opt.path = video_path
                detect_video(opt, Model, f"rtsp://{opt.url[0]}:8554/test")
        # svo
        svo1_2 = [i.absolute() for i in opt.path.iterdir() if i.suffix in SUF3]
        if svo1_2:
            for svo_path in svo1_2:
                opt.path = svo_path
                detect_video(opt, Model, f"rtsp://{opt.url[0]}:8554/test")


if __name__ == '__main__':
    main()
