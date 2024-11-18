# -*- coding: utf-8 -*-
# @Time    : 2024/5/20 5:20
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : Local_funcs.py
# ------❤❤❤------ #

import datetime
import json
import os
import re
import logging
import subprocess
import subprocess as sp
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from pyzed import sl

from .get_result import get_det_result, get_seg_result, polynomial_fit
from .tracker.byte_tracker import BYTETracker

# logging.basicConfig(level=logging.ERROR)

# region
# ------------------------------图像、mp4功能---------------------------------#
def detect_image_or_imgdir(opt, img_path, Model, draw_bezier,saveMask):
    """使用图片或图片目录"""
    saves = f"{opt.base_directory}/detect_image"
    save_path = Path(create_incremental_directory(saves))
    for image in img_path:
        im = cv2.imread(str(image))
        start_time = time.time()
        det, seg, masks = Model(im)
        end_time = time.time()
        res = Model.my_show(det, seg, im, zed_show=True)
        # res = Model.my_show_yellow_color(det, seg, im, svo=True)
        if len(seg[0]) != 0 and draw_bezier:
            box, segpoint = seg[0][0], seg[1][0]
            if box[box[:, -1] == 0].size != 0:  # 根据lebel查看是否有车道线. 0是车道线类别
                for index, j in enumerate(
                        [tensor for tensor, is_true in zip(segpoint, box[:, -1] == 0) if is_true]):  # 只要车道线的数据
                    sorted_indices = np.argsort(j[:, 1])[::-1]
                    points_np = j[sorted_indices]
                    fit_xdata = polynomial_fit(list(range(1, len(j) + 1)), points_np.T[0], degree=4)
                    fit_ydata = polynomial_fit(list(range(1, len(j) + 1)), points_np.T[1], degree=4)
                    fit_point = np.array([fit_xdata, fit_ydata])  # 组合
                    for i in range(len(fit_point[0]) - 1):  # 画线条
                        x1 = fit_point[0][i]
                        y1 = fit_point[1][i]
                        x2 = fit_point[0][i + 1]
                        y2 = fit_point[1][i + 1]
                        cv2.line(res, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    indices = np.linspace(0, fit_point.shape[1] - 1, num=15, dtype=int)  # 等距取点的索引
                    new_fit_point = fit_point[:, indices]
                    max_distance = np.linalg.norm(np.vstack((new_fit_point[0], new_fit_point[1])).T[0] -
                                                  np.vstack((new_fit_point[0], new_fit_point[1])).T[-1])
                    for i, (x, y) in enumerate(zip(new_fit_point[0], new_fit_point[1])):
                        if max_distance > 200:
                            cv2.circle(res, (int(x), int(y)), 5, (0, 0, 255), -1)
                        if max_distance < 200 and i == 4 or i == 10:
                            cv2.circle(res, (int(x), int(y)), 5, (0, 0, 255), -1)
        print(f"Use time for {image}: {(end_time - start_time) * 1000:.2f} ms")
        save_img = save_path / image.name
        cv2.imwrite(str(save_img), res)
        print(f"Save in {save_img}")

        if saveMask:
            filename = os.path.join(save_path, f'{os.path.splitext(image.name)[0]}_mask.png')
            # 如果有多个掩膜
            if len(masks) != 0:
                # 根据第一个掩膜的尺寸创建空白彩色图像
                height, width = masks[0].shape[:2]
                combined_mask = np.full((height, width, 3), (114, 114, 114), dtype=np.uint8)  # 初始化为灰色
                # 定义颜色列表，不同掩膜用不同颜色
                colors = generate_colors(len(masks))
                # 遍历每个掩膜
                for i in range(len(masks)):
                    mask_image = (masks[i] * 255).astype(np.uint8)  # 将掩膜转化为 0 或 255 的二值图像
                    color_mask = np.zeros_like(combined_mask)  # 创建与 combined_mask 大小相同的空白彩色图像
                    # 为每个通道分别应用颜色
                    for j in range(3):
                        color_mask[:, :, j] = mask_image * (colors[i % len(colors)][j] // 255)
                    # 将彩色掩膜叠加到 combined_mask
                    combined_mask = cv2.addWeighted(combined_mask, 1, color_mask, 0.5, 0)
                # 保存最终组合的掩膜图像
                cv2.imwrite(filename, combined_mask)


# for get result -->json file
def detect_image_or_imgdir_json(opt, img_path, Model):
    """使用图片或图片目录，并生成对应的json预标注"""
    saves = f"{opt.base_directory}/detect_image/image_res_and_json"
    save_dir = create_incremental_directory(saves)
    sub_dir1 = os.path.join(save_dir, 'images')
    sub_dir2 = os.path.join(save_dir, 'json')

    os.makedirs(sub_dir1)
    os.makedirs(sub_dir2)

    for image in img_path:
        file_name = os.path.splitext(os.path.basename(str(image)))[0]
        json_file_path = f"{sub_dir2}/{file_name}.json"
        data = {
            "version": "5.4.1",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.join("..\\images", f"{file_name}.jpg"),  # 根据文件名生成图片路径
            "imageData": None,
            "imageHeight": 1080,
            "imageWidth": 1920
        }
        im = cv2.imread(str(image))
        start_time = time.time()
        det, seg = Model(im)
        end_time = time.time()
        res = Model.my_show(det, seg, im, zed_show=True)
        if len(det[0]) != 0:
            for i in det[0]:
                shape_points = []
                shape_points.append([i[0], i[1]])
                shape_points.append([i[2], i[3]])
                shape_label = Model.label_dict[i[-1]]
                data["shapes"].append({
                    "label": shape_label,
                    "points": shape_points,
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                    "mask": None
                })

        if len(seg[0]) != 0:
            for i, j in zip(seg[0][0], seg[1][0]):
                shape_label = Model.classes[int(i[-1])]
                shape_points = []
                stride = 10
                j = j[::stride]
                j = j.tolist()
                for k in range(0, len(j)):
                    x = j[k][0]
                    y = j[k][1]
                    shape_points.append([x, y])
                data["shapes"].append({
                    "label": shape_label,
                    "points": shape_points,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {},
                    "mask": None
                })
        with open(json_file_path, 'w') as outfile:
            json.dump(data, outfile, indent=2)
        print(f"Use time for {image}: {(end_time - start_time) * 1000:.2f} ms")
        save_img = f"{sub_dir1}/{image.name}"
        cv2.imwrite(str(save_img), res)
        print(f"Save in {save_img}")


def detect_video(opt, Model):
    """使用一般的视频"""
    saves = f"{opt.base_directory}/detect_video"
    save_path = Path(create_incremental_directory(saves))
    save_name = os.path.splitext(os.path.basename(opt.path))[0] + '_infer.mp4'
    save_video = save_path / save_name

    capture = cv2.VideoCapture(str(opt.path))
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(str(save_video), fourcc, 30, size)

    cv2.namedWindow('trt_result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('trt_result', 1344, 768)
    while True:
        ref, frame = capture.read()
        if not ref:
            break
        det, seg, mask = Model(frame)
        frame = Model.my_show(det, seg, frame)
        cv2.imshow("trt_result", frame)
        c = cv2.waitKey(1) & 0xff
        out.write(frame)
        if c == 27:
            capture.release()
            break
    capture.release()
    out.release()
    print("Save path :" + str(save_video))
# endregion

# 创建递增目录
def create_incremental_directory(base_dir, subfolder_name=None, save_img=False):
    """创建递增目录"""
    # 拼接完整的目标目录路径
    target_dir = os.path.join(base_dir, subfolder_name) if subfolder_name else os.path.join(base_dir)
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # 获取目标目录下的所有子目录，并过滤出符合命名规则的目录
    existing_dirs = [d for d in os.listdir(target_dir) if
                     os.path.isdir(os.path.join(target_dir, d)) and re.match(r'runs\d+', d)]
    # 获取已有目录的编号
    existing_numbers = sorted(int(re.findall(r'\d+', d)[0]) for d in existing_dirs)
    # 找到第一个缺失的编号
    next_num = 1
    for num in existing_numbers:
        if num != next_num:
            break
        next_num += 1
    # 新目录的名称
    new_dir_name = f"runs{next_num}"
    new_dir_path = os.path.join(target_dir, new_dir_name)
    # 创建新目录
    os.makedirs(new_dir_path)
    if save_img:
        os.mkdir(f"{new_dir_path}/imgs")

    return new_dir_path

# ------------------------------svo---------------------------------#
# 初始化相机
def init_camera(opt, svo_real_time_mode=True):
    """初始化相机"""
    saves = f"{opt.base_directory}/detect_svo"
    if not os.path.exists(saves):
        os.makedirs(saves)
    if opt.path != 'camera':
        mode = 'LocalFile'
    else:
        mode = 'RealTime'
    new_directory = create_incremental_directory(saves, mode, opt.save_img)
    zed = sl.Camera()

    if opt.path != 'camera':
        input_type = sl.InputType()
        input_type.set_from_svo_file(str(opt.path.resolve()))  # 得到绝对路径
        filename, extension = os.path.splitext(os.path.basename(opt.path))
        save_txt_path = os.path.join(new_directory, f'{filename}_infer.txt')
        save_video_path = os.path.join(new_directory, f'{filename}_infer.mp4')
        save_svo_path = None
        save_imgs_path = os.path.join(new_directory, 'imgs')
        init_params = sl.InitParameters(input_t=input_type,
                                        svo_real_time_mode=svo_real_time_mode)  ## svo_real_time_mode设置为False后，会处理每一帧
    else:
        save_txt_path = os.path.join(new_directory, 'Record_my.txt')
        save_video_path = os.path.join(new_directory, 'Record_myinfer.mp4')
        save_svo_path = os.path.join(new_directory, 'Record_orin.svo')
        save_imgs_path = os.path.join(new_directory, 'imgs')
        init_params = sl.InitParameters(svo_real_time_mode=svo_real_time_mode)
        init_params.camera_resolution = sl.RESOLUTION.HD1080

    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
    else:
        print('Total Frame: ', zed.get_svo_number_of_frames())
    # for save record svo
    if opt.save_orin_svo and save_svo_path:
        print(save_svo_path)
        recording_param = sl.RecordingParameters(save_svo_path, sl.SVO_COMPRESSION_MODE.H264)
        err = zed.enable_recording(recording_param)
        if err != sl.ERROR_CODE.SUCCESS:
            print('recording:', err)
            exit(1)

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True

    zed.enable_object_detection(obj_param)
    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    point_cloud_res = sl.Resolution(min(camera_res.width, 1920), min(camera_res.height, 1080))
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)  # 点云容器
    cam_w_pose = sl.Pose()  # 姿态容器
    sensors_data = sl.SensorsData()  # 传感器容器
    py_orientation = sl.Orientation()  # 角
    image_left_tmp = sl.Mat()  # 装image的容器
    # 不需要返回save_svo_path
    return (
        new_directory, save_txt_path, save_video_path, save_imgs_path, zed, runtime_params, image_left_tmp,
        sensors_data,
        point_cloud, point_cloud_res, cam_w_pose, py_orientation, objects, obj_runtime_param)


# 跟踪iou区分
def iou(box: np.ndarray, boxes: np.ndarray):
    xy_max = np.minimum(boxes[:, 2:], box[2:])
    xy_min = np.maximum(boxes[:, :2], box[:2])
    inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
    inter = inter[:, 0] * inter[:, 1]

    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_box = (box[2] - box[0]) * (box[3] - box[1])

    return inter / (area_box + area_boxes - inter)


# 获取当前帧的信息
def get_current_frame_info(zed, point_cloud, point_cloud_res, cam_w_pose, py_orientation, sensors_data):
    # -- Get the point_cloud
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
    # -- Get the ox oy oz ow
    zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)
    ox = round(cam_w_pose.get_orientation(py_orientation).get()[0], 3)  # 获取ZED相机在世界坐标系中的位置和姿态
    oy = round(cam_w_pose.get_orientation(py_orientation).get()[1], 3)
    oz = round(cam_w_pose.get_orientation(py_orientation).get()[2], 3)
    ow = round(cam_w_pose.get_orientation(py_orientation).get()[3], 3)
    # --Get the magnetic_heading
    zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT)  # 获取传感器数据，指定时间参考为当前时间
    magnetometer_data = sensors_data.get_magnetometer_data()  # 拿到磁力计数据，储存在magnetometer_data中
    magnetic_heading = round(magnetometer_data.magnetic_heading, 4)  # 磁力计值四舍五入
    # -- Get the camera time
    timestamp = int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).data_ns / (10 ** 6))
    timeStamp = float(timestamp) / 1000
    ret_datetime = datetime.datetime.utcfromtimestamp(timeStamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return ox, oy, oz, ow, magnetic_heading, timeStamp, ret_datetime


def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        # 均匀分布在 HSV 空间的色调上，然后转换为 BGR
        hue = int(i * 180 / num_colors)  # 取值范围为 0 到 180（OpenCV 中 H 通道范围为 0-180）
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color))  # 转换为 BGR 元组
    return colors


# 检测svo使用了跟踪，rtsp可选
def detect_svo_track(opt, Model, rtspUrl, use_RTSP):
    # Init camera
    (new_directory, save_txt_path, save_video_path, save_imgs_path, zed, runtime_params, image_left_tmp,
     sensors_data, point_cloud, point_cloud_res, cam_w_pose, py_orientation, objects, obj_runtime_param) \
        = init_camera(opt, svo_real_time_mode=True)
    # track
    det_tracker = BYTETracker(opt, frame_rate=30)
    seg_tracker = BYTETracker(opt, frame_rate=30)
    file = open(save_txt_path, 'w')  # Open local txt file
    # Set video writer
    video_writer = None
    if opt.save_video:
        if opt.save_video_ffmpeg:
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24', '-s', f'{1920}x{1080}', '-r', str(25),
                '-i', '-', '-an', '-vcodec', 'mpeg4', '-qscale:v', '5', save_video_path
            ]  # -qscale:v 是控制视频质量的一个重要参数，取值范围为 1 到 31，数值越小视频质量越高
            ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        else:
            video_writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter.fourcc(*'MP4V'), 25,
                                           (1920, 1080))  # axc1好像终止程序保存的视频打不开
    # RTSP
    if use_RTSP:
        command = [
            'ffmpeg',
            # 're',#
            # '-y', # 无需询问即可覆盖输出文件
            '-f', 'rawvideo',  # 强制输入或输出文件格式
            '-vcodec', 'rawvideo',  # 设置视频编解码器。这是-codec:v的别名
            '-pix_fmt', 'bgr24',  # 设置像素格式
            '-s', '1920*1080',  # 设置图像大小
            '-r', '30',  # 设置帧率
            '-i', '-',  # 输入
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-f', 'rtsp',  # 强制输入或输出文件格式
            rtspUrl]
        pipe = sp.Popen(command, stdin=sp.PIPE)

    cv2.namedWindow('||    Result    ||', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('||    Result    ||', 1080, 720)
    try:
        while 1:
            grab = time.time()
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # -- Get the image and info
                zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
                image_net = image_left_tmp.get_data()
                image_net = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)
                ori=image_net.copy()
                ox, oy, oz, ow, magnetic_heading, timeStamp, ret_datetime = get_current_frame_info(zed, point_cloud, point_cloud_res, cam_w_pose, py_orientation, sensors_data)
                grabend=time.time()
                if use_RTSP:
                    pipe.stdin.write(image_net.tostring())  # 推流zed相机画面
                # Infer and get the result
                det, seg, masks = Model(image_net)
                # for det track.
                if len(det[0]) != 0:
                    new_det_output = np.zeros((det[0].shape[0], 7))
                    new_det_output[:, :6] = det[0][:, :6]
                    track_det = det[0].copy()
                    track_det[:, 4] = 0.95
                    tracks = det_tracker.update(track_det[:, :5])
                    # 将id和框对应起来
                    for track in tracks:
                        box_iou = iou(track.tlbr, track_det[:, :4])
                        maxindex = np.argmax(box_iou)
                        new_det_output[maxindex, :6] = det[0][maxindex, :6]
                        new_det_output[maxindex, 6] = track.track_id
                    # new_det_output  x1y1x2y2 conf class id
                    # 为跟踪丢失的手动分配id
                    if 0 in new_det_output[:,6]:
                        for i in range(len(new_det_output[:,5])):
                            if new_det_output[:,6][i] == 0:
                                new_det_output[:,6][i] = det_tracker.nextid()
                    det[0] = new_det_output
                # for seg track
                if len(seg[0]) != 0:
                    new_seg_output = np.zeros((seg[0][0].shape[0], 7))  # seg是个list2 seg[0]是个list seg[0][0]里面是框
                    # 使用seg[0][0]填充，然后再填充id
                    new_seg_output[:, :5] = seg[0][0][:, :5]
                    new_seg_output[:, 6] = seg[0][0][:, 5]
                    track_seg = seg[0][0].copy()
                    track_seg[:, 4] = 0.95
                    seg_track = seg_tracker.update(track_seg[:, :5])
                    # 将id和框对应起来
                    for track in seg_track:
                        box_iou = iou(track.tlbr, track_seg[:, :4])
                        maxindex = np.argmax(box_iou)
                        new_seg_output[maxindex, :5] = seg[0][0][maxindex, :5]
                        new_seg_output[maxindex, 5] = track.track_id
                        new_seg_output[maxindex, 6] = seg[0][0][maxindex, 5]
                    # new_seg_output  x1y1x2y2 conf id class
                    if 0 in new_seg_output[:,5]:
                        for i in range(len(new_seg_output[:,5])):
                            if new_seg_output[:,5][i] == 0:
                                new_seg_output[:,5][i] = seg_tracker.nextid()
                    seg[0] = [new_seg_output]
                '''
                mask = np.zeros_like(image, dtype=np.uint8)
                for label, color in zip(labels, colors):
                    mask[sem_seg[0] == label, :] = color
                
                
                color_seg = (image * (1 - self.alpha) + mask * self.alpha).astype(
                np.uint8)
            
                '''
                # deal img for show
                bgr_image = Model.my_show_track(det, seg, image_net)

                # write sth
                writestart = time.time()
                file.write(f"time:{ret_datetime},{ox} {oy} {oz} {ow} {magnetic_heading}\n")
                get_det_result(det, point_cloud, file, Model)
                get_seg_result(seg, point_cloud, bgr_image, file, Model, ret_datetime)
                writeend = time.time()

                cvshowsave = time.time()
                # Put timestamp
                cv2.putText(bgr_image, str(ret_datetime), (800, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 1, cv2.LINE_AA)
                # Visualization with opencv
                cv2.imshow("||    Result    ||", bgr_image)
                # pipe.stdin.write(bgr_image.tostring())# 推流结果

                # Save video
                if opt.save_video:
                    if opt.save_video_ffmpeg:
                        ffmpeg_proc.stdin.write(bgr_image.tobytes())  # 通过管道传递给 FFmpeg
                    else:
                        video_writer.write(bgr_image)
                # Save picture
                if opt.save_img:
                    filename = datetime.datetime.utcfromtimestamp(timeStamp).strftime("%Y-%m-%d_%H.%M.%S.%f")[:-3]
                    cv2.imwrite(f"{save_imgs_path}/{filename}.jpg", bgr_image)
                    cv2.imwrite(f"{save_imgs_path}/{filename}.jpeg", ori)
                if opt.save_mask:
                    save_imgs_path = os.path.splitext(save_video_path)[0]
                    os.makedirs(save_imgs_path, exist_ok=True)
                    timeStamp = datetime.datetime.utcfromtimestamp(timeStamp).strftime("%Y-%m-%d_%H.%M.%S.%f")[:-3]
                    filename = os.path.join(save_imgs_path, f'{timeStamp}.png')
                    # 如果有多个掩膜
                    if len(masks) != 0 or masks is not None:
                        # 根据第一个掩膜的尺寸创建空白彩色图像
                        height, width = masks[0].shape[:2]
                        combined_mask = np.full((height, width, 3), (114, 114, 114), dtype=np.uint8)  # 初始化为灰色
                        # 定义颜色列表，不同掩膜用不同颜色
                        colors = generate_colors(len(masks))
                        # 遍历每个掩膜
                        for i in range(len(masks)):
                            mask_image = (masks[i] * 255).astype(np.uint8)  # 将掩膜转化为 0 或 255 的二值图像
                            color_mask = np.zeros_like(combined_mask)  # 创建与 combined_mask 大小相同的空白彩色图像
                            # 为每个通道分别应用颜色
                            for j in range(3):
                                color_mask[:, :, j] = mask_image * (colors[i % len(colors)][j] // 255)
                            # 将彩色掩膜叠加到 combined_mask
                            combined_mask = cv2.addWeighted(combined_mask, 1, color_mask, 0.5, 0)
                        # 保存最终组合的掩膜图像
                        cv2.imwrite(filename, combined_mask)
                # Exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                end=time.time()
                print(f"grab: {(grabend- grab) * 1000:6.2f}ms, "
                      f"write:{(writeend - writestart) * 1000:6.2f}ms, "
                      f"showsave:{(end - cvshowsave) * 1000:6.2f}ms, "
                      f"total:{(end - grab) * 1000:6.2f}ms"
                      )

            else:
                break

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()  # 打印详细的错误跟踪信息
        # logging.error("发生错误", exc_info=True)  # 记录错误信息和堆栈跟踪
    finally:
        print('result save at ', new_directory)
        cv2.destroyAllWindows()
        if opt.save_video_ffmpeg:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
        else:
            video_writer.release()
        zed.close()
        file.close()
