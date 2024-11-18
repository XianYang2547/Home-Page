# -*- coding: utf-8 -*-
# @Time    : 2024/5/20 5:20
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : My_infer.py
# ------❤❤❤------ #

import argparse
import importlib.util
import os
import re
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

import datetime
from xy import *
from pyzed import sl
import subprocess
import subprocess as sp

from xy.get_result import get_information, write_guide_line

from Train_Folder.ultralytics.ultralytics import YOLO

classes = {0: 'Lane', 1: 'Isolation_fence', 2: 'Green_belt', 3: 'Sound_insulation_tape',
           4: 'Isolation_net', 5: 'Anti_throwing_net', 6: 'Cement_guardrail'}
label_dict = {0: 'Signs', 1: 'Camera', 2: 'Box', 3: 'Signal_light', 4: 'Bridge', 5: 'Bridge_columns',
              6: 'Electronic_screen', 7: 'Anti_collision_bucket', 8: 'Speed_indicator',
              9: 'gan_zi', 10: 'jia_zi', 11: 'Warning_lights', 12: 'Speed_limit',
              13: 'Right_turn', 14: 'Sr_turn', 15: 'Straight_arrow'}


######拟合函数#####
def my_fit(j, image, cutoff=50, color=(0, 255, 0)):
    # 排序再拟合
    """j.shape (577, 2)"""
    sorted_indices = np.argsort(j[:, 1])[::-1]
    points_np = j[sorted_indices]
    # 拟合
    fit_xdata = polynomial_fit(list(range(1, len(j) + 1)), points_np.T[0], degree=4)
    fit_ydata = polynomial_fit(list(range(1, len(j) + 1)), points_np.T[1], degree=4)

    fit_point = np.array([fit_xdata, fit_ydata])  # 组合
    positive_mask = (fit_point >= 0).all(axis=0)
    fit_point = fit_point[:, positive_mask]  # 拟合出来的点有负值，去掉这组点

    if fit_point.shape[1] > 130:
        fit_point = fit_point[:, cutoff:-cutoff]  # 去掉前后的一些点
    # 画线
    # points = np.array(fit_point).T.astype(int).reshape((-1, 1, 2))   # 将 fit_point 转换为适合 cv2.polylines 的格式
    # cv2.polylines(image, [points], isClosed=False, color=color, thickness=2)

    # 使用步长选取点
    indices = np.linspace(0, fit_point.shape[1] - 1, num=15, dtype=int)  # 等距取点的索引
    return fit_point[:, indices], fit_point


def polynomial_fit(xarray, yarray, degree=3):
    parameters = np.polyfit(xarray, yarray, degree)
    return fit_curve(parameters, xarray)


def fit_curve(parameters, xarray):
    return np.polyval(parameters, xarray)


def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        # 均匀分布在 HSV 空间的色调上，然后转换为 BGR
        hue = int(i * 180 / num_colors)  # 取值范围为 0 到 180（OpenCV 中 H 通道范围为 0-180）
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color))  # 转换为 BGR 元组
    return colors


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


def detect_image_or_imgdir(opt, img_path, Det, Seg, draw_bezier, saveMask):
    """使用图片或图片目录"""
    saves = f"{opt.base_directory}/detect_image"
    save_path = Path(create_incremental_directory(saves))
    for image in img_path:
        im = cv2.imread(str(image))
        start_time = time.time()
        det = Det.predict(im)
        seg = Seg.predict(im)
        end_time = time.time()
        annotated_frame = seg[0].plot(boxes=False)
        det[0].orig_img = annotated_frame
        annotated_frame = det[0].plot()
        # res = Model.my_show_yellow_color(det, seg, im, svo=True)
        if len(seg[0]) != 0 and draw_bezier:
            box, segpoint = seg[0].boxes.cpu().numpy().data, seg[0].masks.xy
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
                        cv2.line(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    indices = np.linspace(0, fit_point.shape[1] - 1, num=15, dtype=int)  # 等距取点的索引
                    new_fit_point = fit_point[:, indices]
                    max_distance = np.linalg.norm(np.vstack((new_fit_point[0], new_fit_point[1])).T[0] -
                                                  np.vstack((new_fit_point[0], new_fit_point[1])).T[-1])
                    for i, (x, y) in enumerate(zip(new_fit_point[0], new_fit_point[1])):
                        if max_distance > 200:
                            cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                        if max_distance < 200 and i == 4 or i == 10:
                            cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        print(f"Use time for {image}: {(end_time - start_time) * 1000:.2f} ms")
        save_img = save_path / image.name
        cv2.imwrite(str(save_img), annotated_frame)
        print(f"Save in {save_img}")

        if saveMask:
            filename = os.path.join(save_path, f'{os.path.splitext(image.name)[0]}_mask.png')
            # 如果有多个掩膜
            if len(seg[0].masks.data) != 0:
                # 根据第一个掩膜的尺寸创建空白彩色图像
                height, width = seg[0].masks.orig_shape[0], seg[0].masks.orig_shape[1]
                combined_mask = np.full((height, width, 3), (114, 114, 114), dtype=np.uint8)  # 初始化为灰色
                # 定义颜色列表，不同掩膜用不同颜色
                colors = generate_colors(len(seg[0].masks.data))
                # 遍历每个掩膜
                for i in range(len(seg[0].masks.data)):
                    mask_image = (seg[0].masks.data[i].cpu().numpy() * 255).astype(np.uint8)
                    mask_image = cv2.resize(mask_image, (width, height), interpolation=cv2.INTER_LINEAR)
                    # 将掩膜转化为 0 或 255 的二值图像
                    color_mask = np.zeros_like(combined_mask)  # 创建与 combined_mask 大小相同的空白彩色图像
                    # 为每个通道分别应用颜色
                    for j in range(3):
                        color_mask[:, :, j] = mask_image * (colors[i % len(colors)][j] // 255)

                    # 将彩色掩膜叠加到 combined_mask
                    combined_mask = cv2.addWeighted(combined_mask, 1, color_mask, 0.5, 0)
                # 保存最终组合的掩膜图像
                cv2.imwrite(filename, combined_mask)


def detect_video(opt, Det, Seg):
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
        det = Det.predict(frame)
        seg = Seg.predict(frame)
        annotated_frame = seg[0].plot(boxes=False)
        det[0].orig_img = annotated_frame
        annotated_frame = det[0].plot()
        cv2.imshow("trt_result", annotated_frame)
        c = cv2.waitKey(1) & 0xff
        out.write(annotated_frame)
        if c == 27:
            capture.release()
            break
    capture.release()
    out.release()
    print("Save path :" + str(save_video))


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


def get_seg_result(seg, point_cloud, bgr_image, file, color_palette, ret_datetime):
    """Write segmentation information"""
    if len(seg) != 0:
        box, segpoint = seg.boxes.data.cpu().numpy(), seg.masks.xy
        s_line = ''
        for key, value in classes.items():  # for each seg class
            if len(box[box[:, -1] == key]) != 0 and (value == 'Anti_throwing_net'):
                file.write(f"{value}s:{len(box[box[:, -1] == key])}\n")
            # Lane
            if len(box[box[:, -1] == key]) != 0 and value == 'Lane':
                get_information(box, segpoint, key, value, bgr_image, file, point_cloud, color_palette, ret_datetime,
                                lane=True, point_size=5)
            # Isolation_fence
            if len(box[box[:, -1] == key]) != 0 and value == 'Isolation_fence':
                get_information(box, segpoint, key, value, bgr_image, file, point_cloud, color_palette, ret_datetime,
                                lane=False, point_size=3)
            # Green_belt
            if len(box[box[:, -1] == key]) != 0 and value == 'Green_belt':
                get_information(box, segpoint, key, value, bgr_image, file, point_cloud, color_palette, ret_datetime,
                                lane=False, point_size=2)
            # Sound_insulation_tape
            if len(box[box[:, -1] == key]) != 0 and value == 'Sound_insulation_tape':
                get_information(box, segpoint, key, value, bgr_image, file, point_cloud, color_palette, ret_datetime,
                                lane=False, point_size=2)
            # Cement_guardrail
            if len(box[box[:, -1] == key]) != 0 and value == 'Cement_guardrail':
                get_information(box, segpoint, key, value, bgr_image, file, point_cloud, color_palette, ret_datetime,
                                lane=False, point_size=2)
            # Isolation_net
            if len(box[box[:, -1] == key]) != 0 and value == 'Guide_line':
                write_guide_line(box, segpoint, key, value, file, point_cloud)

        if s_line:
            return s_line
        else:
            return 0
    else:
        return 0


def detect_svo_track(opt, Det, Seg, rtspUrl, use_RTSP):
    # Init camera
    (new_directory, save_txt_path, save_video_path, save_imgs_path, zed, runtime_params, image_left_tmp,
     sensors_data, point_cloud, point_cloud_res, cam_w_pose, py_orientation, objects, obj_runtime_param) \
        = init_camera(opt, svo_real_time_mode=True)
    color_palette = np.random.uniform(50, 255, size=(len(label_dict) + len(classes), 3))
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
    cv2.namedWindow('||    Result    ||', cv2.WINDOW_GUI_NORMAL)
    # cv2.namedWindow('||    Result    ||', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('||    Result    ||', 1080, 720)
    try:
        while 1:
            grab = time.time()
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # -- Get the image and info
                zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
                image_net = image_left_tmp.get_data()
                image_net = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)
                ori = image_net.copy()
                ox, oy, oz, ow, magnetic_heading, timeStamp, ret_datetime = get_current_frame_info(zed, point_cloud,
                                                                                                   point_cloud_res,
                                                                                                   cam_w_pose,
                                                                                                   py_orientation,
                                                                                                   sensors_data)
                # print('-------------------------------------------')
                # print(f"grab: {(time.time() - grab) * 1000:6.2f}ms")
                if use_RTSP:
                    pipe.stdin.write(image_net.tostring())  # 推流zed相机画面
                # Infer and get the result
                seg_results = Seg.track(image_net, persist=True, tracker='bytetrack.yaml', conf=0.5, iou=0.5, )
                det_results = Det.track(image_net, persist=True, tracker='bytetrack.yaml', conf=0.5, iou=0.5, )
                annotated_frame = seg_results[0].plot(boxes=False)
                det_results[0].orig_img = annotated_frame
                annotated_frame = det_results[0].plot()

                # write sth
                lane = time.time()
                file.write(f"time:{ret_datetime},{ox} {oy} {oz} {ow} {magnetic_heading}\n")
                if len(det_results[0]) != 0:
                    if det_results[0].boxes.id != None:
                        for (bbox, score, label, id) in zip(det_results[0].boxes.data, det_results[0].boxes.conf,
                                                            det_results[0].boxes.cls, det_results[0].boxes.id):
                            x_center = bbox[0] + (bbox[2] - bbox[0]) / 2
                            y_center = bbox[1] + (bbox[3] - bbox[1]) / 2
                            point = point_cloud.get_data()[
                                        int(round(y_center.item(), 0)), int(round(x_center.item(), 0))][
                                    :3]
                            text = f"label:{label_dict[label.item()]},{int(id.item())},{point[0]:.3f} {point[1]:.3f} {point[2]:.3f}\n"
                            if 'nan' not in text and 'inf' not in text:  # 太远处的没有深度值值
                                file.write(text)

                if len(seg_results[0]) != 0:
                    if seg_results[0].boxes.id != None:
                        get_seg_result(seg_results[0], point_cloud, annotated_frame, file, color_palette, ret_datetime)

                print(f"结果处理-------------------------------: {(time.time() - lane) * 1000:6.2f}ms")

                show = time.time()
                # Put timestamp
                cv2.putText(annotated_frame, str(ret_datetime), (800, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255),
                            1, cv2.LINE_AA)
                # Visualization with opencv
                cv2.imshow("||    Result    ||", annotated_frame)
                # pipe.stdin.write(bgr_image.tostring())# 推流结果

                # Save video
                if opt.save_video:
                    if opt.save_video_ffmpeg:
                        ffmpeg_proc.stdin.write(annotated_frame.tobytes())  # 通过管道传递给 FFmpeg
                    else:
                        video_writer.write(annotated_frame)
                # Save picture
                if opt.save_img:
                    filename = datetime.datetime.utcfromtimestamp(timeStamp).strftime("%Y-%m-%d_%H.%M.%S.%f")[:-3]
                    cv2.imwrite(f"{save_imgs_path}/{filename}.jpg", annotated_frame)
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
                # print(f"show: {(time.time() - show) * 1000:6.2f}ms")
                # print(f"total: {(time.time() - grab) * 1000:6.2f}ms")
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


def main(opt):
    Det = YOLO(opt.det, task='detect')
    Seg = YOLO(opt.seg, task='segment')
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
            detect_image_or_imgdir(opt, images, Det, Seg, draw_bezier=True, saveMask=False)
        else:  # 传入文件
            if opt.path.suffix in SUF1:  # image
                images = [opt.path.absolute()]
                detect_image_or_imgdir(opt, images, Det, Seg, draw_bezier=True, saveMask=True)

            elif opt.path.suffix in SUF2:  # video
                detect_video(opt, Det, Seg)

            elif opt.path.suffix in SUF3:  # svo
                detect_svo_track(opt, Det, Seg, f"rtsp://{opt.ip_add[0]}:8554/test", use_RTSP=False)

            else:
                print('Input Err')
    else:
        "use camera"
        detect_svo_track_socket_rstp(opt, Model, '172.16.20.68', 9001, f"rtsp://{opt.ip_add[0]}:8554/zed")


def make_parser():
    # model config
    parser = argparse.ArgumentParser()
    parser.add_argument('--det', type=str,
                        default='/home/xianyang/Desktop/YFZH_Road_Assets/model_files/v8/linuxtrt/gaosu/s/yolomodel/gaosudet.engine')
    parser.add_argument('--seg', type=str,
                        default='/home/xianyang/Desktop/YFZH_Road_Assets/model_files/v8/linuxtrt/gaosu/s/yolomodel/gaosuseg.engine')
    parser.add_argument('--iou_threshold', type=str, default=0.45)
    parser.add_argument('--conf_threshold', type=str, default=0.45)
    # get ip for rtsp
    parser.add_argument('--ip_add', type=str, default=get_ip_addresses())
    # input output path
    parser.add_argument('--path', type=str, default=f"./test_data/t00.svo")
    # parser.add_argument('--path', type=str, default='/home/xianyang/Desktop/chengya1.svo')
    # parser.add_argument('--path', type=str, default='camera')
    parser.add_argument('--base_directory', type=str, default='./output_folder')
    # save config
    parser.add_argument('--save_orin_svo', type=str, default=True)
    parser.add_argument('--save_video', type=str, default=True)
    parser.add_argument('--save_video_ffmpeg', type=str, default=True)
    parser.add_argument('--save_img', type=str, default=False)
    parser.add_argument('--save_mask', default=False)

    return parser


if __name__ == '__main__':
    opt = make_parser().parse_args()
    main(opt)
