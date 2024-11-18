# -*- coding: utf-8 -*-
# @Time    : 2024/5/20 5:20
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : RealTime_funcs.py
# ------❤❤❤------ #

import datetime
import json
import os
import platform
import queue
import re
import signal
import socket
import subprocess
import subprocess as sp
import sys
import threading
import time
import traceback

import cv2
import numpy as np
from pyzed import sl

from .get_result import get_det_result, get_seg_result
from .tracker.byte_tracker import BYTETracker

message_queue = queue.Queue()
DATA = {'txt': '../output_folder/detect_svo_socker_Plus',
        'video': '../output_folder/detect_svo_socker_Plus',
        'svo': '../output_folder/detect_svo_socker_Plus',
        'taskId': 'test',
        'command': 'nothing'}


# region
# 获取ip
def get_ip_addresses():
    # 判断操作系统
    system = platform.system()
    if system == "Windows":
        # Windows 使用 ipconfig
        result = subprocess.run(['ipconfig'], capture_output=True, text=True)
        output = result.stdout
        # 使用正则表达式提取所有 IPv4 地址
        ip_pattern = r'IPv4 地址[. ]*: (\d+\.\d+\.\d+\.\d+)'
    else:
        # Linux 或 Mac 使用 ifconfig
        result = subprocess.run(['ifconfig'], capture_output=True, text=True)
        output = result.stdout
        # 使用正则表达式提取所有 IP 地址
        ip_pattern = r'inet (\d+\.\d+\.\d+\.\d+)'

    # 查找匹配的 IP 地址
    matches = re.findall(ip_pattern, output)
    # 排除回环地址
    return [ip for ip in matches if ip != '127.0.0.1']


# 跟踪iou区分
def iou(box: np.ndarray, boxes: np.ndarray):
    xy_max = np.minimum(boxes[:, 2:], box[2:])
    xy_min = np.maximum(boxes[:, :2], box[:2])
    inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
    inter = inter[:, 0] * inter[:, 1]

    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_box = (box[2] - box[0]) * (box[3] - box[1])

    return inter / (area_box + area_boxes - inter)


# 初始化相机，没有递增目录
def init_camera_socket(opt, svo_real_time_mode=True):
    zed = sl.Camera()
    if opt.path != 'camera':
        input_type = sl.InputType()
        input_type.set_from_svo_file(str(opt.path.resolve()))  # 得到绝对路径
        init_params = sl.InitParameters(input_t=input_type,
                                        svo_real_time_mode=svo_real_time_mode)  ## svo_real_time_mode设置为False后，会处理每一帧
    else:
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
        print('Local File Total Frame: ', zed.get_svo_number_of_frames())

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
    return (zed, runtime_params, image_left_tmp, sensors_data,
            point_cloud, point_cloud_res, cam_w_pose, py_orientation, objects, obj_runtime_param)


# 发送消息
def send_messages(sock):
    while True:
        try:
            message = message_queue.get()  # 从队列中获取消息
            sock.sendall(message.encode('utf-8'))
        except Exception as e:
            print(f"Error sending data: {e}")
            break


# 接收消息
def receive_messages_rtsp(sock):
    global DATA
    while True:
        try:
            DATA = sock.recv(1024).decode('utf-8')
            if not DATA:
                break
            json_start = DATA.find('{')
            json_end = DATA.rfind('}')
            # 提取 JSON 字符串
            json_str = DATA[json_start:json_end + 1]
            # 将 JSON 字符串解析为 Python 字典
            DATA = json.loads(json_str)
            required_keys = ['txt', 'video', 'svo', 'taskId', 'command']
            missing_keys = [key for key in required_keys if key not in DATA]
            if missing_keys:
                raise ValueError(f"Missing required keys: {missing_keys}")
            print(f"-------------------Received: {DATA}-------------------------")

        except Exception as e:
            print(f"Error receiving data: {e}")
            break


# 获取文件的创建时间
def get_creation_time(file_path):
    return os.path.getctime(file_path)


# 合并mp4
def merge_videos_by_creation_time(input_directory, output_file_path, taskId, stop_cont):
    file_list_path = 'filelist.txt'
    mp4_files = [f for f in os.listdir(input_directory) if f.endswith('.mp4')] if os.path.isdir(
        output_file_path) else []
    # 有2个mp4就合并一次，也可以全部一起合
    if len(mp4_files) < 2:
        print("文件数量不足以进行合并。")
        return
    stop_cont[0] += 1  # 更新列表中的值
    # 按创建时间排序文件
    mp4_files.sort(key=lambda f: get_creation_time(os.path.join(input_directory, f)))

    with open(file_list_path, 'w') as file:
        for mp4_file in mp4_files:
            file.write(f"file '{os.path.join(input_directory, mp4_file)}'\n")

    output_file = f"{output_file_path}/{taskId}{stop_cont[0]}.mp4"
    # 执行 FFmpeg 合并
    command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', file_list_path,
        '-c', 'copy',
        output_file
    ]
    sp.run(command, check=True)
    for mp4_file in mp4_files:
        file_path = os.path.join(input_directory, mp4_file)
        os.remove(file_path)
        print(f"已删除文件: {file_path}")

    # 删除 filelist.txt
    os.remove(file_list_path)

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

# 捕获中断
def signal_handler(signal, frame):
    global ffmpeg_proc
    print('Signal received, shutting down gracefully...')
    try:
        # 关闭 stdin 并等待 ffmpeg 进程完成
        if ffmpeg_proc.stdin:
            ffmpeg_proc.stdin.close()  # 关闭 stdin 让 ffmpeg 知道输入完成
        ffmpeg_proc.wait()  # 等待 ffmpeg 进程完成
        print('FFmpeg process cleaned up.')
    except Exception as e:
        print(f"Error during cleanup: {e}")
    sys.exit(0)  # 退出程序


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
# endregion


# 检测svo使用了跟踪，使用了trsp，使用socker
def detect_svo_track_socket_rstp(opt, Model, host, port, rtspUrl):
    global DATA
    # 创建socket对象
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            print(f"Connected to {host}:{port}")
        except Exception as e:
            print(f"Could not connect to {host}:{port}: {e}")
            return

        # 启动接收消息的线程
        receive_thread = threading.Thread(target=receive_messages_rtsp, args=(s,))
        receive_thread.daemon = True
        receive_thread.start()

        # 启动发送消息的线程
        send_thread = threading.Thread(target=send_messages, args=(s,))
        send_thread.daemon = True
        send_thread.start()

        main_logic_rtsp(opt, Model, rtspUrl)


# socker收发具体实现
def main_logic_rtsp(opt, Model, rtspUrl):
    global DATA, ffmpeg_proc
    assert opt.path == 'camera', 'must be use camera'
    (zed, runtime_params, image_left_tmp, sensors_data, point_cloud, point_cloud_res, cam_w_pose,
     py_orientation, objects, obj_runtime_param) = init_camera_socket(opt, svo_real_time_mode=True)  # Init camera
    det_tracker = BYTETracker(opt, frame_rate=30)
    seg_tracker = BYTETracker(opt, frame_rate=30)
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
    # 控制变量
    stop_handled = False
    pause_handled = False
    first_message = False
    resume_svo = 1
    resume_mp4 = 0
    creat_path = True
    creat = True
    ffmpeg_proc = None
    new_dir_path = None
    task_old = ''
    stop_cont = [0]
    try:
        while True:
            grab = time.time()
            error_code = zed.grab(runtime_params)
            if error_code == sl.ERROR_CODE.SUCCESS:
                # -- Get the image and info
                zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
                image_net = image_left_tmp.get_data()
                image_net = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)
                image_orin = image_net.copy()
                ox, oy, oz, ow, magnetic_heading, timeStamp, ret_datetime = get_current_frame_info(zed, point_cloud,point_cloud_res,cam_w_pose,py_orientation,sensors_data)

                grab_end = time.time()
                pipe.stdin.write(image_net.tostring())  # 推流zed相机画面
                if DATA['command'] == 'start' and error_code == sl.ERROR_CODE.SUCCESS:
                    pause_handled = False
                    stop_handled = False
                    if creat_path:  # mp4和txt能够续写，每次stop后释放资源。重新start后再新建mp4文件，txt不受影响
                        # 解析DATA
                        resume_mp4 += 1
                        if task_old == '':
                            task_old = DATA['taskId']
                        if task_old != DATA['taskId']:  # 根据任务来区别视频的编号是否连续
                            task_old = DATA['taskId']
                            resume_mp4, resume_svo = 1, 1

                        save_txt_path = f"{DATA['txt']}/txt/AiData.txt"
                        save_video_path = f"{DATA['video']}/Video_{resume_mp4}.mp4"
                        save_svo_path = f"{DATA['svo']}/ZED_.svo"
                        save_imgs_path = None
                        new_dir_path = f"{DATA['video']}/full_mp4/"
                        for path in [save_txt_path, save_video_path, save_svo_path, new_dir_path]:
                            directory = os.path.dirname(path)  # 提取目录部分
                            if not os.path.exists(directory) and directory:
                                os.makedirs(directory)

                        ffmpeg_cmd = [
                            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                            '-pix_fmt', 'bgr24', '-s', f'{1920}x{1080}', '-r', str(25),
                            '-i', '-', '-an', '-vcodec', 'mpeg4', '-qscale:v', '6', save_video_path
                        ]  # -qscale:v 是控制视频质量的一个重要参数，取值范围为 1 到 31，数值越小视频质量越高

                        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
                        file = open(save_txt_path, 'a')
                        creat_path = False

                    if creat:  # svo文件不能续写，每次pause后都需要重新新建和保存
                        base_path, filename = os.path.split(save_svo_path)
                        name, ext = os.path.splitext(filename)
                        new_filename = f"{name}{resume_svo}{ext}"
                        new_save_svo_path = os.path.join(base_path, new_filename)
                        print('crate new svo: ', new_save_svo_path)
                        resume_svo += 1
                        recording_param = sl.RecordingParameters(new_save_svo_path, sl.SVO_COMPRESSION_MODE.H264)
                        zed.enable_recording(recording_param)
                        creat = False
                    # region
                    # Infer and get the result
                    det, seg, mask = Model(image_net)
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
                        if 0 in new_det_output[:, 6]:
                            for i in range(len(new_det_output[:, 5])):
                                if new_det_output[:, 6][i] == 0:
                                    new_det_output[:, 6][i] = det_tracker.nextid()
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
                        if 0 in new_seg_output[:, 5]:
                            for i in range(len(new_seg_output[:, 5])):
                                if new_seg_output[:, 5][i] == 0:
                                    new_seg_output[:, 5][i] = seg_tracker.nextid()
                        seg[0] = [new_seg_output]
                    # deal img for show
                    bgr_image = Model.my_show_track(det, seg, image_net, svo=True)
                    # write and send sth
                    file.write(f"time:{ret_datetime},{ox} {oy} {oz} {ow} {magnetic_heading}\n")

                    message = 'type:city' + '\n'
                    message += f"time:{ret_datetime},{ox} {oy} {oz} {ow} {magnetic_heading}\n"
                    message += 'label:'

                    send_det_msg = get_det_result(det, point_cloud, file, Model)
                    if send_det_msg != 0 and len(send_det_msg) != 0:
                        message += send_det_msg

                    s_line = get_seg_result(seg, point_cloud, bgr_image, file, Model, ret_datetime)

                    if s_line != 0 and len(s_line) != 0:
                        message += s_line
                    message += '\n'
                    message += '<AI>'
                    message_queue.put(message)
                    # endregion
                    show = time.time()
                    # Put timestamp
                    cv2.putText(bgr_image, str(ret_datetime), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 225), 1, cv2.LINE_AA)

                    # Visualization with opencv
                    global_image = cv2.hconcat([image_orin, bgr_image])
                    cv2.namedWindow('love', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('love', 900, 300)
                    cv2.imshow("love", global_image)
                    cv2.waitKey(1)
                    # Save video
                    if opt.save_video:
                        ffmpeg_proc.stdin.write(bgr_image.tobytes())

                    # Save picture
                    if opt.save_img:
                        filename = datetime.datetime.utcfromtimestamp(timeStamp).strftime("%Y-%m-%d_%H.%M.%S.%f")[:-3]
                        cv2.imwrite(f"{save_imgs_path}/{filename}.jpg", bgr_image)

                    print(f"grab: {(grab_end - grab) * 1000:6.2f}ms")
                    print(f"show: {(time.time() - show) * 1000:6.2f}ms")
                    print(f"total: {(time.time() - grab) * 1000:6.2f}ms")

                elif DATA['command'] == 'pause' and not pause_handled:
                    cv2.destroyAllWindows()
                    zed.disable_recording()
                    print('Pause ...')
                    creat = True
                    pause_handled = True

                elif DATA['command'] == 'stop' and not stop_handled:
                    zed.disable_recording()
                    cv2.destroyAllWindows()
                    if ffmpeg_proc != None:
                        ffmpeg_proc.stdin.close()
                        ffmpeg_proc.wait()
                        file.close()

                    merge_videos_by_creation_time(DATA['video'], f"{new_dir_path}", DATA['taskId'], stop_cont)
                    creat_path = True
                    creat = True
                    print('Stop ...')
                    stop_handled = True

                elif DATA['command'] == 'exit':
                    break
                # 初始状态
                elif DATA['command'] == 'nothing' and not first_message:
                    print('Running ... waite input ...')
                    first_message = True

            if str(error_code) == 'CAMERA MOTION SENSORS NOT DETECTED':
                # 关闭相机
                zed.disable_recording()
                # 停止FFmpeg进程
                pipe.stdin.close()
                pipe.terminate()
                pipe.wait()  # 确保FFmpeg进程完全停止
                # 释放mp4写入
                ffmpeg_proc.stdin.close()
                ffmpeg_proc.wait()
                file.close()
                # 变量置位
                DATA['command'] = 'pause'
                creat = True
                pause_handled = True
                # 尝试重新连接相机
                reconnect_camera_success = False
                while not reconnect_camera_success:
                    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                        print("Camera reconnected, restarting stream...")
                        pipe = sp.Popen(command, stdin=sp.PIPE)  # 重新启动FFmpeg推流进程
                        reconnect_camera_success = True

    except Exception as e:
        print(f"Error occurred during processing: {e}")
        traceback.print_exc()

    finally:
        zed.disable_recording()
        zed.close()
        try:
            if ffmpeg_proc and ffmpeg_proc.stdin and not ffmpeg_proc.stdin.closed:  # 确认 stdin 还没关闭
                ffmpeg_proc.stdin.close()
            if ffmpeg_proc and ffmpeg_proc.poll() is None:  # 确认进程还在运行
                ffmpeg_proc.wait()
            if pipe and pipe.stdin:
                pipe.stdin.close()
            if pipe:
                pipe.wait()
        except Exception as e:
            print(f"Error during final cleanup: {e}")

