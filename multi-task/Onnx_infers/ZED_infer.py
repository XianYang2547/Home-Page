# -*- coding: utf-8 -*-
# @Time    : 2024/2/5 08:48
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : ZED_infer.py
# ------❤❤❤------ #


import argparse
import datetime
from queue import Queue
from threading import Lock, Thread
from time import sleep

import onnxruntime as ort

import Zed_tools.tracking_viewer as cv_viewer
from my_utils import *

lock = Lock()
run_signal = False
exit_signal = False

# label_list = ['Straight arrow', 'Turn left', 'Turn right', 'Straight and left', 'Straight and right',
#               'Turn left and back', 'Rhomboid', 'Zebra crossing', 'Traffic light', 'Traffic sign',
#               'Drivable area', 'Lane line']
label_list = ['lta', 'straight line', 'zebra crossing', 'left turn',
              'sr turn', 'sl turn', 'right turn', 'Rhomboid', 'turn around',
              'line']


def xywh2abcd(xywh):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5 * xywh[2])  # * im_shape[1]
    x_max = (xywh[0] + 0.5 * xywh[2])  # * im_shape[1]
    y_min = (xywh[1] - 0.5 * xywh[3])  # * im_shape[0]
    y_max = (xywh[1] + 0.5 * xywh[3])  # * im_shape[0]

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


def x1y1x2y2_to_xywh(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2
    y = y1 + h / 2
    return [x, y, w, h]


def detections_to_custom_box(detections, im0):
    output = []
    for i, det in enumerate(detections):
        xywh = x1y1x2y2_to_xywh(det[0], det[1], det[2], det[3])
        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh)
        obj.label = det[5]
        obj.probability = det[4]
        obj.is_grounded = False
        output.append(obj)
    return output
'''
cv2.error: OpenCV(4.9.0) /io/opencv/modules/highgui/src/window.cpp:1272: error: (-2:Unspecified error) 
The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. 
If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'
'''
def find_closest_horizontal(points):
    # 按照横坐标（x坐标）从小到大排序
    sorted_points = sorted(enumerate(points), key=lambda x: x[1][0])
    # 输出最远的点对
    farthest_horizontal_indices = (sorted_points[0][0], sorted_points[-1][0])
    farthest_horizontal_distance = np.abs(sorted_points[0][1][0] - sorted_points[-1][1][0])

    return farthest_horizontal_indices, farthest_horizontal_distance


def num_line(lane_seg):
    image = cv2.cvtColor(lane_seg, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(image.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((8, 8), np.uint8)
    morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_DILATE, kernel)
    # 连通区域分析[数量 标签 状态 质心]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph_image, connectivity=8)
    # 设置像素数量阈值
    pixel_threshold = 3000
    # 统计像素数量超过阈值的连通区域
    lane_lines_count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > pixel_threshold:
            lane_lines_count += 1
    # 如果centroids区域>=2,找到最左边和最右边车道的质心点
    centroids = np.delete(centroids, 0, axis=0)  # 删除第一个元素，因为它是图像的中心点，固定值
    if len(centroids) >= 2 and lane_lines_count >= 2:
        closest_horizontal, closest_horizontal_distance = find_closest_horizontal(centroids)
        point1, point2 = centroids[closest_horizontal[0]], centroids[closest_horizontal[1]]  # 左边point1, 右边point2
        cv2.circle(image, (int(point1[0]), int(point1[1])), 5, (0, 0, 0), -1)
        cv2.circle(image, (int(point2[0]), int(point2[1])), 5, (0, 0, 0), -1)
        return image, lane_lines_count, [point1, point2]

    return image, lane_lines_count, [None, None]


def torch_thread(weights, img_size, result_queue, conf_thres, iou_thres):
    global image_net, exit_signal, run_signal, detections

    while not exit_signal:
        if run_signal:
            lock.acquire()
            image = img_preprocess(image_net, img_size)
            preds = weights.run(None, {'images': image})
            results = postprocess(preds, image, image_net, conf_thres, iou_thres)
            det, lane_seg = results[0], results[1]
            im, lane_lines_count, left_and_right_Point = num_line(lane_seg)

            cv2.imshow("lane_seg", im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = detections_to_custom_box(det, image_net)
            lock.release()
            run_signal = False
            result_queue.put([lane_lines_count, left_and_right_Point, det, im])
        sleep(0.01)


def main():
    global image_net, exit_signal, run_signal, detections, depth
    file = open(opt.info_txt_path, 'w')
    result_queue = Queue()
    capture_thread = Thread(target=torch_thread,
                            kwargs={'weights': ort_session, 'img_size': opt.img_size, "conf_thres": opt.conf_thres,
                                    'iou_thres': opt.iou_thres, "result_queue": result_queue})
    capture_thread.start()
    # region
    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat()

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    # Create OpenGL viewer
    # viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 1920), min(camera_res.height, 1080))
    point_cloud_render = sl.Mat()
    # viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image_left = sl.Mat()
    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_res.width, 1080), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # Utilities for tracks view
    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps,
                                                    init_params.depth_maximum_distance)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
    # Camera pose
    cam_w_pose = sl.Pose()  # 创建一个对象，用于存储相机的姿态信息
    depth = sl.Mat()  # 创建一个对象，用于存储深度信息
    sensors_data = sl.SensorsData()  # 创建一个对象，存储各种传感器的数据
    py_orientation = sl.Orientation()
    # endregion
    count=1
    while not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            lock.release()
            run_signal = True

            # 得到相机数据
            zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT)  # 获取传感器数据，指定时间参考为当前时间
            magnetometer_data = sensors_data.get_magnetometer_data()  # 拿到磁力计数据，储存在magnetometer_data中
            magnetic_heading = round(magnetometer_data.magnetic_heading, 4)  # 磁力计值四舍五入
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)  # 得到point_cloud
            # 得到oxoyozow
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)
            ox = round(cam_w_pose.get_orientation(py_orientation).get()[0], 3)  # 获取ZED相机在世界坐标系中的位置和姿态
            oy = round(cam_w_pose.get_orientation(py_orientation).get()[1], 3)
            oz = round(cam_w_pose.get_orientation(py_orientation).get()[2], 3)
            ow = round(cam_w_pose.get_orientation(py_orientation).get()[3], 3)
            # 得到相机时间点
            timestamp = int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).data_ns / (10 ** 6))
            timeStamp = float(timestamp) / 1000
            ret_datetime = datetime.datetime.utcfromtimestamp(timeStamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections
            lock.acquire()

            result = result_queue.get()
            file.write("\n"
                "time:" + str(ret_datetime) + ',' + str(ox) + ' ' + str(oy) + ' ' + str(oz) + " " + str(ow) + " " + str(
                    magnetic_heading) + "\n")
            if result[1][0] is None:
                file.write(
                    f"line:{result[0]}, None,None,0 ----------------------------------------------------{count}\n")
            else:
                lane_line_number = result[0]
                left_point = point_cloud.get_data()[int(round(result[1][0][1], 0)), int(round(result[1][0][0], 0))][:3]
                right_point = point_cloud.get_data()[int(round(result[1][1][1], 0)), int(round(result[1][1][0], 0))][:3]
                lane_width = (right_point[0] - left_point[0]) / (lane_line_number - 1)
                file.write(
                    f"line:{lane_line_number},"
                    f"{left_point[0]:.3f} {left_point[1]:.3f} {left_point[2]:.3f}, "
                    f"{right_point[0]:.3f} {right_point[1]:.3f} {right_point[2]:.3f}, {round(lane_width, 3)} --------------{count}\n")
            # 写入检测到的目标信息
            for i in result[2]:
                # [array([448.65, 636.46, 596.03, 735.64, 0.92032, 0]),
                #  array([796.03, 627.28, 1041.7, 723.7, 0.70855, 3])]
                x_center = i[0] + (i[2] - i[0]) / 2
                y_center = i[1] + (i[3] - i[1]) / 2
                point = point_cloud.get_data()[
                            int(round(y_center, 0)), int(round(x_center, 0))][:3]
                file.write(
                    f"label:{label_list[int(i[5])]}, {point[0]:.3f} {point[1]:.3f} {point[2]:.3f}\n")

            # -- Ingest detections
            zed.ingest_custom_box_objects(detections)
            lock.release()
            zed.retrieve_objects(objects, obj_runtime_param)

            # -- Display
            # Retrieve display data
            # zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            point_cloud.copy_to(point_cloud_render)
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # 3D rendering
            # viewer.updateData(point_cloud_render, objects)
            # 2D rendering
            np.copyto(image_left_ocv, image_left.get_data())
            cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking, label_list)
            # global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            # Tracking view
            track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)
            cv2.imwrite(f"runs/imgs/{count}.jpg",image_left_ocv)
            cv2.imwrite(f"runs/imgs/{count}--{result[0]}.png",result[3])
            count+=1
            cv2.imshow("||    ZED    || ", image_left_ocv)
            key = cv2.waitKey(10)
            if key == 27:
                exit_signal = True
        else:
            exit_signal = True

    # viewer.exit()
    exit_signal = True
    zed.close()
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/best.onnx', )
    parser.add_argument('--svo', type=str, default=r"../test.svo", help='optional svo file')
    parser.add_argument('--info_txt_path', type=str, default='./runs/zed_information.txt')
    parser.add_argument('--img_size', type=int, default=(384, 672), help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.45)
    parser.add_argument('--iou_thres', type=float, default=0.45)
    opt = parser.parse_args()

    ort_session = ort.InferenceSession(opt.weights, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    main()
