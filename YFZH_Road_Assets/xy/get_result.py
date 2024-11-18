# -*- coding: utf-8 -*-
# @Time    : 2024/5/29 13:33
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : get_result.py
# ------❤❤❤------ #

import math
import time
from sympy.codegen.ast import continue_

import cv2
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=np.RankWarning)# 去掉拟合警告

# for det
def get_det_result(det, point_cloud, file, Model):
    """Use center point ---> 3D point  and write file"""
    if len(det[0]) != 0:
        str_accum = []
        for obj in det[0]:
            x_center = obj[0] + (obj[2] - obj[0]) / 2
            y_center = obj[1] + (obj[3] - obj[1]) / 2
            point = point_cloud.get_data()[int(round(y_center, 0)), int(round(x_center, 0))][:3]
            text = f"label:{Model.label_dict[int(obj[5])]},{point[0]:.3f} {point[1]:.3f} {point[2]:.3f}\n" if len(obj)==6 \
                else f"label:{Model.label_dict[int(obj[5])]},{int(obj[6])},{point[0]:.3f} {point[1]:.3f} {point[2]:.3f}\n"
            if 'nan' not in text and 'inf' not in text:# 太远处的没有深度值值
                file.write(text)
                str_accum.append(f"{Model.label_dict[int(obj[5])]},{point[0]:.3f} {point[1]:.3f} {point[2]:.3f}")
        result_str = ';'.join(str_accum)
        return result_str
    else:
        return 0


# for seg
def get_seg_result(seg, point_cloud, bgr_image, file, Model, ret_datetime):
    """Write segmentation information"""
    if len(seg[0]) != 0:
        box, segpoint = seg[0][0], seg[1][0]
        s_line = ''
        for key, value in Model.classes.items():  # for each seg class
            if len(box[box[:, -1] == key]) != 0 and (value == 'Anti_throwing_net'):
                file.write(f"{value}s:{len(box[box[:, -1] == key])}\n")
            # Lane
            if len(box[box[:, -1] == key]) != 0 and value == 'Lane':
                get_information(box, segpoint, key, value, bgr_image, file, point_cloud, Model.color_palette,ret_datetime,lane=True,point_size=5)
            # Isolation_fence
            if len(box[box[:, -1] == key]) != 0 and value == 'Isolation_fence':
                get_information(box, segpoint, key, value, bgr_image, file, point_cloud, Model.color_palette,ret_datetime,lane=False,point_size=3)
            # Green_belt
            if len(box[box[:, -1] == key]) != 0 and value == 'Green_belt':
                get_information(box, segpoint, key, value, bgr_image, file, point_cloud, Model.color_palette,ret_datetime,lane=False,point_size=2)
            # Sound_insulation_tape
            if len(box[box[:, -1] == key]) != 0 and value == 'Sound_insulation_tape':
                get_information(box, segpoint, key, value, bgr_image, file, point_cloud, Model.color_palette,ret_datetime,lane=False,point_size=2)
            # Cement_guardrail
            if len(box[box[:, -1] == key]) != 0 and value == 'Cement_guardrail':
                get_information(box, segpoint, key, value, bgr_image, file, point_cloud, Model.color_palette,ret_datetime,lane=False,point_size=2)
            # Isolation_net
            if len(box[box[:, -1] == key]) != 0 and value == 'Guide_line':
                write_guide_line(box, segpoint, key, value, file, point_cloud)

        if s_line:
            return s_line
        else:
            return 0
    else:
        return 0


def get_information(box, segpoint, key, value, bgr_image, file, point_cloud, color_palette, ret_datetime, lane, point_size):
    # 传入 ret_datetime 方便定位排查
    lane0 = time.time()
    points = get_up_down_point(box, segpoint, key, bgr_image, point_cloud, color_palette, point_size, value, ret_datetime, lane)
    if len(points)!=0 and lane:
        write_lane(file, value, points)
    if len(points)!=0 and not lane:
        write_except_lane(file, value, points)
    # print(f"{value}: {(time.time() - lane0) * 1000:6.2f}ms")


def get_up_down_point(box, segpoint, key, bgr_image, point_cloud, color_palette, point_size, value, ret_datetime, lane=False):
    total=[]
    # 获取当前类别的所有数量的3d点
    for b, j in zip([bb for bb, is_true in zip(box, box[:, -1] == key) if is_true],[tensor for tensor, is_true in zip(segpoint, box[:, -1] == key) if is_true]):
        j=j[~np.any(j < 0, axis=1)]# 过滤掉负值
        if lane:
            '''
            # 按 y 值进行排序
            sorted_points = j[np.argsort(j[:, 1])]
            # 初始化字典来记录每个 y 值的最大 x 和最小 x
            upper_contour = {}
            lower_contour = {}
            # 遍历排序后的点集
            for point in sorted_points:
                x, y = point
                if y not in upper_contour:
                    upper_contour[y] = x
                    lower_contour[y] = x
                else:
                    upper_contour[y] = max(upper_contour[y], x)
                    lower_contour[y] = min(lower_contour[y], x)
            # 上边缘和下边缘的轮廓点
            upper_contour_points = np.array([[x, y] for y, x in upper_contour.items()])
            lower_contour_points = np.array([[x, y] for y, x in lower_contour.items()])'''
            # 按 y 值进行排序
            sorted_points = j[j[:, 1].argsort()]
            # 获取唯一的 y 值，以及对应的每个 y 的第一个和最后一个点
            y_unique, indices_first = np.unique(sorted_points[:, 1], return_index=True)
            indices_last = np.unique(sorted_points[:, 1], return_index=True, return_counts=True)[1] + \
                           np.unique(sorted_points[:, 1], return_counts=True)[1] - 1
            # 获取上边缘和下边缘的轮廓点
            upper_contour_points = sorted_points[indices_last]
            lower_contour_points = sorted_points[indices_first]
        else:
            sorted_points = j[np.argsort(j[:, 0])]
            # 初始化[字典]来记录每个 x 值的最大 y 和最小 y，直接过滤掉端点处多个x点相等的冗余情况
            upper_contour = {}
            lower_contour = {}
            # 遍历排序后的点集
            for point in sorted_points:
                x, y = point
                if x not in upper_contour:
                    upper_contour[x] = y
                    lower_contour[x] = y
                else:
                    upper_contour[x] = max(upper_contour[x], y)
                    lower_contour[x] = min(lower_contour[x], y)
            # 上边缘和下边缘的轮廓点
            # 轮廓构建逻辑对每个 x 都做了对称的处理。如果某些 x 坐标没有相应的上、下点，
            # 这些 x 也不会被添加到 upper_contour 或 lower_contour，从而避免了长度不一致的情况。
            upper_contour_points = np.array([[x, y] for x, y in upper_contour.items()])
            lower_contour_points = np.array([[x, y] for x, y in lower_contour.items()])

        # 掐头去尾 此处拟合防止midpoints的点几个几个的挤在一起
        upper_contour_points = upper_contour_points[3:-10] if len(upper_contour_points)>20 else upper_contour_points
        lower_contour_points = lower_contour_points[3:-10] if len(lower_contour_points)>20 else lower_contour_points
        _, fit_pointU = my_fit(upper_contour_points, bgr_image,cutoff=1)
        upper_contour_points =  fit_pointU.T
        _, fit_pointL = my_fit(lower_contour_points, bgr_image, cutoff=1)
        lower_contour_points = fit_pointL.T
        # 等距从 upper_contour_points 中取 20 个点
        num_points = 20
        indices = np.linspace(0, len(upper_contour_points) - 1, num=num_points, dtype=int)
        sampled_upper_points = upper_contour_points[indices]
        # 从 lower_contour_points 中找到离 sampled_upper_points 最近的点
        nearest_lower_points = []
        for point in sampled_upper_points:
            # 计算距离
            distances = np.linalg.norm(lower_contour_points - point, axis=1)
            # 找到最近点
            nearest_idx = np.argmin(distances)
            nearest_lower_points.append(lower_contour_points[nearest_idx])
        nearest_lower_points = np.array(nearest_lower_points)
        # 计算每对点的中间点
        midpoints = np.int32((sampled_upper_points + nearest_lower_points) / 2)
        # 计算20个点首尾距离
        distance = np.linalg.norm(midpoints[0] - midpoints[-1])
        if distance < 80: # 近处的距离才会小，理论上来说都会有点云值
            # 计算等间距的索引
            num_points = 5
            indices = np.linspace(0, len(midpoints) - 1, num=num_points, dtype=int)
            # 提取等间距的点
            midpoints = midpoints[indices]
        # 取出3D点
        point_data = np.array([point_cloud.get_value(int(point[0].item()), int(point[1].item()))[1][:3] for point in midpoints])
        # 过滤掉无效的数据
        point_data = point_data[~np.isnan(point_data).any(axis=1)]
        point_data = point_data[~np.isinf(point_data).any(axis=1)]
        point_data = point_data[~np.all(point_data == 0, axis=1)]

        if len(point_data)>1:
            # 排序
            point_data = point_data[point_data[:, 2].argsort()]
            # 相邻点的x范围限制  todo 可能多余步骤
            filtered_data = [point_data[1]]  # 第一个不准，从第二个开始，最后一个也不要
            for i in range(2, len(point_data) - 1):
                x_prev = filtered_data[-1][0]  # 获取已保留数据的上一行 x 值
                x_curr = point_data[i, 0]  # 当前行的 x 值
                x_next = point_data[i + 1, 0]  # 下一行的 x 值
                # 判断当前行的 x 是否同时与前后行的 x 差值在 0.5 以内
                if abs(x_curr - x_prev) <= 0.5 and abs(x_curr - x_next) <= 0.5:
                    filtered_data.append(point_data[i])  # 满足条件则保留当前行
            # 取前15个（可选）
            if len(filtered_data)>15:
                filtered_data = filtered_data[:15]
            # 添加id------------------------------------------------------------------
            pos = np.vstack((filtered_data,np.array([[b[-2],0,np.inf]])))
            # 画线点
            if lane:
                # cv2.polylines(bgr_image, [np.int32([midpoints])], isClosed=False, color=(0, 255, 0), thickness=3)
                for point in midpoints[:-1]:
                    cv2.circle(bgr_image, tuple(point), radius=point_size, color=(0, 0, 255), thickness=-1)
                    # cv2.putText(bgr_image,str(f"{int(point[0]),int(point[1])}"),tuple(point),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,0,255),1)
            else:
                for point in midpoints[:-1]:
                    cv2.circle(bgr_image, tuple(point), radius=point_size, color=color_palette[key], thickness=-1)
                    # cv2.putText(bgr_image,str(f"{int(point[0]),int(point[1])}"),tuple(point),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,0,255),1)
            # 绘制线条
            # cv2.polylines(bgr_image, [np.int32([midpoints])], isClosed=False, color=(255, 0, 255), thickness=2)
            # cv2.polylines(bgr_image, [np.int32([upper_contour_points])], isClosed=False, color=(255, 0, 0), thickness=3)
            # cv2.polylines(bgr_image, [np.int32([lower_contour_points])], isClosed=False, color=(0, 255, 0), thickness=3)

            total.append(pos)

    Points=[]
    # 进行拟合
    for filtered_data in total:
        # 拟合------------------------------------------------------------------
        x_coords = [point[0] for point in filtered_data[:-1]]
        y_coords = [point[1] for point in filtered_data[:-1]]
        z_coords = [point[2] for point in filtered_data[:-1]]
        coefficients = np.polyfit(z_coords, x_coords, 2)
        polynomial = np.poly1d(coefficients)
        z_c_fit = np.linspace(z_coords[0], z_coords[-1], 100)
        x_c_fit = polynomial(z_c_fit)
        num_points = len(y_coords)  # y_coords 的数量
        indices = np.linspace(0, len(z_c_fit) - 1, num_points, dtype=int)  # 生成等间距的索引

        # 使用生成的索引选取 x 和 z 的值
        x_c_sampled = x_c_fit[indices]
        z_c_sampled = z_c_fit[indices]
        # 使用列表推导式将 x_c_sampled, y_coords, z_c_sampled 组合成一个新的点列表
        new_point_list = [np.array([x, y, z]) for x, y, z in zip(x_c_sampled, y_coords, z_c_sampled)]

        # 添加id------------------------------------------------------------------
        pos = np.vstack((new_point_list,filtered_data[-1]))
        # 处理成字符串
        result = ['{:.3f} {:.3f} {:.3f}'.format(row[0], row[1], row[2]) for row in pos]
        result = [s + ',' for s in result]# 加个逗号

        Points.append(result)

    return Points


def write_lane(file, value, _coordinates):
    file.write(f"{value}s:{len(_coordinates)}\n")
    # 3维点从左往右排， 再写入
    sorted_lists = sorted(_coordinates, key=lambda x: float(x[0].split()[0]) if x else float('inf'))
    for index, i in enumerate(sorted_lists):
        if len(i) != 0:
            # [entry for entry in i if 'inf' not in entry and 'nan' not in entry] 去掉inf nan
            # i[:-1] 不让增加的ID参与
            line = ' '.join([entry for entry in i[:-1] if 'inf' not in entry and 'nan' not in entry])
            formatted_data = line.replace(', ', ',')
            formatted_data = formatted_data.rstrip(',')
            # i[-1]为'6.0 0 inf' 通过int(float(i[-1].rstrip(',').split()[0]))得到 ID 6
            file.write(f"{value.lower()}{index + 1}:{int(float(i[-1].rstrip(',').split()[0]))},{formatted_data}\n")
        else:
            file.write(f"{value.lower()}{index + 1}:{0}\n")


def write_except_lane(file, value, _coordinates):
    # 按左右排序
    sorted_lists = sorted(_coordinates, key=lambda x: float(x[0].split()[0]) if x else float('inf'))
    RL_list = []
    # 筛选 区分左右
    x_positive = []
    x_negative = []
    # 遍历 sorted_list 中的每个子列表
    for sublist in sorted_lists: # sublist[:-1] 不让最后一个元素参与
        lt_0_sublist = [item for item in sublist[:-1] if
                        is_valid_number(item.strip().split()[0]) and float(item.strip().split()[0]) < 0]
        gt_0_sublist = [item for item in sublist[:-1] if
                        is_valid_number(item.strip().split()[0]) and float(item.strip().split()[0]) > 0]
        if lt_0_sublist:
            lt_0_sublist.append(sublist[-1]) # 加回去
            x_negative.append(lt_0_sublist)
        if gt_0_sublist:
            gt_0_sublist.append(sublist[-1]) # 加回去
            x_positive.append(gt_0_sublist)
    # 取近处的对象，只取1个
    if len(x_positive)!=0:
        if len(x_positive)>=2:
            x_positive = get_min_z_sublist(x_positive)
        else:
            x_positive=x_positive[0]
        RL_list.append(x_positive)
    if len(x_negative)!=0:
        if len(x_negative)>=2:
            x_negative = get_min_z_sublist(x_negative)
        else:
            x_negative=x_negative[0]
        RL_list.append(x_negative)

    file.write(f"{value}s:{len(RL_list)}\n")
    RL_list = sorted(RL_list, key=lambda x: float(x[0].split()[0]) if x else float('inf'))
    for index, i in enumerate(RL_list):
        if len(i) != 0:
            # [entry for entry in i if 'inf' not in entry and 'nan' not in entry] 去掉inf nan
            # i[:-1] 不让增加的ID参与
            line = ' '.join([entry for entry in i[:-1] if 'inf' not in entry and 'nan' not in entry])
            formatted_data = line.replace(', ', ',')
            # data_groups = formatted_data.split(',')
            # # 保留非全0的组
            # filtered_groups = []
            # for group in data_groups:
            #     if group.strip():  # 确保组不为空
            #         numbers = list(map(float, group.split()))
            #         if not all(num == 0 for num in numbers):
            #             filtered_groups.append(group)
            # filtered_data = ','.join(filtered_groups)
            formatted_data = formatted_data.rstrip(',')
            # i[-1]为'6.0 0 inf' 通过int(float(i[-1].rstrip(',').split()[0]))得到 ID 6
            file.write(f"{value.lower()}{index + 1}:{int(float(i[-1].rstrip(',').split()[0]))},{formatted_data}\n")
        else:
            file.write(f"{value.lower()}{index + 1}:{0}\n")


# 导流线
def write_guide_line(box, segpoint, key, value, file, point_cloud):
    file.write(f"{value}s:{len([tensor for tensor, is_true in zip(segpoint, box[:, -1] == key) if is_true])}\n")
    for i, j in enumerate([tensor for tensor, is_true in zip(segpoint, box[:, -1] == key) if is_true]):
        j=j[~np.any(j < 0, axis=1)]# 过滤掉负值
        pos = np.array([point_cloud.get_value(int(point[0].item()), int(point[1].item()))[1][:3] for point in j])
        inf_indices = np.isinf(pos).any(axis=1)
        nan_indices = np.isnan(pos).any(axis=1)
        # 将 inf 和 nan 的行索引合并
        invalid_indices = np.logical_or(inf_indices, nan_indices)
        # 过滤掉这些行
        pos_filtered = pos[~invalid_indices]
        # 排序
        pos_sorted = sorted(pos_filtered, key=lambda x: x[2])
        if len(pos_sorted)>30:
            step = math.ceil(len(pos_sorted) / 30)
            sampled_data = [pos_sorted[i] for i in range(0, len(pos_sorted), step)]
        else:
            sampled_data=pos_sorted
        formatted_data = ','.join(['{:.3f} {:.3f} {:.3f}'.format(x[0], x[1], x[2]) for x in sampled_data])
        file.write(f"{value.lower()}{i + 1}:{formatted_data}\n")


# 杂项
def is_valid_number(value):
    try:
        num = float(value)
        if math.isnan(num) or math.isinf(num):
            return False
        return True
    except ValueError:
        return False


def get_min_z_sublist(group):
    min_z_sublist = None
    min_z_value = float('inf')

    for sublist in group:
        # 获取每个子列表第一个元素的z值  sublist[:-1] 不让最后一个元素参与
        first_z_values = [float(item.strip().split()[2].strip(',')) for item in sublist[:-1] if is_valid_number(item.strip().split()[2].strip(','))]
        if first_z_values:
            first_z = first_z_values[0]  # 第一个元素的z值
            if first_z < min_z_value:
                min_z_value = first_z
                min_z_sublist = sublist

    return min_z_sublist


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








def get_seg_result0(seg, point_cloud, bgr_image, file, Model, ret_datetime):
    """Write segmentation information"""
    if len(seg[0]) != 0:
        box, segpoint = seg[0][0], seg[1][0]
        s_line = ''
        for key, value in Model.classes.items():  # for each seg class
            if len(box[box[:, -1] == key]) != 0 and (value == 'Anti_throwing_net'):
                file.write(f"{value}s:{len(box[box[:, -1] == key])}\n")
            # Lane
            if len(box[box[:, -1] == key]) != 0 and value == 'Lane':
                lane = time.time()
                s_line = gets_lane(box, segpoint, key, value, bgr_image, file, point_cloud)
                get_up_down_point(box, segpoint, key, bgr_image, point_cloud, Model.color_palette, 2, value, ret_datetime,lane=True)
                # print(f"gets_lane: {(time.time() - lane) * 1000:6.2f}ms")
            # Isolation_fence
            if len(box[box[:, -1] == key]) != 0 and value == 'Isolation_fence':
                get_information(box, segpoint, key, value, bgr_image, file, point_cloud, Model.color_palette,ret_datetime,point_size=3)
            # Green_belt
            if len(box[box[:, -1] == key]) != 0 and value == 'Green_belt':
                get_information(box, segpoint, key, value, bgr_image, file, point_cloud, Model.color_palette,ret_datetime,point_size=2)
            # Sound_insulation_tape
            if len(box[box[:, -1] == key]) != 0 and value == 'Sound_insulation_tape':
                get_information(box, segpoint, key, value, bgr_image, file, point_cloud, Model.color_palette,ret_datetime,point_size=2)
            # Cement_guardrail
            if len(box[box[:, -1] == key]) != 0 and value == 'Cement_guardrail':
                get_information(box, segpoint, key, value, bgr_image, file, point_cloud, Model.color_palette,ret_datetime,point_size=2)
            # Isolation_net
            if len(box[box[:, -1] == key]) != 0 and value == 'Guide_line':
                write_guide_line(box, segpoint, key, value, file, point_cloud)

        if s_line:
            return s_line
        else:
            return 0
    else:
        return 0

def get_up_down_point0(box, segpoint, key, bgr_image, point_cloud, color_palette, point_size, value, ret_datetime, lane=False):
    Points=[]
    for b, j in zip([bb for bb, is_true in zip(box, box[:, -1] == key) if is_true],[tensor for tensor, is_true in zip(segpoint, box[:, -1] == key) if is_true]):
        j=j[~np.any(j < 0, axis=1)]# 过滤掉负值
        if lane:
            # 按 y 值进行排序
            sorted_points = j[np.argsort(j[:, 1])]
            # 初始化字典来记录每个 y 值的最大 x 和最小 x
            upper_contour = {}
            lower_contour = {}
            # 遍历排序后的点集
            for point in sorted_points:
                x, y = point
                if y not in upper_contour:
                    upper_contour[y] = x
                    lower_contour[y] = x
                else:
                    upper_contour[y] = max(upper_contour[y], x)
                    lower_contour[y] = min(lower_contour[y], x)
            # 上边缘和下边缘的轮廓点
            upper_contour_points = np.array([[x, y] for y, x in upper_contour.items()])
            lower_contour_points = np.array([[x, y] for y, x in lower_contour.items()])
        else:
            sorted_points = j[np.argsort(j[:, 0])]
            # 初始化[字典]来记录每个 x 值的最大 y 和最小 y，直接过滤掉端点处多个x点相等的冗余情况
            upper_contour = {}
            lower_contour = {}
            # 遍历排序后的点集
            for point in sorted_points:
                x, y = point
                if x not in upper_contour:
                    upper_contour[x] = y
                    lower_contour[x] = y
                else:
                    upper_contour[x] = max(upper_contour[x], y)
                    lower_contour[x] = min(lower_contour[x], y)
            # 上边缘和下边缘的轮廓点
            # 轮廓构建逻辑对每个 x 都做了对称的处理。如果某些 x 坐标没有相应的上、下点，
            # 这些 x 也不会被添加到 upper_contour 或 lower_contour，从而避免了长度不一致的情况。
            upper_contour_points = np.array([[x, y] for x, y in upper_contour.items()])
            lower_contour_points = np.array([[x, y] for x, y in lower_contour.items()])
        # 掐头去尾
        upper_contour_points = upper_contour_points[3:-10] if len(upper_contour)>20 else upper_contour_points
        lower_contour_points = lower_contour_points[3:-10] if len(lower_contour)>20 else lower_contour_points
        # 等距从 upper_contour_points 中取 20 个点
        num_points = 20
        indices = np.linspace(0, len(upper_contour_points) - 1, num=num_points, dtype=int)
        sampled_upper_points = upper_contour_points[indices]
        # 从 lower_contour_points 中找到离 sampled_upper_points 最近的点
        nearest_lower_points = []
        for point in sampled_upper_points:
            # 计算距离
            distances = np.linalg.norm(lower_contour_points - point, axis=1)
            # 找到最近点
            nearest_idx = np.argmin(distances)
            nearest_lower_points.append(lower_contour_points[nearest_idx])
        nearest_lower_points = np.array(nearest_lower_points)
        # 计算每对点的中间点
        midpoints = np.int32((sampled_upper_points + nearest_lower_points) / 2)
        # 计算20个点首尾距离
        distance = np.linalg.norm(midpoints[0] - midpoints[-1])
        if distance < 100: # 近处的距离才会小，理论上来说都会有点云值
            # 计算等间距的索引
            num_points = 5
            indices = np.linspace(0, len(midpoints) - 1, num=num_points, dtype=int)
            # 提取等间距的点
            midpoints = midpoints[indices]
        # 取出3D点
        point_data = np.array([point_cloud.get_value(int(point[0].item()), int(point[1].item()))[1][:3] for point in midpoints])
        # 过滤掉无效的数据
        point_data = point_data[~np.isnan(point_data).any(axis=1)]
        point_data = point_data[~np.isinf(point_data).any(axis=1)]
        point_data = point_data[~np.all(point_data == 0, axis=1)]

        # 较远处的都为inf，过滤后point_data可能为空，防止索引报错
        if len(point_data)>1:
            # 排序
            point_data = point_data[point_data[:, 2].argsort()]
            # 相邻点的x范围限制
            filtered_data = [point_data[1]]  # 第一个不准，从第二个开始，最后一个也不要
            for i in range(2, len(point_data) - 1):
                x_prev = filtered_data[-1][0]  # 获取已保留数据的上一行 x 值
                x_curr = point_data[i, 0]  # 当前行的 x 值
                x_next = point_data[i + 1, 0]  # 下一行的 x 值
                # 判断当前行的 x 是否同时与前后行的 x 差值在 0.5 以内
                if abs(x_curr - x_prev) <= 0.5 and abs(x_curr - x_next) <= 0.5:
                    filtered_data.append(point_data[i])  # 满足条件则保留当前行
            # 取前15个（可选）
            if len(filtered_data)>15:
                filtered_data = filtered_data[:15]

            # 拟合------------------------------------------------------------------
            x_coords = [point[0] for point in filtered_data]
            y_coords = [point[1] for point in filtered_data]
            z_coords = [point[2] for point in filtered_data]
            coefficients = np.polyfit(z_coords, x_coords, 2)
            polynomial = np.poly1d(coefficients)
            z_c_fit = np.linspace(z_coords[0], z_coords[-1], 100)
            x_c_fit = polynomial(z_c_fit)
            num_points = len(y_coords)  # y_coords 的数量
            indices = np.linspace(0, len(z_c_fit) - 1, num_points, dtype=int)  # 生成等间距的索引

            # 使用生成的索引选取 x 和 z 的值
            x_c_sampled = x_c_fit[indices]
            z_c_sampled = z_c_fit[indices]
            # 使用列表推导式将 x_c_sampled, y_coords, z_c_sampled 组合成一个新的点列表
            new_point_list = [np.array([x, y, z]) for x, y, z in zip(x_c_sampled, y_coords, z_c_sampled)]

            # 添加id------------------------------------------------------------------
            pos = np.vstack((new_point_list,np.array([[b[-2],0,np.inf]])))
            # 处理成字符串
            result = ['{:.3f} {:.3f} {:.3f}'.format(row[0], row[1], row[2]) for row in pos]
            result = [s + ',' for s in result]# 加个逗号

            for point in midpoints[:-1]:
                cv2.circle(bgr_image, tuple(point), radius=point_size, color=color_palette[key], thickness=-1)

            # 绘制线条
            # cv2.polylines(bgr_image, [np.int32([midpoints])], isClosed=False, color=(255, 0, 255), thickness=2)
            # cv2.polylines(bgr_image, [np.int32([upper_contour_points])], isClosed=False, color=(255, 0, 0), thickness=3)
            # cv2.polylines(bgr_image, [np.int32([lower_contour_points])], isClosed=False, color=(0, 255, 0), thickness=3)
            Points.append(result)

    return Points
# 其他的元素
def get_infermations0(box, segpoint, key, value, bgr_image, file, point_cloud, color_palette, ret_datetime, point_size):
    # 传入 ret_datetime 方便定位排查
    points = get_up_down_point(box, segpoint, key, bgr_image, point_cloud, color_palette, point_size, value, ret_datetime)
    if len(points)!=0:# 不写入没有3d点的类别
        write_except_lane(file, value, points)
# 车道线
def gets_lane(box, segpoint, key, value, bgr_image, file, point_cloud):# 耗时

    _2D_3D_point = []  # 某一条线上的某个3d点,有3条线就有3条记录，每条线都要取一个，因为不知道这条线在哪边，用于得到后面的p
    _coordinates = []  # 某一条线上的15个3d点，用于后面从左向右排
    # 取出框和点，一一对应
    for b, j in zip([bb for bb, is_true in zip(box, box[:, -1] == key) if is_true],[tensor for tensor, is_true in zip(segpoint, box[:, -1] == key) if is_true]):
        j=j[~np.any(j < 0, axis=1)]# 过滤掉负值
        new_fit_point, fit_point = my_fit(j, bgr_image)
        # 区域小的情况--拟合的这些点可能挤在一堆 判断距离
        max_distance = np.linalg.norm(np.vstack((new_fit_point[0], new_fit_point[1])).T[0] - np.vstack((new_fit_point[0], new_fit_point[1])).T[-1])
        # 画点
        for index, (x, y) in enumerate(zip(new_fit_point[0], new_fit_point[1])):
            if max_distance > 100:
                cv2.circle(bgr_image, (int(x), int(y)), 5, (0, 0, 255), -1)
            if max_distance < 100 and index == 4 or index == 10:
                cv2.circle(bgr_image, (int(x), int(y)), 5, (0, 0, 255), -1)

        if max_distance > 100:
            crent = []  # 用来装当前目标的三维点
            flag = False
            counter = 2  # 从第3个点开始
            for index, (x, y) in enumerate(zip(new_fit_point[0], new_fit_point[1])):
                text = f"{' '.join(['{:.3f}'.format(i) for i in point_cloud.get_value(int(x), int(y))[1][:3]])},"
                # 这15个点取点云值有的可能是nan，然后在第2个点后才开始保存，一条线保存一个点就行
                # 当这个点是nan时，就寻找下一个，依次类推
                if 'nan' not in text and 'inf' not in text:
                    numbers = [float(num) for num in text.replace(',', '').split()]
                    if not all(num == 0 for num in numbers):# 过滤掉0 0 0 这种
                        if not flag and counter==index:  # 在每条线上取一个点，再取出它的二维点用于后面画图，再取出B_point的点用于后面筛选，i好像没用上
                            _2D_3D_point.append(
                                [numbers, [x, y], fit_point[:, 1:-1].transpose(1, 0)]
                            )
                            cv2.circle(bgr_image, (int(x), int(y)), 5, (255, 0, 0), -1) # 画这个点
                            flag = True
                            counter = None  # 停止后续尝试
                        crent.append(text)
                if counter is not None and index >= counter:
                    counter += 1
            if len(crent) != 0:
                crent.append(int(b[-2]))
                _coordinates.append(crent)  # 遍历一个就加入一个
        else:
            crent = []
            flag1, flag2, flug = False, False, False
            for i in range(0, len(new_fit_point[0]) - 1, 2):  # 依次取出两个点，并循环处理这些点
                x1, y1 = new_fit_point[0][i], new_fit_point[1][i]
                x2, y2 = new_fit_point[0][i + 1], new_fit_point[1][i + 1]
                text1 = f"{' '.join(['{:.3f}'.format(i) for i in point_cloud.get_value(int(x1), int(y1))[1][:3]])},"
                text2 = f"{' '.join(['{:.3f}'.format(i) for i in point_cloud.get_value(int(x2), int(y2))[1][:3]])},"
                # 存点 _2D_3D_point只执行一次
                if 'nan' not in text1 and 'inf' not in text1 and not flag1:
                    numbers = [float(num) for num in text1.replace(',', '').split()]
                    if not all(num == 0 for num in numbers):
                        if not flug:
                            _2D_3D_point.append(
                                [numbers, [x1, y1], fit_point[:, 1:-1].transpose(1, 0)]
                            )
                            flug = True
                        flag1 = True
                        crent.append(text1)

                if 'nan' not in text2 and 'inf' not in text2 and not flag2:
                    numbers = [float(num) for num in text2.replace(',', '').split()]
                    if not all(num == 0 for num in numbers):
                        if not flug:
                            # _2D_3D_point.append(
                            #     [[float(num) for num in text2.replace(',', '').split() if
                            #       num.replace('.', '', 1).lstrip('-').isdigit()],
                            #      [x2, y2], fit_point[:, 1:- 1].transpose(1, 0)])
                            _2D_3D_point.append(
                                [numbers, [x2, y2], fit_point[:, 1:-1].transpose(1, 0)]
                            )
                            flug = True
                        flag2 = True
                        crent.append(text2)
            if len(crent) != 0:
                crent.append(int(b[-2]))
                _coordinates.append(crent)  # 遍历一个就加入一个
        ########## 检测到线，但是从拟合的点中提取不到相应的点云值，就不写？
        #########  比如说两条线，其中一条有点云值，其中一条没有 _2D_3D_point长度为1 _coordinates为2 因为还添加了个crent
    lane = time.time()
    if len(_2D_3D_point) != 0 and len(_coordinates) != 0:
        s = ''
        sorted_line_2D_3D_point = sorted([point for point in _2D_3D_point if point[0]], key=lambda x: x[0][0])  # 从小到大排序 第一个和最后一个是最左边和最右边的点
        # 以最左边线上某个点为准 取出后面几条线z接近的点
        p = [sorted_line_2D_3D_point[0][0]] # 最左边的点，以它为基准，从旁边的线上找出z接近的点[0:len(i[-1])-len(i[-1])//3]
        target_z = sorted_line_2D_3D_point[0][0][2] # p中的z
        # 得到sorted_line_2D_3D_point除最左边外，其他车道的拟合点的点云数据
        pos = [np.array([point_cloud.get_value(int(point[0].item()), int(point[1].item()))[1][:3] for point in i[-1]]) for i in sorted_line_2D_3D_point[1:]]
        for i in pos:
            if i.ndim == 2 and i.shape[1] == 3:
                z_values = i[:, -1]
                closest_point = i[np.argmin(np.abs(z_values - target_z))] # 找到p中距离z最近的车道的点
                p.append(closest_point)
            else:
                print(i)
                print(i.ndim,i.shape)

        file.write(f"{value}s:{len(box[box[:, -1] == key])},")
        s += f"\n{value}:{len(box[box[:, -1] == key])},"
        for i, values in enumerate(p):
            line = ' '.join([f'{value:.3f}' for value in values])
            s += line
            file.write(line)
            if i < len(p) - 1:
                file.write(',')
                s += ','
        file.write('\n')#  Lanes:3,-4.086 -1.289 8.911,-1.398 -1.277 9.052,1.723 -1.306 9.194
        # 3维点从左往右排， 再写入
        sorted_lists = sorted(_coordinates, key=lambda x: float(x[0].split()[0]))
        for index, i in enumerate(sorted_lists):
            # region
            # 拟合------------------------------------------------------------------
            filtered_data = np.array([list(map(float, s.strip(',').split())) for s in i[:-1]])
            x_coords = [point[0] for point in filtered_data]
            y_coords = [point[1] for point in filtered_data]
            z_coords = [point[2] for point in filtered_data]
            coefficients = np.polyfit(z_coords, x_coords, 2)
            polynomial = np.poly1d(coefficients)
            z_c_fit = np.linspace(z_coords[0], z_coords[-1], 100)
            x_c_fit = polynomial(z_c_fit)
            num_points = len(y_coords)  # y_coords 的数量
            indices = np.linspace(0, len(z_c_fit) - 1, num_points, dtype=int)  # 生成等间距的索引

            # 使用生成的索引选取 x 和 z 的值
            x_c_sampled = x_c_fit[indices]
            z_c_sampled = z_c_fit[indices]
            # 使用列表推导式将 x_c_sampled, y_coords, z_c_sampled 组合成一个新的点列表
            new_point_list = [f"{x:.3f} {y:.3f} {z:.3f}," for x, y, z in zip(x_c_sampled, y_coords, z_c_sampled)]
            # endregion
            line = ' '.join(new_point_list)
            formatted_data = line.replace(', ', ',')  # 去掉空格
            formatted_data = formatted_data.rstrip(',')  # 去掉最后一个逗号
            file.write(f"{value.lower()}{index + 1}:{i[-1]},{formatted_data}\n")
            s += f";{formatted_data}"
        # print(f"lane: {(time.time() - lane) * 1000:6.2f}ms")
        return s

    return 0
#显示和写入2d点   测试代码
def gets_lane_p(box, segpoint, key, value, bgr_image, file, point_cloud):# 耗时
    D2=[]
    _2D_3D_point = []  # 某一条线上的某个3d点,有3条线就有3条记录，每条线都要取一个，因为不知道这条线在哪边，用于得到后面的p
    _coordinates = []  # 某一条线上的15个3d点，用于后面从左向右排
    # 取出框和点，一一对应
    for b, j in zip([bb for bb, is_true in zip(box, box[:, -1] == key) if is_true],[tensor for tensor, is_true in zip(segpoint, box[:, -1] == key) if is_true]):
        j=j[~np.any(j < 0, axis=1)]# 过滤掉负值
        new_fit_point, fit_point = my_fit(j, bgr_image)
        # 区域小的情况--拟合的这些点可能挤在一堆 判断距离
        max_distance = np.linalg.norm(np.vstack((new_fit_point[0], new_fit_point[1])).T[0] - np.vstack((new_fit_point[0], new_fit_point[1])).T[-1])
        # 画点
        for index, (x, y) in enumerate(zip(new_fit_point[0], new_fit_point[1])):
            if max_distance > 100:
                cv2.circle(bgr_image, (int(x), int(y)), 5, (0, 0, 255), -1)
            if max_distance < 100 and index == 4 or index == 10:
                cv2.circle(bgr_image, (int(x), int(y)), 5, (0, 0, 255), -1)

        if max_distance > 100:
            crent = []  # 用来装当前目标的三维点
            d2 = []
            flag = False
            counter = 2  # 从第3个点开始
            for index, (x, y) in enumerate(zip(new_fit_point[0], new_fit_point[1])):
                text = f"{' '.join(['{:.3f}'.format(i) for i in point_cloud.get_value(int(x), int(y))[1][:3]])},"
                # 这15个点取点云值有的可能是nan，然后在第2个点后才开始保存，一条线保存一个点就行
                # 当这个点是nan时，就寻找下一个，依次类推
                if 'nan' not in text and 'inf' not in text:
                    numbers = [float(num) for num in text.replace(',', '').split()]
                    if not all(num == 0 for num in numbers):# 过滤掉0 0 0 这种
                        if not flag and counter==index:  # 在每条线上取一个点，再取出它的二维点用于后面画图，再取出B_point的点用于后面筛选，i好像没用上
                            _2D_3D_point.append(
                                [numbers, [x, y], fit_point[:, 1:-1].transpose(1, 0)]
                            )
                            cv2.circle(bgr_image, (int(x), int(y)), 5, (255, 0, 0), -1) # 画这个点
                            flag = True
                            counter = None  # 停止后续尝试
                        crent.append(text)
                        d2.append('{:.0f} {:.0f}'.format(x, y))
                if counter is not None and index >= counter:
                    counter += 1
            if len(crent) != 0:
                crent.append(int(b[-2]))
                _coordinates.append(crent)  # 遍历一个就加入一个
                D2.append(d2)
        else:
            crent = []
            d2 = []
            flag1, flag2, flug = False, False, False
            for i in range(0, len(new_fit_point[0]) - 1, 2):  # 依次取出两个点，并循环处理这些点
                x1, y1 = new_fit_point[0][i], new_fit_point[1][i]
                x2, y2 = new_fit_point[0][i + 1], new_fit_point[1][i + 1]
                text1 = f"{' '.join(['{:.3f}'.format(i) for i in point_cloud.get_value(int(x1), int(y1))[1][:3]])},"
                text2 = f"{' '.join(['{:.3f}'.format(i) for i in point_cloud.get_value(int(x2), int(y2))[1][:3]])},"
                # 存点 _2D_3D_point只执行一次
                if 'nan' not in text1 and 'inf' not in text1 and not flag1:
                    numbers = [float(num) for num in text1.replace(',', '').split()]
                    if not all(num == 0 for num in numbers):
                        if not flug:
                            _2D_3D_point.append(
                                [numbers, [x1, y1], fit_point[:, 1:-1].transpose(1, 0)]
                            )
                            flug = True
                        flag1 = True
                        crent.append(text1)
                        d2.append('{:.0f} {:.0f}'.format(x1, y1))
                if 'nan' not in text2 and 'inf' not in text2 and not flag2:
                    numbers = [float(num) for num in text2.replace(',', '').split()]
                    if not all(num == 0 for num in numbers):
                        if not flug:
                            # _2D_3D_point.append(
                            #     [[float(num) for num in text2.replace(',', '').split() if
                            #       num.replace('.', '', 1).lstrip('-').isdigit()],
                            #      [x2, y2], fit_point[:, 1:- 1].transpose(1, 0)])
                            _2D_3D_point.append(
                                [numbers, [x2, y2], fit_point[:, 1:-1].transpose(1, 0)]
                            )
                            flug = True
                        flag2 = True
                        crent.append(text2)
                        d2.append('{:.0f} {:.0f}'.format(x2, y2))
            if len(crent) != 0:
                crent.append(int(b[-2]))
                _coordinates.append(crent)  # 遍历一个就加入一个
                D2.append(d2)
        ########## 检测到线，但是从拟合的点中提取不到相应的点云值，就不写？
        #########  比如说两条线，其中一条有点云值，其中一条没有 _2D_3D_point长度为1 _coordinates为2 因为还添加了个crent
    lane = time.time()
    if len(_2D_3D_point) != 0 and len(_coordinates) != 0:
        s = ''
        sorted_line_2D_3D_point = sorted([point for point in _2D_3D_point if point[0]], key=lambda x: x[0][0])  # 从小到大排序 第一个和最后一个是最左边和最右边的点
        # 以最左边线上某个点为准 取出后面几条线z接近的点
        p = [sorted_line_2D_3D_point[0][0]] # 最左边的点，以它为基准，从旁边的线上找出z接近的点[0:len(i[-1])-len(i[-1])//3]
        target_z = sorted_line_2D_3D_point[0][0][2] # p中的z
        # 得到sorted_line_2D_3D_point除最左边外，其他车道的拟合点的点云数据
        pos = [np.array([point_cloud.get_value(int(point[0].item()), int(point[1].item()))[1][:3] for point in i[-1]]) for i in sorted_line_2D_3D_point[1:]]
        for i in pos:
            z_values = i[:, 2]
            closest_point = i[np.argmin(np.abs(z_values - target_z))] # 找到p中距离z最近的车道的点
            p.append(closest_point)

        file.write(f"{value}s:{len(box[box[:, -1] == key])},")
        s += f"\n{value}:{len(box[box[:, -1] == key])},"
        for i, values in enumerate(p):
            line = ' '.join([f'{value:.3f}' for value in values])
            s += line
            file.write(line)
            if i < len(p) - 1:
                file.write(',')
                s += ','
        file.write('\n')#  Lanes:3,-4.086 -1.289 8.911,-1.398 -1.277 9.052,1.723 -1.306 9.194
        # 3维点从左往右排， 再写入
        sorted_lists = sorted(_coordinates, key=lambda x: float(x[0].split()[0]))
        # for index, i in enumerate(_coordinates):
        #     line = ' '.join(i[:-1])
        #     formatted_data = line.replace(', ', ',')  # 去掉空格
        #     # data_groups = formatted_data.split(',')
        #     # # 保留非全0的组
        #     # filtered_groups = []
        #     # for group in data_groups:
        #     #     if group.strip():  # 确保组不为空
        #     #         numbers = list(map(float, group.split()))
        #     #         if not all(num == 0 for num in numbers):
        #     #             filtered_groups.append(group)
        #     # filtered_data = ','.join(filtered_groups)
        #     formatted_data = formatted_data.rstrip(',')  # 去掉最后一个逗号
        #     file.write(f"{value.lower()}{index + 1}:{i[-1]},{formatted_data}\n")
        #     s += f";{formatted_data}"
        # print(f"lane: {(time.time() - lane) * 1000:6.2f}ms")

        for index, (i, j) in enumerate(zip(_coordinates, D2)):
            line = ' '.join(i[:-1])
            formatted_data = line.replace(', ', ',')  # 去掉空格
            formatted_data = formatted_data.rstrip(',')  # 去掉最后一个逗号
            file.write(f"{value.lower()}{index + 1}:{i[-1]},{formatted_data}\n")
            file.write(f"{value.lower()}{index + 1}_twoD:")
            for t in j:
                file.write(t+',')
                xx, yy = map(int, t.split())
                cv2.putText(bgr_image,t,(xx,yy),cv2.FONT_HERSHEY_SIMPLEX, 0.55,color=(255,0,255),thickness=1)
            file.write('\n')
        # cv2.putText(im, f'{self.classes[cls_]}:{conf:.3f} ID: {int(track_id)}',
        #             ([int(num) for num in box][0], [int(num) for num in box][1] - 2),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        #             self.color_palette[int(cls_) + len(self.label_dict)], thickness=2)
        return s

    return 0
