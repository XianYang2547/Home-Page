# -*- coding: utf-8 -*-
# @Time    : 2024/7/24 下午4:55
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : __init__.py
# ------❤❤❤------ #


from .models import My_dection, Build_TRT_model, Build_Ort_model
from .RealTime_funcs import detect_svo_track_socket_rstp, get_ip_addresses
from .Local_funcs import detect_image_or_imgdir, detect_image_or_imgdir_json, detect_video, detect_svo_track

__all__ = (
    "My_dection",
    "Build_TRT_model",
    "Build_Ort_model",
    "detect_image_or_imgdir",
    "detect_image_or_imgdir_json",
    "detect_video",
    "detect_svo_track",
    "detect_svo_track_socket_rstp",
    "get_ip_addresses"
)
