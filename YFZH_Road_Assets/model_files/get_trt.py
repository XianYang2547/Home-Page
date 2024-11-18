# -*- coding: utf-8 -*-
# @Time    : 2024/5/20 5:20
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : get_trt.py
# ------❤❤❤------ #

import os
import argparse
import tensorrt as trt
from cuda import cudart


def convert_onnx_to_engine(onnx_path, engine_path,fp16=False):
    # 获取可用和总显存
    _, free_mem, total_mem = cudart.cudaMemGetInfo()
    # 创建 logger：日志记录器
    logger = trt.Logger(trt.Logger.WARNING)
    # 创建构建器 builder
    builder = trt.Builder(logger)
    # 预创建网络
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # 加载 onnx 解析器
    parser = trt.OnnxParser(network, logger)
    # 解析 ONNX 文件
    success = parser.parse_from_file(onnx_path)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    if not success:
        print("Failed to parse ONNX file.")
        return

    # builder 配置
    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    # 分配显存作为工作区间
    config.max_workspace_size = total_mem // 2

    # 构建序列化的网络
    serialized_engine = builder.build_serialized_network(network, config)

    # 检查引擎文件是否已存在
    if os.path.exists(engine_path):
        overwrite = input("The engine file already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print("Operation aborted.")
            return

    # 保存引擎文件
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
        print("Engine file saved successfully!")


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX model to TensorRT engine')
    parser.add_argument('--onnx', type=str, required=False, default=None, help='Path to the ONNX model file')
    parser.add_argument('--engine', type=str, required=False, default=None,
                        help='Path to save the TensorRT engine file')
    parser.add_argument('--fp16', type=str, default=False)

    args = parser.parse_args()

    convert_onnx_to_engine(args.onnx, args.engine)


if __name__ == '__main__':
    # onnx文件路径设置
    onnx = r"/home/xianyang/Desktop/Project/Infer_Py/model_files/v11/onnx/gaosu/n/gaosudet.onnx"
    # 引擎文件保存路径设置
    engine = r"/home/xianyang/Desktop/Project/Infer_Py/model_files/v11/linuxtrt/gaosu/n/gaosudet.plan"
    main()


