<a align="left">
  <a href="https://github.com/XianYang2547">
    <img src="https://img.shields.io/badge/Author-@XianYang-000000.svg?logo=GitHub" alt="GitHub">
  </a>
</a>

## 目录结构
```
.
├── assets             # 测试数据
│ ├── rtsp             # 传送门下载的rtsp file
│ ├── test.jpg
│ ├── test.avi
│ └── test.svo
├── C_code          # c++ TODO
│ ├── bytetrack
│ ├── CMakeLists.txt
│ ├── main.cpp
│ ├── segment
│ └── xytools
├── Infer_Python       # Py推理
│ ├── demo.py          # 示例脚本
│ ├── My_infer.py      # ...
│ └── xy               # 包
├── models             # 模型文件
│ ├── get_trt.py       # 转换脚本1
│ ├── get_trt.cpp      # 转换脚本2
│ └── get_engine       # cpp--->二进制文件
├── output             # 输出文件夹
│ └── detect_image
├── README.md
└── requirements.txt
```
## 快速开始 on ubuntu22.04
### 0. Clone it
```bash
git clone http://172.16.10.64/xianyang/Road_Assets.git
cd Road_Assets
```

### 1. 环境安装
```bash
pip install -r requirements.txt
cat requirements.txt
```
除demo.py外，其余需要ros2环境
### 2. 模型转换(if you run it with onnx model, skip this)
- 2.1 use py script
```bash
python ./model_files/get_trt.py --onnx model.onnx --engine model.plan --fp16 True
```
- 2.2 use c++ script
```bash
g++ -o get_engine get_trt.cpp -I/home/xianyang/Documents/TensorRT-8.6.1.6/include -L/home/xianyang/Documents/TensorRT-8.6.1.6/lib -lnvinfer -I/usr/local/cuda/include -L/usr/local/cuda/lib64  -lcudart -lnvonnxparser
./get_engine your_onnx_path your_engine_path 
```

### 3. 运行示例
all arg in demo.py ---> make_parser()
 - for image/avi/svo/directory
```bash
cd Infer_Python
python demo.py --path ../assets/test.jpg
python demo.py --path ../assets/test.avi
python demo.py --path ../assets/test.svo
python demo.py --path [your directory]
```
some arg as follows:
```bash
python demo.py --model [your model] --path [your file/dir path] --iou_threshold [your iou num] --conf_threshold [your conf num] --base_directory [your custom save dir] --rtsp [use rtsp?] --url [your rtsp ip] ... 
```

 - for camera ---> before this, you must run camera instance
```bash
git clone http://172.16.10.64/xianyang/camera_gnss.git
cd camera_gnss
cat README.md
```
then
```bash
cd Infer_Python
python My_infer.py
```
#### 3.1 rtsp file
****
[传送门](https://github.com/bluenviron/mediamtx/releases/tag/v1.9.3)
****

## 🎖 贡献者
<a href="https://github.com/XianYang2547/Internship-Item/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=XianYang2547/Internship-Item" />
</a>
