<a align="left">
  <a href="https://github.com/XianYang2547">
    <img src="https://img.shields.io/badge/Author-@XianYang-000000.svg?logo=GitHub" alt="GitHub">
  </a>
</a>

## 目录结构
```
.
├── assets             # 测试数据
│ ├── test1.jpg
│ └── test2.jpg
├── C_code          # c++ TODO
│ ├── bytetrack
│ ├── CMakeLists.txt
│ ├── main.cpp
│ ├── segment
│ └── xytools
├── Infer_Python       # Py推理
│ ├── demo.py
│ ├── My_infer.py
│ └── xy               # 包
├── models             # 模型文件
│ ├── get_trt.py
│ ├── get_trt.cpp
│ └── get_engine    
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
```bash
python ./model_files/get_trt.py --onnx model.onnx --engine model.plan --fp16 True
```
### 3. 运行示例
传入参数见make_parser()
 - for image or mp4/avi  ---> support directory
```bash
cd Infer_Python
python demo.py --path ../assets
```
 - for camera ---> before this, you must run camera instance
```bash
cd Infer_Python
python My_infer.py
```



## 🎖 贡献者
<a href="https://github.com/XianYang2547/Internship-Item/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=XianYang2547/Internship-Item" />
</a>
