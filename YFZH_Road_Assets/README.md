<a align="left">
  <a href="https://github.com/XianYang2547">
    <img src="https://img.shields.io/badge/Author-@XianYang-000000.svg?logo=GitHub" alt="GitHub">
  </a>
</a>

## 说明...

## 目录结构
```
.
├── Infer_Py                | python推理
    │── xy                    | 工具包
    │     ├── ...py             
    │     └── tracker           | 跟踪工具
    ├── model_files             | 模型文件
    │ └── linuxtrt              | engine文件夹
    ├── output_folder           | 输出文件夹
    ├── RTSP_file               | 推流服务器
    └── test_data               | 测试数据
```
## 快速开始
### 1. 环境安装
```bash
pip install -r requirements.txt
```
### 2. 模型转换
```bash
python ./model_files/get_trt.py --onnx /path/to/model.onnx --engine /path/to/model.plan --use_fp16_mode True
```

### 3. 运行示例
传入参数见make_parser()

3.1 normal
 - for image 
```bash
python My_infer.py --path ./test_data/test.jpg
```
 - for mp4/avi
```bash
python My_infer.py --path ./test_data/test.avi
```
 - for svo
```bash
python My_infer.py --path ./test_data/test.svo
```
3.2 rtsp socker

for camera (only support rtsp、socker)
```bash
python My_infer.py --path camera
```

## 🎖 贡献者
<a href="https://github.com/XianYang2547/Internship-Item/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=XianYang2547/Internship-Item" />
</a>

