# Python 车辆识别系统（YOLOv8 + OpenCV）

这是一个可直接运行的车辆识别示例项目，使用 `Ultralytics YOLOv8` 模型识别以下常见车辆类别：

- car
- bus
- truck
- motorcycle
- bicycle

## 1. 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 运行方式

### 图片识别

```bash
python vehicle_recognition.py --source path/to/image.jpg --show --save
```

### 视频识别

```bash
python vehicle_recognition.py --source path/to/video.mp4 --show --save
```

### 摄像头实时识别

```bash
python vehicle_recognition.py --source 0 --show
```

## 3. 常用参数

- `--model`：模型路径（默认 `yolov8n.pt`）
- `--conf`：置信度阈值（默认 `0.35`）
- `--iou`：NMS IoU 阈值（默认 `0.45`）
- `--save`：保存识别结果到 `runs/vehicle_recognition/`
- `--show`：可视化窗口显示

## 4. 输出说明

程序会输出：
- 每帧车辆检测框
- 类别与置信度
- 每帧车辆数量统计（按类别）

> 首次运行会自动下载 YOLOv8 预训练权重（需要网络）。
