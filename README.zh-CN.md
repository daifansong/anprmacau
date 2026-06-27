# 澳门地区车牌自动识别系统（Macau ANPR）

[English version](README.md) | [中文说明](README.zh-CN.md)

这是一个专为澳门地区车牌定制的端到端车牌自动识别（ANPR）系统。系统包含车辆检测、车牌定位、字符分割和光学字符识别（OCR）等核心模块。

---

## 🌟 核心特性

1. **可插拔的目标检测器**：
   系统支持 4 种主流深度学习检测器来识别车辆并定位车牌：
   - **YOLOv5**（强烈推荐，速度与精度的最佳平衡点）
   - **SSD**（轻量级 MobileNetV2 SSD-Lite，适用于低算力设备）
   - **EfficientDet**（基于 EfficientDet-D1 骨干网络）
   - **Faster R-CNN**（基于 Detectron2 框架）
2. **鲁棒的坐标重投影**：
   自动提取车辆裁剪图中的车牌相对坐标，并精准投影还原到原始大图的坐标系中。
   传统二值化与形态学算法。
3. **传统 CV 字符分割**：
   利用先进的 OpenCV 图像处理技术（自适应阈值、高斯滤波、腐蚀、连通域分析）对车牌图像进行单字符切分，能较好应对复杂光照。
4. **ResNet OCR 识别**：
   利用 PyTorch 搭建的 ResNet 神经网络分类器，对切分出的单个字符进行高精度预测（支持 A-Z, 0-9 字符集）。
5. **实时黑名单预警与通行记录**：
   - 将识别到的通行车辆信息（日期、时间、车型、车牌号、颜色）追记到 `files/veh.csv` 日志中。
   - 实时将识别车牌与嫌疑黑名单库 `files/suspected.csv` 进行比对，若匹配则在控制台打印高警警告。

---

## 🛠️ 安装与环境配置

请确保已安装 Python 3.10+。请按照以下步骤配置您的本地运行环境（支持 macOS/Linux/Windows）：

### 1. 创建并激活虚拟环境
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# Windows 用户请执行: venv\Scripts\activate
```

### 2. 拉取第三方算法仓库
本项目依靠特定的检测算法实现。运行我们为您准备的克隆脚本，将所需的算法源码拉取至 `experiments/` 目录：
```bash
chmod +x setup_dependencies.sh
./setup_dependencies.sh
```

### 3. 安装 Python 依赖包
使用 pip 在当前虚拟环境中安装依赖包：
```bash
pip install -r requirements.txt
pip install -r experiments/yolov5/requirements.txt
```
*(注：如果您打算使用 Faster R-CNN 后端，需要手动安装 Detectron2：`pip install 'git+https://github.com/facebookresearch/detectron2.git'`)*

---

## 💾 权重文件配置

在运行推理前，请下载对应的模型权重文件并放置于指定目录下：

1. **OCR 字符识别模型**：
   - 将您的 ResNet OCR 分类器模型命名为 `ocr.pth` 并放置在：
     `models/ocr.pth`
2. **YOLOv5 模型权重**：
   - 车辆检测器：`models/yolo/yolov5s.pt`
   - 车牌检测器：`models/yolo/yolov5np_samhui.pt`
3. **SSD 模型权重**：
   - 车辆检测器：`models/ssd/mb2-ssd-lite-mp-0_686.pth`
   - 车牌检测器：`models/ssd/mb2-ssd-lite-np-epoch401-loss0.906.pth`
4. **EfficientDet 模型权重**：
   - 车辆检测器：`models/efficientdet/efficientdet-d1.pth`
   - 车牌检测器：`models/efficientdet/efficientdet-d1_296_7699.pth`

---

## 🚀 运行方法

重构后的系统由统一的命令行入口 `predict.py` 控制。

### 1. 执行单模型识别
使用 `--model` 指定所选模型，`--source` 指定输入源（支持单张图片、图片目录、视频文件或摄像头索引）：
```bash
# 使用 YOLOv5 识别单张示例图片
python predict.py --model yolo --source images/021.jpg

# 使用 SSD 识别视频文件
python predict.py --model ssd --source path/to/video.mp4

# 指定自定义 OCR 权重或嫌疑人名单路径
python predict.py --model yolo --source images/ --ocr-model path/to/ocr.pth --suspected path/to/suspected.csv
```

### 2. 一键跑完所有模型（对比效果）
您可以使用 `quickrun.sh` 脚本依次运行所有 4 种检测模型，对比定位效果：
```bash
./quickrun.sh images/021.jpg
```
所有绘制了检测框和车牌标签的图像/视频都将保存在 `output/` 文件夹下。

---

## 📂 项目结构

```text
├── character.py          # 传统图像处理字符分割与 ResNet OCR 识别
├── predict.py            # 统一的车牌识别命令行入口
├── setup_dependencies.sh # 一键拉取外部子仓库的代码脚本
├── quickrun.sh           # 依次用 4 种模型运行推理的对比脚本
├── requirements.txt      # 全局 Python 第三方库依赖列表
├── experiments/
│   ├── adapters.py       # 模型接口适配层 (YOLO, SSD, Edet, RCNN)
│   ├── yolov5/           # YOLOv5 子仓库目录
│   ├── pytorch-ssd/      # SSD 子仓库目录
│   └── ...
├── files/
│   ├── suspected.csv     # 嫌疑车牌黑名单数据库
│   └── veh.csv           # 通行车辆识别记录日志表
├── models/               # 各模型权重存放目录 (yolo/, ssd/, efficientdet/, rcnn/)
└── images/               # 示例测试图片
```
