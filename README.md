# Macau Automatic Number Plate Recognition (Macau ANPR)

[中文说明](README.zh-CN.md) | [English version](README.md)

An end-to-end Automatic Number Plate Recognition (ANPR) system tailored for Macau vehicle license plates. It performs vehicle detection, license plate localization, character segmentation, and optical character recognition (OCR).

---

## 🌟 Key Features

1. **Pluggable Object Detectors**: 
   Supports 4 mainstream deep learning frameworks to detect vehicles and localize license plates:
   - **YOLOv5** (Highly recommended, optimized for speed and accuracy)
   - **SSD** (MobileNetV2 SSD-Lite, extremely lightweight)
   - **EfficientDet** (EfficientDet-D1 backbone)
   - **Faster R-CNN** (Detectron2 backbone)
2. **Robust Coordinate Projection**:
   Automatically calculates cropped bounding boxes and re-projects license plate locations back to the original coordinate space.
3. **Traditional CV Character Segmentation**:
   Uses advanced OpenCV image processing (adaptive thresholding, Gaussian blur, erosion, and connected components) to isolate individual plate characters under varying lighting conditions.
4. **ResNet OCR Classification**:
   Uses a PyTorch-based ResNet neural network to classify segmented characters (A-Z, 0-9).
5. **Real-time Alerting & Logging**:
   - Records all recognized vehicles with timestamps, vehicle classes, plate numbers, and colors in `files/veh.csv`.
   - Cross-references plates against a suspected blacklist database (`files/suspected.csv`) and prints instant warnings.

---

## 🛠️ Installation & Setup

Ensure you have Python 3.10+ installed. Follow these steps to set up the project on your local machine (macOS/Linux/Windows):

### 1. Set Up Virtual Environment
Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Pull Third-Party Code Repositories
The project uses specific open-source implementations for the detection backends. Run the provided script to clone them into the `experiments/` directory:
```bash
chmod +x setup_dependencies.sh
./setup_dependencies.sh
```

### 3. Install Package Dependencies
Install all package requirements for the pipeline and sub-modules:
```bash
pip install -r requirements.txt
pip install -r experiments/yolov5/requirements.txt
```
*(Note: If you want to use the Faster R-CNN backend, you need to install Detectron2 manually via `pip install 'git+https://github.com/facebookresearch/detectron2.git'`)*

---

## 💾 Model Weights Setup

Before running the inference, download the weights files and place them in the correct directories:

1. **OCR Character recognition weights**:
   - Place your custom-trained ResNet OCR model `ocr.pth` in:
     `models/ocr.pth`
2. **YOLOv5 weights**:
   - Vehicle detector: `models/yolo/yolov5s.pt`
   - Plate detector: `models/yolo/yolov5np_samhui.pt`
3. **SSD weights**:
   - Vehicle detector: `models/ssd/mb2-ssd-lite-mp-0_686.pth`
   - Plate detector: `models/ssd/mb2-ssd-lite-np-epoch401-loss0.906.pth`
4. **EfficientDet weights**:
   - Vehicle detector: `models/efficientdet/efficientdet-d1.pth`
   - Plate detector: `models/efficientdet/efficientdet-d1_296_7699.pth`

---

## 🚀 How to Run

The refactored pipeline is controlled through a single CLI tool: `predict.py`.

### 1. Run Detection & OCR
Specify the backend model using `--model` and the input file using `--source` (supports images, folders of images, video files, or a camera index):
```bash
# Detect vehicles and recognize plates using YOLOv5 on a sample image
python predict.py --model yolo --source images/021.jpg

# Detect using SSD on a video file
python predict.py --model ssd --source path/to/video.mp4

# Custom OCR model path or custom suspected blacklist CSV
python predict.py --model yolo --source images/ --ocr-model path/to/ocr.pth --suspected path/to/suspected.csv
```

### 2. Quick Run Compare
You can also run all four detection models sequentially on a single source file to compare performance:
```bash
./quickrun.sh images/021.jpg
```
All annotated images/videos will be saved in the `output/` directory.

---

## 📂 Project Structure

```text
├── character.py          # Traditional CV character segmentation & ResNet OCR
├── predict.py            # Unified ANPR inference CLI
├── setup_dependencies.sh # Helper script to clone sub-repositories
├── quickrun.sh           # Script to run all 4 models sequentially
├── requirements.txt      # Global Python dependencies
├── experiments/
│   ├── adapters.py       # Pluggable model wrappers (YOLO, SSD, Edet, RCNN)
│   ├── yolov5/           # YOLOv5 submodule directory
│   ├── pytorch-ssd/      # SSD submodule directory
│   └── ...
├── files/
│   ├── suspected.csv     # Target suspected vehicle blacklist database
│   └── veh.csv           # Output traffic recognition log database
├── models/               # Model weights directory (yolo/, ssd/, efficientdet/, rcnn/)
└── images/               # Sample test images
```
