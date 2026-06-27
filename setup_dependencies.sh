#!/bin/bash

# Setup dependencies for Macau ANPR project

echo "========================================="
# Check and clone YOLOv5
if [ ! -d "experiments/yolov5" ]; then
    echo "Cloning YOLOv5 repository..."
    git clone https://github.com/ultralytics/yolov5.git experiments/yolov5
else
    echo "YOLOv5 repository already exists."
fi

# Check and clone EfficientDet
if [ ! -d "experiments/Yet-Another-EfficientDet-Pytorch" ]; then
    echo "Cloning Yet-Another-EfficientDet-Pytorch repository..."
    git clone https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git experiments/Yet-Another-EfficientDet-Pytorch
else
    echo "Yet-Another-EfficientDet-Pytorch repository already exists."
fi

# Check and clone SSD
if [ ! -d "experiments/pytorch-ssd" ]; then
    echo "Cloning pytorch-ssd repository..."
    git clone https://github.com/dusty-nv/pytorch-ssd.git experiments/pytorch-ssd
else
    echo "pytorch-ssd repository already exists."
fi

echo "========================================="
echo "Dependencies setup completed!"
echo "To install the python packages, run:"
echo "pip install -r requirements.txt"
echo ""
echo "If you need to run Faster R-CNN, please install detectron2 manually:"
echo "pip install 'git+https://github.com/facebookresearch/detectron2.git'"
echo "========================================="
