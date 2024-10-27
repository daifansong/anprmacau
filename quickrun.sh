#!/bin/bash

echo -e "\t\tYOLOv5"
python experiments/yolo_entire.py "$1"

echo -e "\t\tEfficientDet"
python experiments/edet_entire.py "$1"

echo -e "\t\tSSD"
python experiments/ssd_entire.py "$1"

echo -e "\t\tFaster R-CNN"
python experiments/rcnn_entire.py "$1"