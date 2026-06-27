#!/bin/bash

echo -e "\t\tYOLOv5"
python predict.py --model yolo --source "$1"

echo -e "\t\tEfficientDet"
python predict.py --model edet --source "$1"

echo -e "\t\tSSD"
python predict.py --model ssd --source "$1"

echo -e "\t\tFaster R-CNN"
python predict.py --model rcnn --source "$1"