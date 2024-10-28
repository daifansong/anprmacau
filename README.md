# Automatic Number Plate Recognition for Macau

Final Year (naive) Project in MUST, April 2023.

Applied YOLOv5, Single Shot Detection, Faster R-CNN, EfficientDet for vehicle and number plate detection, self-trained Resnet for optical character recognition.

Train datasets are self-photoed images in Macau, and partical char74k with Gil Sans fonts.

Quick Start, provided test images in directory `images`.
```bash
./quickrun.sh /path/to/your/image.jpg
