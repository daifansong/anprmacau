import sys
import torch
import numpy as np
import cv2
from pathlib import Path

class BaseDetectorAdapter:
    def __init__(self, device):
        self.device = device

    def detect_vehicles(self, image):
        """
        Runs vehicle detection on the input image.
        Returns a list of tuples: (box, confidence, class_name)
        where box is [x1, y1, x2, y2] (coordinates in original image).
        """
        raise NotImplementedError

    def detect_plates(self, vehicle_image):
        """
        Runs license plate detection on the cropped vehicle image.
        Returns a list of tuples: (box, confidence)
        where box is [x1, y1, x2, y2] (coordinates relative to vehicle_image).
        """
        raise NotImplementedError

    def recalculate_plate_coords(self, plate_box, vehicle_box):
        """
        Recalculates local plate box coordinates back to the original image coordinate space.
        """
        vx1, vy1, vx2, vy2 = vehicle_box
        px1, py1, px2, py2 = plate_box
        bias = 2
        return [
            int(vx1 + px1 - bias),
            int(vy1 + py1 - bias),
            int(vx1 + px2 + bias),
            int(vy1 + py2 + bias)
        ]


class YOLOv5Adapter(BaseDetectorAdapter):
    def __init__(self, device, weights_vehicle='models/yolo/yolov5s.pt', weights_np='models/yolo/yolov5np_samhui.pt'):
        super().__init__(device)
        self.root = Path(__file__).resolve().parent
        yolo_root = self.root / 'yolov5'
        if str(yolo_root) not in sys.path:
            sys.path.insert(0, str(yolo_root))
        
        if not (yolo_root / 'models' / 'common.py').exists():
            raise FileNotFoundError(
                "YOLOv5 repository not found under 'experiments/yolov5'. "
                "Please run: git clone https://github.com/ultralytics/yolov5.git experiments/yolov5"
            )
        from models.common import DetectMultiBackend

        self.model_veh = DetectMultiBackend(weights_vehicle, device=self.device)
        self.model_np = DetectMultiBackend(weights_np, device=self.device)
        self.classes_vehicle = [2, 3, 5, 6, 7]  # car, motorcycle, bus, truck, etc.
        self.conf_thres = 0.25
        self.iou_thres = 0.45

    def _resize_img(self, im, model):
        from utils.augmentations import letterbox
        tmp_img = np.array(im)
        im_padded = letterbox(tmp_img, 640, stride=model.stride, auto=True)[0]
        im_transposed = im_padded.transpose((2, 0, 1))
        im_contiguous = np.ascontiguousarray(im_transposed)
        im_tensor = torch.from_numpy(im_contiguous).to(model.device)
        im_tensor = im_tensor.half() if model.fp16 else im_tensor.float()
        im_tensor /= 255.0
        if len(im_tensor.shape) == 3:
            im_tensor = im_tensor[None]
        return im_tensor

    def detect_vehicles(self, image):
        from utils.general import non_max_suppression, scale_boxes
        im = self._resize_img(image, self.model_veh)
        pred = self.model_veh(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes_vehicle, agnostic=False, max_det=1000)
        
        vehicles = []
        det = pred[0]
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], image.shape).round()
            for *xyxy, conf, cls in reversed(det):
                box = [int(x.item()) for x in xyxy]
                vehicles.append((box, conf.item(), self.model_veh.names[int(cls)]))
        return vehicles

    def detect_plates(self, vehicle_image):
        from utils.general import non_max_suppression, scale_boxes
        im = self._resize_img(vehicle_image, self.model_np)
        pred = self.model_np(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, agnostic=False, max_det=1000)
        
        plates = []
        det = pred[0]
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], vehicle_image.shape).round()
            for *xyxy, conf, cls in reversed(det):
                box = [int(x.item()) for x in xyxy]
                plates.append((box, conf.item()))
        return plates

    def recalculate_plate_coords(self, plate_box, vehicle_box):
        vx1, vy1, vx2, vy2 = vehicle_box
        px1, py1, px2, py2 = plate_box
        width = px2 - px1
        height = py2 - py1
        bias = 3
        ax1 = px1 + vx1 - 12 - bias
        ay1 = py1 + vy1 - 12 - bias
        ax2 = ax1 + width + bias
        ay2 = ay1 + height + bias
        return [int(ax1), int(ay1), int(ax2), int(ay2)]


class SSDAdapter(BaseDetectorAdapter):
    def __init__(self, device, weights_vehicle='models/ssd/mb2-ssd-lite-mp-0_686.pth', weights_np='models/ssd/mb2-ssd-lite-np-epoch401-loss0.906.pth'):
        super().__init__(device)
        self.root = Path(__file__).resolve().parent
        ssd_root = self.root / 'pytorch-ssd'
        if str(ssd_root) not in sys.path:
            sys.path.insert(0, str(ssd_root))
        
        if not (ssd_root / 'vision' / 'ssd' / 'mobilenet_v2_ssd_lite.py').exists():
            raise FileNotFoundError(
                "SSD repository not found under 'experiments/pytorch-ssd'. "
                "Please run: git clone https://github.com/dusty-nv/pytorch-ssd.git experiments/pytorch-ssd"
            )
        from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

        self.default_classes = ['BACKGROUND', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                                'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.veh_id = [6, 7, 14]  # bus, car, motorbike
        
        self.net = create_mobilenetv2_ssd_lite(len(self.default_classes), is_test=True)
        self.net.load(weights_vehicle)
        self.predictor = create_mobilenetv2_ssd_lite_predictor(self.net, candidate_size=200, device=self.device)
        
        self.det_class = ['BACKGROUND', 'Licence-Plate']
        self.np_model = create_mobilenetv2_ssd_lite(len(self.det_class), is_test=True)
        self.np_model.load(weights_np)
        self.np_detector = create_mobilenetv2_ssd_lite_predictor(self.np_model, candidate_size=200, device=self.device)

    def detect_vehicles(self, image):
        RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = self.predictor.predict(RGB_image, 10, 0.2)
        vehicles = []
        for i in range(boxes.size(0)):
            lbl = labels[i].item()
            if lbl in self.veh_id:
                box = [int(x) for x in boxes[i, :].numpy().tolist()]
                conf = probs[i].item()
                vehicles.append((box, conf, self.default_classes[lbl]))
        return vehicles

    def detect_plates(self, vehicle_image):
        RGB_img = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = self.np_detector.predict(RGB_img, 10, 0.4)
        plates = []
        for i in range(boxes.size(0)):
            box = [int(x) for x in boxes[i, :].numpy().tolist()]
            conf = probs[i].item()
            plates.append((box, conf))
        return plates

    def recalculate_plate_coords(self, plate_box, vehicle_box):
        vx1, vy1, vx2, vy2 = vehicle_box
        px1, py1, px2, py2 = plate_box
        bias = 2
        nx1 = int(px1 + vx1 - bias)
        ny1 = int(py1 + vy1 - bias)
        nx2 = int(px2 + vx1 + bias)
        ny2 = int(py2 + vy1 + bias)
        return [nx1, ny1, nx2, ny2]


class EfficientDetAdapter(BaseDetectorAdapter):
    def __init__(self, device, weights_vehicle='models/efficientdet/efficientdet-d1.pth', weights_np='models/efficientdet/efficientdet-d1_296_7699.pth'):
        super().__init__(device)
        self.root = Path(__file__).resolve().parent
        edet_root = self.root / 'Yet-Another-EfficientDet-Pytorch'
        if str(edet_root) not in sys.path:
            sys.path.insert(0, str(edet_root))
            
        if not (edet_root / 'backbone.py').exists():
            raise FileNotFoundError(
                "EfficientDet repository not found under 'experiments/Yet-Another-EfficientDet-Pytorch'. "
                "Please run: git clone https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git experiments/Yet-Another-EfficientDet-Pytorch"
            )
        from backbone import EfficientDetBackbone

        self.compound_coef = 1
        self.anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        self.anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.input_size = self.input_sizes[self.compound_coef]

        self.obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.veh_id = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.det_list = ['Licence-Plate']

        self.model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=len(self.obj_list), ratios=self.anchor_ratios, scales=self.anchor_scales)
        self.model.load_state_dict(torch.load(weights_vehicle, map_location=str(self.device)), strict=False)
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(self.device)

        self.np_model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=len(self.det_list), ratios=self.anchor_ratios, scales=self.anchor_scales)
        self.np_model.load_state_dict(torch.load(weights_np, map_location=str(self.device)), strict=False)
        self.np_model.requires_grad_(False)
        self.np_model.eval()
        self.np_model.to(self.device)

        self.threshold = 0.25
        self.iou_threshold = 0.25

    def _preprocess(self, image):
        from utils.utils import aspectaware_resize_padding
        h, w = image.shape[:2]
        framed_img, new_w, new_h, pad_h, pad_w = aspectaware_resize_padding(image, self.input_size, self.input_size)
        
        framed_img = framed_img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        framed_img = (framed_img - mean) / std
        
        framed_img = np.transpose(framed_img, (2, 0, 1))
        framed_img = torch.from_numpy(framed_img).unsqueeze(0).to(self.device)
        
        meta = {'mean': mean, 'std': std, 'scale': self.input_size / max(h, w)}
        return framed_img, meta

    def detect_vehicles(self, image):
        from efficientdet.utils import BBoxTransform, ClipBoxes
        from utils.utils import postprocess
        
        framed_img, meta = self._preprocess(image)
        with torch.no_grad():
            features, regression, classification, anchors = self.model(framed_img)
            
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            
            out = postprocess(framed_img,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              self.threshold, self.iou_threshold)
            
        vehicles = []
        if len(out) > 0 and len(out[0]['rois']) > 0:
            rois = out[0]['rois']
            from utils.utils import invert_affine
            rois = invert_affine([meta], rois)[0]
            
            class_ids = out[0]['class_ids']
            scores = out[0]['scores']
            
            for j in range(len(rois)):
                cls_id = int(class_ids[j])
                if cls_id in self.veh_id:
                    box = [int(x) for x in rois[j]]
                    conf = float(scores[j])
                    vehicles.append((box, conf, self.obj_list[cls_id]))
        return vehicles

    def detect_plates(self, vehicle_image):
        from efficientdet.utils import BBoxTransform, ClipBoxes
        from utils.utils import postprocess
        
        framed_img, meta = self._preprocess(vehicle_image)
        with torch.no_grad():
            features, regression, classification, anchors = self.np_model(framed_img)
            
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            
            out = postprocess(framed_img,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              self.threshold, self.iou_threshold)
            
        plates = []
        if len(out) > 0 and len(out[0]['rois']) > 0:
            rois = out[0]['rois']
            from utils.utils import invert_affine
            rois = invert_affine([meta], rois)[0]
            scores = out[0]['scores']
            for j in range(len(rois)):
                box = [int(x) for x in rois[j]]
                conf = float(scores[j])
                plates.append((box, conf))
        return plates

    def recalculate_plate_coords(self, plate_box, vehicle_box):
        vx1, vy1, vx2, vy2 = vehicle_box
        px1, py1, px2, py2 = plate_box
        bias = 2
        return [
            int(vx1 + px1 - bias),
            int(vy1 + py1 - bias),
            int(vx1 + px2 + bias),
            int(vy1 + py2 + bias)
        ]


class RCNNAdapter(BaseDetectorAdapter):
    def __init__(self, device, weights_vehicle='models/rcnn/faster_default_final.pkl', weights_np='models/rcnn/faster_model_np3000.pth'):
        super().__init__(device)
        try:
            import detectron2
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2 import model_zoo
            from detectron2.data import MetadataCatalog
        except ImportError:
            raise ImportError(
                "Detectron2 not installed. Please install it using: "
                "pip install 'git+https://github.com/facebookresearch/detectron2.git'"
            )

        str_device = 'cuda' if 'cuda' in str(device) else 'cpu'
        
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = weights_vehicle
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
        cfg.MODEL.DEVICE = str_device
        self.predictor_default = DefaultPredictor(cfg)

        cfg_np = get_cfg()
        cfg_np.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg_np.MODEL.WEIGHTS = weights_np
        cfg_np.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
        cfg_np.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg_np.MODEL.DEVICE = str_device
        self.predictor_np = DefaultPredictor(cfg_np)

        self.class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        self.classes_vehicle = [2, 3, 5, 6, 7]

    def detect_vehicles(self, image):
        outputs = self.predictor_default(image)
        instances = outputs["instances"].to('cpu')
        pred_classes = instances.pred_classes.numpy().tolist()
        pred_boxes = instances.pred_boxes.tensor.numpy().tolist()
        scores = instances.scores.numpy().tolist()
        
        vehicles = []
        for i in range(len(pred_classes)):
            cls_id = pred_classes[i]
            if cls_id in self.classes_vehicle:
                box = [int(x) for x in pred_boxes[i]]
                conf = scores[i]
                vehicles.append((box, conf, self.class_names[cls_id]))
        return vehicles

    def detect_plates(self, vehicle_image):
        outputs = self.predictor_np(vehicle_image)
        instances = outputs["instances"].to('cpu')
        pred_boxes = instances.pred_boxes.tensor.numpy().tolist()
        scores = instances.scores.numpy().tolist()
        
        plates = []
        for i in range(len(pred_boxes)):
            box = [int(x) for x in pred_boxes[i]]
            conf = scores[i]
            plates.append((box, conf))
        return plates

    def recalculate_plate_coords(self, plate_box, vehicle_box):
        vx1, vy1, vx2, vy2 = vehicle_box
        px1, py1, px2, py2 = plate_box
        return [
            int(px1 + vx1),
            int(py1 + vy1),
            int(px2 + vx1),
            int(py2 + vy1)
        ]
