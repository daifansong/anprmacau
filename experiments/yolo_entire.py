from pathlib import Path
from PIL import Image
import sys
import torch
import os
import time
import numpy as np
from character import segment, csv_related

'''
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
LIB_ROOT = ROOT / 'lib'
sys.path.append(str(LIB_ROOT))
'''

FILE = Path(__file__).resolve()
ROOT = FILE.parent
YOLO_ROOT = ROOT / 'yolov5'
sys.path.append(str(YOLO_ROOT))

from yolov5 import utils, models
from utils.general import increment_path, check_img_size, Profile, non_max_suppression, scale_boxes, xyxy2xywh, cv2, LOGGER, colorstr, strip_optimizer
from utils.augmentations import letterbox
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.plots import Annotator, save_one_box, colors


source = 'zidane.png'
project = ROOT
weights_vehicle = 'models/yolo/yolov5s.pt'
weights_np = 'models/yolo/yolov5np_samhui.pt'
data = YOLO_ROOT / 'data/coco128.yaml'
name = 'exp'
classes_vehicle = [2,3,5,6,7]
classes_np = None
exist_ok = False
save_txt = False
conf_thres = 0.25
iou_thres = 0.45
agnostic_nms = False
max_det = 1000


def recalculate_coordinate(nmnm, xyxy):
    # print(nmnm, xyxy)
    width = nmnm[2] - nmnm[0]
    height = nmnm[3] - nmnm[1]
    bias = torch.tensor(3)
    # - torch.tensor(12)

    nmnm[0] = nmnm[0] + xyxy[0] - torch.tensor(12) - bias
    nmnm[1] = nmnm[1] + xyxy[1] - torch.tensor(12) - bias
    nmnm[2] = nmnm[0] + width + bias
    nmnm[3] = nmnm[1] + height + bias

    return nmnm


def get_orgnp(det, im, im0):
    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
    #cor, conf = [], []
    cor = None
    conf = 0
    for *xyxy, conf, cls in reversed(det):
        cor= xyxy
        conf = conf
    return cor, conf


def resize_img(im, model):
    tmp_img = np.array(im)

    im = letterbox(tmp_img, 640, stride=model.stride, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]
    return im


def detect_np(model, im, p, im0):   # im: vehicle的框 im0: original image
    stride_np, names_np, pt_np = model.stride, model.names, model.pt
    xyxy, conf = None, 0
    dt = (Profile(), Profile(), Profile())
    veh_img = im
    im = resize_img(im, model)

    with dt[1]:
        pred_np = model(im, augment=False,visualize=False)

    with dt[2]:    
        pred_np = non_max_suppression(pred_np, conf_thres, iou_thres, None, agnostic_nms, max_det=max_det)

    for i, det in enumerate(pred_np):
        #annotator_np = Annotator(veh_img, line_width=3, example=str(model.names))
        gn = torch.tensor(veh_img.shape)[[1,0,1,0]]
        if len(det):
            #det_veh = det.copy()
            #det_veh[:, :4] = scale_boxes(im.shape[2:], det[:, :4], veh_img.shape).round()
            #det_real = det_veh.copy()
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], veh_img.shape).round()

            for *xyxy, conf, cls in reversed(det):
                # print('NP ', xyxy)  # NP [tensor(349.), tensor(648.), tensor(458.), tensor(712.)]
                # print('pic: ', names_np[int(cls)], conf, xyxy)
                veh_img_copy = veh_img.copy()
                partial_np = save_one_box(xyxy, veh_img_copy, file=Path(project) / 'crops' / names_np[int(cls)] / f'{p.stem}.jpg', BGR=True, save=False)
                #cv2.imwrite('number plate.png', partial_np)
                #annotator_np.box_label(xyxy, 'np', color=colors(int(cls), True))

    return len(pred_np[0]), pred_np, xyxy, conf


def set_device():
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def run(source, weights=weights_vehicle, classes=classes_vehicle):

    imgsz = (640,640)
    dnn = False
    half = False
    vid_stride=1
    visualize = False
    augment = False
    save_crop = False
    
    source = str(source)
    save_dir = ROOT
    save_output_dir = ROOT / 'output'
    save_veh_dir = ROOT / 'vehicles'
    try:
        os.mkdir(save_veh_dir)
    except FileExistsError:
        pass
    try:
        os.mkdir(save_output_dir)
    except FileExistsError:
        pass
    #save_dir = increment_path(Path(project)/ 'output', exist_ok=exist_ok)
    #save_dir.mkdir(parents=True, exist_ok=True)
    device = set_device()
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    model_np = DetectMultiBackend(weights_np, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt #print(stride, names, pt)
    
    imgsz = check_img_size(imgsz, s=stride)

    bs = 1
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None]*bs, [None]*bs

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        # im0s: cv2.imread
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
        
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)      
        # print('Start Picture', seen+1)

        #pred: list pred[0]:tensor len(pred[0]):numbers of detections
        #print(type(pred[0]), len(pred[0]), pred[0].size())
        #print('pred: ', pred)

        for i, det in enumerate(pred):  # per image
            seen += 1

            #  im0: cv2.imread copy
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            #save_path = str(save_output_dir/+'yolo_'+p.name)  # im.jpg
            save_path = f'{save_output_dir}/yolo_{p.name}'
            # save_path = str(p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            im_copy = im0.copy()
            annotator = Annotator(im0, line_width=3, example=str(names))
            veh_count = 0
            lp = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                # print('im: ', im0.shape, end=' ') # (738, 689, 3)
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                #print(s)
                # Write results
                for *xyxy, conf, cls in reversed(det):

                    imcc = im_copy.copy()
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                    #xyxy2 = torch.tensor(xyxy).view(-1, 4)
                    #b = xyxy2xywh(xyxy2)

                    inside = -1

                    cc = save_one_box(xyxy, imcc, file=save_dir / 'crops' / names[int(cls)] / f'{p.stem}.jpg', BGR=True, save=False)
                    cv2.imwrite('2.png', cc)

                    # inside: how many; nmnm: np's xyxy; np_conf: conf
                    inside, pred_np, nmnm, np_conf = detect_np(model_np, cc, p, im0)
                    if nmnm is not None:
                        nmnm = recalculate_coordinate(nmnm, xyxy)                      
                        #print('new np cor: ','\033[1m' + str(nmnm) + '\033[0m')
                        #nmnm, np_conf = get_orgnp(det_np, im, im0)
                        #print('np: ', nmnm)

                    if save_crop:
                        cc = save_one_box(xyxy, imc, file=save_dir / 'crops' / names[int(cls)] / f'{p.stem}.jpg', BGR=True, save=False)
                        cv2.imwrite("test.png", cc)
                    c = int(cls)  # integer class
                    label =  f'{names[c]} {conf:.2f}'
                    if inside != 0:
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        np = save_one_box(nmnm, imcc, file=save_dir / 'crops' / names[int(cls)] / f'{p.stem}.jpg', BGR=True, save=False)
                        # label_np = '0'
                        # cv2.imwrite('mm.png', np)
                        plate, char_color, attribute = segment(np)
                        license_plate = ''.join(str(v) for v in plate)
                        lp.append(license_plate)
                        # print('\033[1m' + license_plate + '\033[0m')
                        label_np = f'{names[c]} {license_plate} {char_color} {attribute}'
                        annotator.box_label(nmnm, label_np, color=colors(-1, True))

                        csv_related(license_plate, names[c], char_color)

                        veh_count += 1
                        cc = save_one_box(xyxy, imcc, file=save_dir / 'crops' / names[int(cls)] / f'{p.stem}.jpg', BGR=True, save=False)
                        cv2.imwrite(f'{save_veh_dir}/ {names[c]}_{veh_count:02d}.png', cc)
            
            for npi in lp:
                print('\033[1m' + npi + '\033[0m', end=' ')

            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path[i] != save_path:
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w, h = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    save_path = str(Path(save_path).with_suffix('.mp4'))
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        source_path = 'images/011.jpg'
    elif len(sys.argv) > 2:
        print('Redo')
    else:
        source_path = sys.argv[1]
    #image_path = '/content/drive/MyDrive/Projects/samples/selected/001.jpg'
    #video_path = '/content/drive/MyDrive/Projects/samples/videos/bus_short.mp4'
    start = time.perf_counter()
    run(source_path)
    end = time.perf_counter()
    print(f'\nDone Time: {end - start:.4f}s')