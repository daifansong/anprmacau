import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import time
import numpy as np
import cv2
import os
import sys
import imghdr
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from character import segment, csv_related
from pathlib import Path
from tqdm import tqdm

str_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_path = 'models/rcnn/faster_model_np3000.pth'

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = 'models/rcnn/faster_default_final.pkl'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
cfg.MODEL.DEVICE = str_device
predictor_default = DefaultPredictor(cfg)

output_video = 'output_test.mp4'
class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
# print(class_names)
classes_vehicle = [2,3,5,6,7]

cfg_np = get_cfg()
cfg_np.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg_np.MODEL.WEIGHTS = model_path
cfg_np.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
cfg_np.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg_np.MODEL.DEVICE = str_device
predictor_np = DefaultPredictor(cfg_np)

FILE = Path(__file__).resolve()
ROOT = FILE.parent
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

#outputs_default = predictor_default(image)
#outputs_np = predictor_np(image)


def detect_file_type(source):
    img_type = imghdr.what(source)
    if img_type:
        return 'Image'
    try:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return 'Video'
    except:
        pass
    return None


def recalculate(nx1, ny1, nx2, ny2, vx1, vy1):
    return nx1+vx1, ny1+vy1, nx2+vx1, ny2+vy1


def run(image, isVideo=False):
    imgc = image.copy()
    use_index = []
    start_index = 0
    outputs_default = predictor_default(image)
    img_veh_numpy_list = outputs_default["instances"].to('cpu').pred_classes.numpy().tolist()
    boxes_numpy_list  = outputs_default["instances"].to('cpu').pred_boxes.tensor.numpy().tolist()
    veh_scores_numpy_list  = outputs_default["instances"].to('cpu').scores.numpy().tolist()
    for i in img_veh_numpy_list:
        if i in classes_vehicle:
            use_index.append(start_index)
        start_index += 1
    # print(use_index)
    veh_boxes, veh_class, veh_score = [], [], []
    for index in use_index:
        veh_boxes.append(boxes_numpy_list[index])
        veh_class.append(img_veh_numpy_list[index])
        veh_score.append(veh_scores_numpy_list[index])
    veh_related = [veh_boxes, veh_class, veh_score]
    # print(veh_related)
    i = 0
    lp_list = []
    for veh_boxx in veh_boxes:
        
        veh_ttype, veh_predict_score = veh_class[i], veh_score[i]
        i += 1
        [x1, y1, x2, y2] = veh_boxx
        vx1, vy1, vx2, vy2 = round(x1), round(y1), round(x2), round(y2)
        veh_img = imgc[vy1:vy2,vx1:vx2]
        result = detect_np(veh_img)
        if result is False:
            pass
        else:
            np_boxes, score_list = result
            veh_label = f'{class_names[veh_ttype]} {veh_predict_score:.4f}'
            w1, h1 = cv2.getTextSize(veh_label, 0, fontScale=1, thickness=2)[0]
            cv2.rectangle(imgc, (vx1,vy1),(vx2,vy2), (255, 255, 0), 2)
            cv2.putText(imgc, veh_label, (vx1, vy1+h1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)
            ii = 0
            for npbb in np_boxes:
                np_score = score_list[ii]
                [nx1, ny1, nx2, ny2] = npbb
                nx1, ny1, nx2, ny2 = recalculate(nx1, ny1, nx2, ny2, vx1, vy1)
                imgcc = image.copy()
                cropped_np = imgcc[ny1:ny2, nx1:nx2]

                plate, char_color, attribute = segment(cropped_np)
                license_plate = ''.join(plate)
                lp_list.append(license_plate)
                #np_label = '0'
                np_label = f'{license_plate} {char_color} {attribute} {np_score:.4f}'

                csv_related(license_plate, class_names[veh_ttype], char_color)

                cv2.rectangle(imgc, (nx1,ny1),(nx2,ny2), (0, 255, 255), 2)
                cv2.putText(imgc, np_label, (nx1, ny1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            
            veh_img = imgc.copy()[vy1:vy2, vx1:vx2]
            cv2.imwrite(f'{save_veh_dir}/{img_prefix}_{class_names[veh_ttype]}_{i:02d}.png', veh_img)

    #print(use_index)
    #v = Visualizer(image[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
    #v = v.draw_instance_predictions(outputs_default['instances'].to('cpu')[use_index])
    #res_img = v.get_image()[:, :, ::-1]
    if isVideo:
        return imgc
    else:
        if len(lp_list) == 0:
            print('NO Licence Plates Found')
        else:
            for npi in lp_list:
                print('\033[1m' + npi + '\033[0m', end=' ')

        cv2.imwrite(f'{save_output_dir}/faster_{img_name}', imgc)
        print(f'\nFound {i} vehicles. Result stores in {save_output_dir}/faster_{img_name}')


def detect_np(image):
    use_index = []
    start_index = 0
    outputs_np = predictor_np(image)
    img_class_numpy_list = outputs_np["instances"].to('cpu').pred_classes.numpy().tolist()
    for p_cls in img_class_numpy_list:
        if p_cls != 0:
            use_index.append(start_index)
        start_index += 1
    # print(use_index)
    num = len(outputs_np["instances"].to('cpu').pred_boxes)
    if num == 0:
        return False
    else:
        np_boxes = []
        np_boxes_numpy_list  = outputs_np["instances"].to('cpu').pred_boxes.tensor.numpy().tolist()
        np_scores_numpy_list  = outputs_np["instances"].to('cpu').scores.numpy().tolist()

        score_list, box_list = [], []
        for index in use_index:
            score_list.append(np_scores_numpy_list[index])
            box_list.append(np_boxes_numpy_list[index])

        for box in box_list:
            [x1, y1, x2, y2] = box
            x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
            nbox = [x1, y1, x2, y2]
            np_boxes.append(nbox)
        return np_boxes, score_list
    

def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames, cap_fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    interval = int(cap_fps / 10)
    desired_fps = 10
    ret, frame1 = cap.read()
    height, width, channels = frame1.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, desired_fps, (width, height))
    # print(cap_fps)
    frame_count = 0
    with tqdm(range(total_frames), desc='Video Processing: ') as pbar:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            pbar.update(1)
            if frame_count % interval == 0:
                res_frame = run(frame, isVideo=True)
                cv2.imwrite('test.png', res_frame)

                video.write(res_frame)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        source_path = 'images/011.jpg'
    elif len(sys.argv) > 2:
        print('Redo')
    else:
        source_path = sys.argv[1]
    source_type = detect_file_type(source_path)
    global img_name, img_prefix
    img_name = Path(source_path).name
    img_prefix = Path(source_path).stem
    start = time.perf_counter()
    if source_type == 'Image':
        print('Processing Image...')
        image = cv2.imread(source_path)
        run(image)
    elif source_type == 'Video':
        print('Processing Video...')
        detect_video(source_path)
    end = time.perf_counter()
    print(f'Done Time: {end - start:.4f}s')