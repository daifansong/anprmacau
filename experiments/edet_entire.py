import time
import torch
from torch.backends import cudnn
from matplotlib import colors
from pathlib import Path
import cv2
import numpy as np
import imghdr
import sys
import os
from character import segment, csv_related
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parent
EDET_ROOT = ROOT / 'Yet-Another-EfficientDet-Pytorch'
# print(EDET_ROOT)
sys.path.append(str(EDET_ROOT))

output_video = 'output_test.mp4'

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import aspectaware_resize_padding, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box, preprocess_video

compound_coef = 1
force_input_size = None  # set None to use default size
img_path = 'images/TCM_10B.jpg'
default_model_path = 'models/efficientdet/efficientdet-d1.pth'
np_model_path = 'models/efficientdet/efficientdet-d1_296_7699.pth'

anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.25
iou_threshold = 0.25

use_cuda = True if torch.cuda.is_available() else False
str_decive = 'cuda:0' if torch.cuda.is_available() else 'cpu'
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
veh_id = [2, 3, 5, 7]
det_list = ['Licence-Plate']
np_id = [0]

color_list = standard_to_bgr(STANDARD_COLORS)

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list), ratios=anchor_ratios, scales=anchor_scales)
#model = model[18:27] # 2 3 5 7
model.load_state_dict(torch.load(default_model_path, map_location=str_decive), strict=False)
model.requires_grad_(False)
model.eval()

np_model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(det_list), ratios=anchor_ratios, scales=anchor_scales)
#model = model[18:27] # 2 3 5 7
np_model.load_state_dict(torch.load(np_model_path, map_location=str_decive), strict=False)
np_model.requires_grad_(False)
np_model.eval()

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

if use_cuda:
    model = model.cuda()
    np_model = np_model.cuda()
if use_float16:
    model = model.half()
    np_model = np_model.cuda()


def recalculate(np_cor, veh_cor):
    bias = 2
    [vx1, vy1, vx2, vy2] = veh_cor
    [nx1, ny1, nx2, ny2] = np_cor
    #width, height = nx2 - nx1, ny2 - ny1
    ab_cor = [vx1+nx1-bias, vy1+ny1-bias, vx1+nx2+bias, vy1+ny2+bias]
    return ab_cor


def display(preds, imgs, imshow=False, imwrite=False, is_video=False):
    # print(f'Inside: {len(preds)}, {len(imgs)}')
    for i in range(len(imgs)):
        img_cc = imgs[i].copy()
        if is_video:
            img_c2 = imgs[i].copy()
        else:
            imgs[i] = imgs[i].copy()
        if len(preds[i]['rois']) == 0:
            continue
        #veh_cor_list, type_list, prob_list = [], [], []
        lp_list = []
        veh_count = 0
        for j in range(len(preds[i]['rois'])):
            obj_id = preds[i]['class_ids'][j]

            if obj_id in veh_id:
                x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int32)
                current_veh_cor = [x1, y1, x2, y2]
                #veh_cor_list.append(temp_cor)
                np_detection = detect_np(imgs[i][y1:y2,x1:x2])
                #print(np_detection, end=' ')
                if np_detection is None:
                    pass
                else:
                    relative_cor, np_score = np_detection
                    absolute_cor = recalculate(relative_cor, current_veh_cor)
                    crop_cor = [0 if i < 0 else i for i in absolute_cor]
                    [nx1, ny1, nx2, ny2] = crop_cor
                    plate_img = img_cc[ny1:ny2, nx1:nx2]
                    # cv2.imwrite('2.png', plate_img)
                    # print(j, crop_cor)

                    plate_list, char_color, attribute = segment(plate_img)
                    plate = ''.join(plate_list)
                    lp_list.append(plate)

                    obj = obj_list[obj_id]
                    score = float(preds[i]['scores'][j])
                    # print(f'{obj}: {x1, y1, x2, y2}\tpred: {score:.4f}')
                    plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)], line_thickness=2)
                    #np_label = '0'
                    np_label = f'{obj} {plate} {char_color} {attribute}'
                    plot_one_box(imgs[i], absolute_cor, label=np_label, score=np_score, color=color_list[-2], line_thickness=2)

                    csv_related(plate, obj, char_color)

                    veh_count += 1
                    cc = imgs[i][y1:y2,x1:x2]
                    cv2.imwrite(f'{save_veh_dir}/edet_{img_prefix}_{obj}_{veh_count:02d}.png', cc)

        if len(lp_list) == 0:
            print('NO Licence Plates Found')
        else:
            for npi in lp_list:
                print('\033[1m' + npi + '\033[0m', end=' ')
        #print('Total: ', len(preds[i]['rois']))        


        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite(f'{save_output_dir}/edet_{img_name}', imgs[i])
            print(f'\nSaved in {save_output_dir}/edet_{img_name}')
            
        return imgs[i]


def preprocess(*cv2_imgs, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    ori_imgs = [cv2_img for cv2_img in cv2_imgs]
    #ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    normalized_imgs = [(img[..., ::-1] / 255 - mean) / std for img in ori_imgs]

    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size, means=None) for img in normalized_imgs]

    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def detect_np(img):
    ori_imgs, framed_imgs, framed_metas = preprocess(img, max_size=input_size)
    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = np_model(x)
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
    out = invert_affine(framed_metas, out)
    # print('out', out)
    if len(out[0]['rois']) == 0:
        return None
    else:
        x1, y1, x2, y2 = out[0]['rois'][0].astype(np.int32)
        cor = [x1, y1, x2, y2]
        score = out[0]['scores'][0]
        return cor, score


def run(img):
    #img = cv2.imread(img_path)
    #img2 = cv2.imread('/content/drive/MyDrive/Projects/samples/selected/002.jpg')
    #img3 = cv2.imread('/content/drive/MyDrive/Projects/samples/images/TCM_10B.jpg')

    ori_imgs, framed_imgs, framed_metas = preprocess(img, max_size=input_size)
    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
        
    # print('1', out)
    out = invert_affine(framed_metas, out)
    # print('2', out)
    display(out, ori_imgs, imshow=False, imwrite=True)
    # print('Done')


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


def detect_video(video_path):
    regressBoxes, clipBoxes = BBoxTransform(), ClipBoxes()
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
                ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)
                if use_cuda:
                    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
                else:
                    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
                x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

                with torch.no_grad():
                    features, regression, classification, anchors = model(x)

                    out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
                out = invert_affine(framed_metas, out)
                new_vid_frame = display(out, ori_imgs, imshow=False, is_video=True)
                #print('len: ', len(new_vid_frames))
                video.write(new_vid_frame)
'''
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % interval == 0:
            ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)
            if use_cuda:
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
            x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

            with torch.no_grad():
                features, regression, classification, anchors = model(x)

                out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
            out = invert_affine(framed_metas, out)
            new_vid_frame = display(out, ori_imgs, imshow=False, is_video=True)
            #print('len: ', len(new_vid_frames))
            video.write(new_vid_frame)
'''


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        source_path = 'images/011.jpg'
    elif len(sys.argv) > 2:
        print('Redo')
    else:
        source_path = sys.argv[1]
    global img_name, img_prefix
    img_name = Path(source_path).name
    img_prefix = Path(source_path).stem
    source_type = detect_file_type(source_path)
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