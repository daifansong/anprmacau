from pathlib import Path
import cv2
import sys
import torch
from character import segment, csv_related
from tqdm import tqdm
import imghdr
import time
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parent
YOLO_ROOT = ROOT / 'pytorch-ssd'
sys.path.append(str(YOLO_ROOT))

from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

default_classes = ['BACKGROUND', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
veh_id = [6, 7, 14]
det_class = ['BACKGROUND', 'Licence-Plate']
np_id = [0]
model_path = 'models/ssd/mb2-ssd-lite-mp-0_686.pth'
np_model_path = 'models/ssd/mb2-ssd-lite-np-epoch401-loss0.906.pth'

output_video = 'output_test.mp4'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

net = create_mobilenetv2_ssd_lite(len(default_classes), is_test=True)
net.load(model_path)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=device)

np_model = create_mobilenetv2_ssd_lite(len(det_class), is_test=True)
np_model.load(np_model_path)
np_detector = create_mobilenetv2_ssd_lite_predictor(np_model, candidate_size=200, device=device)

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
#save_path = save_output_dir


def detect_np(img): # number plate cv2
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = np_detector.predict(RGB_img, 10, 0.4)
    if len(probs) == 0:
        return False
    else:
        detect_result = [boxes, probs[0]]
        return detect_result


def recalculate(box, x1, y1):
    bias = 2
    box = box[0]
    nx1, ny1, nx2, ny2 = box.numpy().tolist()
    nx1, ny1, nx2, ny2 = round(nx1)+x1-bias, round(ny1)+y1-bias, round(nx2)+x1+bias, round(ny2)+y1+bias
    np_cor = [nx1, ny1, nx2, ny2]
    return np_cor
    #new_cor = [nx1, ny1, nx2, ny2]
    #return new_cor


def detect_ssd(ori_image, isVideo=False):

    RGB_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    # image = image.to(device)
    boxes, labels, probs = predictor.predict(RGB_image, 10, 0.2)
    image = ori_image.copy()
    # print('boxes', boxes)
    # print('labels', labels)
    # print('probs', probs)
    imgc = image.copy()
    lp_list = []
    veh_count, np_count = 0, 0
    for i in range(boxes.size(0)):
        img_cc = image.copy()
        if labels[i] in veh_id:
            veh_count += 1
            box = boxes[i, :]
            real_box = box.numpy().tolist()
            x1, y1, x2, y2 = real_box
            
            non_zero_box = [0 if i < 0 else i for i in real_box]
            x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
            ax1, ay1, ax2, ay2= non_zero_box
            ax1, ay1, ax2, ay2 = round(ax1), round(ay1), round(ax2), round(ay2)
            img_copy = image.copy()
            cropped_vehicle = img_copy[ay1:ay2, ax1:ax2]
            #cv2.imwrite(f'crop_{i}.png', cropped_vehicle) 
            #print('cor: ',x1, y1)
            detection = detect_np(cropped_vehicle)
            #print('det: ', detection)
            #detection = True
            if detection is False:
                pass
            else:
                imgc2 = image.copy()
                [np_box, np_score] = detection
                # print('np_box', np_box)
                absolute_cor = recalculate(np_box, ax1, ay1)
                bx1, by1, bx2, by2 = absolute_cor
                crop_cor = [0 if i < 0 else i for i in absolute_cor]
                nx1, ny1, nx2, ny2 = crop_cor
                '''
                np_probs = [np_possibilities for np_possibilities in detection[1]]
                np_boxes = [np_box for np_box in detection[0]]
                '''
                np_count += 1
                # imgc2 = imgc.copy()
                # print(nx1, ny1, nx2, ny2)
                cropped_np = img_cc[ny1:ny2, nx1:nx2]
                #cv2.imwrite('2.png', cropped_np)
                # print(cropped_np.shape)

                plate, char_color, attribute = segment(cropped_np)
                license_plate = ''.join(plate)
                lp_list.append(license_plate)

                veh_label = f'{default_classes[labels[i]]} {probs[i]:.4f}'
                w1, h1 = cv2.getTextSize(veh_label, 0, fontScale=1, thickness=2)[0]
                #np_label = '0'
                np_label = f'{license_plate} {char_color} {attribute} {np_score:.4f}'

                csv_related(license_plate, default_classes[labels[i]], char_color)

                cv2.rectangle(imgc, (x1,y1),(x2,y2), (255, 255, 0), 2)
                cv2.putText(imgc, veh_label, (x1, y1+h1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)
                cv2.rectangle(imgc, (bx1,by1),(bx2,by2), (0, 255, 255), 2)
                cv2.putText(imgc, np_label, (bx1, ny1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                
                veh_img = imgc.copy()[ay1:ay2, ax1:ax2]
                #cv2.imwrite('pp.png', veh_img)
                cv2.imwrite(f'{save_veh_dir}/{img_prefix}_{default_classes[labels[i]]}_{np_count:02d}.png', veh_img)
            # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""

    if isVideo:
        cv2.imwrite('yu.png', imgc)
        return imgc
        
    else:
        if len(lp_list) == 0:
            print('NO Licence Plates Found')
        else:
            for npi in lp_list:
                print('\033[1m' + npi + '\033[0m', end=' ')

        cv2.imwrite(f'{save_output_dir}/ssd_{img_name}', imgc)

        print(f'\nFound {veh_count} vehicles {np_count} plates. The output image is {save_output_dir}/ssd_{img_name}')
        #imgc = cv2.cvtColor(imgc, cv2.COLOR_RGB2BGR)


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
                res_frame = detect_ssd(frame, isVideo=True)
                cv2.imwrite('test.png', res_frame)

                video.write(res_frame)
'''
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % interval == 0:
            res_frame = detect_ssd(frame)
            cv2.imwrite('test.png', res_frame)

            video.write(res_frame)
'''

if __name__ == '__main__':
    #source_path = '/content/drive/MyDrive/Projects/samples/videos/holand_Trim.mp4'
    if len(sys.argv) <= 1:
        source_path = 'images/011.jpg'
    elif len(sys.argv) > 2:
        print('Redo')
    else:
        source_path = sys.argv[1]
    global img_name, img_prefix
    img_name = Path(source_path).name
    img_prefix = Path(source_path).stem
    start = time.perf_counter()
    source_type = detect_file_type(source_path)
    if source_type == 'Image':
        print('Processing Image...')
        image = cv2.imread(source_path)
        detect_ssd(image)
    elif source_type == 'Video':
        print('Processing Video...')
        detect_video(source_path)
    end = time.perf_counter()
    print(f'Done Time: {end - start:.4f}s')


'''
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor

from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.utils.misc import Timer

if len(sys.argv) < 5:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path>')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
image_path = sys.argv[4]

class_names = [name.strip() for name in open(label_path).readlines()]

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb3-large-ssd-lite':
    net = create_mobilenetv3_large_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb3-small-ssd-lite':
    net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite' or net_type == "mb3-large-ssd-lite" or net_type == "mb3-small-ssd-lite":
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
'''