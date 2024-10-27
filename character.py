import cv2
import numpy as np
import functools
import torch
from torchvision import transforms
from PIL import Image
import csv
from datetime import date, datetime
from pathlib import Path

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ocr = torch.load('/content/Models/ocr.pth', map_location=device)
ocr = ocr.module.to(device)
ocr.eval()
edge = 500
labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
'''
colors = [
    ("red", (0, 0, 255)),
    ("green", (0, 255, 0)),
    ("blue", (255, 0, 0)),
    ("yellow", (0, 255, 255)),
    ("magenta", (255, 0, 255)),
    ("cyan", (255, 255, 0)),
    ("white", (255, 255, 255)),
    ("black", (0, 0, 0))
]
'''
colors = [
    ("yellow", (0, 255, 255)),
    ("white", (255, 255, 255)),
]
color_list = []

transform = transforms.Compose([  # variable transform
    transforms.Resize(256),  # resize 256,256
    transforms.CenterCrop(224),  # Crop the image to 224×224 pixels about the center
    transforms.ToTensor(),  # Convert the image to PyTorch Tensor data type
    transforms.Normalize(  # Normalize
        mean=[0.485, 0.456, 0.406],  # Mean and std of image as also used when training the network
        std=[0.229, 0.224, 0.225])])


def compare(rect1, rect2):
    if abs(rect1[1] - rect2[1]) > img_h / 4:
        return rect1[1] - rect2[1]
    else:
        return rect1[0] - rect2[0]


def detect_color(mask, img):
    img_specifiedColor = cv2.bitwise_and(img, img, mask=255-mask)
    mean_color = cv2.mean(img_specifiedColor, mask=255-mask)[:3]
    distances = []
    for name, color in colors:
        distance = np.linalg.norm(np.array(mean_color) - np.array(color))
        distances.append((name, distance))

    closest_color = min(distances, key=lambda d: d[1])[0]
    # print('Closest color:', closest_color)
    color_list.append(closest_color)


def resize_img(img):
    h, w = img.shape[0], img.shape[1]
    if w >= 500 or h >= 500:
        return img
    else:
        scale = round(min(edge / h, edge / w))
        width = int(w * scale)
        height = int(h * scale)
        dim = (width, height)
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        # print(resized_img.shape)
        return resized_img


def segment(img):

    resized_img = resize_img(img)
    img = resized_img
    
    global img_h, img_w
    img_h, img_w = img.shape[0], img.shape[1]
    single = True if img_w >= 3 * img_h else False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(255 - gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=1)
    _, labels = cv2.connectedComponents(erosion)
    mask = np.zeros(erosion.shape, dtype="uint8")
    total_pixels = img.shape[0] * img.shape[1]
    lower = total_pixels // 100  # heuristic param, can be fine tuned if necessary
    upper = total_pixels // 10

    for (i, label) in enumerate(np.unique(labels)):
        # If this is the background label, ignore it
        if label == 0:
            continue
        # Otherwise, construct the label mask to display only connected component
        # for the current label
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        # If the number of pixels in the component is between lower bound and upper bound,
        # add it to our mask
        if numPixels > lower and numPixels < upper:
            mask = cv2.add(mask, labelMask)

    img_copy = img.copy()
    mc = mask.copy()
    cnts, _ = cv2.findContours(mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    #print(boundingBoxes)
    boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare))

    i = 1  # Order
    plate = []
    for rect in boundingBoxes:
        img_cc = img_copy.copy()
        x, y, w, h = rect
        cm = cv2.bitwise_not(mc)
        if (single and w > 1.1 * h) or (not single and (h > img_h / 2 or w > img_w / 4 or w > 1.1 * h)):
            pass
        else:
            crop = cm[y:y + h, x:x + w]
            crop_image = img_cc[y:y + h, x:x + w]
            detect_color(crop, crop_image)

            char_img = resize_char(crop)
            possible_char = predict(char_img)
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.putText(img, str(i), (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            #cv2.putText(img, str(possible_char), (x + w - 25, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 ,0, 255), 2)
            plate.append(possible_char)
            i = i + 1
    if len(color_list) == 0:
        char_color = 'Indeterminated'
    else:
        char_color = max(color_list, key=color_list.count)
    attribute = 'tax_free' if char_color == 'yellow' else 'normal'
    color_list.clear()
    return plate, char_color, attribute

file_name = 'veh.csv'
fields = ['date', 'time', 'vehicle', 'plate', 'color']
content = csv.reader(open('suspected.csv', 'r'))

def check():
    if Path(file_name).is_file():
        pass
    else:
        with open(file_name, 'x', newline='') as f:
            write_header = csv.DictWriter(f, fieldnames=fields)
            write_header.writeheader()


def add(row):
    with open('veh.csv', 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


def compare_plate(plate):
    i = 0
    for row in content:
        if plate == row[0]:
            print(f'Warning! {plate} {row[1]}')
            i += 1
    print('All Fine' if i == 0 else f'Found {i} vehicle suspected')


def csv_related(plate, veh_type, np_color):
    check()
    date_today = date.today()
    now = datetime.now()
    h, m, s = now.hour, now.minute, now.second
    time = f'{h:02d}:{m:02d}:{s:02d}'
    new_row = [date_today, time, veh_type, plate, np_color]
    add(new_row)
    compare_plate(plate)


def predict(img):
    img = Image.fromarray(img)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    batch_t = batch_t.to(device)
    out = ocr(batch_t)

    _, indices = torch.sort(out, descending=True)
    index = indices[0][0]
    #print('\033[1m' + labels[index] + '\033[0m', end=' ')
    return labels[index]


def resize_char(img):
    height, width = img.shape[:2]
    scale = min(160 / height, 160 / width)
    img = cv2.resize(img, None, fx=scale, fy=scale)
    background = np.zeros((256, 256), dtype=np.uint8)
    background.fill(255)
    # 计算图片放置位置
    x_offset = int((256 - img.shape[1]) / 2)
    y_offset = int((256 - img.shape[0]) / 2)
    # 将缩放后的图片放置在白色背景图片中央
    background[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img
    resized_img = background
    resized_rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    return resized_rgb_img