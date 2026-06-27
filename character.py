import cv2
import numpy as np
import functools
import torch
from torchvision import transforms
from PIL import Image
import csv
from datetime import date, datetime
from pathlib import Path
import sqlite3
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ocr = None

def get_ocr_model(model_path=None):
    global ocr
    if ocr is not None:
        return ocr
    
    paths_to_try = []
    if model_path:
        paths_to_try.append(Path(model_path))
    
    base_dir = Path(__file__).resolve().parent
    paths_to_try.extend([
        base_dir / 'models' / 'ocr.pth',
        base_dir / 'ocr.pth',
        Path('/content/Models/ocr.pth'),
        Path('/content/ocr_april.pth')
    ])
    
    for p in paths_to_try:
        if p.exists():
            try:
                model = torch.load(str(p), map_location=device)
                if hasattr(model, 'module'):
                    model = model.module
                ocr = model.to(device)
                ocr.eval()
                print(f"Successfully loaded OCR model from: {p}")
                return ocr
            except Exception as e:
                print(f"Error loading OCR model from {p}: {e}")
                
    raise FileNotFoundError(
        f"Could not find or load OCR model 'ocr.pth' in any of the expected paths: "
        f"{[str(x) for x in paths_to_try]}. Please make sure the weights file exists."
    )

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


def validate_and_correct_plate(plate_input):
    if isinstance(plate_input, list):
        plate_str = "".join(str(v) for v in plate_input)
    else:
        plate_str = str(plate_input)
        
    raw = "".join(c for c in plate_str if c.isalnum()).upper()
    
    LETTER_TO_DIGIT = {
        'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7', 'J': '1', 'A': '4', 'D': '0'
    }
    DIGIT_TO_LETTER = {
        '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '7': 'T', '4': 'A'
    }
    
    result = plate_str
    if len(raw) == 6 and (raw.startswith('CM') or (raw[0] in ('C', '0', 'D') and raw[1] in ('M', 'N', 'W'))):
        corrected = ['C', 'M']
        for c in raw[2:]:
            corrected.append(LETTER_TO_DIGIT.get(c, c))
        corrected_str = "".join(corrected)
        result = f"{corrected_str[:2]}-{corrected_str[2:4]}-{corrected_str[4:]}"
    elif len(raw) == 5 and (raw[0] == 'M' or raw[0] in ('N', 'H', 'W', '1', 'V')):
        corrected = ['M']
        for c in raw[1:]:
            corrected.append(LETTER_TO_DIGIT.get(c, c))
        corrected_str = "".join(corrected)
        result = f"{corrected_str[0]}-{corrected_str[1:3]}-{corrected_str[3:]}"
    elif len(raw) == 6 and (raw[0] == 'M' or raw[0] in ('N', 'H', 'W', '1', 'V')):
        corrected = ['M']
        second_char = raw[1]
        corrected.append(DIGIT_TO_LETTER.get(second_char, second_char))
        for c in raw[2:]:
            corrected.append(LETTER_TO_DIGIT.get(c, c))
        corrected_str = "".join(corrected)
        result = f"{corrected_str[:2]}-{corrected_str[2:4]}-{corrected_str[4:]}"
    else:
        if len(raw) == 5:
            result = f"{raw[0]}-{raw[1:3]}-{raw[3:]}"
        elif len(raw) == 6:
            result = f"{raw[:2]}-{raw[2:4]}-{raw[4:]}"
        else:
            result = raw
            
    return list(result)


def filter_and_label_plate(plate_chars, avg_conf):
    plate_str = "".join(plate_chars)
    raw = "".join(c for c in plate_str if c.isalnum()).upper()
    
    # 1. Check length constraints
    if len(raw) < 4 or len(raw) > 8:
        return "无法识别", "invalid"
        
    # 2. Check confidence threshold
    if avg_conf < 0.65:
        return "识别模糊", "low_confidence"
        
    # 3. Check plate type (Macau formats vs other)
    if (len(raw) in (5, 6) and raw.startswith('M')) or (len(raw) == 6 and raw.startswith('CM')):
        return plate_str, "normal"
    else:
        return plate_str, "other"


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
    confidences = []
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
            possible_char, conf = predict(char_img)
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.putText(img, str(i), (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            #cv2.putText(img, str(possible_char), (x + w - 25, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 ,0, 255), 2)
            plate.append(possible_char)
            confidences.append(conf)
            i = i + 1
    if len(color_list) == 0:
        char_color = 'Indeterminated'
    else:
        char_color = max(color_list, key=color_list.count)
    attribute = 'tax_free' if char_color == 'yellow' else 'normal'
    color_list.clear()
    
    avg_conf = np.mean(confidences) if confidences else 0.0
    corrected_plate = validate_and_correct_plate(plate)
    
    plate_str, status_label = filter_and_label_plate(corrected_plate, avg_conf)
    return list(plate_str), char_color, attribute, avg_conf, status_label

BASE_DIR = Path(__file__).resolve().parent
file_name = str(BASE_DIR / 'files' / 'veh.csv')
suspected_path = str(BASE_DIR / 'files' / 'suspected.csv')
db_path = str(BASE_DIR / 'files' / 'anpr.db')

if not Path(suspected_path).is_file() and Path('suspected.csv').is_file():
    suspected_path = 'suspected.csv'
if not Path(file_name).parent.is_dir():
    file_name = 'veh.csv'

fields = ['date', 'time', 'vehicle', 'plate', 'color']

# Initialize SQLite Database
def init_db():
    try:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create traffic logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                time TEXT,
                vehicle TEXT,
                plate TEXT,
                color TEXT
            )
        ''')
        
        # Create suspected plates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS suspected_plates (
                plate TEXT PRIMARY KEY,
                reason TEXT
            )
        ''')
        conn.commit()
        
        # Migrate suspected.csv if database is empty
        cursor.execute("SELECT COUNT(*) FROM suspected_plates")
        count = cursor.fetchone()[0]
        if count == 0 and Path(suspected_path).is_file():
            try:
                with open(suspected_path, 'r', encoding='utf-8') as f:
                    content = csv.reader(f)
                    header = next(content, None)  # skip header
                    to_insert = []
                    for row in content:
                        if len(row) >= 2:
                            to_insert.append((row[0].strip().upper(), row[1]))
                        elif len(row) == 1:
                            to_insert.append((row[0].strip().upper(), ''))
                    if to_insert:
                        cursor.executemany("INSERT OR IGNORE INTO suspected_plates (plate, reason) VALUES (?, ?)", to_insert)
                        conn.commit()
                print(f"Migrated {len(to_insert)} suspected plates from CSV to SQLite database.")
            except Exception as e:
                print(f"Error migrating suspected.csv to SQLite: {e}")
                
        conn.close()
    except Exception as e:
        print(f"Error initializing SQLite database: {e}")

# Call init_db immediately at import
init_db()

# Suspected list memory cache
suspected_cache = {}
last_cache_load_time = 0

def load_suspected_cache():
    global suspected_cache, last_cache_load_time
    db_file = Path(db_path)
    if not db_file.exists():
        return
        
    try:
        current_mtime = os.path.getmtime(db_path)
        if current_mtime > last_cache_load_time or not suspected_cache:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT plate, reason FROM suspected_plates")
            rows = cursor.fetchall()
            new_cache = {row[0].strip().upper(): row[1] for row in rows}
            suspected_cache = new_cache
            last_cache_load_time = current_mtime
            conn.close()
    except Exception as e:
        print(f"Error loading suspected plates cache: {e}")

# Database edit helpers
def db_add_suspected(plate, reason=""):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO suspected_plates (plate, reason) VALUES (?, ?)", (plate.strip().upper(), reason))
        conn.commit()
        conn.close()
        sync_db_to_csv()
        return True
    except Exception as e:
        print(f"Error adding suspected plate to SQLite: {e}")
        return False

def db_delete_suspected(plate):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM suspected_plates WHERE plate = ?", (plate.strip().upper(),))
        conn.commit()
        conn.close()
        sync_db_to_csv()
        return True
    except Exception as e:
        print(f"Error deleting suspected plate from SQLite: {e}")
        return False

def sync_db_to_csv():
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT plate, reason FROM suspected_plates")
        rows = cursor.fetchall()
        conn.close()
        
        with open(suspected_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['plate', 'reason'])
            for row in rows:
                writer.writerow(row)
    except Exception as e:
        print(f"Error syncing SQLite to CSV: {e}")

def check():
    if Path(file_name).is_file():
        pass
    else:
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, 'x', newline='') as f:
            write_header = csv.DictWriter(f, fieldnames=fields)
            write_header.writeheader()


def add(row):
    with open(file_name, 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


def compare_plate(plate):
    load_suspected_cache()
    cleaned_plate = plate.strip().upper()
    
    def normalize(p):
        return p.replace('-', '').replace(' ', '').upper()
        
    normalized_plate = normalize(cleaned_plate)
    
    matches = []
    for cached_plate, reason in suspected_cache.items():
        if normalize(cached_plate) == normalized_plate:
            matches.append((cached_plate, reason))
            
    if len(matches) > 0:
        for cached_plate, reason in matches:
            print(f'Warning! {cached_plate} {reason}')
        print(f'Found {len(matches)} vehicle suspected')
        return True
    else:
        print('All Fine')
        return False


def csv_related(plate, veh_type, np_color, status_label="normal"):
    if status_label in ("invalid", "low_confidence"):
        print(f"Ignored database logging for low-confidence/invalid plate: {plate} (Status: {status_label})")
        if status_label != "invalid":
            compare_plate(plate)
        return
        
    check()
    date_today = date.today()
    now = datetime.now()
    h, m, s = now.hour, now.minute, now.second
    time_str = f'{h:02d}:{m:02d}:{s:02d}'
    
    new_row = [date_today, time_str, veh_type, plate, np_color]
    add(new_row)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO traffic_logs (date, time, vehicle, plate, color)
            VALUES (?, ?, ?, ?, ?)
        ''', (str(date_today), time_str, veh_type, plate, np_color))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving traffic log to SQLite: {e}")
        
    compare_plate(plate)


def predict(img):
    model = get_ocr_model()
    img = Image.fromarray(img)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    batch_t = batch_t.to(device)
    
    with torch.no_grad():
        out = model(batch_t)
        probs = torch.softmax(out, dim=1)
        
    prob, index = torch.max(probs, dim=1)
    char = labels[index.item()]
    confidence = prob.item()
    return char, confidence


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