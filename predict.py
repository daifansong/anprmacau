import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch

# Add root folder to sys.path so we can import character
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import character
from experiments.adapters import YOLOv5Adapter, SSDAdapter, EfficientDetAdapter, RCNNAdapter


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_source_generator(source_path):
    p = Path(source_path)
    if p.is_file():
        suffix = p.suffix.lower()
        if suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']:
            yield p, cv2.imread(str(p)), True
        elif suffix in ['.mp4', '.avi', '.mov', '.mkv']:
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {source_path}")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield p, frame, False
            cap.release()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    elif p.is_dir():
        valid_files = sorted(
            [f for f in p.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']]
        )
        for img_p in valid_files:
            yield img_p, cv2.imread(str(img_p)), True
    else:
        # Check if camera index
        try:
            cam_idx = int(source_path)
            cap = cv2.VideoCapture(cam_idx)
            if not cap.isOpened():
                raise IOError(f"Cannot open webcam: {cam_idx}")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield Path("webcam.jpg"), frame, False
            cap.release()
        except ValueError:
            raise FileNotFoundError(f"Source path not found: {source_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified Macau ANPR Inference Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo",
        choices=["yolo", "ssd", "edet", "rcnn"],
        help="Select detection model model (yolo, ssd, edet, rcnn)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="images/021.jpg",
        help="Path to input image, video, directory of images, or camera index"
    )
    parser.add_argument(
        "--ocr-model",
        type=str,
        default=None,
        help="Path to custom OCR ResNet model weights (defaults to models/ocr.pth)"
    )
    parser.add_argument(
        "--suspected",
        type=str,
        default=None,
        help="Path to suspected plates list CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save annotated outputs"
    )
    args = parser.parse_args()

    # Configure character module paths if provided
    if args.ocr_model:
        # Pre-initialize or override search paths in character.py
        # We can pass model_path directly when predict is called or configure it globally
        pass
    if args.suspected:
        character.suspected_path = args.suspected

    device = get_device()
    print(f"Using device: {device}")

    # Load Adapter
    try:
        if args.model == "yolo":
            print("Initializing YOLOv5 Adapter...")
            adapter = YOLOv5Adapter(device)
        elif args.model == "ssd":
            print("Initializing SSD Adapter...")
            adapter = SSDAdapter(device)
        elif args.model == "edet":
            print("Initializing EfficientDet Adapter...")
            adapter = EfficientDetAdapter(device)
        elif args.model == "rcnn":
            print("Initializing Faster R-CNN Adapter...")
            adapter = RCNNAdapter(device)
    except Exception as e:
        print(f"Error initializing detector '{args.model}': {e}")
        print("Please check if the model repository is cloned and the weights exist.")
        sys.exit(1)

    # Initialize/warm up OCR model safely
    try:
        character.get_ocr_model(args.ocr_model)
    except Exception as e:
        print(f"Warning: OCR model initialization failed: {e}")
        print("You can still run vehicle/plate detection, but OCR character recognition will fail.")

    # Create directories
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    crops_dir = ROOT_DIR / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    vehicles_dir = ROOT_DIR / "vehicles"
    vehicles_dir.mkdir(parents=True, exist_ok=True)

    # Source Processing Loop
    start_time = time.perf_counter()
    
    # For video writing
    vid_writer = None
    last_vid_name = None

    try:
        generator = run_inference(
            model_name=args.model,
            source=args.source,
            ocr_model_path=args.ocr_model,
            suspected_csv_path=args.suspected,
            output_dir=args.output_dir,
            device=device
        )
        
        for path, frame, is_image, detections in generator:
            if is_image:
                out_filename = f"{args.model}_{path.name}"
                out_path = output_path / out_filename
                cv2.imwrite(str(out_path), frame)
                print(f"Saved annotated image to: {out_path}")
            else:
                # Video mode
                video_out_name = f"{args.model}_{path.stem}.mp4"
                if vid_writer is None or last_vid_name != video_out_name:
                    last_vid_name = video_out_name
                    if vid_writer is not None:
                        vid_writer.release()
                    save_path = str(output_path / video_out_name)
                    # Video properties
                    fps = 30.0
                    frame_w, frame_h = frame.shape[1], frame.shape[0]
                    # Try reading FPS from source video
                    cap_temp = cv2.VideoCapture(str(path))
                    if cap_temp.isOpened():
                        fps = cap_temp.get(cv2.CAP_PROP_FPS)
                        cap_temp.release()
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))
                vid_writer.write(frame)
                
            for det in detections:
                print(f"Detected Macau Plate: \033[1m{det['plate_str']}\033[0m (Color: {det['plate_color']}, Attribute: {det['attribute']})")

    except Exception as e:
        print(f"Inference execution failed: {e}")

    if vid_writer is not None:
        vid_writer.release()
        if last_vid_name:
            print(f"Saved annotated video to: {output_path / last_vid_name}")

    end_time = time.perf_counter()
    print(f"Done. Time elapsed: {end_time - start_time:.4f}s")


def run_inference(model_name, source, ocr_model_path=None, suspected_csv_path=None, output_dir="output", device=None):
    if suspected_csv_path:
        character.suspected_path = suspected_csv_path
        
    if device is None:
        device = get_device()
        
    # Load Adapter
    if model_name == "yolo":
        adapter = YOLOv5Adapter(device)
    elif model_name == "ssd":
        adapter = SSDAdapter(device)
    elif model_name == "edet":
        adapter = EfficientDetAdapter(device)
    elif model_name == "rcnn":
        adapter = RCNNAdapter(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    try:
        character.get_ocr_model(ocr_model_path)
    except Exception as e:
        print(f"Warning: OCR model initialization failed: {e}")
        
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    crops_dir = ROOT_DIR / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    vehicles_dir = ROOT_DIR / "vehicles"
    vehicles_dir.mkdir(parents=True, exist_ok=True)

    generator = get_source_generator(source)
    
    for path, frame, is_image in generator:
        if frame is None:
            continue
            
        im0 = frame.copy()
        h, w = frame.shape[:2]
        
        try:
            vehicles = adapter.detect_vehicles(frame)
        except Exception as e:
            print(f"Vehicle detection failed: {e}")
            vehicles = []
            
        detections = []
        veh_count = 0
        for veh_box, veh_conf, veh_class in vehicles:
            vx1, vy1, vx2, vy2 = veh_box
            vx1, vy1, vx2, vy2 = max(0, vx1), max(0, vy1), min(w, vx2), min(h, vy2)
            if vx2 <= vx1 or vy2 <= vy1:
                continue
                
            vehicle_crop = im0[vy1:vy2, vx1:vx2]
            if vehicle_crop.size == 0:
                continue
                
            veh_count += 1
            cv2.imwrite(str(vehicles_dir / f"{veh_class}_{veh_count:02d}.png"), vehicle_crop)
            
            try:
                plates = adapter.detect_plates(vehicle_crop)
            except Exception as e:
                print(f"Plate detection failed for vehicle: {e}")
                plates = []
                
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
            cv2.putText(frame, f"{veh_class} {veh_conf:.2f}", (vx1, max(vy1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
            for plate_box, plate_conf in plates:
                abs_plate_box = adapter.recalculate_plate_coords(plate_box, [vx1, vy1, vx2, vy2])
                px1, py1, px2, py2 = abs_plate_box
                px1, py1, px2, py2 = max(0, px1), max(0, py1), min(w, px2), min(h, py2)
                if px2 <= px1 or py2 <= py1:
                    continue
                    
                plate_crop = im0[py1:py2, px1:px2]
                if plate_crop.size == 0:
                    continue
                    
                cv2.imwrite(str(crops_dir / f"{path.stem}_plate.jpg"), plate_crop)
                
                license_plate_str = ""
                char_color = "Indeterminated"
                attribute = "normal"
                is_suspected = False
                try:
                    plate_chars, char_color, attribute = character.segment(plate_crop)
                    license_plate_str = "".join(str(v) for v in plate_chars)
                    character.csv_related(license_plate_str, veh_class, char_color)
                    is_suspected = character.compare_plate(license_plate_str)
                except Exception as e:
                    print(f"OCR/CSV processing failed for plate: {e}")
                    
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                plate_label = f"NP: {license_plate_str} | {char_color} | {attribute}"
                cv2.putText(frame, plate_label, (px1, max(py1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                detections.append({
                    "vehicle_class": veh_class,
                    "vehicle_conf": veh_conf,
                    "plate_str": license_plate_str,
                    "plate_color": char_color,
                    "attribute": attribute,
                    "is_suspected": is_suspected
                })
                
        yield path, frame, is_image, detections


if __name__ == "__main__":
    main()
