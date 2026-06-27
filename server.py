import os
import sys
import time
import shutil
import base64
import sqlite3
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
import cv2

# Add current directory to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import character
import predict

app = FastAPI(title="澳门智能车牌识别系统 API Server")

# Serve static files
static_dir = ROOT_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    index_file = static_dir / "index.html"
    if not index_file.exists():
        return HTMLResponse(content="<h3>Index.html not found. Please create front-end.</h3>")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/blacklist")
def get_blacklist():
    try:
        conn = sqlite3.connect(character.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT plate, reason FROM suspected_plates ORDER BY plate ASC")
        rows = cursor.fetchall()
        conn.close()
        return [{"plate": row[0], "reason": row[1]} for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.post("/api/blacklist")
def add_blacklist(data: dict):
    plate = data.get("plate", "").strip().upper()
    reason = data.get("reason", "").strip()
    if not plate:
        raise HTTPException(status_code=400, detail="车牌号码不能为空")
    
    success = character.db_add_suspected(plate, reason)
    if success:
        character.load_suspected_cache()
        return {"status": "success", "message": f"车牌 {plate} 成功加入布控库"}
    else:
        raise HTTPException(status_code=500, detail="保存失败，请检查数据库状态")

@app.delete("/api/blacklist/{plate}")
def delete_blacklist(plate: str):
    cleaned_plate = plate.strip().upper()
    success = character.db_delete_suspected(cleaned_plate)
    if success:
        character.load_suspected_cache()
        return {"status": "success", "message": f"车牌 {cleaned_plate} 成功移出布控库"}
    else:
        raise HTTPException(status_code=500, detail="删除失败，请检查数据库状态")

@app.get("/api/logs")
def get_logs(plate: str = "", vehicle: str = "", limit: int = 100):
    try:
        conn = sqlite3.connect(character.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT plate FROM suspected_plates")
        suspected_set = {row[0].strip().upper() for row in cursor.fetchall()}
        
        query = "SELECT id, date, time, vehicle, plate, color FROM traffic_logs"
        params = []
        conditions = []
        if plate:
            conditions.append("plate LIKE ?")
            params.append(f"%{plate}%")
        if vehicle:
            conditions.append("vehicle = ?")
            params.append(vehicle)
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        logs = []
        for row in rows:
            logs.append({
                "id": row[0],
                "date": row[1],
                "time": row[2],
                "vehicle": row[3],
                "plate": row[4],
                "color": row[5],
                "is_suspected": row[4].strip().upper() in suspected_set
            })
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...), model: str = Form("yolo")):
    temp_dir = ROOT_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        generator = predict.run_inference(
            model_name=model,
            source=str(file_path),
            device=predict.get_device()
        )
        
        annotated_base64 = ""
        detections_result = []
        
        for path, frame, is_image, detections in generator:
            _, buffer = cv2.imencode('.jpg', frame)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            annotated_base64 = f"data:image/jpeg;base64,{b64_str}"
            detections_result = detections
            break  # Process single image
            
        if file_path.exists():
            file_path.unlink()
            
        return {
            "status": "success",
            "image": annotated_base64,
            "detections": detections_result
        }
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)):
    temp_dir = ROOT_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "success", "file_path": str(file_path)}

@app.get("/api/detect/video")
def stream_video(source_path: str, model: str = "yolo"):
    if not Path(source_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
        
    def video_frame_generator():
        try:
            generator = predict.run_inference(
                model_name=model,
                source=source_path,
                device=predict.get_device()
            )
            for path, frame, is_image, detections in generator:
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.03)  # control framerate
        except Exception as e:
            print(f"Streaming exception: {e}")
            
    return StreamingResponse(
        video_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    # Ensure cache is initialized
    character.load_suspected_cache()
    print("Starting server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
