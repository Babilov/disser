import cv2
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketState
import numpy as np
import asyncio
import json
from ultralytics import YOLO
import logging
import time
from collections import defaultdict
from database import database
from crud import *
import datetime

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
video_path = "rtsp://localhost:8554/mystream"
app = FastAPI()

detector = YOLO("detector.pt").cuda()
classifier = YOLO("classifier.pt").cuda()

client_rois = {}
track_class_cache = {}
client_db_ids = {}  # map client websocket id to db client.id

# Статистика по клиентам и ROI
client_stats = defaultdict(lambda: {
    "rois": defaultdict(lambda: {
        "last_seen_ids": set(),
        "density": 0,
        "intensity": 0,
        "last_update": time.time(),
        "db_roi_id": None,
    })
})

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

async def receive_rois(websocket: WebSocket, client_id: int):
    while True:
        try:
            message = await websocket.receive_text()
            rois = json.loads(message)
            client_rois[client_id] = rois
            logging.debug(f"ROI обновлены для клиента {client_id}: {rois}")
            for idx, roi in enumerate(rois):
                cords = roi.get("cords")
                if cords is not None:
                    existing_roi = await get_roi(client_db_ids[client_id], idx)
                    if not existing_roi:
                        # Вставляем новый ROI
                        roi_db = await create_roi(client_db_ids[client_id], idx, cords)
                        client_stats[client_id]["rois"][idx]["db_roi_id"] = roi_db["id"]
        except Exception as e:
            logging.error(f"[{client_id}] Ошибка при приёме ROI: {type(e).__name__}: {e}")
            break

async def video_stream(websocket: WebSocket, client_id: int):
    await websocket.accept()
    logging.debug(f"Подключение клиента с ID: {client_id} установлено.")
    if client_id not in client_db_ids:
            client = await create_client(str(client_id))
            client_db_ids[client_id] = client["id"]
            print('CREATED')

    roi_task = asyncio.create_task(receive_rois(websocket, client_id))
    cap = cv2.VideoCapture(video_path)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_results = []
            rois = client_rois.get(client_id, [])
            current_time = time.time()

            # Считаем плотность и интенсивность для каждого ROI
            y_offset = 30  # Начальная позиция по вертикали для текста
            line_height = 30  # Расстояние между строками

            for roi_index, roi in enumerate(rois):
                cords = roi["cords"]
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(cords, dtype=np.int32)], 255)

                x, y, w, h = cv2.boundingRect(np.array(cords, dtype=np.int32))
                roi_crop = frame[y:y+h, x:x+w]
                mask_crop = mask[y:y+h, x:x+w]
                roi_crop_masked = cv2.bitwise_and(roi_crop, roi_crop, mask=mask_crop)

                results = detector.track(roi_crop_masked, imgsz=1280, tracker='bytetrack.yaml', device=0, verbose=False)[0]
                current_ids = set()

                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    track_id = int(box.id[0]) if box.id is not None else -1

                    global_x1 = x1 + x
                    global_y1 = y1 + y
                    global_x2 = x2 + x
                    global_y2 = y2 + y
                    car_crop = frame[global_y1:global_y2, global_x1:global_x2]

                    current_ids.add(track_id)

                    if track_id in track_class_cache and track_class_cache[track_id][1] >= 0.85:
                        label = track_class_cache[track_id][0]
                    else:
                        result2 = classifier(car_crop)
                        label = result2[0].names[int(result2[0].probs.top1)]
                        track_class_cache[track_id] = [label, float(result2[0].probs.top1conf)]

                    frame_results.append(((global_x1, global_y1, global_x2, global_y2), cls_id, track_id, label))

                stats = client_stats[client_id]["rois"][roi_index]
                if current_time - stats["last_update"] >= 1.0:
                    new_ids = current_ids - stats["last_seen_ids"]
                    stats["intensity"] = len(new_ids)
                    stats["last_seen_ids"] = current_ids.copy()
                    stats["last_update"] = current_time
                stats["density"] = len(current_ids)

                db_roi_id = stats.get("db_roi_id")
                if db_roi_id:
                    await create_roi_stat(db_roi_id, stats["density"], stats["intensity"], timestamp=datetime.datetime.fromtimestamp(current_time))


                # Выводим статистику для каждого ROI на отдельной строке
                text = f"ROI {roi_index+1} - Density: {stats['density']} Intensity: {stats['intensity']}"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                y_offset += line_height

            # Отрисовка bbox и ROI
            for ((x1, y1, x2, y2), cls_id, track_id, label) in frame_results:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Model:{label} ID:{track_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            for roi in rois:
                cords = roi["cords"]
                cv2.polylines(frame, [np.array(cords, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

            _, buffer = cv2.imencode('.jpg', frame)
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_bytes(buffer.tobytes())
            else:
                break

            await asyncio.sleep(0)

    except Exception as e:
        logging.error(f"Ошибка обработки видео: {e}")
    finally:
        roi_task.cancel()
        client_rois.pop(client_id, None)
        client_stats.pop(client_id, None)
        cap.release()
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()

@app.websocket("/video")
async def websocket_video(websocket: WebSocket):
    client_id = id(websocket)
    logging.debug(f"Подключение клиента с ID: {client_id}")
    await video_stream(websocket, client_id)
