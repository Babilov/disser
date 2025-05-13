import cv2
import numpy as np
import asyncio
import json
from fastapi import FastAPI, WebSocket
from ultralytics import YOLO
from typing import List

app = FastAPI()

# Загружаем модель YOLO
model = YOLO("/home/griwa/Рабочий стол/Dissertation/code/model/cars_detection/runs/detect/train9/weights/best.pt").cuda()

# Словарь с ROI для каждого клиента (по ID WebSocket)
client_rois = {}

def point_in_any_roi(x, y, roi_list: List[List[List[int]]]) -> bool:
    """Проверка, находится ли точка внутри хотя бы одного ROI."""
    for roi in roi_list:
        contour = np.array(roi, dtype=np.int32)
        if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
            return True
    return False

async def receive_rois(websocket: WebSocket, client_id: int):
    while True:
        try:
            message = await websocket.receive_text()
            rois = json.loads(message)
            client_rois[client_id] = rois
            print(f"ROI обновлены для клиента {client_id}: {rois}")
        except Exception as e:
            print(f"Ошибка при приёме ROI: {e}")
            break

async def video_stream(websocket: WebSocket, client_id: int):
    await websocket.accept()

    roi_task = asyncio.create_task(receive_rois(websocket, client_id))
    video_path = '/home/griwa/Рабочий стол/Dissertation/code/test-client-server-app/client/videos/road1.mp4'
    
    try:
        # YOLO сам открывает и 
        results_gen = model.track(
            source=video_path,
            tracker='bytetrack.yaml',
            stream=True,
            imgsz=1280,
            device=0,
            show=False,
            verbose=False
        )

        for result in results_gen:
            frame = result.orig_img.copy()

            rois = client_rois.get(client_id, [])
            if rois:
                for roi in rois:
                    cords = roi["cords"]
                    cv2.polylines(frame, [np.array(cords, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    track_id = int(box.id[0]) if box.id is not None else -1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'ID:{track_id} Class:{cls_id}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            _, buffer = cv2.imencode('.jpg', frame)
            await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0)

    except Exception as e:
        print(f"Ошибка YOLO при обработке видео: {e}")

    finally:
        roi_task.cancel()
        client_rois.pop(client_id, None)
        await websocket.close()


@app.websocket("/video")
async def websocket_video(websocket: WebSocket):
    client_id = id(websocket)  # Уникальный ID клиента
    await video_stream(websocket, client_id)

# Запуск
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
