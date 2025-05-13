from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

model = YOLO('/home/griwa/Рабочий стол/Dissertation/code/model/cars_detection/runs/detect/train9/weights/best.pt')

roi_points = []
drawing = False
entered_ids = {}  # track_id -> time_entered

# Папка для сохранения bbox-кропов
save_dir = 'saved_bbox_crops'
os.makedirs(save_dir, exist_ok=True)
crop_counter = 0

def mouse_callback(event, x, y, flags, param):
    global roi_points, drawing, entered_ids
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
        drawing = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        roi_points.clear()
        entered_ids.clear()
        drawing = False

def is_inside_roi(bbox, roi_polygon):
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    if len(roi_polygon) >= 3:
        return cv2.pointPolygonTest(np.array(roi_polygon), (cx, cy), False) >= 0
    return False

cv2.namedWindow("ROI + YOLO Tracking")
cv2.setMouseCallback("ROI + YOLO Tracking", mouse_callback)

results = model.track(
    source='rtsp://localhost:8554/mystream',
    tracker='bytetrack.yaml',
    device='cuda',
    imgsz=1280,
    stream=True
)

start_time = time.time()

for result in results:
    frame = result.orig_img.copy()
    current_time = time.time()

    # Нарисовать ROI
    if len(roi_points) >= 2:
        pts = np.array(roi_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    inside_ids = set()

    if result.boxes is not None:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            track_id = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = map(int, xyxy)

            if is_inside_roi(xyxy, roi_points):
                inside_ids.add(track_id)

                if track_id not in entered_ids:
                    entered_ids[track_id] = current_time

                # Сохраняем bbox кроп
                crop = result.orig_img[y1:y2, x1:x2]
                crop_filename = os.path.join(save_dir, f'crop_{crop_counter:06d}_id{track_id}.jpg')
                cv2.imwrite(crop_filename, crop)
                crop_counter += 1

                # Отрисовка bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Статистика
    car_count = len(entered_ids)
    elapsed = current_time - start_time
    intensity = car_count / elapsed if elapsed > 0 else 0

    stats_text = [
        f"Current density: {len(inside_ids)}",
        f"Traffic intensity: {intensity:.2f} cars/sec"
    ]
    for i, line in enumerate(stats_text):
        cv2.putText(frame, line, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Показываем кадр
    cv2.imshow("ROI + YOLO Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
