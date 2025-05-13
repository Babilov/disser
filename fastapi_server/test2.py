from ultralytics import YOLO
"""
# Загружаем модель
model = YOLO('/home/griwa/Рабочий стол/Dissertation/code/model/cars_detection/runs/detect/jeston/weights/best.pt')

# Переводим модель в режим FP16 для ускорения
model.half()

# Обработка видео
result = model.track(
    source='/home/griwa/Рабочий стол/Dissertation/code/client-server-app/client/videos/road2.mp4',
    tracker='bytetrack.yaml',
    device=0,
    imgsz=1280,
    show=True,
    half=True  # использование FP16
)
"""
model = YOLO('/home/griwa/Рабочий стол/Dissertation/code/model/cars_detection/runs/detect/train9/weights/best.pt')

result = model.track(
    source='/home/griwa/Рабочий стол/Dissertation/code/test-client-server-app/client/videos/road1.mp4',
    tracker='bytetrack.yaml',
    device=0,
    imgsz=1280, 
    show=True,
)