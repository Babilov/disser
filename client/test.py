import cv2
import websockets
import numpy as np
import asyncio
import threading
import json

class Roi:
    def __init__(self, cords, index):
        self.cords = cords  # [[x1, y1], [x2, y2], ...]
        self.index = index

    def to_dict(self):
        return {
            "cords": self.cords,
            "index": self.index
        }

class VideoStream:
    def __init__(self):
        self.temp_coords = []
        self.rois = []
        self.index = 0
        self.frame = None
        self.new_roi_added = False  # Флаг для отправки ROI

    def add_roi_point(self, x: int, y: int):
        self.temp_coords.append([x, y])
        if len(self.temp_coords) == 4:
            self.rois.append(Roi(self.temp_coords.copy(), self.index))
            self.index += 1
            self.temp_coords.clear()
            self.new_roi_added = True  # Активируем отправку

    def remove_roi_at_point(self, x: int, y: int):
        for idx, roi in enumerate(self.rois):
            contour = np.array(roi.cords, dtype=np.int32)
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                del self.rois[idx]
                self.new_roi_added = True  # Обновляем ROI
                break

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_roi_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.remove_roi_at_point(x, y)

    async def receive_frames(self, websocket):
        while True:
            try:
                frame_data = await websocket.recv()
                nparr = np.frombuffer(frame_data, np.uint8)
                self.frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except websockets.ConnectionClosed:
                print("Соединение закрыто сервером.")
                break

    async def send_rois(self, websocket):
        while True:
            await asyncio.sleep(0.1)
            if self.new_roi_added:
                roi_data = [roi.to_dict() for roi in self.rois]
                await websocket.send(json.dumps(roi_data))
                self.new_roi_added = False

    async def receive_video(self):
        uri = "ws://localhost:8000/video"
        async with websockets.connect(uri) as websocket:
            # Параллельный приём и отправка
            await asyncio.gather(
                self.receive_frames(websocket),
                self.send_rois(websocket)
            )

async def video_thread(video_stream):
    await video_stream.receive_video()

def start_video_stream(video_stream):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(video_thread(video_stream))

# Основной запуск
if __name__ == "__main__":
    video_stream = VideoStream()
    cv2.namedWindow("ROAD", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROAD", video_stream._mouse_callback)

    thread = threading.Thread(target=start_video_stream, args=(video_stream,))
    thread.daemon = True
    thread.start()

    while True:
        if video_stream.frame is not None:
            display_frame = video_stream.frame.copy()
            for roi in video_stream.rois:
                points = np.array(roi.cords, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imshow("ROAD", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
