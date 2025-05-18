import cv2
import websockets
import numpy as np
import asyncio
import threading
import json
from ROI import Roi

class VideoStream:
    def __init__(self):
        self.temp_coords = []
        self.rois = []
        self.index = 0
        self.frame = None
        self.loop = None
        self.running = True
        self.websocket = None
        self.new_roi_added = False
        self.paused = False

    def add_roi_point(self, x: int, y: int):
        self.temp_coords.append([x, y])
        if len(self.temp_coords) == 4:
            self.rois.append(Roi(self.temp_coords.copy(), self.index))
            self.index += 1
            self.temp_coords.clear()
            self.new_roi_added = True

    def remove_roi_at_point(self, x: int, y: int):
        for idx, roi in enumerate(self.rois):
            contour = np.array(roi.cords, dtype=np.int32)
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                del self.rois[idx]
                self.new_roi_added = True
                break

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_roi_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.remove_roi_at_point(x, y)

    async def stop(self):
        self.running = False
        if self.websocket and not self.websocket.close:
            await self.websocket.close()

    async def connect(self, uri="ws://localhost:8000/video"):
        try:
            self.websocket = await websockets.connect(uri)
        except Exception as e:
            print(f"Ошибка подключения: {e}")
            self.running = False

    async def receive_frames(self):
        try:
            while self.running:
                frame_data = await self.websocket.recv()
                try:
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None and not self.paused:
                        self.frame = frame
                except Exception as e:
                    print(f"Ошибка декодирования кадра: {e}")
        except websockets.ConnectionClosed:
            print("Соединение закрыто сервером.")
        except Exception as e:
            print(f"Ошибка получения кадра: {e}")
        finally:
            self.running = False
            if self.websocket and not self.websocket.close:
                await self.websocket.close()

    async def send_rois(self):
        try:
            while self.running:
                await asyncio.sleep(0.1)
                if not self.paused and self.new_roi_added:
                    roi_data = [roi.to_dict() for roi in self.rois]
                    await self.websocket.send(json.dumps(roi_data))
                    self.new_roi_added = False
        except Exception as e:
            print(f"Ошибка отправки ROI: {e}")

    async def run(self):
        await self.connect()
        if self.websocket:
            await asyncio.gather(
                self.receive_frames(),
                self.send_rois()
            )
    
    def draw_rois(self, display_frame):
        for roi in self.rois:
            points = np.array(roi.cords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(display_frame, str(roi.index), tuple(roi.cords[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
    def draw_points(self, display_frame):
        for pt in self.temp_coords:
            cv2.circle(display_frame, tuple(pt), 4, (255, 0, 0), -1)


def video_thread(video_stream):
    loop = asyncio.new_event_loop()
    video_stream.loop = loop
    asyncio.set_event_loop(loop)
    loop.run_until_complete(video_stream.run())


if __name__ == "__main__":
    video_stream = VideoStream()
    cv2.namedWindow("ROAD", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROAD", video_stream._mouse_callback)

    thread = threading.Thread(target=video_thread, args=(video_stream,))
    thread.start()

    try:
        while video_stream.running:
            if video_stream.frame is not None:
                display_frame = video_stream.frame.copy()

                # Рисуем ROI
                video_stream.draw_rois(display_frame)

                # Отображение временных точек
                video_stream.draw_points(display_frame)

                # Надпись "ПАУЗА"
                if video_stream.paused:
                    cv2.putText(display_frame, "ПАУЗА", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                cv2.imshow("ROAD", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                video_stream.running = False
                if video_stream.loop and video_stream.loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(video_stream.stop(), video_stream.loop)
                    future.result()
                break
            elif key == ord('p'):
                video_stream.paused = not video_stream.paused
            elif key == ord('c'):
                video_stream.rois.clear()
                video_stream.new_roi_added = True

    finally:
        thread.join()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
