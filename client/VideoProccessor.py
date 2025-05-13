import time
import cv2
import logging
import numpy as np
from ServerCommunicator import ServerCommunicator
from FrameProcessor import FrameProcessor
import websockets

# Настройка логирования
# logging.exception("Ошибка в работе видео процессора:")

class VideoProcessor:
    def __init__(self, video_path: str, window_name: str, server_url: str, delay: float = 1 / 30):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Не удалось открыть видео: {video_path}")
        self.window_name = window_name
        self.size = (1280, 1280)
        self.delay = delay
        self.frame_processor = FrameProcessor(window_name, self.size)
        self.communicator = ServerCommunicator(server_url)
        self.server_url = server_url
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.size)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.frame_processor.add_roi_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.frame_processor.remove_roi_at_point(x, y)

    def process(self):
        paused = False
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    logging.info("Конец видео или ошибка чтения кадра.")
                    break
                # Отправляем кадр на сервер для предсказания
                self.communicator.send_frame(frame)
            # Отрисовка
            display_frame = self.frame_processor.draw_overlays(frame, self.communicator.bboxes)
            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("Выход по запросу пользователя.")
                break
            elif key == ord(' '):
                paused = not paused
                logging.info("Пауза" if paused else "Продолжение")
            time.sleep(self.delay)
        self.cap.release()
        cv2.destroyAllWindows()
