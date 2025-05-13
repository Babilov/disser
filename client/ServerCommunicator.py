import time
import cv2
import requests
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from Bbox import Bbox

# logging.basicConfig(level=logging.INFO)

class ServerCommunicator:
    def __init__(self, url: str, send_interval: float = 0.5):
        self.url = url
        self.send_interval = send_interval
        self.last_sent_time = 0
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.bboxes = []

    def send_frame(self, frame: np.ndarray):
        """Отправка кадра на сервер с использованием пула потоков."""
        current_time = time.time()
        if current_time - self.last_sent_time < self.send_interval:
            return
        self.last_sent_time = current_time

        # Отправляем кадр в пул потоков
        self.executor.submit(self._post_frame, frame)

    def _post_frame(self, frame: np.ndarray):
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            # logging.error("Не удалось закодировать кадр в JPEG")
            return
        frame_data = buffer.tobytes()
        files = {'file': ('frame.jpg', frame_data, 'image/jpeg')}
        try:
            response = requests.post(self.url, files=files)
            response.raise_for_status()
            self.bboxes = [Bbox(bbox['box'], bbox['confidence'], bbox['class_'], bbox['track_id']) for bbox in response.json()]
            # print(self.bboxes)
            # logging.debug("Ответ от сервера: %s", self.bboxes)
        except requests.RequestException as e:
            # logging.error("Ошибка при отправке кадра: %s", e)
            print("Ошибка при отправке кадра: %s", e)
