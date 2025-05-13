from Roi import Roi
import cv2
import numpy as np
from shapely.geometry import Polygon
from Bbox import Bbox


class FrameProcessor:
    def __init__(self, window_name: str, size=(640, 640)):
        self.window_name = window_name
        self.size = size
        self.rois = []
        self.temp_coords = []
        self.index = 0
    
    def add_roi_point(self, x: int, y: int):
        """Добавить точку для формирования ROI."""
        self.temp_coords.append([x, y])
        if len(self.temp_coords) == 4:
            self.rois.append(Roi(self.temp_coords.copy(), self.index))
            self.index += 1
            self.temp_coords.clear()

    def remove_roi_at_point(self, x: int, y: int):
        """Удалить ROI, если точка (x, y) находится внутри."""
        for idx, roi in enumerate(self.rois):
            contour = np.array(roi.cords, dtype=np.int32)
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                del self.rois[idx]
                break

    def update_roi_counts(self, bboxes: list[Bbox]):
        for roi in self.rois:
            count = 0
            for bbox in bboxes:
                if bbox.cords:
                    if self._is_bbox_in_roi(bbox, roi):
                        count += 1
            roi.car_count = count
            # print(f"ROI {roi.index}: {roi.car_count} машин")
            

    def draw_overlays(self, frame: np.ndarray, bboxes: list):
        """Рисует ROI и боксы на кадре."""
        self.update_roi_counts(bboxes)
        resized = cv2.resize(frame, self.size)
        # Рисуем ROI
        for roi in self.rois:
            points = np.array(roi.cords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(resized, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            xs = [p[0] for p in roi.cords]
            ys = [p[1] for p in roi.cords]
            center = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))
            cv2.putText(resized, f"Count: {roi.car_count}", center, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
        # Рисуем боксы
        for bbox in bboxes:
            if bbox.cords:
                x1, y1, x2, y2 = map(int, bbox.cords)
                confidence = bbox.confidence
                label = f"{bbox.class_} ({confidence:.2f})"
                if hasattr(bbox, 'track_id'):
                    label += f" ID: {bbox.track_id}"
                color = (0, 0, 255)
                if self.rois:
                    for roi in self.rois:
                        if self._is_bbox_in_roi(bbox, roi, threshold=0.35):
                            color = (255, 0, 255)
                            if bbox.track_id not in roi.unique_cars_ids:
                                roi.unique_cars_ids.append(bbox.track_id)
                                print(roi.unique_cars_ids)
                            break
                cv2.rectangle(resized, (x1, y1), (x2, y2), color, 1)
                cv2.putText(resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
        return resized

    def _is_bbox_in_roi(self, bbox: Bbox, roi: Roi, threshold: float = 0.8) -> bool:
        x1, y1, x2, y2 = bbox.cords
        bbox_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        roi_polygon = Polygon(roi.cords)
        intersection_area = bbox_polygon.intersection(roi_polygon).area
        return (intersection_area / bbox_polygon.area) >= threshold
