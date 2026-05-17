import numpy as np
import logging

from controller.dto.tracker_out import TrackedCrop

logger = logging.getLogger(__name__)


class DetectionsToCropMapper:
    """ Приводим сырые объекты supervision к формату  кадров с id"""
    def __init__(self):
        self.frame_counter = 0


    def convert(self, frame: np.ndarray, detections) -> list[TrackedCrop]:
        self.frame_counter += 1
        crops_list = []

        try:
            if detections is None or len(detections) == 0 or detections.tracker_id is None:
                logger.debug(f"[Кадр {self.frame_counter}] Детекции не обнаружены")
                return crops_list

            for xyxy, t_id in zip(detections.xyxy, detections.tracker_id):
                x1, y1, x2, y2 = map(int, xyxy)

                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    logger.warning(f"[Кадр {self.frame_counter}] Пропущен битый bbox (ID: {t_id})")
                    continue

                crop_img = frame[y1:y2, x1:x2]
                crops_list.append(TrackedCrop(int(t_id), crop_img, "price_tag", self.frame_counter))

        except Exception as e:
            logger.error(f"[Кадр {self.frame_counter}] Ошибка при маппинге детекций: {e}", exc_info=True)

        return crops_list