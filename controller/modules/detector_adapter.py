from fileinput import filename

import cv2
import numpy as np
from typing import Optional, List
import logging

from controller.dto.tracker_out import TrackedCrop

logger = logging.getLogger(__name__)

class YoloToCropAdapter:
    def __init__(self, yolo_tracker_instance, crop_margin: int = 12, crop_rotation_angle: int = 90, save_crops=True):
        self.tracker = yolo_tracker_instance
        self.crop_margin = max(0, crop_margin)  # Защита от отрицательных отступов
        self.crop_rotation_angle = crop_rotation_angle % 360
        self._crop_buffer: list[TrackedCrop] = []
        self.save_crops = save_crops

    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        if angle == 0:
            return image

        try:
            if angle == 90:
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif angle == 180:
                return cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            M[0, 2] += (nW / 2) - center[0]
            M[1, 2] += (nH / 2) - center[1]

            return cv2.warpAffine(image, M, (nW, nH))

        except Exception as e:
            logger.error(f"Ошибка cv2 при повороте кропа на {angle} градусов: {e}", exc_info=True)
            return image  # Возвращаем оригинал, чтобы не ронять конвейер из-за угла

    def get_next(self) -> Optional[TrackedCrop]:
        """
        Отдает следующий кроп. Безопасно крутится в цикле, пропуская пустые кадры,
        пока не найдет ценник или пока не закончится видеопоток.
        """
        while True:
            # 1. Если в буфере есть кропы с прошлого кадра - отдаем их
            if self._crop_buffer:
                return self._crop_buffer.pop(0)

            # 2. Буфер пуст. Пробуем получить новый кадр с защитой от обрывов потока/IP-камер
            try:
                result = self.tracker.get_next_frame()
            except Exception as e:
                logger.error(f"Критический сбой источника видео/трекера: {e}", exc_info=True)
                return None

            if result is None:
                logger.info("Источник видео завершен. Адаптер отдает None.")
                return None

            frame, detections = result
            frame_idx = getattr(self.tracker, 'frame_idx', -1)
            tracker_ids = getattr(detections, "tracker_id", None)

            # 3. Если на кадре ничего нет, цикл while пойдет на следующий круг (БЕЗ рекурсии)
            if tracker_ids is None or len(detections) == 0:
                continue

            # 4. Нарезаем кропы с защитой от битых координат
            for i in range(len(detections)):
                try:
                    t_id = int(tracker_ids[i])
                    if t_id == -1:
                        continue

                    x1, y1, x2, y2 = map(int, detections.xyxy[i])
                    h, w = frame.shape[:2]

                    x1 = max(0, x1 - self.crop_margin)
                    y1 = max(0, y1 - self.crop_margin)
                    x2 = min(w, x2 + self.crop_margin)
                    y2 = min(h, y2 + self.crop_margin)

                    crop_img = frame[y1:y2, x1:x2]

                    if crop_img is None or crop_img.size == 0:
                        logger.warning(f"[Кадр {frame_idx}] Получен пустой кроп для ID {t_id}. Пропускаем.")
                        continue

                    if self.crop_rotation_angle != 0:
                        crop_img = self._rotate_image(crop_img, self.crop_rotation_angle)

                    # сохраняем кадры для отладки
                    if self.save_crops:
                        crop_filename = f"_img/{frame_idx}_{t_id}.jpg"
                        cv2.imwrite(crop_filename, crop_img)
                        logger.debug(f"сохраняем изображение ценника {crop_filename}")

                    # Забираем оригинальные дробные координаты от YOLO
                    raw_x1, raw_y1, raw_x2, raw_y2 = detections.xyxy[i]

                    # Считаем timestamp (в миллисекундах)
                    fps = max(self.tracker.src_fps, 1.0)
                    timestamp_ms = (frame_idx / fps) * 1000.0

                    tracked_crop = TrackedCrop(
                        track_id=t_id,
                        crop=crop_img,
                        class_name="price_tag",
                        frame_number=frame_idx,
                        bbox=(raw_x1, raw_y1, raw_x2, raw_y2),
                        frame_timestamp=timestamp_ms
                    )
                    self._crop_buffer.append(tracked_crop)

                except Exception as e:
                    # Если один ценник сломался, логируем и идем к следующему ценнику
                    logger.error(f"[Кадр {frame_idx}] Ошибка обработки кропа для ID {tracker_ids[i]} на кадре {frame_idx}: {e}", exc_info=True)
                    continue

