import cv2
import numpy as np
import os
import random
import logging

from controller.dto.tracker_out import TrackedCrop

logger = logging.getLogger(__name__)


class CropFakeIdGen:
    def __init__(self, image_dir: str, max_frames_per_object: int = 5, class_name: str = "price_tag"):
        """
        Имитатор модуля детекции и трекинга (Mock Tracker).
        Берет реальные картинки из папки и имитирует видеопоток с движущимся роботом.

        :param image_dir: Путь к папке с эталонными картинками (файлы должны называться 1.jpg, 2.jpg...).
        :param frames_per_object: Имитация времени жизни объекта в кадре.
            Пояснение: Робот едет мимо ценника не мгновенно. Ценник висит в кадре какое то время.
            Этот параметр заставляет генератор выдать N вариантов картинки 1.jpg (подряд)
            с одним и тем же track_id=1, прежде чем переключиться на 2.jpg с track_id=2.
        :param class_name: Имя класса, которое "как бы" предсказала нейросеть YOLO.
            Когда прикрутим реальную YOLO, она будет возвращать строки вроде
            'bottle', 'qr_code' или 'price_tag'. Здесь мы жестко задаем, кого имитируем.
        """
        self.image_dir = image_dir
        self.max_frames_per_object = max_frames_per_object
        self.class_name = class_name

        self.current_frame_global = 0
        self.current_track_id = 1
        self.frames_alive_for_current_id = 0

        # Динамический лимит для текущего объекта
        self.current_target_frames = 0

        # Кэш
        self._current_base_image: np.ndarray | None = None

    def _load_base_image(self) -> bool:
        """Загружает новую картинку и вычисляет, сколько кадров она проживет."""
        filename = os.path.join(self.image_dir, f"{self.current_track_id}.jpg")

        if not os.path.exists(filename):
            logger.warning(f"Файл {filename} не найден. Симуляция завершена.")
            return False

        try:
            img = cv2.imread(filename)
            if img is None:
                logger.error(f"Не удалось декодировать картинку: {filename}")
                return False

            self._current_base_image = img

            # Генерируем случайное время жизни для нового объекта (от 3 до максимума)
            self.current_target_frames = random.randint(3, self.max_frames_per_object)
            logger.debug(f"Загружен ID {self.current_track_id}. Он пробудет в кадре {self.current_target_frames} фреймов.")

            return True

        except Exception as e:
            logger.error(f"Ошибка чтения {filename}: {e}", exc_info=True)
            return False

    def _apply_augmentations(self) -> np.ndarray:
        """Берет базовую картинку из self и портит ее."""
        try:
            img = self._current_base_image
            h, w = img.shape[:2]

            # 1. Поворот
            angle = random.uniform(-3, 3)
            center = (w / 2, h / 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)
            aug_img = cv2.warpAffine(img, rot_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)

            # 2. Размытие
            if random.random() > 0.7:
                k_size = random.choice([3, 5])
                aug_img = cv2.GaussianBlur(aug_img, (k_size, k_size), 0)

            # 3. Яркость
            factor = random.uniform(0.9, 1.1)
            aug_img = cv2.convertScaleAbs(aug_img, alpha=factor, beta=0)

            return aug_img
        except Exception as e:
            logger.warning(f"Ошибка при аугментации: {e}")
            return self._current_base_image.copy()

    def get_next(self) -> TrackedCrop | None:
        if self._current_base_image is None or self.frames_alive_for_current_id == 0:
            success = self._load_base_image()
            if not success:
                return None

                # Вызываем без аргументов
        jittered_crop = self._apply_augmentations()

        result = TrackedCrop(
            track_id=self.current_track_id,
            crop=jittered_crop,
            class_name=self.class_name,
            frame_number=self.current_frame_global
        )

        self.current_frame_global += 1
        self.frames_alive_for_current_id += 1

        # Сравниваем со случайным таргетом, а не с фиксированным максимумом
        if self.frames_alive_for_current_id >= self.current_target_frames:
            logger.info(f"Объект ID {self.current_track_id} исчез после {self.frames_alive_for_current_id} кадров.")
            self.current_track_id += 1
            self.frames_alive_for_current_id = 0

        return result