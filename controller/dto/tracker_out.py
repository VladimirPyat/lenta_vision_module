from dataclasses import dataclass

import numpy as np

@dataclass()
class TrackedCrop:
    track_id: int
    crop: np.ndarray
    class_name: str
    frame_number: int
    # Делаем поля необязательными, по умолчанию они равны None.
    bbox: tuple[float, float, float, float] | None = None
    frame_timestamp: float | None = None


