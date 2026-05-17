from dataclasses import dataclass

import numpy as np


@dataclass
class TrackedCrop:
    track_id: int
    crop: np.ndarray
    class_name: str
    frame_number: int