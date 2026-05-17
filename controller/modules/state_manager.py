import logging
import pandas as pd

logger  = logging.getLogger(__name__)

class StateManager:
    """
    Управляет жизненным циклом отслеживаемых объектов (треков) в оперативной памяти.

    Жизненный цикл (status):
    1. 'ACTIVE':  Новый трек. Объект находится в кадре (или недавно пропал).
                  В этом состоянии мы копим для него JSON-ответы от модуля Vision.
    2. 'DONE':    Трек успешно закрыт. Мы собрали нужное количество распознаваний (threshold),
                  провели голосование и сохранили идеальный результат в final_data.
                  Больше мы этот объект не обрабатываем.
    3. 'DROPPED': Трек забракован. Объект надолго пропал из кадра (timeout),
                  а мы так и не смогли собрать для него четкие данные.
                  Больше мы этот объект не обрабатываем.
    """
    def __init__(self, threshold: int = 3, timeout: int = 20):
        self.memory_state = {}
        self.threshold = threshold
        self.timeout = timeout

    def is_processed(self, track_id: int) -> bool:
        """Проверяет, нужно ли еще работать с этим ID."""
        return self.memory_state.get(track_id, {}).get("status") in ["DONE", "DROPPED"]

    def add_result(self, track_id: int, frame_num: int, vision_result: dict):
        """Обновляет время последней активности и добавляет результат, если он есть."""
        if track_id not in self.memory_state:
            self.memory_state[track_id] = {
                "status": "ACTIVE",
                "last_seen_frame": frame_num,
                "success_count": 0,
                "temp_results": [],
                "final_data": None
            }
            logger.debug(f"[Кадр {frame_num}] Создан новый трек ID {track_id} (ACTIVE)")

        self.memory_state[track_id]["last_seen_frame"] = frame_num

        # Если Vision вернул processed: True
        if vision_result and vision_result.get("processed"):
            self.memory_state[track_id]["temp_results"].append(vision_result.get("payload", {}))
            self.memory_state[track_id]["success_count"] += 1
            logger.debug(
                f"[Кадр {frame_num}] Трек {track_id} +1 успешный кадр "
                f"(Собрано: {self.memory_state[track_id]['success_count']})"
            )

    def get_ready_to_close(self, current_frame: int) -> list[int]:
        """Ищет активные треки, которые пора закрывать (по лимиту кадров или по таймауту)."""
        ready_ids = []
        for t_id, data in self.memory_state.items():
            if data["status"] != "ACTIVE":
                continue

            is_threshold_met = data["success_count"] >= self.threshold
            is_timed_out = (current_frame - data["last_seen_frame"]) > self.timeout

            if is_threshold_met or is_timed_out:
                ready_ids.append(t_id)
                reason = "Набран лимит" if is_threshold_met else "Таймаут (ушел из кадра)"
                logger.debug(f"[Кадр {current_frame}] Трек {t_id} готов к закрытию. Причина: {reason}")

        return ready_ids

    def save_final(self, track_id: int, final_data: dict, status: str = "DONE"):
        """Фиксирует статус и результат, очищает временный массив для экономии памяти."""
        self.memory_state[track_id]["status"] = status
        self.memory_state[track_id]["final_data"] = final_data
        self.memory_state[track_id]["temp_results"] = [] # Очищаем сырые данные
        logger.debug(f"Трек {track_id} зафиксирован со статусом {status}")

    def export_to_dataframe(self) -> pd.DataFrame:
        """Выгружает только успешные результаты."""
        done_tracks = {
            k: v["final_data"]
            for k, v in self.memory_state.items()
            if v["status"] == "DONE" and v["final_data"]
        }
        logger.debug(f"Сформирован DataFrame. Успешных записей: {len(done_tracks)}")
        return pd.DataFrame.from_dict(done_tracks, orient='index')