from collections import Counter
from typing import Any
import logging

logger = logging.getLogger(__name__)

class Aggregator:
    """
    Модуль агрегации и мажоритарного голосования.
    Склеивает "грязные" результаты с нескольких кадров в один чистый эталонный JSON,
    а также оценивает уверенность (Confidence) в полученном результате.
    """
    def __init__(self, target_frames_count: int = 3, min_matches_required: int = 2,
                 price_fields_priority: list[str] = ("price_card", "price_discount", "price_default")):
        """
        :param target_frames_count: Целевое количество кадров (лимит), которое мы хотим собрать.
                                    (Сейчас напрямую не блокирует метод, но полезно для логики).
        :param min_matches_required: Сколько одинаковых значений цены должно совпасть,
                                     чтобы признать результат надежным ('good').
        """
        self.target_frames_count = target_frames_count
        self.min_matches_required = min_matches_required


        # Приоритет полей цен для проверки уверенности (от самого важного к базовому)
        self.price_fields_priority = price_fields_priority

    def _evaluate_confidence(self, raw_results: list[dict[str, Any]]) -> str:
        """
        Приватный метод для оценки надежности склеенных данных на основе совпадения цен.

        Логика работы:
        1. Если всего собрано кадров меньше, чем нужно для минимального кворума (min_matches_required),
           то результат автоматически помечается как 'uncertain'.
        2. Последовательно проверяем типы цен по приоритету (карта -> акция -> обычная).
        3. Если находим хотя бы одну цену, которая была распознана одинаково
           min_matches_required раз (или больше), считаем результат 'good'.
        4. Если ни одна из цен не набрала нужного количества совпадений,
           фиксируем конфликт данных — 'uncertain (conflict)'.
        """
        frames_count = len(raw_results)

        # Если кадров меньше, чем требуется для совпадения - мы физически не можем быть уверены
        if frames_count < self.min_matches_required:
            logger.debug(f"[Aggregator] Недостаточно кадров ({frames_count}) для оценки (min {self.min_matches_required})")
            return f"uncertain (only {frames_count} frames)"

        # Идем по полям цен в заданном порядке
        for price_key in self.price_fields_priority:
            # Вытаскиваем все непустые значения для текущего типа цены
            values = [res.get(price_key) for res in raw_results if res.get(price_key) is not None]

            # Если хотя бы попытались распознать это поле нужное количество раз
            if len(values) >= self.min_matches_required:
                # Находим, сколько раз встречается самое частое (популярное) значение
                most_common_count = Counter(values).most_common(1)[0][1]

                # Если самое популярное значение встречается >= min_matches_required раз
                if most_common_count >= self.min_matches_required:
                    logger.debug(f"[Aggregator] Найдено {most_common_count} совпадений для {price_key}")
                    return "good"

        # Если цикл завершился, а кворум совпадений ни по одной цене не набрался
        logger.debug(f"[Aggregator] Нет достаточного кворума для оценки {len(values)}")
        return "uncertain (conflict)"

    def vote(self, raw_results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Основной метод. Склеивает результаты мажоритарным голосованием по всем полям
        и добавляет метаданные (количество кадров и оценку уверенности).
        """
        if not raw_results:
            logger.debug("[Aggregator] Результат пуст")
            return {}

        final_result = {}

        # 1. Голосование по всем ключам (собираем эталонный JSON)
        all_keys = set().union(*(d.keys() for d in raw_results))
        for key in all_keys:
            values = [res.get(key) for res in raw_results if res.get(key) is not None]
            if values:
                # Проверяем тип ПЕРВОГО элемента в собранном списке
                if isinstance(values[0], (list, dict)):
                    # Если это список или словарь (нехэшируемый тип),
                    # голосовать бесполезно. Просто берем первый попавшийся.
                    final_result[key] = values[0]
                else:
                    # Нормальные числа и строки отдаем на голосование
                    final_result[key] = Counter(values).most_common(1)[0][0]
            else:
                final_result[key] = None

        # 2. Оценка уверенности (вызов приватного метода)
        final_result["_frames_analyzed"] = len(raw_results)
        final_result["_confidence"] = self._evaluate_confidence(raw_results)

        logger.debug(f"[Aggregator] Результат: {final_result}")
        return final_result