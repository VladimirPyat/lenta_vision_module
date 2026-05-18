import logging
import time

import pandas as pd

from controller.modules.results_aggregator import Aggregator
from controller.modules.state_manager import StateManager

logger = logging.getLogger(__name__)


class ControllerPipeline:
    """
    Главный управляющий узел (Оркестратор) пайплайна.
    Его задача: запрашивать новые кадры у трекера, отправлять их на распознавание
    в модуль Vision, передавать результаты в менеджер состояний и вовремя
    закрывать обработку объектов.
    """

    def __init__(self, tracker_module, vision_callable, config_dict: dict = None, progress_cb=None):
        """
        Инициализация конвейера.

        :param tracker_module: Объект-источник данных (фейковый генератор или адаптер YOLO).
                               Обязан иметь метод get_next(), возвращающий TrackedCrop или None.
        :param vision_callable: Функция обработки изображения (твоя vision_main).
                                Принимает (crop_image, config_dict), возвращает dict.
        """
        self.tracker = tracker_module
        self.vision_main = vision_callable
        self.config = config_dict or {}
        # Сохраняем коллбэк как атрибут класса
        self.progress_cb = progress_cb

        # ==========================================
        # РАСПАКОВКА НАСТРОЕК
        # ==========================================
        # 1. Настройки для StateManager
        st_thresh = self.config.get("track_success_threshold", 3)
        st_time = self.config.get("track_timeout_frames", 20)

        # 2. Настройки для Aggregator
        agg_matches = self.config.get("min_matches_required", 2)
        agg_prio = self.config.get("price_fields_priority", ["price_card", "price_discount", "price_default"])

        # ==========================================
        # ИНИЦИАЛИЗАЦИЯ МОДУЛЕЙ
        # ==========================================
        # Передаем им конкретные значения, а не весь словарь
        self.state = StateManager(threshold=st_thresh, timeout=st_time)
        self.aggregator = Aggregator(
            target_frames_count=st_thresh,
            min_matches_required=agg_matches,
            price_fields_priority=agg_prio
        )


    def run(self):
        """
        Запускает бесконечный цикл обработки видеопотока.
        Прерывается только когда tracker_module возвращает None (конец видео).
        """
        logger.info("🚀 Запуск пайплайна...")

        try:
            while True:
                # 1. Запрашиваем следующий кадр
                try:
                    obj = self.tracker.get_next()
                except Exception as e:
                    logger.error(f"Критическая ошибка трекера при получении кадра: {e}", exc_info=True)
                    break

                # Если видео/картинки кончились - выходим из цикла
                if not obj:
                    logger.debug(" Источник данных иссяк. Завершаем обработку.")
                    break

                # Сообщаем API на каком мы кадре
                if self.progress_cb:
                    self.progress_cb(obj.frame_number)

                current_frame = obj.frame_number
                t_id = obj.track_id

                # 2. Быстрый фильтр: если ценник уже распознан, не тратим ресурсы VLM
                if self.state.is_processed(t_id):
                    continue

                # 3. Безопасный вызов модуля Vision (самое хрупкое место!)
                vision_result = {"processed": False, "description": "error", "data": None}
                try:
                    logger.debug(f"[Кадр {current_frame}] Отправляем ID {t_id} в Vision...")

                    # Вызываем переданную нам функцию vision_main
                    vision_result = self.vision_main(obj.crop, self.config)
                    if vision_result.get("processed") and "payload" in vision_result:
                        # Координаты (с защитой от None
                        bx = obj.bbox if obj.bbox is not None else (0.0, 0.0, 0.0, 0.0)
                        vision_result["payload"]["x_min"] = round(bx[0], 2)
                        vision_result["payload"]["y_min"] = round(bx[1], 2)
                        vision_result["payload"]["x_max"] = round(bx[2], 2)
                        vision_result["payload"]["y_max"] = round(bx[3], 2)

                        # Время
                        ts = obj.frame_timestamp if obj.frame_timestamp is not None else 0.0
                        vision_result["payload"]["frame_timestamp"] = round(ts, 2)

                    logger.info(f"Результат распознавания ID {t_id} : {vision_result}")

                except Exception as e:
                    # Если VLM упала, мы ловим ошибку, пишем в лог, но скрипт НЕ ПАДАЕТ
                    logger.error(f"❌ Ошибка внутри Vision для ID {t_id}: {e}", exc_info=True)

                # 4. Сохраняем результат в память
                self.state.add_result(t_id, current_frame, vision_result)

                # 5. Проверяем пороги (пора ли закрывать какие-то треки)
                self._check_and_close_tracks(current_frame)

        except KeyboardInterrupt:
            logger.warning("⏸ Работа прервана пользователем (Ctrl+C).")
        except Exception as e:
            logger.critical(f"💀 ГЛОБАЛЬНАЯ ОШИБКА ПАЙПЛАЙНА: {e}", exc_info=True)
        finally:
            # Блок finally выполнится ВСЕГДА.
            # Даже если скрипт упал с ошибкой, мы всё равно спасем данные,
            # которые успели распознать до момента падения!
            final_df = self._export_results()

        return final_df


    def _check_and_close_tracks(self, current_frame: int):
        """
        Приватный метод. Проверяет, есть ли треки, которые набрали нужное
        количество детекций или отвалились по таймауту, и проводит голосование.
        """
        try:
            ready_ids = self.state.get_ready_to_close(current_frame)

            for r_id in ready_ids:
                raw_results = self.state.memory_state[r_id]["temp_results"]

                if raw_results:
                    # Проводим мажоритарное голосование
                    final_data = self.aggregator.vote(raw_results)
                    self.state.save_final(r_id, final_data, status="DONE")
                    logger.info(f"✅ Трек {r_id} УСПЕШНО ЗАКРЫТ. Итог: {final_data}")
                else:
                    # Все кадры были мыльными, или модуль Vision падал с ошибками
                    self.state.save_final(r_id, None, status="DROPPED")
                    logger.warning(f"❌ Трек {r_id} брошен (нет успешных данных).")
        except Exception as e:
            logger.error(f"Ошибка при закрытии треков и голосовании: {e}", exc_info=True)

    def _export_results(self) -> pd.DataFrame:
        """
        Финальная выгрузка результатов. Вызывается при штатном завершении
        или при аварийном падении скрипта.
        """
        logger.info("📊 Формирование итогового отчета...")
        try:
            df = self.state.export_to_dataframe()
            logger.info("\nФИНАЛЬНАЯ ТАБЛИЦА:")
            if not df.empty:
                logger.debug(df.to_string())
                # Тут можно раскомментировать сохранение в файл:
                # df.to_csv("submission_results.csv", index_label="track_id")
            else:
                logger.warning("Таблица пуста. Нет успешно распознанных объектов.")

            return df

        except Exception as e:
            logger.error(f"Не удалось экспортировать результаты: {e}", exc_info=True)
            return pd.DataFrame() # Возвращаем пустышку в случае сбоя
