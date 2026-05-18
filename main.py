import os
import yaml
import logging
import pandas as pd
from pathlib import Path
import shutil


from controller.controller_main import run_pipeline
from controller.modules.detector_adapter import YoloToCropAdapter
from detection.video_gen_tracker import YOLOTracker
from vision.vision_main import vision_main

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger("Main")

# ==========================================
# ЭТАЛОННАЯ СХЕМА CSV
# ==========================================
SUBMISSION_COLUMNS = [
    "filename", "product_name", "price_default", "price_card", "price_discount",
    "barcode", "discount_amount", "id_sku", "print_datetime", "code",
    "additional_info", "color", "special_symbols", "frame_timestamp",
    "x_min", "y_min", "x_max", "y_max", "qr_code_barcode", "price1_qr",
    "price2_qr", "price3_qr", "price4_qr", "wholesale_level_1_coun",
    "wholesale_level_1_price", "wholesale_level_2_count",
    "wholesale_level_2_price", "action_price_qr", "action_code_qr"
]

def prepare_debug_folder(folder_name: str = "_img", clear_if_exists: bool = True):
    """
    Создает папку для отладки.
    Если папка уже есть и clear_if_exists=True — полностью очищает её.
    """
    debug_dir = Path(folder_name).resolve()

    if debug_dir.exists() and clear_if_exists:
        try:
            shutil.rmtree(debug_dir)  # Удаляем папку со всем содержимым
        except Exception as e:
            logging.getLogger("Main").warning(f"Не удалось очистить папку {folder_name}: {e}")

    # Создаем чистую папку
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir

def load_config(config_path: str = "config.yaml") -> dict:
    """Загружает YAML и сплющивает его в 1D словарь с абсолютными путями."""
    BASE_DIR = Path(__file__).resolve().parent
    yaml_file = BASE_DIR / config_path


    with open(yaml_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Делаем пути абсолютными
    if "paths" in config:
        for k, v in config["paths"].items():
            if v and isinstance(v, str) and not v.isdigit():
                config["paths"][k] = str(BASE_DIR / v)

    # Сплющиваем словарь (flat dict)
    flat_config = {}
    for section, values in config.items():
        if isinstance(values, dict):
            flat_config.update(values)

    return flat_config

def process_video_task(video_path: str, rotation_angle: int, output_csv_path: str):
    logger.info(f"Начинаем задачу. Видео: {video_path}, Угол: {rotation_angle}")
    config = load_config("config.yaml")
    
    # создание или очистка папки для сохранения кропов
    prepare_debug_folder("_img", clear_if_exists=True)
    logger.debug("📁 Отладочная папка '_img' очищена и готова к работе.")

    skip_val = config.get("skip_frames", 0)
    logger.info(f"⚙️ Настройки трекера: пропускаем {skip_val} кадров (обрабатываем каждый {skip_val + 1}-й)")

    live_yolo_tracker = YOLOTracker(
        model_path=config.get("tracker_model", "best.pt"),
        video_path=video_path,
        conf=config.get("conf", 0.4),
        iou=config.get("iou", 0.45),
        imgsz=config.get("imgsz", 640),
        skip=config.get("skip_frames", 0)
    )

    adapter = YoloToCropAdapter(
        yolo_tracker_instance=live_yolo_tracker,
        crop_margin=config.get("crop_margin", 15),
        crop_rotation_angle=rotation_angle,
        save_crops=config.get("save_crops", False)
    )

    df = run_pipeline(tracker_module=adapter, vision_callable=vision_main, config_dict=config)

    video_filename = Path(video_path).name
    if not df.empty:
        df["filename"] = video_filename
        for col in SUBMISSION_COLUMNS:
            if col not in df.columns: df[col] = "нет"

        df = df[SUBMISSION_COLUMNS]
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"🏆 Задача завершена. Результат: {output_csv_path}")
    else:
        logger.warning("⚠️ Таблица пуста. Создаем пустой файл шаблона.")
        empty_df = pd.DataFrame(columns=SUBMISSION_COLUMNS)
        empty_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")


# =====================================================================
# ТОЧКА ЗАПУСКА ДЛЯ ТЕСТИРОВАНИЯ В IDE
# =====================================================================
if __name__ == "__main__":
    # Этот блок сработает ТОЛЬКО если ты запустишь этот файл вручную.
    test_config = load_config("config.yaml")
    test_video_path = test_config.get("video_source", "_download/43_15.mp4")

    # 1. Забираем путь к выходной папке из конфига (если нет - берем "outputs")
    out_dir_str = test_config.get("output_dir", "outputs")
    out_dir = Path(out_dir_str).resolve()

    # 2. Создаем директорию, включая родительские (parents=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Формируем финальный абсолютный путь к CSV-файлу
    out_csv_path = out_dir / "submission_local_test.csv"

    logging.getLogger("Main").info(f"📂 Папка для сохранения результатов: {out_dir}")

    # 4. Вызываем ядро с тестовыми параметрами
    process_video_task(
        video_path=test_video_path,
        rotation_angle=test_config.get("crop_rotation_angle", 90),
        output_csv_path=str(out_csv_path) # Обязательно приводим Path к строке
    )