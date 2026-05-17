import logging

from controller.modules.controller import ControllerPipeline
from controller.modules.fake_crop_gen import CropFakeIdGen
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

logger = logging.getLogger("ControllerPipeline")

def run_pipeline(tracker_module, vision_callable, config_dict: dict = None):
    """
    Внешний интерфейс для запуска проверки всего конвейера.
    Принимает на вход путь к файлу (или заглушку папки), трекер, функцию vision и конфиг.
    """
    pipeline = ControllerPipeline(
        tracker_module=tracker_module,
        vision_callable=vision_callable,
        config_dict=config_dict
    )
    return pipeline.run()

if __name__ == '__main__':
    config = {"cpp_folder" : "../vision/cpp_llm"}
    tracker = CropFakeIdGen("_img")
    run_pipeline(
        tracker_module=tracker,
        vision_callable=vision_main,
        config_dict=config
    )