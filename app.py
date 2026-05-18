import os
import sys
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import logging

from main import process_video_task, load_config

app = FastAPI(title="Lenta CV Web")
templates = Jinja2Templates(directory="templates")


API_CONFIG = load_config("config.yaml")

# Достаем уровень логирования (по умолчанию INFO, если забыли указать в конфиге)
log_level_str = API_CONFIG.get("log_level", "INFO").upper()
numeric_level = getattr(logging, log_level_str, logging.INFO)

#  Настраиваем корневой логгер для веб-сервера
logging.basicConfig(
    level=numeric_level,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        # Пишем в файл (обязательно utf-8, чтобы русские названия ценников не сломались)
        logging.FileHandler("api.log", encoding="utf-8"),
        # И дублируем в консоль
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

logger = logging.getLogger("WebAPI")

# (Опционально) Глушим спам от сторонних библиотек, если мы не в DEBUG режиме
if numeric_level != logging.DEBUG:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)

logger.info(f"🌐 Сервер запущен. Уровень логирования: {log_level_str}")

UPLOAD_DIR = Path(API_CONFIG.get("upload_dir", "_download")).resolve()
OUTPUT_DIR = Path(API_CONFIG.get("output_dir", "outputs")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Хранилище состояний задач
TASKS_DB = {}

def background_runner(task_id: str, video_path: str, rotation_angle: int, csv_path: str):
    """Фоновая задача, которая обновляет свой статус"""

    # Эта функция будет вызываться Оркестратором на каждом кадре
    def update_progress(frame_number):
        TASKS_DB[task_id]["current_frame"] = frame_number

    try:
        TASKS_DB[task_id]["status"] = "processing"
        # Передаем наш коллбэк в ядро
        process_video_task(video_path, rotation_angle, csv_path, progress_cb=update_progress)
        TASKS_DB[task_id]["status"] = "done"
    except Exception as e:
        TASKS_DB[task_id]["status"] = "error"
        TASKS_DB[task_id]["error"] = str(e)
        logger.error(f"Ошибка в задаче {task_id}: {e}", exc_info=True)


# --- ЭНДПОИНТ 1: Рендер главной страницы ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(name="index.html", request=request)


# --- ЭНДПОИНТ 2: Загрузка видео ---
@app.post("/api/recognize")
async def start_recognition(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...),
        rotation_angle: int = Form(90)
):
    task_id = str(uuid.uuid4())
    video_ext = Path(video.filename).suffix
    video_path = UPLOAD_DIR / f"{task_id}{video_ext}"

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    csv_path = OUTPUT_DIR / f"{task_id}_result.csv"

    TASKS_DB[task_id] = {
        "status": "starting",
        "current_frame": 0,
        "csv_path": str(csv_path)
    }

    background_tasks.add_task(
        background_runner, task_id, str(video_path), rotation_angle, str(csv_path)
    )
    return {"task_id": task_id}


# --- ЭНДПОИНТ 3: Callback/Статус для фронта ---
@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    task = TASKS_DB.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return task


# --- ЭНДПОИНТ 4: Выгрузка CSV ---
@app.get("/api/download/{task_id}")
async def download_result(task_id: str):
    task = TASKS_DB.get(task_id)
    if task and task["status"] == "done" and os.path.exists(task["csv_path"]):
        return FileResponse(
            path=task["csv_path"],
            filename="lenta_submission.csv",
            media_type="text/csv"
        )
    return JSONResponse(status_code=400, content={"error": "Файл не готов или не найден"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)