import os
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse

# Импортируем нашу функцию из main.py
from main import process_video_task

app = FastAPI(title="Lenta CV Pipeline API")

# Папки для временных файлов
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Простейшая "База данных" в оперативной памяти (для хакатона сойдет)
# Структура: {"task_id_123": {"status": "processing", "csv_path": "outputs/123.csv"}}
TASKS_DB = {}

def background_runner(task_id: str, video_path: str, rotation_angle: int, csv_path: str):
    """Обертка для безопасного запуска в фоне"""
    try:
        # Запускаем тяжелый процесс
        process_video_task(video_path, rotation_angle, csv_path)
        # Отмечаем статус как "готово"
        TASKS_DB[task_id]["status"] = "done"
    except Exception as e:
        TASKS_DB[task_id]["status"] = "error"
        TASKS_DB[task_id]["error"] = str(e)


@app.post("/api/recognize")
async def start_recognition(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...),
        rotation_angle: int = Form(90)  # По умолчанию 90, если фронт не прислал
):
    """Эндпоинт №1: Принимает файл, сохраняет, запускает фон и отдает ID"""
    task_id = str(uuid.uuid4())

    # Сохраняем загруженное видео
    video_ext = Path(video.filename).suffix
    video_path = UPLOAD_DIR / f"{task_id}{video_ext}"

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    csv_path = OUTPUT_DIR / f"{task_id}_result.csv"

    # Регистрируем задачу в нашей "БД"
    TASKS_DB[task_id] = {
        "status": "processing",
        "csv_path": str(csv_path)
    }

    # Запускаем тяжелую функцию в фоне! Бэкенд не блокируется.
    background_tasks.add_task(
        background_runner,
        task_id=task_id,
        video_path=str(video_path),
        rotation_angle=rotation_angle,
        csv_path=str(csv_path)
    )

    # Моментально отвечаем фронтенду
    return {"task_id": task_id, "message": "Видео принято в обработку"}


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """Эндпоинт №2: Фронтенд опрашивает его каждые 3 секунды"""
    task = TASKS_DB.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    return {"task_id": task_id, "status": task["status"]}


@app.get("/api/download/{task_id}")
async def download_result(task_id: str):
    """Эндпоинт №3: Скачивание готового CSV"""
    task = TASKS_DB.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    if task["status"] != "done":
        return JSONResponse(status_code=400, content={"error": "Task is not finished yet"})

    csv_path = task["csv_path"]
    if not os.path.exists(csv_path):
        return JSONResponse(status_code=500, content={"error": "CSV file missing"})

    # Отдаем файл как вложение
    return FileResponse(
        path=csv_path,
        filename=f"lenta_results_{task_id[:8]}.csv",
        media_type="text/csv"
    )

# Для локального запуска:
# uvicorn api:app --host 0.0.0.0 --port 8080 --reload