# Берем легкий образ Python
FROM python:3.12-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /work/lenta_cv

# Ставим системные библиотеки, необходимые для работы OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Копируем список зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь остальной код проекта
COPY . .

# Открываем порт для веб-сервера
EXPOSE 8080

# Команда запуска FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]