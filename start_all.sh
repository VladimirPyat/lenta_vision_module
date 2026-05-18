#!/bin/bash

# Настраиваем красивый цветной вывод
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}🚀 Запускаем Llama Cpp Server...${NC}"
source .venv/bin/activate
# Переходим в папку сервера и запускаем в фоне (символ &)
cd /work/lenta_cv/vision/cpp_llm

python3 -m llama_cpp.server --config_file server_config.json &
LLM_PID=$!

# Даем серверу пару секунд на поднятие
sleep 3

echo -e "${GREEN}🎬 Запускаем FastAPI Оркестратор...${NC}"
cd /work/lenta_cv/
# Запускаем uvicorn (предполагая, что твой файл app.py и объект app)
uvicorn app:app --host 0.0.0.0 --port 8080 &
API_PID=$!

# Магия для чистого закрытия портов при Ctrl+C
trap "echo 'Останавливаем сервисы...'; kill $LLM_PID $API_PID; exit" SIGINT SIGTERM

# Ждем завершения процессов
wait