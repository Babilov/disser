# Используем официальный образ PyTorch с CUDA (версия под твою CUDA, например 11.8)
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Обновим pip и установим зависимости Python
RUN pip install --upgrade pip

COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходники в контейнер
WORKDIR /app
COPY . /app

# Открываем порт FastAPI
EXPOSE 8000

# Запускаем приложение
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
