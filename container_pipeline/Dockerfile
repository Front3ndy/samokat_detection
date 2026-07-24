# Используем официальный легкий образ Python
FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Сначала копируем только requirements.txt для кэширования
COPY requirements.txt .

# Устанавливаем Python зависимости (CPU-версии)
RUN pip install --no-cache-dir -r requirements.txt

# Копируем только необходимые файлы
COPY samoPipeline.py .

# Команда для запуска (с unbuffered output для логов)
CMD ["python", "-u", "samoPipeline.py"]