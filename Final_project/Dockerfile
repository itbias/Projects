FROM python:3.10-slim

# Отключаем буферизацию логов
ENV PYTHONUNBUFFERED=1

# Рабочая директория внутри контейнера
WORKDIR /app

# Обновляем pip
RUN pip install --upgrade pip

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Порт Streamlit
EXPOSE 8501

# Команда по умолчанию
CMD ["streamlit", "run", "app.py"]
