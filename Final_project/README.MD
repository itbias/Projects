# US Oil & Gas Production Forecasting
- Проект посвящён анализу и прогнозированию месячных объёмов добычи нефти и газа в США на федеральных и офшорных территориях на основе данных ONRR (Office of Natural Resources Revenue).
- Решение построено как воспроизводимый аналитический и ML-пайплайн, готовый к запуску в Docker и использованию через интерактивный dashboard.

## 🎯 Цели проекта
- Провести EDA и регуляторно-экономический анализ добычи
- Проверить гипотезу о роли `Disposition Description` и `Commodity`
- Сравнить классические эконометрические модели и ML-подходы
- Построить устойчивую модель прогнозирования
- Подготовить воспроизводимый production-ready пайплайн

## 🧠 Используемые модели
- VAR / VARX (базовый эконометрический анализ)
- Exponential Smoothing (baseline)
- Prophet (основная модель)
- CatBoost / LightGBM / XGBoost (классификация)

## 📁 Структура проекта
```text
oil-gas-forecasting/
│
├── app.py                  # Streamlit dashboard
├── main.py                 # Pipeline обучения / оценки моделей
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── predictions/
│   └── metrics/
│
├── configs/
│   ├── prophet_oil.yaml
│   └── prophet_gas.yaml
│
├── notebooks/
│   ├── US Oil & Gas P & D (EDA).ipynb
│   ├── US Oil & Gas P & D (Parsing).ipynb
│   ├── US Oil & Gas P & D (Classification).ipynb
│   └── US Oil & Gas P & D (Modeling).ipynb
│
└── src/
    ├── evaluation/
    ├── models/
    ├── pipelines/
    ├── preprocessing/
    └── utils/
```

### Особенность анализа и моделирования временных рядов
В аналитическом блоке (US Oil & Gas P & D (Modeling).ipynb) моделирование проводится на данных после структурного сдвига (2022-07), что позволяет сравнивать модели в актуальном экономическом режиме.

В production-пайплайне (main.py) модели обучаются на всей доступной истории с целью максимального использования информации и демонстрации промышленного подхода, при этом качество оценивается через backtest.

## 🚀 Как запустить проект

### Требования
- Docker
- Docker Compose

### 🔧 Сборка контейнера
```bash
docker compose build
```

### ▶️ Запуск dashboard и pipeline
```bash
docker compose up
```

После запуска:
- Streamlit dashboard будет доступен по адресу
👉 http://localhost:8501
- Modeling pipeline автоматически выполняется в отдельном контейнере
(обучение моделей, оптимизация, сохранение прогнозов и метрик)

### ▶️ Запуск pipeline отдельно (опционально)
```bash
docker compose run pipeline python main.py
```

## 📊 Dashboard Preview

![Dashboard demo](https://github.com/itbias/Projects/blob/main/Final_project/assets/dashboard.gif)

### 📊 Что показывает dashboard
1. Историческая динамика и качество модели
- Фактические данные
- Backtest прогноз
- Метрики качества (MAE / RMSE / MAPE)
- Визуализация структурного сдвига (2022-07)
2. Прогноз (планирование)
- Выбор горизонта прогноза
- Прогноз только на будущие даты
- Таблица прогнозных значений

### 🧾 Основные выводы
- VAR / VARX нестабильны на коротком пост-2022 периоде
- Exponential Smoothing показал себя как сильный baseline
- Prophet продемонстрировал наилучшее качество и устойчивость
- Экзогенные факторы не дали статистически значимого улучшения
- Структурный сдвиг 2022-07 критически важен для корректного моделирования

### 🧩 Технологический стек
- Python 3.10
- pandas, numpy, scikit-learn, statsmodels
- Prophet + Optuna
- Streamlit + Altair
- Docker / Docker Compose

### 📌 Статус проекта
- Проект выполнен в рамках итоговой аналитической работы и реализован в формате production-ready ML/DS приложения.
