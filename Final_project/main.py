#main.py

import numpy as np
import pandas as pd
import joblib

from src.preprocessing.loader import load_raw_data
from src.preprocessing.preprocess import get_monthly_data, prepare_for_prophet
from src.utils.config import upload_config
from src.pipelines.forecast_pipeline import run_prophet, run_es
from src.evaluation.es_cv import es_cv
from src.evaluation.prophet_optuna import prophet_optuna
from src.evaluation.metrics import evaluate

# 1. Загрузка
df = load_raw_data(path="data/raw/OGORB.csv", delimiter=',')

# 2. Агрегация
oil_monthly, gas_monthly = get_monthly_data(df)

# 3. Приведение к формату prophet
oil_monthly = prepare_for_prophet(oil_monthly)
gas_monthly = prepare_for_prophet(gas_monthly)

# 4.1 Моделирование ES (oil)
alpha_oil = es_cv(oil_monthly, alphas=np.arange(0.1, 1.0, 0.1), periods=24)
predictions_oil_es = run_es(oil_monthly, alpha_oil, periods=24)

# сохраняем бэктест
predictions_oil_es[:-24].to_csv(
    'data/predictions/backtest_oil_es.csv',
    index=False
)

# сохраняем прогноз
predictions_oil_es[-24:].to_csv(
    'data/predictions/forecast_oil_es.csv',
    index=False
)

# 4.2 Моделирование ES (gas)
alpha_gas = es_cv(gas_monthly, alphas=np.arange(0.1, 1.0, 0.1), periods=24)
predictions_gas_es = run_es(gas_monthly, alpha_gas, periods=24)

# сохраняем бэктест
predictions_gas_es[:-24].to_csv(
    'data/predictions/backtest_gas_es.csv',
    index=False
)

# сохраняем прогноз
predictions_gas_es[-24:].to_csv(
    'data/predictions/forecast_gas_es.csv',
    index=False
)

# 5.1 Моделирование Prophet (oil)
config_oil = prophet_optuna(oil_monthly, periods=24, n_trials=20, freq='M')
upload_config(config_oil, "configs/prophet_oil.yaml")

model_oil_prophet, predictions_oil_prophet = run_prophet(oil_monthly, config_oil, periods=24, return_model=True)

# сохраняем модель
joblib.dump(model_oil_prophet, 'src/models/oil_prophet.pkl')

# сохраняем бэктест
predictions_oil_prophet[:-24].to_csv(
    'data/predictions/backtest_oil_prophet.csv',
    index=False
)

# сохраняем прогноз
predictions_oil_prophet[-24:].to_csv(
    'data/predictions/forecast_oil_prophet.csv',
    index=False
)

# 5.2 Моделирование Prophet (gas)
config_gas = prophet_optuna(gas_monthly, periods=24, n_trials=20, freq='M')
upload_config(config_gas, "configs/prophet_gas.yaml")

model_gas_prophet, predictions_gas_prophet = run_prophet(gas_monthly, config_gas, periods=24, return_model=True)

# сохраняем модель
joblib.dump(model_gas_prophet, 'src/models/gas_prophet.pkl')

# сохраняем бэктест
predictions_gas_prophet[:-24].to_csv(
    'data/predictions/backtest_gas_prophet.csv',
    index=False
)

# сохраняем прогноз
predictions_gas_prophet[-24:].to_csv(
    'data/predictions/forecast_gas_prophet.csv',
    index=False
)

# 6. Оценка
pd.DataFrame([evaluate(
    oil_monthly['y'].values,
    predictions_oil_es[:-24]['yhat'].values
)]).to_csv('data/metrics/metrics_oil_es.csv', index=False)

pd.DataFrame([evaluate(
    gas_monthly['y'].values,
    predictions_gas_es[:-24]['yhat'].values
)]).to_csv('data/metrics/metrics_gas_es.csv', index=False)

pd.DataFrame([evaluate(
    oil_monthly['y'].values,
    predictions_oil_prophet[:-24]['yhat'].values
)]).to_csv('data/metrics/metrics_oil_prophet.csv', index=False)

pd.DataFrame([evaluate(
    gas_monthly['y'].values,
    predictions_gas_prophet[:-24]['yhat'].values
)]).to_csv('data/metrics/metrics_gas_prophet.csv', index=False)