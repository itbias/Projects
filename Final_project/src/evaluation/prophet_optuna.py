# src/evaluation/prophet_optuna.py

import optuna
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

def prophet_optuna(train, periods, n_trials, freq='M'):

    # Целевая функция для Optuna
    def objective(trial):

        params = {
            'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
            'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
            'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 100, log=True),
            'changepoint_range': trial.suggest_float('changepoint_range', 0.8, 0.95),
            'n_changepoints': trial.suggest_int('n_changepoints', 10, 50),
            'yearly_seasonality': trial.suggest_categorical('yearly_seasonality', [True]),
            'weekly_seasonality': trial.suggest_categorical('weekly_seasonality', [False]),
            'daily_seasonality': trial.suggest_categorical('daily_seasonality', [False])
        }
        
        try:
            model = Prophet(**params)
            
            model.fit(train[:-periods])

            future_val = model.make_future_dataframe(
                periods=periods, 
                freq=freq
            )
            forecast_val = model.predict(future_val)
            
            val_predictions = forecast_val.tail(periods)['yhat'].values
            val_actual = train['y'][-periods:].values
            
            error = mean_absolute_percentage_error(val_actual, val_predictions)
            
            return error
        
        except Exception as e:
            # В случае ошибки возвращаем большое значение
            return float('inf')
    
    # Создаем исследование Optuna
    study = optuna.create_study(
        direction='minimize',
        study_name='prophet_monthly_optimization',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )

    print(f"\nЗапуск оптимизации с {n_trials} испытаниями...")
    print("Это может занять несколько минут...")

    # Запускаем оптимизацию
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nОптимизация завершена!")
    print(f"Лучшее значение MAPE: {study.best_value:.4f}")
    print("\nЛучшие гиперпараметры:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study.best_params.copy()