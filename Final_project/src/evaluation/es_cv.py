# src/evaluation/es_cv.py
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_absolute_percentage_error

def es_cv(train_df, alphas, periods):
    values = train_df['y'][:-periods].values
    tscv = TimeSeriesSplit(n_splits=3)

    best_alpha = None
    best_mape = float("inf")

    for alpha in alphas:
        scores = []
        for tr_idx, val_idx in tscv.split(values):
            model = SimpleExpSmoothing(values[tr_idx])
            fit = model.fit(smoothing_level=alpha, optimized=False)
            forecast = fit.forecast(len(val_idx))
            scores.append(
                mean_absolute_percentage_error(
                    values[val_idx], forecast
                )
            )
        avg = np.mean(scores)
        if avg < best_mape:
            best_mape = avg
            best_alpha = alpha

    return best_alpha
