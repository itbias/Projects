#es_model.py

import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from src.models.base import ForecastModel

class ExponentialSmoothingModel(ForecastModel):

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.model = None
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        self.model = SimpleExpSmoothing(df['y'].values)
        self.fitted_model = self.model.fit(
            smoothing_level=self.alpha,
            optimized=False
        )
        self.first_date = df['ds'].iloc[0]
        self.last_date = df['ds'].iloc[-1]
        self.freq = pd.infer_freq(df['ds'])
        self.fitted = True

    def predict(self, periods: int, return_history: bool = True) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Model must be fitted")
        
        if return_history:

            dates_history = pd.date_range(
                start=self.first_date,
                end=self.last_date,
                freq=self.freq
            )

            history_df = pd.DataFrame({
                "ds": dates_history,
                "yhat": self.fitted_model.fittedvalues
            })

        forecast = self.fitted_model.forecast(periods)

        forecast_dates = pd.date_range(
            start=self.last_date,
            periods=periods + 1,
            freq=self.freq
        )[1:]

        forecast_df = pd.DataFrame({
            "ds": forecast_dates,
            "yhat": forecast
        })

        if return_history:
            return pd.concat([history_df, forecast_df], ignore_index=True)

        return forecast_df
