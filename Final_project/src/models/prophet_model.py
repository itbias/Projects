import pandas as pd
from prophet import Prophet
from src.models.base import ForecastModel

class ProphetModel(ForecastModel):

    def __init__(self, config: dict):
        self.config = config
        self.model = Prophet(**config)
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        """
        df должен иметь колонки:
        - ds (datetime)
        - y  (float)
        """
        self.model.fit(df)
        self.fitted = True

    def predict(self, periods: int, freq: str = "MS") -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")

        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
