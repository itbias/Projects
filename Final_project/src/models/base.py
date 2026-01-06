from abc import ABC, abstractmethod
import pandas as pd

class ForecastModel(ABC):
    """
    Базовый интерфейс для моделей прогнозирования
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """Обучение модели"""
        pass

    @abstractmethod
    def predict(self, periods: int) -> pd.DataFrame:
        """Прогноз на periods шагов вперед"""
        pass
