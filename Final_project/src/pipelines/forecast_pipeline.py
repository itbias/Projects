from src.models.prophet_model import ProphetModel
from src.models.es_model import ExponentialSmoothingModel

def run_prophet(df, config, periods, return_model=False):
    model = ProphetModel(config)
    model.fit(df)
    if return_model:
        return model, model.predict(periods)
    return model.predict(periods)

def run_es(df, alpha, periods):
    model = ExponentialSmoothingModel(alpha)
    model.fit(df)
    return model.predict(periods)
