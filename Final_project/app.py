import pandas as pd
import streamlit as st
import yaml
import altair as alt

st.set_page_config(page_title="US Oil & Gas Forecast", layout="wide")

st.sidebar.header("Параметры")

resource = st.sidebar.selectbox("Ресурс", ["Oil", "Gas"])
model_type = st.sidebar.selectbox("Модель", ["Exponential Smoothing", "Prophet"])
horizon = st.sidebar.slider("Горизонт прогноза (мес.)", 3, 24, 12)

@st.cache_data
def load_history(resource):
    path = (
        "data/processed/oil_monthly.csv"
        if resource == "Oil"
        else "data/processed/gas_monthly.csv"
    )
    df = pd.read_csv(path, parse_dates=["ds"])
    return df

@st.cache_data
def load_backtest(resource, model):
    path = f"data/predictions/backtest_{resource.lower()}_{model}.csv"
    return pd.read_csv(path, parse_dates=["ds"])

@st.cache_data
def load_forecast(resource, model):
    path = f"data/predictions/forecast_{resource.lower()}_{model}.csv"
    return pd.read_csv(path, parse_dates=["ds"])

@st.cache_data
def load_metrics(resource, model):
    path = f"data/metrics/metrics_{resource.lower()}_{model}.csv"
    return pd.read_csv(path)

@st.cache_data
def load_config():
    with open("configs/app.yaml") as f:
        return yaml.safe_load(f)

history = load_history(resource)
model_key = "prophet" if model_type == "Prophet" else "es"
backtest = load_backtest(resource, model_key)
forecast = load_forecast(resource, model_key).head(horizon)
metrics = load_metrics(resource, model_key)

config = load_config()
break_date = pd.to_datetime(config["structural_break"])

st.header("📊 Историческая динамика и качество модели")

hist_plot_df = pd.concat([
    history.assign(type="History"),
    backtest.rename(columns={"yhat": "y"}).assign(type="Backtest")
])

chart_hist = (
    alt.Chart(hist_plot_df)
    .mark_line(strokeWidth=3)
    .encode(
        x="ds:T",
        y="y:Q",
        color=alt.Color(
            "type:N",
            scale=alt.Scale(
                domain=["History", "Backtest"],
                range=["#1f77b4", "#ff7f0e"]
            )
        )
    )
)

vline = (
    alt.Chart(pd.DataFrame({"ds": [break_date]}))
    .mark_rule(color="red", strokeDash=[6, 6], size=2)
    .encode(x="ds:T")
)

st.altair_chart( (chart_hist + vline).properties(height=450, title="US Oil & Gas Production: History and Backtest" ), use_container_width=True)

st.metric("MAE", metrics['MAE'].round(0))
st.metric("RMSE", metrics['RMSE'].round(0))
st.metric("MAPE", metrics['MAPE'].round(3))

merged = history.merge(
    backtest.rename(columns={"yhat": "y_pred"}),
    on="ds",
    how="inner"
)

st.subheader("Факт vs прогноз (история)")
st.dataframe(
    merged[["ds", "y", "y_pred"]]
    .rename(columns={
        "ds": "Дата",
        "y": "Факт",
        "y_pred": "Прогноз"
    })
)

st.header("📈 Прогноз (планирование)")

forecast_plot_df = pd.concat([
    history.assign(type="History"),
    forecast.rename(columns={"yhat": "y"}).assign(type="Forecast")
])

chart_forecast = (
    alt.Chart(forecast_plot_df)
    .mark_line(strokeWidth=3)
    .encode(
        x="ds:T",
        y="y:Q",
        color=alt.Color(
            "type:N",
            scale=alt.Scale(
                domain=["History", "Forecast"],
                range=["#1f77b4", "#d62728"]
            )
        )
    )
)

st.altair_chart( (chart_forecast + vline).properties(height=450, title="US Oil & Gas Production: History and Forecast" ), use_container_width=True)

st.subheader("Прогнозные значения")

st.dataframe(
    forecast[["ds", "yhat"]]
    .rename(columns={
        "ds": "Дата",
        "yhat": "Прогноз"
    })
)

st.markdown(
    """
**Обозначения:**
- **History** — фактические данные добычи  
- **Backtest** — прогноз модели (исторческие данные)
- **Forecast** — прогноз модели (будущее) 
- **Красная пунктирная линия** — структурный сдвиг (2022-07)
"""
)
