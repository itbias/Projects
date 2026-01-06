# preprocess.py
import pandas as pd
from typing import Tuple

def get_monthly_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    oil_df = df[df['Commodity'] == 'Oil (bbl)'].reset_index(drop=True)
    gas_df = df[df['Commodity'] == 'Gas (Mcf)'].reset_index(drop=True)

    oil_monthly = oil_df.groupby('Production Date', as_index=False)['Volume'].sum()
    gas_monthly = gas_df.groupby('Production Date', as_index=False)['Volume'].sum()

    print("=" * 70)
    print("Месячные объемы")
    print("=" * 70)
    print(f'Размерность данных (Нефть): {oil_monthly.shape}')
    print(f'Размерность данных (Газ): {gas_monthly.shape}')
    print(f'Период данных: {oil_monthly["Production Date"][0]} - {oil_monthly["Production Date"].iloc[-1]}')

    return oil_monthly, gas_monthly

def prepare_for_prophet(df: pd.DataFrame) -> pd.DataFrame:
    df['ds'] = df['Production Date'].dt.to_timestamp()
    df = df.sort_values('ds')

    return df[['ds', 'Volume']].rename(columns={'Volume': 'y'})