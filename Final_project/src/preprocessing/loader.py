#loader.py

import pandas as pd

def load_raw_data(path: str, delimiter: str) -> pd.DataFrame:

    df = pd.read_csv(path, thousands=delimiter)

    df['Production Date'] = pd.to_datetime(df['Production Date']).dt.to_period('M')

    print("✅ Датасет загружен!")
    print("\nПервые пять строк:")

    print(df.head())

    print("\nОбщая информация:")
    print("="*50)
    df.info()

    print("\nОписательная статистика:")
    print(df.describe(include='all'))

    print("\n📊 Основные характеристики")
    print(f"• Объем данных: {df.shape[0]:,} записей, {df.shape[1]} признаков")
    print(f"• Покрытие: данные за {df['Production Date'].iloc[0]} - {df['Production Date'].iloc[-1]}")
    print(f"• География: {df['State'].nunique()} штатов, {df['County'].nunique()} округов")
    print(f"• Уникальных операций: {df['Disposition Description'].nunique()}")

    print("\n🔍 Анализ пропущенных значений:")
    print("="*50)
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100

    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })

    print(missing_df[missing_df['Missing Count'] > 0])

    print(f"\n🔍 Анализ дубликатов (все столбцы одинаковые)")
    print("-" * 50)
        
    print(f"• Количество полных дубликатов: {len(df) - len(df.drop_duplicates())}")
    print(f"• Уникальных записей: {len(df.drop_duplicates())}")

    return df
