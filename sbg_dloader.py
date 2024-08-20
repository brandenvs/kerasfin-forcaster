import os
import numpy as np
import pandas as pd

def sbg_dloader(data_dir=str, start_date=int) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir))

    df.rename(columns={'Closing (c)': 'Close'}, inplace=True)
    df.rename(columns={'High (c)': 'High'}, inplace=True)
    df.rename(columns={'Low (c)': 'Low'}, inplace=True)

    # Date convert
    df['Date'] = pd.to_datetime(df['Date'])

    # Select date range
    df = df[(df['Date'].dt.year >= start_date)].copy()

    df.index = range(len(df))

    df = df.sort_values(by='Date')

    return pd.DataFrame(df)
