import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import datetime, timedelta

from sbg_dloader import sbg_dloader
from sbg_fitter import model_loader, model_predict, model_train
from sbg_preprocesser import sbg_dropper, sbg_splitter
from sbg_indicators import moving_averages, relative_strength_idx, macd

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

import os



def get_datasets() -> str:
    directory = r'data/'

    dataset_dirs = []

    for filename in os.listdir(directory):
        # Check if it is a file (and not a directory)
        if os.path.isfile(os.path.join(directory, filename)):
            dataset_dirs.append(os.path.join(directory, filename))
    return dataset_dirs

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:

    df = moving_averages(df, 'Close')
    
    target = 'Close'
    
    df['RSI'] = relative_strength_idx(df)

    ema_12, ema_26 = macd(df)

    df['Close'] = df['Close'].shift(-1)

    df = df.iloc[33:]
    df = df[:-1]
    df.index = range(len(df))

    return df

def feed_df(df: pd.DataFrame, id) -> pd.DataFrame:
    target = 'Close'
    train_set, valid_set, test_set = sbg_splitter(df, .85, .15)

    drop_cols = ['Date', 'Volume', 'Low', 'High', '# Deals','Value (R)', 'Move (%)', 'DY', 'EY', 'PE']

    train_df, valid_df, test_df = sbg_dropper(train_set, valid_set, test_set, drop_cols)

    model = model_train(
        train_df=train_df,
        valid_df=valid_df,
        target=target,
        epochs=85
    ).save(f'temp{id}.keras')


def forecast_df(model_dir, df):
    target = 'Close'

    loaded_model = model_loader(model_dir)

    start_date = datetime(2024, 8, 27)
    end_date = datetime(2024, 9, 1)

    # Set the current_date to the start_date
    current_date = start_date
    forecasted_df = df.copy()
    
    train_set, valid_set, test_set = sbg_splitter(df, .85, .15)

    drop_cols = ['Date', 'Volume', 'Low', 'High', '# Deals','Value (R)', 'Move (%)', 'DY', 'EY', 'PE']

    train_df, valid_df, test_df = sbg_dropper(train_set, valid_set, test_set, drop_cols)
    y_pred = model_predict(test_set, test_df, target, loaded_model)

    new_row = pd.DataFrame([{
        'Date': pd.to_datetime(current_date).strftime('%Y-%m-%d 00:00:00'),
        'Close': y_pred[0,0]
    }])

    forecasted_df = pd.concat([forecasted_df, new_row], ignore_index=True)

    forecasted_values = pd.DataFrame()

    # Loop through the dates from start_date to end_date
    while current_date <= end_date:
        # Apply moving averages and other indicators to forecasted_df
        forecasted_df = moving_averages(forecasted_df, target)
        
        forecasted_df['RSI'] = relative_strength_idx(forecasted_df)
        
        ema_12, ema_26 = macd(forecasted_df)
        
        forecasted_df['Close'] = forecasted_df['Close'].shift(-1)
        
        forecasted_df = forecasted_df.iloc[33:]
        forecasted_df = forecasted_df[:-1]

        forecasted_df.index = range(len(forecasted_df))

        train_set, valid_set, test_set = sbg_splitter(forecasted_df, .85, .15)

        # Drop unnecessary columns after appending the new row
        drop_cols = ['Date', 'Volume', 'Low', 'High', '# Deals','Value (R)', 'Move (%)', 'DY', 'EY', 'PE']

        train_df, valid_df, test_df = sbg_dropper(train_set, valid_set, test_set, drop_cols)
        
        model = model_train(
            train_df=train_df,
            valid_df=valid_df,
            target=target,
            epochs=85
        )

        y_pred = model_predict(test_set, test_df, target, model)

        new_row = pd.DataFrame([{
            'Date': pd.to_datetime(current_date).strftime('%Y-%m-%d 00:00:00'),
            'Close': y_pred[0,0]
        }])

        forecasted_values = pd.concat([forecasted_values, new_row], ignore_index=True)

        # forecasted_df = pd.concat([forecasted_df, forecasted_values], ignore_index=True)

        # Increment the current date
        current_date += timedelta(days=1)

    # Export to CSV
    forecasted_values.to_csv(f'forecasted_{model_dir}.csv', index=False)

    # Plot a line chart
    plt.figure(figsize=(10, 6))
    plt.plot(forecasted_values['Date'], forecasted_values['Close'], marker='o')

    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Close (c)')
    plt.title('Sample Line Chart')
    plt.grid(True)

    # Save the plot as a JPG file
    plt.savefig(f'forecasted_chart_{model_dir}.jpg', format='jpg')

def main():
    dataset_dirs = get_datasets()
    model_names = [data_dir for data_dir in dataset_dirs]

    dfs = [sbg_dloader(data_dir, 2020) for data_dir in dataset_dirs]

    preprocessed_dfs = [preprocess_df(df) for df in dfs]

    items = {}
    count = -1
    for df in preprocessed_dfs:
        count += 1
        items.update({model_names[count]: count})
        feed_df(df, count)
        forecast_df(f'temp{count}.keras', df)
    print(items)

if __name__ == "__main__":
    main()
