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


# Load in dataset
data_dir = r'data/test.csv'

df = sbg_dloader(data_dir, 2020)
df1 = df.copy()

target = 'Close'

df = moving_averages(df, target)

df['RSI'] = relative_strength_idx(df)

ema_12, ema_26 = macd(df)

df['Close'] = df['Close'].shift(-1)

df = df.iloc[33:]
df = df[:-1]

df.index = range(len(df))

train_set, valid_set, test_set = sbg_splitter(df, .15, .15)

drop_cols = ['Date', 'Volume', 'Low', 'High', '# Deals','Value (R)', 'Move (%)', 'DY', 'EY', 'PE']

train_df, valid_df, test_df = sbg_dropper(train_set, valid_set, test_set, drop_cols)

model = model_train(
    train_df=train_df,
    valid_df=valid_df,
    target=target,
    epochs=85
).save('test_model.keras')

loaded_model = model_loader('test_model.keras')

start_date = datetime(2023, 8, 8)
end_date = datetime(2023, 8, 13)

# Set the current_date to the start_date
current_date = start_date
forecasted_df = df.copy()

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

    train_set, valid_set, test_set = sbg_splitter(forecasted_df, .15, .15)

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
forecasted_values.to_csv('forecasted.csv', index=False)

# Plot a line chart
plt.figure(figsize=(10, 6))
plt.plot(forecasted_values['Date'], forecasted_values['Close'], marker='o')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Close (c)')
plt.title('Sample Line Chart')
plt.grid(True)

# Save the plot as a JPG file
plt.savefig('forecasted_chart.jpg', format='jpg')
