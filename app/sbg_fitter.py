import tensorflow as tf
from datetime import datetime, timedelta
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path
import random
import numpy as np

# set seed
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)
BASE_DIR = Path(__file__).resolve().parent

def model_loader(model_name: str):
    model_dir = f'{BASE_DIR}/models/' +  model_name + '.keras'

    model = tf.keras.models.load_model(model_dir)
    return model

def model_train(train_df=pd.DataFrame, valid_df=pd.DataFrame, target=str, epochs=50, batch=34):
    y_train = train_df[target].copy()
    X_train = train_df.drop([target], axis=1)

    y_valid = valid_df[target].copy()
    X_valid = valid_df.drop([target], axis=1)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    nn_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(34, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    nn_model.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError(), metrics=['accuracy'])

    # Train the model
    nn_model.fit(
        X_train, 
        y_train,
        epochs=epochs, 
        batch_size=batch,
        validation_data=(X_valid, y_valid)
    )

    return nn_model

def model_train_other(train_df=pd.DataFrame, valid_df=pd.DataFrame, target=str, epochs=50, batch=34):
    y_train = train_df[target].copy()
    X_train = train_df.drop([target], axis=1)

    y_valid = valid_df[target].copy()
    X_valid = valid_df.drop([target], axis=1)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    nn_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(X_train_scaled.shape[1],1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dropout(0.2),        
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])

    nn_model.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError())

    # Train the model
    nn_model.fit(
        X_train, 
        y_train,
        epochs=epochs, 
        batch_size=batch,
        validation_data=(X_valid, y_valid)
    )

    return nn_model

def model_predict(test_set=pd.DataFrame, test_df=pd.DataFrame, target=str, trained_model=any, model_name=str):
    y_test  = test_df[target].copy()
    X_test  = test_df.drop([target], axis=1)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_test)
    y_pred = trained_model.predict(X_train_scaled)

    plt.figure(figsize=(12, 6))
    plt.plot(test_set.Date, y_test, label='Actual')
    plt.plot(test_set.Date, y_pred, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.savefig(f'{BASE_DIR.parent}/plots/{model_name}.jpg', format='jpg')

    
    return y_pred
