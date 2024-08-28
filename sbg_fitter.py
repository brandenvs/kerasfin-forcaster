import tensorflow as tf
from datetime import datetime, timedelta
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

def model_loader(model_dir):
    return tf.keras.models.load_model(os.path.join(model_dir))

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

def model_predict(test_set=pd.DataFrame, test_df=pd.DataFrame, target=str, trained_model=any):
    y_test  = test_df[target].copy()
    X_test  = test_df.drop([target], axis=1)

    y_pred = trained_model.predict(X_test)

    return y_pred
