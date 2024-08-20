import tensorflow as tf
from datetime import datetime, timedelta
import pandas as pd
import os

def model_loader(model_dir):
    return tf.keras.models.load_model(os.path.join(model_dir))

def model_train(train_df=pd.DataFrame, valid_df=pd.DataFrame, target=str, epochs=50, batch=34):
    y_train = train_df[target].copy()
    X_train = train_df.drop([target], axis=1)

    y_valid = valid_df[target].copy()
    X_valid = valid_df.drop([target], axis=1)

    nn_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
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

def model_predict(test_set=pd.DataFrame, test_df=pd.DataFrame, target=str, trained_model=any):
    y_test  = test_df[target].copy()
    X_test  = test_df.drop([target], axis=1)

    y_pred = trained_model.predict(X_test)

    return y_pred
