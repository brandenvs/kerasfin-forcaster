import pandas as pd

def moving_averages(df=pd.DataFrame, feature=str):
    df['EMA_9'] = df[feature].ewm(9).mean().shift()
    df['SMA_5'] = df[feature].rolling(5).mean().shift()
    df['SMA_10'] = df[feature].rolling(10).mean().shift()
    df['SMA_15'] = df[feature].rolling(15).mean().shift()
    df['SMA_30'] = df[feature].rolling(30).mean().shift()

    return df

def relative_strength_idx(df, n=14):
    close = df['Close']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi

def macd(df=pd.DataFrame):
    ema_12 = pd.Series(df['Close'].ewm(span=12, min_periods=12).mean())
    ema_26 = pd.Series(df['Close'].ewm(span=26, min_periods=26).mean())

    df['MACD'] = pd.Series(ema_12 - ema_26)
    df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())

    # plot_macd(df, ema_12, ema_26)

    return ema_12, ema_26

def sbg_splitter(df=pd.DataFrame, test_size=float, valid_size=float):
    test_split_idx  = int(df.shape[0] * (1-test_size))
    valid_split_idx = int(df.shape[0] * (1-(valid_size+test_size)))

    train_df = df.loc[:valid_split_idx].copy()
    valid_df = df.loc[valid_split_idx+1:test_split_idx].copy()
    test_df  = df.loc[test_split_idx+1:].copy()

    return train_df, valid_df, test_df

