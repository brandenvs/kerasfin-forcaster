import pandas as pd

def sbg_dropper(train_df=pd.DataFrame, valid_df=pd.DataFrame, test_df=pd.DataFrame, columns=list):
    train_df = train_df.drop(columns, axis=1)
    valid_df = valid_df.drop(columns, axis=1)
    test_df  = test_df.drop(columns, axis=1)

    return train_df, valid_df, test_df

def sbg_splitter(df=pd.DataFrame, test_size=float, valid_size=float):
    test_split_idx  = int(df.shape[0] * (1-test_size))
    valid_split_idx = int(df.shape[0] * (1-(valid_size+test_size)))

    train_df = df.loc[:valid_split_idx].copy()
    valid_df = df.loc[valid_split_idx+1:test_split_idx].copy()
    test_df  = df.loc[test_split_idx+1:].copy()

    return train_df, valid_df, test_df
