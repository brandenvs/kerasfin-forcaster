from statsmodels.tsa.seasonal import seasonal_decompose

def sbg_decomposer(df=pd.DataFrame, feature=str, period=int):
    decompose_df = df[['Date', feature]].copy()
    decompose_df = decompose_df.set_index('Date')

    decompose = seasonal_decompose(decompose_df, period=period)

    return decompose
