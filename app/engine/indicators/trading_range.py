def from_52(df):
    df_copy = df.copy()

    df_copy['max'] = df_copy['high'].rolling(window=52, min_periods=1).max()
    df_copy['min'] = df_copy['low'].rolling(window=52, min_periods=1).min()

    df['off_high'] = round((df_copy['max'] - df_copy['close']) / df_copy['max'] * 100)
    df['off_low'] = round((df_copy['close'] - df_copy['min']) / df_copy['min'] * 100)