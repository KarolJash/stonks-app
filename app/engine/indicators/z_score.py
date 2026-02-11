def calc_z_score(df, window=20):
    avg = df['close'].rolling(window=window).mean()

    std = df['close'].rolling(window=window).std()

    df['z_score'] = round((df['close'] - avg) / std, 5)