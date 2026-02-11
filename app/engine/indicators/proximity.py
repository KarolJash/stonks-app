def proximity(df, sma1, sma2, label):
    df[label] = round((df[sma1] - df[sma2]) / df[sma2] * 100, 2)