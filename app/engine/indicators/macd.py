def macd(df):
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()

    macd_line = ema_12 - ema_26

    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    macd_histogram = macd_line - signal_line

    df['macd'] = round(macd_histogram, 10)

def macd_slope(df, loockback=3):
    df['macd_slope'] = round((df['macd'].diff(loockback)) / (df['macd'].abs() + 1e-6), 5)