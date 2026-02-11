def rsi_stock_v_market(market, df, period=20):
    df_copy = df.copy()

    df_copy['relative price ratio'] = df['close'] / market['Close']

    alpha = 2 / (period + 1)

    df['rsi_weighted'] = round(df_copy['relative price ratio'].ewm(alpha).mean(), 10)

    df['rsi_momentum'] = round(df['rsi_weighted'].pct_change(periods=5), 10)

def rsi(df):
    delta = df['close'].diff()

    # standard rsi up/down but adding a decay so that information escapes overtime

    up = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()

    df['rsi'] = round(100 - (100 / (1 + (up / (down + 1e-6)))), 2)