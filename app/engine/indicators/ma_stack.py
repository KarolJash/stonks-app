import numpy as np

def bullish_stack(df):
    bullish_cond = (df['sma_4'] > df['sma_10']) & (df['sma_10'] > df['sma_40'])

    df['is_bullish'] = np.where(bullish_cond, 1, 0)

def bearish_stack(df):
    bearish_cond = (df['sma_4'] < df['sma_10']) & (df['sma_10'] < df['sma_40'])

    df['is_bearish'] = np.where(bearish_cond, 1, 0)

def price_above_all(df):
    cond = df['close'] > df[['sma_4', 'sma_10', 'sma_40']].max(axis=1)

    df['price_above'] = np.where(cond, 1, 0)
