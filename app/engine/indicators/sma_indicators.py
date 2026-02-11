from .ma_stack import bearish_stack, bullish_stack, price_above_all
from .proximity import proximity

def sma_indicators(df):
    df['sma_4'] = round(df['close'].rolling(window=4).mean(), 2)
    df['sma_10'] = round(df['close'].rolling(window=10).mean(), 2)
    df['sma_40'] = round(df['close'].rolling(window=40).mean(), 2)

    #check if sma is bullish or bearish
    bullish_stack(df)
    bearish_stack(df)

    #check if price is above all averages
    price_above_all(df)

    #proximity_a = (4MA - 10MA) / 10MA
    proximity(df, 'sma_4', 'sma_10', 'proximity_a')

    #proximity_b = (10MA - 40MA) / 40MA
    proximity(df, 'sma_10', 'sma_40', 'proximity_b')

    #proximity_c = (Price - 40MA) / 40MA
    proximity(df, 'close', 'sma_40', 'proximity_c')