import numpy as np
import pandas as pd

def create_class(df, period=13):
    change = (df['close'].shift(periods=-period) - df['close']) / df['close'] * 100

    cond = [
        (change <= -15),
        (-15 < change) & (change <= -5),
        (-5 < change) & (change <= 5),
        (5 < change) & (change <= 15),
        (change > 15),
        (change != change)
    ]

    df['class_performance'] = np.select(cond, [*list(range(5)), float('nan')])

def beat_sp(df, market, period=13):
    change_market = ((market['Close'].shift(periods=-period) - market['Close']) / market['Close']) * 100
    change_stock = ((df['close'].shift(periods=-period) - df['close']) / df['close']) * 100

    combined = pd.concat([change_market, change_stock], axis=1, join='inner')
    combined.columns = ['market', 'stock']

    df['beat_sp'] = np.where(combined['stock'] > combined['market'], 1, 0)