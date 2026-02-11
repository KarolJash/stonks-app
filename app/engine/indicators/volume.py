import numpy as np

def get_volume_expansion(df, period=12):
    avg_vol_20 = df['volume'].rolling(window=4).mean()
    avg_vol_denom = df['volume'].rolling(window=period).mean()

    df['volume_expansion'] = round(avg_vol_20/avg_vol_denom, 2)

def get_energy(df):
    #Calculate volume expansion component
    avg_vol_20 = df['volume'].rolling(window=4).mean()
    avg_vol_60 = df['volume'].rolling(window=12).mean()

    vol_exp = round(avg_vol_20/avg_vol_60, 2)
    norm_vol_exp = (vol_exp/3).clip(upper=1)

    #Calculae Volatility expansion component
    atr_15 = df['close'].rolling(window=3).mean()
    atr_60 = df['close'].rolling(window=12).mean()

    atr = round(atr_15/atr_60, 2)
    norm_atr = (atr/2).clip(upper=1)

    #Calculate Spread Expansion Component
    spread = df['high'] - df['low']

    avg_spread_10 = spread.rolling(window=2).mean()
    avg_spread_40 = spread.rolling(window=10).mean()

    spread = round(avg_spread_10/avg_spread_40, 2)
    norm_spread = (spread/2).clip(upper=1)

    df['energy'] = round(0.5*norm_vol_exp + 0.3*norm_atr + 0.2*norm_spread, 5)


def volume_perc(df, period=10):
    avg_vol_50 = df['volume'].rolling(window=period).mean()

    df["volume_change"] = round((df['volume'] - avg_vol_50) / avg_vol_50 * 100, 2)

# accumulation/distribution wighted
def a_d(df, period=20):
    df_copy = df.copy()

    df_copy['avg_vol'] = df_copy['volume'].rolling(window=period).mean()
    df_copy['prev_vol'] = df_copy['volume'].shift(1)

    acc_cond = ((df_copy['close'] >= df_copy['open'] * 1.01) & (df_copy['volume'] > df_copy['prev_vol']) & (df_copy['volume'] > df_copy['avg_vol']))
    dis_cond = ((df_copy['close'] <= df_copy['open'] * 0.99) & (df_copy['volume'] > df_copy['prev_vol']) & (df_copy['volume'] > df_copy['avg_vol']))

    df_copy['acc'] = np.where(acc_cond, 1, 0)
    df_copy['dis'] = np.where(dis_cond, 1, 0)

    alpha = 2 / (period + 1)
    df['acc_weighted'] = round(df_copy['acc'].ewm(alpha=alpha, adjust=False).mean(), 10)
    df['dis_weighted'] = round(df_copy['dis'].ewm(alpha=alpha, adjust=False).mean(), 10)

    df['acc_dis'] = round((df['acc_weighted'] - df['dis_weighted']) / (df['acc_weighted'] + df['dis_weighted'] + 1e-6), 10)

    del df_copy