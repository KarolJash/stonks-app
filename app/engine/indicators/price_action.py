import numpy as np

def get_slope(df, period=10):
    # 1. Use Log Prices for scale invariance
    y = np.log(df['close'].values)
    n = period
    
    # 2. Pre-calculate x values (0, 1, 2...n-1)
    x = np.arange(n)
    x_mean = np.mean(x)
    
    # 3. Use the simplified formula for rolling linear regression slope:
    # slope = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
    def fast_slope(y_window):
        y_mean = np.mean(y_window)
        numerator = np.sum((x - x_mean) * (y_window - y_mean))
        denominator = np.sum((x - x_mean)**2)
        return numerator / denominator

    df['normalized_slope'] = round(df['close'].rolling(window=period).apply(fast_slope), 10)
    return df
        

def get_bb_width(df, period=10):
    subset = df['close'].rolling(window=period)
    std_dev = subset.std()
    ma = subset.mean()
    
    upper = ma + (2 * std_dev)
    lower = ma - (2 * std_dev)

    df['bandwidth'] = round(((upper - lower) / ma) * 100, 2)

def get_range_position(df, period=10):
    subset_high = df['high'].rolling(window=period)
    subset_low = df['low'].rolling(window=period)

    high = subset_high.max()
    low = subset_low.min()

    current_price = df['close']
    
    df['price_range_position'] = round((current_price - low) / (high - low), 5)


# ------------ As of right now, not used in code. Needs to be upgraded anyways ----------------
def consecutive_days_up_and_down(data, num):
    # 0 = up, 1 = down
    #direction
    up_arr = []
    up = 0

    down_arr = []
    down = 0

    for i in range(52):
        if num - i == -1:
            break

        if data.iloc[num - i]['Open'] < data.iloc[num - i]['Close']:
            if up == 0 and down != 0:
                down_arr.append(down)
                down = 0

            up += 1
        else:
            if down == 0 and up != 0:
                up_arr.append(up)
                up = 0
   
            down += 1
    
    if up != 0:
        up_arr.append(up)
    if down != 0:
        down_arr.append(down)
    
    return [up_arr, down_arr]