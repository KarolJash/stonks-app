def rs(df, market):
    stock_change = round((df['close'] - df['open']) / df['open'] * 100, 2)
    market_change = round((market['Close'] - market['Open']) / market['Open'] * 100, 2)

    df['rs'] = round(stock_change - market_change, 2)