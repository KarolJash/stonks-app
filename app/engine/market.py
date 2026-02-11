import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import string
import numpy as np
from dotenv import load_dotenv
from curl_cffi import requests
from pytickersymbols import PyTickerSymbols
from yahoo_fin import stock_info as si
import time

from app.engine.indicators.sma_indicators import sma_indicators
from app.engine.indicators.banner import gen_banner
from app.engine.indicators.animation import start_animation, end_animation
from app.engine.data_manager import save_to_db
from app.models import MarketData

"""
Market Strength:
proximity_A = (4MA - 10MA) / 10MA
proximity_B = (10MA - 40MA) / 40MA
proximity_C = (Price - 40MA) / 40MA

Bullish_stack = 4MA > 10MA > 40Ma (1 or 0)
Bearish_stack = 4MA < 10MA < 40MA (1 or 0)
Price_above_all = Price > ALL MA (1 or 0)

AD = # off stocks up / # off stocks down (SP 500)
Above_200MA = (# of stocks above 40MA / # off stocks) * 100 (SP500)
NH_NL = # off stocks that made a new high / # off stocks that made a new low

VIX_level = VIX price
VIX_change = (VIX Close - VIX Open) / VIX Open * 100

TNX
DXY
"""

def get_market_tickers():
    # This initialization is local and doesn't hit the web for every call
    stock_data = PyTickerSymbols()
    
    # Get S&P 500 Tickers
    sp500_stocks = stock_data.get_stocks_by_index('S&P 500')
    sp500_tickers = [s['symbol'].replace('.', '-') for s in sp500_stocks]
            
    return sp500_tickers

def calc_up_down(stock, df):
    if 'up' not in df.columns:
        df['up'] = 0
        df['down'] = 0

    df['up'] = df['up'].add(stock['Close'] > stock['Open'], fill_value=0)
    df['down'] = df['down'].add(stock['Close'] <= stock['Open'], fill_value=0)

def clean_data(df, ticker):
    clean_df = df[['open', 'high', 'low', 'close', 'volume', 'sma_4', 'sma_10', 'sma_40', 'proximity_a', 'proximity_b', 'proximity_c', 'is_bullish', 'is_bearish', 'price_above']].copy()

    clean_df['ticker'] = ticker

    clean_df['asc_desc_ratio'] = df['up'] / (df['down'] + 1e-6)
    clean_df['asc_desc_diff'] = df['up'] - df['down']

    roll_mean = df['stocks above 40'].rolling(window=12, min_periods=1).mean()
    roll_std  = df['stocks above 40'].rolling(window=12, min_periods=1).std()

    roll_std = roll_std.replace(0, 1e-9).fillna(1e-9)

    clean_df['z_score_40'] = (df['stocks above 40'] - roll_mean) / roll_std
    clean_df['pct_above_40'] = df['stocks above 40'] / df['count']

    diff = df['new high'] - df['new low']
    roll_mean = diff.rolling(window=12, min_periods=1).mean()
    roll_std  = diff.rolling(window=12, min_periods=1).std()
    roll_std = roll_std.replace(0, 1e-9).fillna(1e-9)

    clean_df['nh_nl_z'] = (diff - roll_mean) / roll_std
    clean_df['nh_nl_pct'] = (df['new high'] - df['new low'])/ df['count']
    clean_df['nh_nl_ratio'] = np.log((df['new high'] + 1) / (df['new low'] + 1))

    return clean_df
        
def stocks_above_40(stock, df):
    if 'stocks above 40' not in df.columns:
        df['stocks above 40'] = 0

    avg = stock['Close'].rolling(window=40, min_periods=1).mean()
    df['stocks above 40'] = df['stocks above 40'].add(stock['Close'] >= avg, fill_value=0)

def new_high(stock, df):
    if 'new high' not in df.columns:
        df['new high'] = 0

    high = stock['High'].rolling(window=52, min_periods=1).max()

    df['new high'] = df['new high'].add((stock['High'] == high).astype(int), fill_value=0)

def new_low(stock, df):
    if 'new low' not in df.columns:
        df['new low'] = 0
    
    low = stock['Low'].rolling(window=52, min_periods=1).min()

    df['new low'] = df['new low'].add((stock['Low'] == low).astype(int), fill_value=0)


def market():
    load_dotenv()
    gen_banner()

    [sl, t] = start_animation("Loading TICKERS ")

    sp_tickers = get_market_tickers()
    nasdaq_tickers = si.tickers_nasdaq()

    end_animation(sl, t)

    session = requests.Session(impersonate="chrome120")
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0 Safari/537.36",
        "Referer": "https://finance.yahoo.com/"
    })

    sp = yf.download(tickers=["^GSPC"], period="12y", group_by="ticker", interval="1wk", session=session, threads=True)["^GSPC"]
    nasdaq = yf.download(tickers=["^IXIC"], period="12y", group_by="ticker", interval="1wk", session=session, threads=True)["^IXIC"]
    vix = yf.download(tickers=["^VIX"], period="12y", group_by="ticker", interval="1wk", session=session, threads=True)["^VIX"]

    sp.columns = sp.columns.str.lower()
    nasdaq.columns = nasdaq.columns.str.lower()
    vix.columns = vix.columns.str.lower()

    vix['% Change'] = round((vix['close'] - vix['open']) / vix['open'] * 100, 2)
    sma_indicators(vix)
    sma_indicators(sp)
    sma_indicators(nasdaq)

    sp_download = yf.download(tickers=sp_tickers, period="12y", group_by="ticker", interval="1wk", session=session, threads=True)
    sp_dic = {ticker: sp_download.xs(ticker, axis=1, level=0) for ticker in sp_tickers}

    for ticker, data in sp_dic.items():
        if (data.empty):
            print("🚩🚩🚩 " + ticker + " failed")
            continue
        
        calc_up_down(data, sp)
        stocks_above_40(data, sp)
        new_high(data, sp)
        new_low(data, sp)

        if 'count' not in sp.columns:
            sp['count'] = 0

        sp['count'] += 1

    for index, ticker in enumerate(nasdaq_tickers):
        if index % 10 == 0:
            print(f"{index}/{len(nasdaq_tickers)}")

        data = yf.Ticker(ticker=ticker).history(period="12y", interval="1wk")

        if (data.empty):
            print("🚩🚩🚩 " + ticker + " failed")
            continue

        data.index = data.index.tz_localize(None)

        calc_up_down(data, nasdaq)
        stocks_above_40(data, nasdaq)
        new_high(data, nasdaq)
        new_low(data, nasdaq)

        if 'count' not in nasdaq.columns:
            nasdaq['count'] = 0

        nasdaq['count'] += 1

        time.sleep(0.1)

    sp_final = clean_data(sp, 'S&P 500')
    save_to_db(sp_final, MarketData.__tablename__)

    nasdaq_final = clean_data(nasdaq, 'NASDAQ')
    save_to_db(nasdaq_final, MarketData.__tablename__)

    print("downloaded and completed successfully")

if __name__ == "__main__":
    market()

