import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import string
import numpy as np
from dotenv import load_dotenv
from curl_cffi import requests

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

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # Spoof a real browser header to bypass the 403 error
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # 1. Fetch the content with the header
    response = requests.get(url, headers=headers)
    
    # 2. Pass the HTML text to pandas
    # read_html returns a list of DataFrames; the S&P table is index 0
    tables = pd.read_html(response.text)
    df = tables[0]
    
    # 3. Clean tickers for Yahoo Finance (replace dots with hyphens)
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    
    return tickers

def get_nyse_tickers():
    all_tickers = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Categories on Wikipedia: 0-9 and then A through Z
    categories = ['0-9'] + list(string.ascii_uppercase)
    
    for char in categories:
        url = f'https://en.wikipedia.org/wiki/Companies_listed_on_the_New_York_Stock_Exchange_({char})'
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                # The ticker table is the first wikitable on these pages
                tables = pd.read_html(response.text)
                df = tables[0]
                
                # Column is usually named 'Symbol'
                if 'Symbol' in df.columns:
                    # Clean the tickers for Yahoo Finance format
                    clean_tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
                    all_tickers.extend(clean_tickers)
        except Exception as e:
            pass
            
    return sorted(list(set(all_tickers))) # Remove duplicates and sort

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

    df['new high'] = df['new high'].add(stock['High'] == high, fill_value=0)

def new_low(stock, df):
    if 'new low' not in df.columns:
        df['new low'] = 0
    
    low = stock['Low'].rolling(window=52, min_periods=1).min()

    df['new low'] = df['new low'].add(stock['Low'] == low, fill_value=0)


def market():
    load_dotenv()
    gen_banner()

    [sl, t] = start_animation("Loading SP 500 ")
    sp_tickers = get_sp500_tickers()
    end_animation(sl, t)
    
    [sl, t] = start_animation("Loading NYSE ")
    nyse_tickers = get_nyse_tickers()

    delete = np.load('storage/models/delisted.npy')
    nyse_tickers = [x for x in nyse_tickers if x not in delete]

    end_animation(sl, t)

    session = requests.Session(impersonate="chrome120")
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0 Safari/537.36",
        "Referer": "https://finance.yahoo.com/"
    })

    sp = yf.download(tickers=["^GSPC"], period="12y", group_by="ticker", interval="1wk", session=session, threads=True)["^GSPC"]
    nyse = yf.download(tickers=["^NYA"], period="12y", group_by="ticker", interval="1wk", session=session, threads=True)["^NYA"]
    vix = yf.download(tickers=["^VIX"], period="12y", group_by="ticker", interval="1wk", session=session, threads=True)["^VIX"]

    sp.columns = sp.columns.str.lower()
    nyse.columns = nyse.columns.str.lower()
    vix.columns = vix.columns.str.lower()

    vix['% Change'] = round((vix['close'] - vix['open']) / vix['open'] * 100, 2)
    sma_indicators(vix)
    sma_indicators(sp)
    sma_indicators(nyse)

    nyse_download = yf.download(tickers=nyse_tickers, period="12y", group_by="ticker", interval="1wk", session=session)
    nyse_dic = {ticker: nyse_download.xs(ticker, axis=1, level=0) for ticker in nyse_tickers}

    for ticker, data in nyse_dic.items():
        if (data.empty):
            print("🚩🚩🚩 " + ticker + " failed")
            continue

        calc_up_down(data, nyse)
        stocks_above_40(data, nyse)
        new_high(data, nyse)
        new_low(data, nyse)

        if 'count' not in nyse.columns:
            nyse['count'] = 0

        nyse['count'] += 1
    
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

    sp_final = clean_data(sp, 'S&P 500')
    save_to_db(sp_final, MarketData.__tablename__)

    nyse_final = clean_data(nyse, 'NYSE')
    save_to_db(nyse_final, MarketData.__tablename__)

    print("downloaded and completed successfully")

if __name__ == "__main__":
    market()

