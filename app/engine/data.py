import yfinance as yf
import pandas as pd

from app.engine.indicators.quarter import quarter_v_quarter
from app.engine.indicators.trading_range import from_52
from app.engine.indicators.sma_indicators import sma_indicators
from app.engine.indicators.rs import rs
from app.engine.indicators.performance import create_class, beat_sp
from app.engine.indicators.volume import get_volume_expansion, get_energy, volume_perc, a_d
from app.engine.indicators.rsi import rsi, rsi_stock_v_market
from app.engine.indicators.macd import macd, macd_slope
from app.engine.indicators.price_action import get_bb_width, get_range_position, get_slope
from app.engine.indicators.z_score import calc_z_score
from app.engine.indicators.banner import gen_banner
from app.engine.indicators.animation import start_animation, end_animation
from app.engine.data_manager import save_to_db
from app.models import StockData

"""
Stock strength:
proximity_A = (4MA - 10MA) / 10MA
proximity_B = (10MA - 40MA) / 40MA
proximity_C = (Price - 40MA) / 40MA

Bullish_stack = 4MA > 10MA > 40Ma (1 or 0)
Bearish_stack = 4MA < 10MA < 40MA (1 or 0)
Price_above_all = Price > ALL MA (1 or 0)

Relative Strength = (% change for stock this week) - (% change for market this week)
"""

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

def download_ticker(ticker: str):
    gen_banner()

    [sl, t] = start_animation(f'Loding {ticker} data ')
    data = yf.Ticker(ticker).history(period="25y", interval="1wk")
    market = yf.Ticker("^GSPC").history(period="25y", interval="1wk")
    end_animation(sl, t)

    index = data.index.to_numpy()

    df = pd.DataFrame(0, index=index, columns=[])

    #pd.set_option('display.max_rows', None)

    [sl, t] = start_animation(f'Creating {ticker} dataframe ')

    #Add all basic information into thd df
    df['close'] = round(data['Close'], 2)
    df['open'] = round(data['Open'], 2)
    df['high'] = round(data['High'], 2)
    df['low'] = round(data['Low'], 2)
    df['volume'] = data['Volume']

    #
    quarter_v_quarter(yf.Ticker(ticker).earnings_dates, df)

    # add % off 52 week high and % off 52 week low every week
    from_52(df)
    
    # simple moving average + indicators
    sma_indicators(df)
    
    # 
    a_d(df)
    
    #
    rs(df, market)
    
    #
    rsi(df)
    
    #
    rsi_stock_v_market(market, df)
    
    #
    create_class(df)
    
    #
    beat_sp(df, market)
    
    #
    get_volume_expansion(df)
    
    #
    get_energy(df)
    
    #
    volume_perc(df)
    
    #
    macd(df)
    
    #
    macd_slope(df)
    
    #
    get_bb_width(df)
    
    #
    get_range_position(df)
    
    #
    get_slope(df)
    
    #
    calc_z_score(df)

    df['ticker'] = ticker
    df.index = df.index.date

    end_animation(sl, t)

    save_to_db(df, StockData.__tablename__)
    print("Successfully uploaded dataframe to database!")

if __name__ == "__main__":
    download_ticker(input("Ticker: "))