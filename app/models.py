from sqlalchemy import Integer, String, ForeignKey, UniqueConstraint, BigInteger
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from datetime import datetime
from typing import Optional

from sqlalchemy.dialects.postgresql import JSONB

from app.db import Base

class StockData(Base):
    __tablename__ = "stocks"

    index: Mapped[int] = mapped_column(primary_key=True, index=True)
    date: Mapped[datetime] = mapped_column(index=True)
    ticker: Mapped[str] = mapped_column(index=True)

    close: Mapped[float]
    open: Mapped[float]
    high: Mapped[float]
    low: Mapped[float]
    volume: Mapped[int] = mapped_column(BigInteger)

    q1_yoy: Mapped[Optional[float]]
    q2_yoy: Mapped[Optional[float]]
    q3_yoy: Mapped[Optional[float]]
    q4_yoy: Mapped[Optional[float]]
    
    surprise_q1: Mapped[Optional[float]]
    surprise_q2: Mapped[Optional[float]]
    surprise_q3: Mapped[Optional[float]]
    surprise_q4: Mapped[Optional[float]]

    off_high: Mapped[float]
    off_low: Mapped[float]

    sma_4: Mapped[Optional[float]]
    sma_10: Mapped[Optional[float]]
    sma_40: Mapped[Optional[float]]

    is_bullish: Mapped[int]
    is_bearish: Mapped[int]
    price_above: Mapped[int]

    proximity_a: Mapped[Optional[float]]
    proximity_b: Mapped[Optional[float]]
    proximity_c: Mapped[Optional[float]]
    
    acc_weighted: Mapped[float]
    dis_weighted: Mapped[float]
    acc_dis: Mapped[float]
    
    rs: Mapped[float]
    rsi: Mapped[Optional[float]]
    rsi_weighted: Mapped[float]
    rsi_momentum: Mapped[Optional[float]]
    
    volume_expansion: Mapped[Optional[float]]
    energy: Mapped[Optional[float]]
    volume_change: Mapped[Optional[float]] 
    
    macd: Mapped[float]
    macd_slope: Mapped[Optional[float]]

    bandwidth: Mapped[Optional[float]]
    price_range_position: Mapped[Optional[float]]
    normalized_slope: Mapped[Optional[float]]
    z_score: Mapped[Optional[float]]
    
    class_performance: Mapped[Optional[float]]
    beat_sp: Mapped[int]

    __table_args__ = (UniqueConstraint('date', 'ticker', name='_date_ticker_uc_stocks_'),)

class MarketData(Base):
    __tablename__ = "market"

    index: Mapped[int] = mapped_column(primary_key=True, index=True)
    date: Mapped[datetime] = mapped_column(index=True)
    ticker: Mapped[str] = mapped_column(index=True)

    close: Mapped[float]
    open: Mapped[float]
    high: Mapped[float]
    low: Mapped[float]
    volume: Mapped[int] = mapped_column(BigInteger)

    sma_4: Mapped[Optional[float]]
    sma_10: Mapped[Optional[float]]
    sma_40: Mapped[Optional[float]]

    is_bullish: Mapped[int]
    is_bearish: Mapped[int]
    price_above: Mapped[int]

    proximity_a: Mapped[Optional[float]]
    proximity_b: Mapped[Optional[float]]
    proximity_c: Mapped[Optional[float]]

    asc_desc_ratio: Mapped[float]
    asc_desc_diff: Mapped[int]

    z_score_40: Mapped[float]
    pct_above_40: Mapped[float]

    nh_nl_z: Mapped[float]
    nh_nl_pct: Mapped[float]
    nh_nl_ratio: Mapped[float]

    __table_args__ = (UniqueConstraint('date', 'ticker', name='_date_ticker_uc_market_'),)

class XGBoostData(Base):
    __tablename__ = "xgboost"

    index: Mapped[int] = mapped_column(primary_key=True, index=True)
    date: Mapped[datetime] = mapped_column(index=True)
    ticker: Mapped[str] = mapped_column(index=True)

    start_training: Mapped[datetime]
    end_training: Mapped[datetime]
    start_testing: Mapped[datetime]
    end_testing: Mapped[datetime]
    output: Mapped[str]
    trials: Mapped[int]
    severity: Mapped[float]
    inputs: Mapped[list] = mapped_column(JSONB)

    best_params: Mapped[dict] = mapped_column(JSONB)
    fixed_params: Mapped[dict] = mapped_column(JSONB)
    picture: Mapped[str]

    rmse: Mapped[float]
    accuracy: Mapped[float]
    score: Mapped[float]