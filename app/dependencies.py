import pandas as pd
from sqlalchemy import select
import yfinance as yf
from fastapi import status, HTTPException

from app.schemas import TickerRequest, XgboostTrainingRequest, XgboostPredictionRequest
from app.db import engine
from app.models import StockData, XGBoostData


def db_validate_model(payload: XgboostPredictionRequest):
    stmt = select(XGBoostData).where(XGBoostData.index == payload.model_id).limit(1)
    result = pd.read_sql(stmt, con=engine)

    if result.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{payload.model_id}' not found.",
        )

    if result.iloc[0].ticker != payload.ticker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"The ticker of the model, {result.ticker} does not match inputed ticker {payload.ticker}",
        )

    return payload.ticker


def db_validate_ticker(payload: XgboostTrainingRequest):
    # This runs BEFORE the route handler
    stmt = select(StockData).where(StockData.ticker == payload.ticker)
    result = pd.read_sql(stmt, con=engine)

    if result.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticker '{payload.ticker}' not found.",
        )
    return payload.ticker


def real_ticker(payload: TickerRequest):
    data = yf.Ticker(payload.ticker).history(period="5d")

    if data.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticker '{payload.ticker}' not found.",
        )
    return payload.ticker
