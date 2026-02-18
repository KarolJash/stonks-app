from fastapi import FastAPI, BackgroundTasks, status, Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordRequestForm

from datetime import timedelta
from typing import Annotated

from app.engine.data import download_ticker
from app.engine.delist import check_delisted, add_delisted
from app.engine.market import market
from app.engine.main import train_xgboost, make_pred
from app.dependencies import db_validate_ticker, real_ticker, db_validate_model
from app.engine.auth import authenticate_user, create_access_token

from app.schemas import (
    TickerRequest,
    DelistCheckRequest,
    DelistAddRequest,
    XgboostTrainingRequest,
    XgboostPredictionRequest,
    Token,
)

ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI()


@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "scope": " ".join(form_data.scopes)},
        expires_delta=access_token_expires,
    )
    return Token(access_token=access_token, token_type="bearer")


@app.post("/download/stock", status_code=status.HTTP_202_ACCEPTED)
async def download_stock(
    payload: TickerRequest,
    background_tasks: BackgroundTasks,
    valid_ticker: str = Depends(real_ticker),
):
    background_tasks.add_task(download_ticker, payload.ticker)

    return {"message": f"Task {payload.ticker} accepted. Check status later."}


@app.post("/download/market", status_code=status.HTTP_202_ACCEPTED)
async def download_market(background_tasks: BackgroundTasks):
    background_tasks.add_task(market)

    return {
        "message": f"Task to update index infromation accpeted. Check status later."
    }


@app.post("/delist/add", status_code=status.HTTP_202_ACCEPTED)
async def add_delisted_func(
    payload: DelistAddRequest, background_tasks: BackgroundTasks
):
    background_tasks.add_task(add_delisted, payload.ticker)

    return {
        "message": f"Task to add {payload.ticker} to delisted securities accepted. Check status later."
    }


@app.post("/delist/check", status_code=status.HTTP_202_ACCEPTED)
async def delisted_func(payload: DelistCheckRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(check_delisted, payload.count)

    return {
        "message": f"Task to check delisted securities accpeted. Check status later."
    }


@app.post("/xgboost/predict", status_code=status.HTTP_202_ACCEPTED)
def xgboost_pred(
    payload: XgboostPredictionRequest, valid_ticker: str = Depends(db_validate_model)
):
    result = make_pred(payload)

    print(result)

    return {
        "message": f"Model successfully was able to make predictions on ticker: {valid_ticker} from {payload.start_pred} to {payload.end_pred}",
        "status": "success",
        "data": result,
    }


@app.post("/xgboost/train", status_code=status.HTTP_202_ACCEPTED)
async def xgboost_train(
    payload: XgboostTrainingRequest,
    background_tasks: BackgroundTasks,
    valid_ticker: str = Depends(db_validate_ticker),
):
    background_tasks.add_task(train_xgboost, payload)

    return {
        "message": f"Task to check delisted securities accpeted. Check status later."
    }
