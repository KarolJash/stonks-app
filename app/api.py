from fastapi import (
    FastAPI,
    BackgroundTasks,
    status,
    Depends,
    HTTPException,
    Security,
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.security import OAuth2PasswordRequestForm
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import psutil
import pandas as pd
from typing import List
import uuid
import json
import asyncio

from datetime import timedelta, datetime
from typing import Annotated
from pwdlib import PasswordHash

from app.engine.data import download_ticker
from app.engine.delist import check_delisted, add_delisted
from app.engine.market import market
from app.engine.main import train_xgboost, make_pred
from app.dependencies import db_validate_ticker, real_ticker, db_validate_model
from app.engine.auth import authenticate_user, create_access_token, create_user
from app.engine.data_manager import (
    get_user,
    update_user_password,
    get_current_price,
    create_task,
    get_latest_task,
    get_ticker_data,
    get_xgboost_importance,
    get_xgboost_table,
    get_stock_table,
    get_market_table,
    get_tasks_table,
    publish_model,
    get_publish,
    update_task,
)
from app.engine.auth import get_current_active_user, init_admin

from app.schemas import (
    TickerRequest,
    XgboostTrainingRequest,
    XgboostPredictionRequest,
    Token,
    UserCreate,
    User,
    PasswordUpdate,
    XgboostFilterRequest,
    StockFilterRequest,
    MarketFilterRequest,
    PublishModelRequest,
)

ACCESS_TOKEN_EXPIRE_MINUTES = 30

password_hash = PasswordHash.recommended()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_admin()

    try:
        yf.Ticker("SPY").fast_info
    except Exception as e:
        print(f"Failed to pre-warm yfinance cache: {e}")

    yield


app = FastAPI(lifespan=lifespan)

# 1. Define the allowed origins (your frontend's IP and port)
origins = [
    "http://172.24.10.91:3000",
    "http://localhost:3000",
]

# 2. Add the middleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows your frontend IP
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, etc.
    allow_headers=["*"],  # Allows all headers (Content-Type, etc.)
)


@app.post("/user/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    user = authenticate_user(form_data.username, form_data.password)

    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "scope": " ".join(user.scopes)},
        expires_delta=access_token_expires,
    )
    return Token(access_token=access_token, token_type="bearer")


@app.post("/user/register")
def register_user(
    user: UserCreate,
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["auth:cretae_user"])
    ],
):
    db_user = get_user(username=user.username)

    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    result = create_user(user=user)

    if result is False:
        raise HTTPException(status_code=400, detail="Blocked from creating user")

    return {"message": "New user successfully created!"}


@app.post("/user/change-password")
async def change_password(
    password_data: PasswordUpdate,
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["auth:change_password"])
    ],
):
    hashed_password = password_hash.hash(password_data.new_password)

    result = update_user_password(password_data.username, hashed_password)

    if result is False:
        raise HTTPException(
            status_code=400,
            detail=(f"Could not update password for user {password_data.username}"),
        )

    return {"message": "Password successfully updated!"}


@app.post("/download/stock", status_code=status.HTTP_202_ACCEPTED)
def download_stock(
    payload: TickerRequest,
    background_tasks: BackgroundTasks,
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["stock:download"])
    ],
    ticker: str = Depends(real_ticker),
):
    task_id = create_task(
        f"Download Data For {ticker}", ticker, current_user.full_name, "0/1", "Download"
    )

    background_tasks.add_task(download_ticker, ticker, task_id)

    return {"message": f"Task {ticker} accepted. Check status later."}


@app.post("/market", status_code=status.HTTP_202_ACCEPTED)
async def fetch_market_price(
    payload: TickerRequest,
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["market:price:live"])
    ],
    ticker: str = Depends(real_ticker),
):
    t = yf.Ticker(ticker)
    info = t.info or {}
    # Try common name fields
    name = info.get("longName") or info.get("shortName") or info.get("companyName")

    # Price fields
    current = info.get("regularMarketPrice")
    prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")

    # Fallback to history if info fields are missing
    if current is None or prev_close is None:
        hist = t.history(period="2d", interval="1d")
        if not hist.empty:
            # last available close and previous close
            if current is None:
                current = hist["Close"].iloc[-1]
            if prev_close is None and len(hist) > 1:
                prev_close = hist["Close"].iloc[-2]
            elif prev_close is None:
                prev_close = hist["Close"].iloc[0]

    if current is None or prev_close is None:
        raise ValueError(
            "Could not retrieve current price or previous close for " + ticker
        )

    pct_change = (current - prev_close) / prev_close * 100

    if pct_change > 0:
        dir = True
    else:
        dir = False

    return {
        "message": "Success",
        "ticker": ticker,
        "price": current,
        "pct_change": round(abs(pct_change), 2),
        "up": dir,
        "name": name,
    }


@app.get("/system/stats", status_code=status.HTTP_200_OK)
async def get_system_stats(
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["system:overview"])
    ],
):
    """
    Returns real-time server health metrics.
    interval=1 is required for psutil to calculate the change over time accurately.
    """
    try:
        # Get CPU usage percentage
        # Setting interval=1 makes it wait 1s to get a true reading,
        # but since this is an async dashboard call, we'll use None or 0.1 for speed.
        cpu_pct = psutil.cpu_percent(interval=0.1)

        return {
            "message": "Success",
            "cpu_percentage": cpu_pct,
            "status": "healthy" if cpu_pct < 85 else "strained",
        }
    except Exception as e:
        return {"message": "Error retrieving system stats", "detail": str(e)}


@app.post("/download/market", status_code=status.HTTP_202_ACCEPTED)
def download_market(
    background_tasks: BackgroundTasks,
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["market:download"])
    ],
):
    task_id = create_task(
        f"Download Data For Market", "Market", current_user.full_name, "0/1", "Download"
    )
    background_tasks.add_task(market, task_id)

    return {
        "message": f"Task to update index infromation accpeted. Check status later."
    }


@app.post("/xgboost/predict", status_code=status.HTTP_202_ACCEPTED)
async def xgboost_pred(
    payload: XgboostPredictionRequest,
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["model:predict"])
    ],
    valid_ticker: str = Depends(db_validate_model),
):
    result = await make_pred(payload)

    print(result)

    return {
        "message": f"Model successfully was able to make predictions on ticker: {valid_ticker} from {payload.start_pred} to {payload.end_pred}",
        "status": "success",
        "data": result,
    }


@app.post("/xgboost/train", status_code=status.HTTP_202_ACCEPTED)
def xgboost_train(
    payload: XgboostTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["model:train"])
    ],
    valid_ticker: str = Depends(db_validate_ticker),
):
    try:
        task_id = create_task(
            f"Training Model For {valid_ticker}",
            valid_ticker,
            current_user.full_name,
            f"0/{payload.trials}",
            "Training",
        )
        background_tasks.add_task(train_xgboost, payload, task_id)
    except Exception as e:
        # Update your DB so you can see the error in your UI!
        update_task(task_id, f"Error: {str(e)}")
        print(f"PROCESS CRASHED: {e}")

    return {
        "message": f"Task to check delisted securities accpeted. Check status later."
    }


@app.get("/data/status", status_code=status.HTTP_202_ACCEPTED)
def get_data_status(
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["system:overview"])
    ],
):
    db_sp = get_current_price("S&P 500")
    db_nasdaq = get_current_price("NASDAQ")

    yf_sp = yf.Ticker("^GSPC").history(interval="1wk", period="1wk")
    yf_nasdaq = yf.Ticker("^IXIC").history(interval="1wk", period="1wk")

    return {
        "status": "success",
        "data": bool(
            yf_sp.iloc[-1]["Close"] == db_sp.close
            and yf_nasdaq.iloc[-1]["Close"] == db_nasdaq.close
        ),
    }


@app.get("/task/status", status_code=status.HTTP_202_ACCEPTED)
def get_task_status(
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["system:overview"])
    ],
):
    result = get_latest_task()

    return {"status": "success", "data": result}


@app.get("/stock/{ticker}", status_code=status.HTTP_200_OK)
def get_stock_info(
    ticker: str,
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["stock:read"])
    ],
):
    result = get_ticker_data(ticker)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No data found for ticker: {ticker}",
        )

    return {"status": "success", "data": result[::-1]}


@app.get("/xgboost/{ticker}", status_code=status.HTTP_200_OK)
def get_modal_importance(
    ticker: str,
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["model:read_metadata"])
    ],
):
    result = get_xgboost_importance(ticker)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No data found for ticker: {ticker}",
        )

    return {"status": "success", "data": result}


@app.post("/data/xgboost")
def fetch_xgboost_data(
    payload: XgboostFilterRequest,
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["table:view:xgboost"])
    ],
):
    try:
        [data, total] = get_xgboost_table(payload)
        return {"status": "success", "data": data, "total": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/stock")
def fetch_stock_data(
    payload: StockFilterRequest,
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["table:view:stocks"])
    ],
):
    try:
        [data, total] = get_stock_table(payload)
        return {"status": "success", "data": data, "total": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/market")
def fetch_market_data(
    payload: MarketFilterRequest,
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["table:view:market"])
    ],
):
    try:
        [data, total] = get_market_table(payload)
        return {"status": "success", "data": data, "total": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks", status_code=status.HTTP_202_ACCEPTED)
def get_task_status(
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["table:view:tasks"])
    ],
):
    result = get_tasks_table()

    return {"status": "success", "data": result}


@app.post("/publish", status_code=status.HTTP_202_ACCEPTED)
def get_task_status(
    payload: PublishModelRequest,
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["model:publish"])
    ],
):
    ticker = payload.ticker
    model_id = payload.model_id
    publish_model(ticker, model_id)

    return {"status": "success", "model_id": model_id, "ticker": ticker}


@app.get("/publish/leaderboard", status_code=status.HTTP_202_ACCEPTED)
def get_task_status(
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["model:read_metadata"])
    ],
):
    top_publish = get_publish(5, "top")
    bottom_publish = get_publish(5, "bottom")

    return {
        "status": "success",
        "data": {"top 5": top_publish, "bottom 5": bottom_publish},
    }


@app.get("/publish/all", status_code=status.HTTP_202_ACCEPTED)
def get_task_status(
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["model:read_metadata"])
    ],
):
    result = get_publish(-1, "all")

    return {
        "status": "success",
        "data": result,
    }


@app.get("/user/name", status_code=status.HTTP_202_ACCEPTED)
async def get_task_status(
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["user:read"])
    ],
):
    return {
        "status": "success",
        "data": current_user.full_name,
    }
