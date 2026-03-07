from fastapi import FastAPI, BackgroundTasks, status, Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordRequestForm
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import psutil

from datetime import timedelta
from typing import Annotated
from pwdlib import PasswordHash

from app.engine.data import download_ticker
from app.engine.delist import check_delisted, add_delisted
from app.engine.market import market
from app.engine.main import train_xgboost, make_pred
from app.dependencies import db_validate_ticker, real_ticker, db_validate_model
from app.engine.auth import authenticate_user, create_access_token, create_user
from app.engine.data_manager import get_user, update_user_password
from app.engine.auth import get_current_active_user, init_admin

from app.schemas import (
    TickerRequest,
    XgboostTrainingRequest,
    XgboostPredictionRequest,
    Token,
    UserCreate,
    User,
    PasswordUpdate,
)

ACCESS_TOKEN_EXPIRE_MINUTES = 30

password_hash = PasswordHash.recommended()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_admin()
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
        User, Security(get_current_active_user, scopes=["system:admin"])
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
async def download_stock(
    payload: TickerRequest,
    background_tasks: BackgroundTasks,
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["stock:download"])
    ],
    ticker: str = Depends(real_ticker),
):
    background_tasks.add_task(download_ticker, ticker)

    return {"message": f"Task {ticker} accepted. Check status later."}


@app.post("/market", status_code=status.HTTP_202_ACCEPTED)
async def fetch_market_price(
    payload: TickerRequest,
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
async def get_system_stats():
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
async def download_market(
    background_tasks: BackgroundTasks,
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["market:download"])
    ],
):
    background_tasks.add_task(market)

    return {
        "message": f"Task to update index infromation accpeted. Check status later."
    }


@app.post("/xgboost/predict", status_code=status.HTTP_202_ACCEPTED)
def xgboost_pred(
    payload: XgboostPredictionRequest,
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["model:predict"])
    ],
    valid_ticker: str = Depends(db_validate_model),
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
    current_user: Annotated[
        User, Security(get_current_active_user, scopes=["model:train"])
    ],
    valid_ticker: str = Depends(db_validate_ticker),
):
    background_tasks.add_task(train_xgboost, payload)

    return {
        "message": f"Task to check delisted securities accpeted. Check status later."
    }
