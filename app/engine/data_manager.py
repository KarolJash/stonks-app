import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import select
from datetime import datetime

from app.db import engine, Base, SessionLocal
from app.models import StockData, MarketData, XGBoostData


def insert_on_conflict(table, conn, keys, data_iter):
    data = [dict(zip(keys, row)) for row in data_iter]

    stmt = insert(table.table).values(data)

    update_dict = {
        col.name: stmt.excluded[col.name]
        for col in StockData.__table__.columns
        if col.name
        not in ["date", "ticker", "index"]  # Optional: Skip keys if you want
    }

    on_conflict_stmt = stmt.on_conflict_do_update(
        index_elements=["date", "ticker"], set_=update_dict
    )

    conn.execute(on_conflict_stmt)
    conn.commit()


def save_to_db(df: pd.DataFrame, tablename):
    df = df.reset_index(names="date")

    Base.metadata.create_all(bind=engine)

    df.to_sql(
        name=tablename,
        con=engine,
        if_exists="append",
        index=False,
        method=insert_on_conflict,
    )


def import_from_db(ticker, table, start, end):
    query = (
        select(table)
        .where(table.ticker == ticker)
        .where(table.date > start)
        .where(table.date < end)
    )

    return pd.read_sql(query, con=engine)


def import_model(index):
    query = select(XGBoostData).where(XGBoostData.index == index).limit(1)

    return pd.read_sql(query, con=engine)


def save_xgboost(payload, best_params, fixed_params, rmse, accuracy, score, pic_name):
    Base.metadata.create_all(bind=engine)

    data_to_save = {
        "ticker": payload.ticker,
        "start_training": payload.start_training,
        "end_training": payload.end_training,
        "start_testing": payload.start_testing,
        "end_testing": payload.end_testing,
        "output": payload.output,
        "inputs": payload.inputs,
        "trials": payload.trials,
        "severity": payload.severity,
        "fixed_params": fixed_params,
        "best_params": best_params,
        "rmse": rmse,
        "accuracy": accuracy,
        "score": score,
        "date": datetime.now(),
        "picture": pic_name,
    }

    entry = XGBoostData(**data_to_save)

    with SessionLocal() as session:
        session.add(entry)
        session.commit()
