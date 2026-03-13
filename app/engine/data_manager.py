import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import select, update, select, desc, case, func, asc, desc
from datetime import datetime

from app.db import engine, Base, SessionLocal
from app.models import StockData, MarketData, XGBoostData, UserData, TasksData
from app.schemas import (
    User,
    XgboostFilterRequest,
    StockFilterRequest,
    MarketFilterRequest,
)


def insert_on_conflict(table, conn, keys, data_iter):
    data = [dict(zip(keys, row)) for row in data_iter]

    stmt = insert(table.table).values(data)

    update_dict = {
        col: stmt.excluded[col]
        for col in keys
        if col not in ["date", "ticker", "index"]  # Optional: Skip keys if you want
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


def get_user(username: str):
    query = select(UserData).where(UserData.username == username)

    with SessionLocal() as session:
        return session.execute(query).scalar_one_or_none()


def update_user_password(username: str, hashed_password: str):
    with SessionLocal() as session:
        query = (
            update(UserData)
            .where(UserData.username == username)
            .values(hashed_password=hashed_password)
        )

        session.execute(query)
        session.commit()

    return True


def upload_new_user(user):
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()

    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()


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


def save_xgboost(
    payload,
    best_params,
    fixed_params,
    rmse,
    accuracy,
    score,
    pic_name,
    hyper_data,
    feature_data,
):
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
        "feature_importance": feature_data,
        "hyperparameters_importance": hyper_data,
    }

    entry = XGBoostData(**data_to_save)

    with SessionLocal() as session:
        session.add(entry)
        session.commit()


def get_current_price(ticker: str):
    query = (
        select(MarketData)
        .where(MarketData.ticker == ticker)
        .order_by(desc(MarketData.date))
        .limit(1)
    )

    with SessionLocal() as session:
        result = session.execute(query)
        return result.scalar_one_or_none()


def create_task(action, ticker, current_user, epoch, task_type):
    stmt = (
        insert(TasksData)
        .values(
            action=action,
            ticker=ticker,
            epoch=epoch,
            task_type=task_type,
            full_name=current_user,
            status="in progress",
            created_at=datetime.now(),
            error_message="",
        )
        .returning(TasksData.index)
    )

    with SessionLocal() as session:
        result = session.execute(stmt)
        session.commit()
        return result.scalar_one()


def update_task(
    task_id: int, status: str = None, epoch: str = None, error_message: str = None
):
    update_data = {}
    if status is not None:
        update_data["status"] = status
    if epoch is not None:
        update_data["epoch"] = epoch
    if error_message is not None:
        update_data["error_message"] = error_message

    if not update_data:
        return

    stmt = update(TasksData).where(TasksData.index == task_id).values(**update_data)

    with SessionLocal() as session:
        session.execute(stmt)
        session.commit()


def get_latest_task():
    status_priority = case(
        (TasksData.status == "in progress", 1),
        (TasksData.status == "completed", 2),
        else_=3,
    )

    stmt = (
        select(TasksData)
        .order_by(
            status_priority,
            TasksData.created_at.desc(),
        )
        .limit(1)
    )

    with SessionLocal() as session:
        result = session.execute(stmt)
        return result.scalar_one_or_none()


def get_ticker_data(ticker):
    query = (
        select(StockData.date, StockData.close)
        .where(StockData.ticker == ticker)
        .where(StockData.class_performance != None)
        .order_by(desc(StockData.date))
        .limit(370)
    )

    with SessionLocal() as session:
        result = session.execute(query)
        return result.mappings().all()


def get_xgboost_importance(ticker):
    query = (
        select(
            XGBoostData.hyperparameters_importance,
            XGBoostData.feature_importance,
            XGBoostData.index,
            XGBoostData.ticker,
            XGBoostData.best_params,
            XGBoostData.inputs,
        )
        .where(XGBoostData.ticker == ticker)
        .where(XGBoostData.hyperparameters_importance.is_not(None))
        .where(XGBoostData.hyperparameters_importance != {})
        .order_by(desc(XGBoostData.created_at))
        .limit(20)
    )

    with SessionLocal() as session:
        result = session.execute(query)
        return result.mappings().all()


def get_xgboost_table(filters: XgboostFilterRequest):
    # 1. DYNAMIC COLUMN SELECTION
    # If the user selected specific fields, map them to the actual SQLAlchemy columns
    if filters.fields:
        selected_columns = []
        for field in filters.fields:
            # Safely check if the column exists on the model to prevent crashes
            if hasattr(XGBoostData, field):
                selected_columns.append(getattr(XGBoostData, field))

        # Fallback to all columns if something went wrong
        if not selected_columns:
            selected_columns = [XGBoostData]

        query = select(*selected_columns)
    else:
        # Default: Select everything
        query = select(XGBoostData)

    # 2. APPLY ADVANCED FILTERS dynamically
    if filters.ticker:
        # .ilike() allows for case-insensitive partial searches (e.g., typing "app" finds "AAPL")
        query = query.where(XGBoostData.ticker.ilike(f"%{filters.ticker}%"))

    if filters.output and filters.output != "All Outputs":
        query = query.where(XGBoostData.output == filters.output)

    if filters.severity is not None:
        query = query.where(XGBoostData.severity == filters.severity)

    # Date Ranges
    if filters.date_start:
        query = query.where(XGBoostData.date >= filters.date_start)
    if filters.date_end:
        query = query.where(XGBoostData.date <= filters.date_end)

    # Exact Dates (Based on your UI having single inputs for these)
    if filters.start_training:
        query = query.where(XGBoostData.start_training == filters.start_training)
    if filters.start_testing:
        query = query.where(XGBoostData.start_testing == filters.start_testing)

    # Numeric Ranges (Using `is not None` because 0 is a valid score/accuracy!)
    if filters.rmse_min is not None:
        query = query.where(XGBoostData.rmse >= filters.rmse_min)
    if filters.rmse_max is not None:
        query = query.where(XGBoostData.rmse <= filters.rmse_max)

    if filters.score_min is not None:
        query = query.where(XGBoostData.score >= filters.score_min)
    if filters.score_max is not None:
        query = query.where(XGBoostData.score <= filters.score_max)

    if filters.accuracy_min is not None:
        query = query.where(XGBoostData.accuracy >= filters.accuracy_min)
    if filters.accuracy_max is not None:
        query = query.where(XGBoostData.accuracy <= filters.accuracy_max)

    if filters.sort_by and hasattr(XGBoostData, filters.sort_by):
        sort_col = getattr(XGBoostData, filters.sort_by)
        if filters.sort_direction == "asc":
            query = query.order_by(asc(sort_col))
        else:
            query = query.order_by(desc(sort_col))
    else:
        # 2. THE FIX: Make sure the fallback sort also uses XGBoostData!
        query = query.order_by(desc(XGBoostData.created_at))

    count_query = select(func.count()).select_from(query.subquery())

    # 3. PAGINATION
    # Calculate how many rows to skip based on the current page
    offset_amount = (filters.page - 1) * filters.size

    query = (
        query.order_by(desc(XGBoostData.created_at))
        .offset(offset_amount)
        .limit(filters.size)
    )

    # 4. EXECUTION
    with SessionLocal() as session:
        # 1. Execute the count query
        total_count = session.execute(count_query).scalar()

        # 2. Execute the data query
        result = session.execute(query)

        if filters.fields:
            data = result.mappings().all()
        else:
            data = result.scalars().all()

        # 3. Return BOTH back to FastAPI
        return [data, total_count]


def get_market_table(filters: MarketFilterRequest):
    # 1. DYNAMIC COLUMN SELECTION
    if filters.fields:
        selected_columns = []
        for field in filters.fields:
            # Map frontend names with spaces (if they send them) to backend snake_case
            clean_field = field.replace(" ", "_")
            if hasattr(MarketData, clean_field):
                selected_columns.append(getattr(MarketData, clean_field))

        if not selected_columns:
            selected_columns = [MarketData]

        query = select(*selected_columns)
    else:
        query = select(MarketData)

    # 2. EXACT MATCH & BOOLEAN FILTERS
    if filters.ticker:
        query = query.where(MarketData.ticker.ilike(f"%{filters.ticker}%"))

    if filters.date_start:
        query = query.where(MarketData.date >= filters.date_start)
    if filters.date_end:
        query = query.where(MarketData.date <= filters.date_end)

    if filters.is_bullish is not None:
        query = query.where(MarketData.is_bullish == filters.is_bullish)
    if filters.is_bearish is not None:
        query = query.where(MarketData.is_bearish == filters.is_bearish)
    if filters.price_above is not None:
        query = query.where(MarketData.price_above == filters.price_above)

    # 3. AUTOMATED MIN/MAX RANGE LOOP
    # The exact database column names for all numeric ranges in the Market table
    range_columns = [
        "close",
        "open",
        "high",
        "low",
        "volume",
        "sma_4",
        "sma_10",
        "sma_40",
        "proximity_a",
        "proximity_b",
        "proximity_c",
        "asc_desc_ratio",
        "asc_desc_diff",
        "z_score_40",
        "pct_above_40",
        "nh_nl_z",
        "nh_nl_pct",
        "nh_nl_ratio",
    ]

    for col_name in range_columns:
        if hasattr(MarketData, col_name):
            model_col = getattr(MarketData, col_name)

            # Dynamically grab the min/max values from the Pydantic payload
            min_val = getattr(filters, f"{col_name}_min", None)
            max_val = getattr(filters, f"{col_name}_max", None)

            if min_val is not None:
                query = query.where(model_col >= min_val)
            if max_val is not None:
                query = query.where(model_col <= max_val)

    if filters.sort_by and hasattr(MarketData, filters.sort_by):
        sort_col = getattr(MarketData, filters.sort_by)
        if filters.sort_direction == "asc":
            query = query.order_by(asc(sort_col))
        else:
            query = query.order_by(desc(sort_col))
    else:
        # Fallback to a default sort if they haven't clicked a column yet
        query = query.order_by(desc(MarketData.date))

    count_query = select(func.count()).select_from(query.subquery())

    # 4. PAGINATION
    offset_amount = (filters.page - 1) * filters.size

    query = (
        query.order_by(desc(MarketData.date))  # Assuming you want newest first
        .offset(offset_amount)
        .limit(filters.size)
    )

    # 5. EXECUTION
    with SessionLocal() as session:
        # 1. Execute the count query
        total_count = session.execute(count_query).scalar()

        # 2. Execute the data query
        result = session.execute(query)

        if filters.fields:
            data = result.mappings().all()
        else:
            data = result.scalars().all()

        # 3. Return BOTH back to FastAPI
        return [data, total_count]


def get_stock_table(filters: StockFilterRequest):
    # 1. DYNAMIC COLUMN SELECTION
    if filters.fields:
        selected_columns = []
        for field in filters.fields:
            if hasattr(StockData, field):
                selected_columns.append(getattr(StockData, field))

        if not selected_columns:
            selected_columns = [StockData]

        query = select(*selected_columns)
    else:
        query = select(StockData)

    # 2. EXACT MATCH & BOOLEAN FILTERS
    if filters.ticker:
        query = query.where(StockData.ticker.ilike(f"%{filters.ticker}%"))

    if filters.date_start:
        query = query.where(StockData.date >= filters.date_start)
    if filters.date_end:
        query = query.where(StockData.date <= filters.date_end)

    if filters.is_bullish is not None:
        query = query.where(StockData.is_bullish == filters.is_bullish)
    if filters.is_bearish is not None:
        query = query.where(StockData.is_bearish == filters.is_bearish)
    if filters.price_above is not None:
        query = query.where(StockData.price_above == filters.price_above)
    if filters.beat_sp is not None:
        query = query.where(StockData.beat_sp == filters.beat_sp)

    # Array matching for the Class Performance pills
    if filters.class_performance:
        query = query.where(StockData.class_performance.in_(filters.class_performance))

    # 3. AUTOMATED MIN/MAX RANGE LOOP
    # The exact database column names for all numeric ranges
    range_columns = [
        "close",
        "open",
        "high",
        "low",
        "volume",
        "q1_yoy",
        "q2_yoy",
        "q3_yoy",
        "q4_yoy",
        "surprise_q1",
        "surprise_q2",
        "surprise_q3",
        "surprise_q4",
        "off_high",
        "off_low",
        "sma_4",
        "sma_10",
        "sma_40",
        "proximity_a",
        "proximity_b",
        "proximity_c",
        "acc_weighted",
        "dis_weighted",
        "acc_dis",
        "rs",
        "rsi",
        "rsi_weighted",
        "rsi_momentum",
        "volume_expansion",
        "energy",
        "volume_change",
        "macd",
        "macd_slope",
        "bandwidth",
        "price_range_position",
        "normalized_slope",
        "z_score",
    ]

    for col_name in range_columns:
        if hasattr(StockData, col_name):
            model_col = getattr(StockData, col_name)

            # Dynamically grab the min/max values from the Pydantic payload
            min_val = getattr(filters, f"{col_name}_min", None)
            max_val = getattr(filters, f"{col_name}_max", None)

            if min_val is not None:
                query = query.where(model_col >= min_val)
            if max_val is not None:
                query = query.where(model_col <= max_val)

    if filters.sort_by and hasattr(StockData, filters.sort_by):
        sort_col = getattr(StockData, filters.sort_by)
        if filters.sort_direction == "asc":
            query = query.order_by(asc(sort_col))
        else:
            query = query.order_by(desc(sort_col))
    else:
        # Fallback to a default sort if they haven't clicked a column yet
        query = query.order_by(desc(StockData.date))

    count_query = select(func.count()).select_from(query.subquery())

    # 4. PAGINATION
    offset_amount = (filters.page - 1) * filters.size

    query = (
        query.order_by(desc(StockData.date)).offset(offset_amount).limit(filters.size)
    )

    # 5. EXECUTION
    with SessionLocal() as session:
        # 1. Execute the count query
        total_count = session.execute(count_query).scalar()

        # 2. Execute the data query
        result = session.execute(query)

        if filters.fields:
            data = result.mappings().all()
        else:
            data = result.scalars().all()

        # 3. Return BOTH back to FastAPI
        return [data, total_count]


def get_tasks_table():
    query = select(TasksData).order_by(desc(TasksData.created_at)).limit(50)

    with SessionLocal() as session:
        result = session.execute(query)
        return result.scalars().all()


def publish_model(ticker: str, model_id: int):
    unpublish_stmt = (
        update(XGBoostData).where(XGBoostData.ticker == ticker).values(published=False)
    )

    publish_stmt = (
        update(XGBoostData).where(XGBoostData.index == model_id).values(published=True)
    )

    with SessionLocal() as session:
        try:
            session.execute(unpublish_stmt)
            session.execute(publish_stmt)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e


def get_publish(limit: int, specifier: str):
    if specifier == "top":
        query = (
            select(
                XGBoostData.ticker,
                XGBoostData.updated_at,
                XGBoostData.accuracy,
                XGBoostData.rmse,
            )
            .order_by(desc(XGBoostData.accuracy))
            .where(XGBoostData.published == True)
            .limit(limit)
        )
    elif specifier == "bottom":
        query = (
            select(
                XGBoostData.ticker,
                XGBoostData.updated_at,
                XGBoostData.accuracy,
                XGBoostData.rmse,
            )
            .order_by(XGBoostData.accuracy)
            .where(XGBoostData.published == True)
            .limit(limit)
        )
    elif specifier == "all":
        query = select(XGBoostData.ticker, XGBoostData.updated_at).where(
            XGBoostData.published == True
        )

    with SessionLocal() as session:
        result = session.execute(query)
        return result.mappings().all()
