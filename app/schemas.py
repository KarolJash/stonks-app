from pydantic import BaseModel, Field, model_validator
from datetime import date
from typing import Dict, List, Optional, Union
from enum import Enum


class TickerRequest(BaseModel):
    ticker: str


class PasswordUpdate(BaseModel):
    new_password: str
    username: str


class XgboostPredictionRequest(BaseModel):
    ticker: str
    start_training: date
    end_training: date
    start_pred: date
    end_pred: date
    model_id: int

    @model_validator(mode="after")
    def check_dates(self):
        if (
            self.start_pred >= self.end_pred
            or self.start_pred <= self.end_training
            or self.start_training >= self.end_training
        ):
            raise ValueError(
                "start has to be before end and training must come before testing"
            )
        return self


class HyperparameterRange(BaseModel):
    min: int | float | None = None
    max: int | float | None = None
    categorical: List[str] | None = None
    log: bool = False


class XgboostTrainingRequest(BaseModel):
    ticker: str = Field(
        description="Unique ticker for desired stock (ex. AAPL HOOD MSFT)"
    )
    start_training: date = Field(
        description="Start of the training window (YYYY-MM-DD)"
    )
    end_training: date = Field(description="End of the training window (YYYY-MM-DD)")
    start_testing: date = Field(description="Start of the training window (YYYY-MM-DD)")
    end_testing: date = Field(description="Start of the training window (YYYY-MM-DD)")
    output: str = Field(
        description="Field that is being guessed (i.e. class_performance)"
    )
    inputs: List[str]
    trials: int
    n_estimators: int = Field(default=1000)
    severity: float
    hyperparameter_space: Dict[str, HyperparameterRange] = Field(
        example={
            "learning_rate": {"min": 0.001, "max": 0.1},
            "max_depth": {"min": 3, "max": 15},
            "subsample": {"min": 0.5, "max": 1.0},
            "colsample_bytree": {"min": 0.5, "max": 1.0},
            "gamma": {"min": 1e-8, "max": 0.001},
            "reg_lambda": {"min": 1e-8, "max": 0.1},
            "reg_alpha": {"min": 1e-8, "max": 0.1},
            "min_child_weight": {"min": 1, "max": 7},
            "max_leaves": {"min": 32, "max": 256},
            "max_bin": {"min": 50, "max": 1024},
        }
    )

    @model_validator(mode="after")
    def check_dates(self):
        if (
            self.start_testing >= self.end_testing
            or self.start_testing <= self.end_training
            or self.start_training >= self.end_training
        ):
            raise ValueError(
                "start has to be before end and training must come before testing"
            )
        return self


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None
    scopes: list[str] = []


class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None
    hashed_password: str


class UserCreate(BaseModel):
    username: str
    email: str
    full_name: str
    password: str
    scopes: List[str]


class XgboostFilterRequest(BaseModel):
    # Pagination
    page: int = 1
    size: int = 20

    # Displayed Fields (The checkboxes)
    fields: List[str] = []

    # Advanced Filters
    date_start: Optional[date] = None
    date_end: Optional[date] = None
    ticker: Optional[str] = None
    start_training: Optional[date] = None
    end_training: Optional[date] = None
    start_testing: Optional[date] = None
    end_testing: Optional[date] = None
    output: Optional[str] = None
    severity: Optional[float] = None

    # Min/Max ranges
    rmse_min: Optional[float] = None
    rmse_max: Optional[float] = None
    accuracy_min: Optional[float] = None
    accuracy_max: Optional[float] = None
    score_min: Optional[float] = None
    score_max: Optional[float] = None

    sort_by: Optional[str] = None
    sort_direction: Optional[str] = "desc"


class StockFilterRequest(BaseModel):
    # Pagination & Display
    page: int = 1
    size: int = 20
    fields: List[str] = []

    # Base Filters
    date_start: Optional[date] = None
    date_end: Optional[date] = None
    ticker: Optional[str] = None

    # Dropdowns / Booleans (None = "All")
    is_bullish: Optional[bool] = None
    is_bearish: Optional[bool] = None
    price_above: Optional[bool] = None
    beat_sp: Optional[bool] = None

    # Multi-select array for pills (e.g., [0, 1, 4] or empty for all)
    class_performance: List[int] = []

    # --- Min/Max Ranges ---
    close_min: Optional[float] = None
    close_max: Optional[float] = None
    open_min: Optional[float] = None
    open_max: Optional[float] = None
    high_min: Optional[float] = None
    high_max: Optional[float] = None
    low_min: Optional[float] = None
    low_max: Optional[float] = None
    volume_min: Optional[float] = None
    volume_max: Optional[float] = None

    q1_yoy_min: Optional[float] = None
    q1_yoy_max: Optional[float] = None
    q2_yoy_min: Optional[float] = None
    q2_yoy_max: Optional[float] = None
    q3_yoy_min: Optional[float] = None
    q3_yoy_max: Optional[float] = None
    q4_yoy_min: Optional[float] = None
    q4_yoy_max: Optional[float] = None

    surprise_q1_min: Optional[float] = None
    surprise_q1_max: Optional[float] = None
    surprise_q2_min: Optional[float] = None
    surprise_q2_max: Optional[float] = None
    surprise_q3_min: Optional[float] = None
    surprise_q3_max: Optional[float] = None
    surprise_q4_min: Optional[float] = None
    surprise_q4_max: Optional[float] = None

    off_high_min: Optional[float] = None
    off_high_max: Optional[float] = None
    off_low_min: Optional[float] = None
    off_low_max: Optional[float] = None

    sma_4_min: Optional[float] = None
    sma_4_max: Optional[float] = None
    sma_10_min: Optional[float] = None
    sma_10_max: Optional[float] = None
    sma_40_min: Optional[float] = None
    sma_40_max: Optional[float] = None

    proximity_a_min: Optional[float] = None
    proximity_a_max: Optional[float] = None
    proximity_b_min: Optional[float] = None
    proximity_b_max: Optional[float] = None
    proximity_c_min: Optional[float] = None
    proximity_c_max: Optional[float] = None

    acc_weighted_min: Optional[float] = None
    acc_weighted_max: Optional[float] = None
    dis_weighted_min: Optional[float] = None
    dis_weighted_max: Optional[float] = None
    acc_dis_min: Optional[float] = None
    acc_dis_max: Optional[float] = None

    rs_min: Optional[float] = None
    rs_max: Optional[float] = None
    rsi_min: Optional[float] = None
    rsi_max: Optional[float] = None
    rsi_weighted_min: Optional[float] = None
    rsi_weighted_max: Optional[float] = None
    rsi_momentum_min: Optional[float] = None
    rsi_momentum_max: Optional[float] = None

    volume_expansion_min: Optional[float] = None
    volume_expansion_max: Optional[float] = None
    energy_min: Optional[float] = None
    energy_max: Optional[float] = None
    volume_change_min: Optional[float] = None
    volume_change_max: Optional[float] = None

    macd_min: Optional[float] = None
    macd_max: Optional[float] = None
    macd_slope_min: Optional[float] = None
    macd_slope_max: Optional[float] = None
    bandwidth_min: Optional[float] = None
    bandwidth_max: Optional[float] = None

    price_range_position_min: Optional[float] = None
    price_range_position_max: Optional[float] = None
    normalized_slope_min: Optional[float] = None
    normalized_slope_max: Optional[float] = None
    z_score_min: Optional[float] = None
    z_score_max: Optional[float] = None

    sort_by: Optional[str] = None
    sort_direction: Optional[str] = "desc"


class MarketFilterRequest(BaseModel):
    # Pagination & Display
    page: int = 1
    size: int = 20
    fields: List[str] = []

    # Base Filters
    date_start: Optional[date] = None
    date_end: Optional[date] = None
    ticker: Optional[str] = None

    # Dropdowns / Booleans (None = "All")
    is_bullish: Optional[bool] = None
    is_bearish: Optional[bool] = None
    price_above: Optional[bool] = None

    # --- Min/Max Ranges ---
    close_min: Optional[float] = None
    close_max: Optional[float] = None
    open_min: Optional[float] = None
    open_max: Optional[float] = None
    high_min: Optional[float] = None
    high_max: Optional[float] = None
    low_min: Optional[float] = None
    low_max: Optional[float] = None
    volume_min: Optional[float] = None
    volume_max: Optional[float] = None

    sma_4_min: Optional[float] = None
    sma_4_max: Optional[float] = None
    sma_10_min: Optional[float] = None
    sma_10_max: Optional[float] = None
    sma_40_min: Optional[float] = None
    sma_40_max: Optional[float] = None

    proximity_a_min: Optional[float] = None
    proximity_a_max: Optional[float] = None
    proximity_b_min: Optional[float] = None
    proximity_b_max: Optional[float] = None
    proximity_c_min: Optional[float] = None
    proximity_c_max: Optional[float] = None

    asc_desc_ratio_min: Optional[float] = None
    asc_desc_ratio_max: Optional[float] = None
    asc_desc_diff_min: Optional[float] = None
    asc_desc_diff_max: Optional[float] = None

    z_score_40_min: Optional[float] = None
    z_score_40_max: Optional[float] = None
    pct_above_40_min: Optional[float] = None
    pct_above_40_max: Optional[float] = None

    nh_nl_z_min: Optional[float] = None
    nh_nl_z_max: Optional[float] = None
    nh_nl_pct_min: Optional[float] = None
    nh_nl_pct_max: Optional[float] = None
    nh_nl_ratio_min: Optional[float] = None
    nh_nl_ratio_max: Optional[float] = None

    sort_by: Optional[str] = None
    sort_direction: Optional[str] = "desc"


class PublishModelRequest(BaseModel):
    ticker: str
    model_id: int
