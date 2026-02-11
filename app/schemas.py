from pydantic import BaseModel, Field, model_validator
from datetime import date
from typing import Dict, List, Optional, Union
from enum import Enum

class TickerRequest(BaseModel):
    ticker: str

class DelistCheckRequest(BaseModel):
    count: int

class DelistAddRequest(BaseModel):
    ticker: str

class HyperparameterRange(BaseModel):
    min: Union[int, float]
    max: Union[int, float]
    log: bool = False

class XgboostTrainingRequest(BaseModel):
    ticker: str = Field(description="Unique ticker for desired stock (ex. AAPL HOOD MSFT)")
    start_training: date = Field(description="Start of the training window (YYYY-MM-DD)")
    end_training: date = Field(description="End of the training window (YYYY-MM-DD)")
    start_testing: date = Field(description="Start of the training window (YYYY-MM-DD)")
    end_testing: date = Field(description="Start of the training window (YYYY-MM-DD)")
    output: str = Field(description="Field that is being guessed (i.e. class_performance)")
    inputs: List[str]
    trials: int
    n_estimators: int = Field(
        default=1000
    )
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
            "max_bin": {"min": 50, "max": 1024}
        }
    )

    @model_validator(mode="after")
    def check_dates(self):
        if self.start_testing >= self.end_testing or self.start_testing <= self.end_training or self.start_training >= self.end_training:
            raise ValueError('start has to be before end and training must come before testing')
        return self