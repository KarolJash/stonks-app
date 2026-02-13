import matplotlib
matplotlib.use('Agg')

import uuid
import numpy as np
import optuna
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor, plot_importance, plot_tree, to_graphviz
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select

from app.models import StockData, MarketData
from app.db import SessionLocal
from app.engine.data_manager import import_from_db, save_xgboost

def import_data(ticker):
    session = SessionLocal()
    stmt = select(StockData).where(StockData.ticker == ticker)

    result = session.query(StockData).filter_by(StockData.ticker == ticker).all()
    
    print(result)

def split_data(csv):
    scaler = StandardScaler().set_output(transform="pandas")
    num = round(len(csv) * 0.8)

    #X = csv[['Q1', 'Q2', 'Q3', 'Q4', 'SMA_4', 'SMA_10', 'SMA_40', 'per off high', 'per off low', 'per change rsi', 'current rsi', 'stock rsi', 'price slope', 'bandwidth', 'price range position', 'macd', 'macd slope', 'volume year over year']]
    X = csv[[
        'off high',
        'off low',
        'is_bullish',
        'is_bearish',
        'price_above',
        'proximity_A',
        'proximity_B',
        'proximity_C',
        'A/D',
        'rs',
        'rsi',
        'weighted rsi',
        'rsi momentum',
        'Volume Expansion',
        'Energy',
        'volume perc change',
        'macd',
        'macd slope',
        'bandwidth',
        'price range position',
        'normalized_slope',
        'z_score']]
    Y = csv[['Class performance']]

    if len(csv) > 52*5:
        print("here")
        X_train = scaler.fit_transform(X.iloc[-169:-65])
        Y_train = Y.iloc[-169:-65]

        X_test = scaler.fit_transform(X.iloc[-65:-13])
        Y_test = Y.iloc[-65:-13]
    else:
        X_train = scaler.fit_transform(X.iloc[62:num])
        Y_train = Y.iloc[62:num]

        X_test = scaler.fit_transform(X.iloc[num:-13])
        Y_test = Y.iloc[num:-13]

    return [X_train, Y_train, X_test, Y_test]

def move_data(X_train, Y_train, X_test, Y_test):
    #Take the imported CSV table and transfer over to GPU using DeviceQuantileDMatrix
    print("Moving data to GPU...")
    d_train = xgb.DMatrix(X_train, label=Y_train)
    d_test = xgb.DMatrix(X_test, label=Y_test)

    return [d_train, d_test]

def define_params(trial):
    return {
        # --- GPU Params ---
        'device':'cuda',
        'tree_method':'hist',
        'objective':'reg:squarederror',

        # --- Optuna Params ---
        'learning_rate':trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth':trial.suggest_int('max_depth', 3, 12),
        'subsample':trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma':trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_lambda':trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'reg_alpha':trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 15)
    }


def define_model(trial, n_estimators, hyperparameter_space):
    print('Creating new Model')

    params = {}
    for name, space in hyperparameter_space.items():
        # Choose suggestion type based on whether min/max are ints or floats
        if isinstance(space.min, int) and isinstance(space.max, int):
            params[name] = trial.suggest_int(name, space.min, space.max, log=space.log)
        elif isinstance(space.min, float) and isinstance(space.max, float):
            params[name] = trial.suggest_float(name, space.min, space.max, log=space.log)
        else:
            params[name] = trial.suggest_categorical(name, space.categorical)

    return XGBRegressor(
        # --- GPU Params ---
        device='cpu',
        tree_method='hist',
        objective='reg:squarederror',
        n_estimators=n_estimators,
        n_jobs=-1,
        enable_categorical=True,

        **params

        #quantile_alpha=[0.05, 0.5, 0.95],
        #objective='reg:quantileerror',

        # --- Optuna Params ---
        #grow_policy=trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
    )
    

def train_model_cpu(model, X_train, Y_train):
    print("Training New model")
    model.fit(X_train, Y_train)

def train_model_gpu(params, d_train, d_test, trial):
    return xgb.train(params, d_train, num_boost_round=1000, evals=[(d_test, 'Test')], early_stopping_rounds=100, verbose_eval=False)

def eval_performance(model, d_test, y_test, severity):
    print("Evaluating Performance")
    preds = model.predict(d_test)

    accuracy = accuracy_score(y_test, np.round(preds))
    rmse = mean_squared_error(y_test, preds)

    print('Accuracy was: ' + str(round(accuracy* 100, 2)) + '%')
    print('RMSE: ' + str(rmse))

    return accuracy - severity * rmse * rmse

def test_predictions(model, x_test, y_test):
    preds = model.predict(x_test)
    prediction_data = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    real_data = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}

    for i in range(len(y_test)):
        prediction_data[str(round(preds[i]))] += 1
        real_data[str(round(y_test.iloc[i]))] += 1
    
    print(f"Prediction by model: {prediction_data}")
    print(f"Actual stock performance: {real_data}")


def main_cpu(trial, X_train, Y_train, X_test, Y_test, severity, n_estimators, hyperparameter_space):
    model = define_model(trial, n_estimators, hyperparameter_space)
    train_model_cpu(model, X_train, Y_train)

    score = eval_performance(model, X_test, Y_test, severity)

    return score

def main_gpu(trial, d_train, d_test, Y_test):
    params = define_params(trial)
    model = train_model_gpu(params, d_train, d_test, trial)

    score = eval_performance(model, d_test, Y_test)

    return score

def get_all_scores(model, d_test, y_test, severity):
    preds = model.predict(d_test)

    accuracy = accuracy_score(y_test, np.round(preds))
    rmse = mean_squared_error(y_test, preds)

    return [rmse, round(accuracy * 100, 2), accuracy - severity * rmse * rmse]

def train_xgboost(payload):
    stock_train = import_from_db(payload.ticker, StockData, payload.start_training, payload.end_training).set_index(['date']).drop('index', axis=1)
    stock_test = import_from_db(payload.ticker, StockData, payload.start_testing, payload.end_testing).set_index(['date']).drop('index', axis=1)

    sp_train = import_from_db("S&P 500", MarketData, payload.start_training, payload.end_training).set_index(['date']).drop('index', axis=1).add_prefix("sp_")
    sp_test = import_from_db("S&P 500", MarketData, payload.start_testing, payload.end_testing).set_index(['date']).drop('index', axis=1).add_prefix("sp_")

    nasdaq_train = import_from_db("NASDAQ", MarketData, payload.start_training, payload.end_training).set_index(['date']).drop('index', axis=1).add_prefix("nasdaq_")
    nasdaq_test = import_from_db("NASDAQ", MarketData, payload.start_testing, payload.end_testing).set_index(['date']).drop('index', axis=1).add_prefix("nasdaq_")

    train = stock_train.merge(sp_train, on="date").merge(nasdaq_train, on="date")
    test = stock_test.merge(sp_test, on="date").merge(nasdaq_test, on="date")

    X_train = train[payload.inputs]
    Y_train = train[payload.output]

    X_test = test[payload.inputs]
    Y_test = test[payload.output]

    study = optuna.create_study(direction="maximize")

    # --- CPU ---
    study.optimize(lambda trial: main_cpu(trial, X_train, Y_train, X_test, Y_test, payload.severity, payload.n_estimators, payload.hyperparameter_space), n_trials=payload.trials)
    
    trial_params = study.best_trial.params
    fixed_params = {        
        'device':'cpu',
        'tree_method':'hist',
        'objective':'reg:squarederror',
        'n_estimators':payload.n_estimators,
        'n_jobs':-1
    }
    
    best_model = XGBRegressor(**fixed_params, **trial_params)
    best_model.fit(X_train, Y_train)

    pic_name = uuid.uuid4()

    plot_importance(best_model, max_num_features=12)
    plt.savefig(f'/app/storage/output_images/{pic_name}.png')

    test_predictions(model=best_model, x_test=X_test, y_test=Y_test)
    print(eval_performance(best_model, X_test, Y_test, payload.severity))

    [rmse, accuracy, score] = get_all_scores(best_model, X_test, Y_test, payload.severity)

    save_xgboost(payload, study.best_trial.params, fixed_params, rmse, accuracy, score, pic_name)

if __name__ == "__main__":
    train_xgboost(input('Ticker: '), 'test', 'test'), 
    
"""    csv = import_data(ticker)
    X_train, Y_train, X_test, Y_test = split_data(csv)

    #d_train, d_test = move_data(X_train, Y_train, X_test, Y_test)

    study = optuna.create_study(direction='maximize')

    # --- CPU ---
    study.optimize(lambda trial: main_cpu(trial, X_train, Y_train, X_test, Y_test), n_trials=250)

    trial_params = study.best_trial.params
    fixed_params = {        
        'device':'cpu',
        'tree_method':'hist',
        'objective':'reg:squarederror',
        'n_estimators':1000,
        'n_jobs':-1
    }

    best_model = XGBRegressor(**fixed_params, **trial_params)
    best_model.fit(X_train, Y_train)

    plot_importance(best_model)
    plt.savefig('my_figure.png')

    test_predictions(model=best_model, x_test=X_test, y_test=Y_test)
    print(eval_performance(best_model, X_test, Y_test))

    # --- GPU ---
    # study.optimize(lambda trial: main_gpu(trial, d_train, d_test, Y_test), n_trials=250)"""