import os
import json
from lightgbm import early_stopping
from tabulate import tabulate
from functools import partial
from IPython.display import display
from tqdm.auto import tqdm
import numpy as np
import xgboost as xgb
from .logger import logger  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from src.params import get_params
from .cfg import CFG
from .models import fetch_model, Valid_Metrics
import pandas as pd 
import optuna

optuna.logging.set_verbosity(optuna.logging.INFO)


def auto_output_folder(filename):
    base_path = os.path.dirname(os.getcwd())
    directory = os.path.join(base_path, filename)
    if not os.path.exists(directory):
        logger.info(f"Create folder name : {filename} folder")
        os.mkdir(directory)
    else:
        logger.info("Folder already exists, create new one")
        raise Exception("Folder already exists, specify new one")

def reduce_mem_usage(df):
    """ 
    iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    logger.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logger.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def null_checker(data):
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            display(f"null col: {col} >> total null : {data[col].isnull().sum()}")
            data[col] = data[col].replace(
                np.NaN, data[col].mean()
            )
            display(f"after fillup null: {col} >> total null : {data[col].isnull().sum()}")
		

def label_encode():
	pass

def categorical_data():
	pass

def normal_data_split(df, label, random_state, shuffle, test_size):
    X = df.drop(label, axis=1)
    y = df[label]
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, 
        random_state=random_state, 
        shuffle=shuffle, 
        test_size=test_size
    )
    return xtrain, xtest, ytrain, ytest

# Function for mean value of metrics
def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict




def optimize(trial, clf_model, use_predict_proba, eval_metric, model_config):
    
    scores = []
    clf_model, use_predict_proba, direction, eval_metric = fetch_model(model_config)

    metrics = Valid_Metrics(model_config)

    params = get_params(trial, model_config)

    early_stopping_rounds = params["early_stopping_rounds"]
    del params["early_stopping_rounds"]

    for fold in range(model_config["no_of_fold"]):
        train_feather = pd.read_feather(os.path.join(model_config["path"], f"{model_config['output_path']}/train_fold_{fold}.feather"))
        test_feather = pd.read_feather(os.path.join(model_config["path"], f"{model_config['output_path']}/test_fold_{fold}.feather"))
        xtrain = train_feather[model_config["features"]]
        xtest = test_feather[model_config["features"]]

        ytrain = train_feather[model_config["label"]].values
        ytest = test_feather[model_config["label"]].values

        if model_config["model_name"] == "xgb":
            model = clf_model(
                **params,
                use_label_encoder=False,
                eval_metric=eval_metric,
                random_state=model_config["random_state"]
            )

        elif model_config["model_name"] == "lgb":
            model = clf_model(
                **params,
                random_state=model_config["random_state"]
            )
        
        elif model_config["model_name"] == "ExtraTree":
            model = clf_model(
                **params,
                random_state=model_config["random_state"]
            )

        model.fit(
            xtrain,
            ytrain,
            verbose=False, 
            early_stopping_rounds=early_stopping_rounds,
            eval_set=[(xtest, ytest)]
        )
        
        if use_predict_proba:
            ypred = model.predict_proba(xtest)
        else:
            ypred = model.predict(xtest)
        
        # metrics calculation
        '''
        models.py : we have to create a function calculate, that will measure model performance
        '''
        metrics_dict = metrics.calculate(ytest, ypred) # TODO : write this function
        scores.append(metrics_dict)

    # create a function that take mean separately take all values from list and print the eval_metrics
    mean_metrics = dict_mean(scores)
    logger.info(f"Metrics: {mean_metrics}")
    return mean_metrics[eval_metric]


def train_model(model_config):
    clf_model, use_predict_proba, direction, eval_metric = fetch_model(model_config)
    if model_config["compare"]:
        optimize_func = partial(
            optimize,
            clf_model = clf_model,
            model_config = model_config,
            eval_metric = eval_metric,
            use_predict_proba = use_predict_proba
        )
    else:
        optimize_func = partial(
            optimize,
            clf_model = clf_model,
            model_config = model_config,
            eval_metric = eval_metric,
            use_predict_proba = use_predict_proba
        )
    db_path = os.path.join(model_config['path'], f"{model_config['output_path']}/params.db")
    study = optuna.create_study(
        direction="minimize",
        study_name=model_config["study_name"],
        storage=f"sqlite:///{db_path}",
        load_if_exists=True
    )
    study.optimize(optimize_func, n_trials=model_config["n_trails"])
    return study.best_params

'''
prediction test => fold by fold prediction using there best params
'''
def predict_model(model_config, best_params):
    scores = []
    test_prediction = []
    clf_model, use_predict_proba, direction, eval_metric = fetch_model(model_config)

    metrics = Valid_Metrics(model_config)
    
    early_stopping_rounds = best_params["early_stopping_rounds"]
    del best_params["early_stopping_rounds"]

    for fold in range(model_config["no_of_fold"]):
        logger.info(f">> Train and predict for fold : {fold}")
        train_feather = pd.read_feather(os.path.join(model_config["path"], f"{model_config['output_path']}/train_fold_{fold}.feather"))
        test_feather = pd.read_feather(os.path.join(model_config["path"], f"{model_config['output_path']}/test_fold_{fold}.feather"))
        xtrain = train_feather[model_config["features"]]
        xtest = test_feather[model_config["features"]]

        ytrain = train_feather[model_config["label"]].values
        ytest = test_feather[model_config["label"]].values

        if model_config["test_path"] is not None:
            test_file = pd.read_feather(f'{os.path.join(model_config["path"], model_config["output_path"])}/test_file.feather')
            X_test = test_file[model_config["features"]]

        if model_config["model_name"] == "xgb":
            model = clf_model(
                **best_params,
                use_label_encoder=False,
                eval_metric=eval_metric,
                random_state=model_config["random_state"]
            )

        elif model_config["model_name"] == "lgb":
            model = clf_model(
                **best_params,
                random_state=model_config["random_state"]
            )
        
        elif model_config["model_name"] == "ExtraTree":
            model = clf_model(
                **best_params,
                random_state=model_config["random_state"]
            )

        model.fit(
            xtrain,
            ytrain,
            verbose=False, 
            early_stopping_rounds=early_stopping_rounds,
            eval_set=[(xtest, ytest)]
        )
        if use_predict_proba:
            ypred = model.predict_proba(xtest)
        else:
            ypred = model.predict(xtest)
            if model_config["test_path"] is not None:
                logger.info(">> Preiction on test file")
                test_pred = model.predict(X_test)
                test_prediction.append(test_pred)

        # metrics calculation
        '''
        models.py : we have to create a function calculate, that will measure model performance
        '''
        metrics_dict = metrics.calculate(ytest, ypred) # TODO : write this function
        scores.append(metrics_dict)
        logger.info(f">> Fold {fold} done")
    # create a function that take mean separately take all values from list and print the eval_metrics
    mean_metrics = dict_mean(scores)
    logger.info(f"Metrics: {mean_metrics}")