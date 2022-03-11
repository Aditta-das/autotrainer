import optuna
from .cfg import CFG
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

def without_compare():
    model_params = {
        "SVM": {
            "model": SVC(gamma="auto"),
            "params":{
                "C": [1, 10, 20],
                "kernel": ["rbf", "linear"]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(),
            "params": {
                "n_estimators": [1, 5, 10]
            }
        },
        "LogisticRegression":{
            "model": LogisticRegression(solver="liblinear", multi_class="auto"),
            "params": {
                "C": [1, 10, 20]
            }
        },
        "ExtraTree": {
            "model": ExtraTreesClassifier(),
            "params": {
                "n_estimators" : [1, 5, 10]
            }
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(),
            "params": {
                "criterion": ["gini", "entropy"]
            }
        },
    }
    return model_params

def get_params(trial, model_config):
    if model_config["model_name"] == "ExtraTree":
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [7000, 15000, 20000]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 1, 9),
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 100, 500),
            "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"]),
        }

    elif model_config["model_name"] == "xgb":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.25, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "max_depth": trial.suggest_int("max_depth", 1, 9),
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 100, 500),
            "n_estimators": trial.suggest_categorical("n_estimators", [7000, 15000, 20000]),
        }
        if model_config["use_gpu"]:
            params["tree_method"] = "gpu_hist"
            params["gpu_id"] = 0
            params["predictor"] = "gpu_predictor"
        else:
            pass
            # params["tree_method"] = trial.suggest_categorical("tree_method", ["exact", "approx", "hist"])
            # params["booster"] = trial.suggest_categorical("booster", ["gbtree", "gblinear"])
            # if params["booster"] == "gbtree":
            #     params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            #     params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
                
    elif model_config["model_name"] == "lgb":
        params = {
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int('num_leaves', 20, 3000, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
            "max_bin": trial.suggest_int("max_bin", 200, 300),
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.2, 0.95, step=0.1
            ),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.2, 0.95, step=0.1
            ),
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 100, 500),
        }

        if model_config["use_gpu"] == True:
            params["device_type"] = "gpu"
    return params