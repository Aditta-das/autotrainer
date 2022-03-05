import xgboost as xgb
import lightgbm as lgb
from sklearn import metrics as skmet
from sklearn.ensemble import ExtraTreesClassifier

def fetch_model(model_config):
    if model_config["problem_type"] == "binary_classification" or model_config["problem_type"] == "multi_classification":
        if model_config["model_name"] == "xgb":
            clf_model = xgb.XGBClassifier
            use_predict_proba = True
            direction = "minimize"
            eval_metric = "logloss"
        elif model_config["model_name"] == "lgb":
            clf_model = lgb.LGBMClassifier
            use_predict_proba = True
            direction = "minimize"
            eval_metric = "logloss"
        elif model_config["model_name"] == "ExtraTree":
            clf_model = ExtraTreesClassifier
            use_predict_proba = True
            direction = "minimize"
            eval_metric = "logloss"
    return clf_model, use_predict_proba, direction, eval_metric


class Valid_Metrics:
    def __init__(self, model_config):
        self.problem = model_config["problem_type"]
        if self.problem == "binary_classification":
            self.valid_metrics = {
                "auc": skmet.roc_auc_score,
                "logloss": skmet.log_loss,
                "f1": skmet.f1_score,
                "accuracy": skmet.accuracy_score,
                "precision": skmet.precision_score,
                "recall": skmet.recall_score,
            }
        elif self.problem == "multi_classification":
            self.valid_metrics = {
                "logloss": skmet.log_loss,
                "accuracy": skmet.accuracy_score,
                "mlogloss": skmet.log_loss,
            }

    def calculate(self, ytest, ypred):
        metrics = {}
        for met_name, met_func in self.valid_metrics.items():
            if self.problem == "binary_classification":
                if met_name == "auc":
                    metrics[met_name] = met_func(ytest, ypred[:, 1])
                elif met_name == "logloss":
                    metrics[met_name] = met_func(ytest, ypred)
                else:
                    metrics[met_name] = met_func(ytest, ypred[:, 1] >= 0.5)
        
        return metrics


