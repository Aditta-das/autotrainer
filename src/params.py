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
        }
    }
    return model_params