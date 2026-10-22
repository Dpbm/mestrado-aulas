"""
List of models using during this project.

This file, holds each model by category in order
to allow us easily retrieve the models as we wish.
"""

from typing import Dict,Any
from functools import partial
from multiprocessing import Pool, Manager

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, StackingClassifier

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

import pandas as pd

from constants import RANDOM_STATE
from format_data import (
    load_and_format_dataset
)

"""
    The combinations WE GONNA RUN are:

    | AUG | params   | columns      |
    |-----|----------|--------------|
    | NO  | default  | raw          |
    | NO  | best     | raw          |
    | NO  | default  | ours         |
    | NO  | best     | ours         |
    | NO  | default  | raw-CAIDE    |
    | NO  | best     | raw-CAIDE    |
    |-----|----------|--------------|
    | YES | default  | raw          |
    | YES | best     | raw          |
    | YES | default  | ours         |
    | YES | best     | ours         |
    | YES | default  | raw-CAIDE    |
    | YES | best     | raw-CAIDE    |

    so 12 experiments, across 16 models
"""

keys_amounts = {"baseline":3, "boosting":4, "forest":1, "bagging":3, "stacking":5}

def get_models_by_key(key:str) -> Dict[str,Any]:
    """
    Get the models obj by its class key
    """

    models = {
        "baseline":{
            "logistic-regression": LogisticRegression,
            "svm-classifier": SVC,
            "decision-tree": DecisionTreeClassifier,
        },
        "boosting":{
            "xgboost": XGBClassifier,
            "adaboost":AdaBoostClassifier,
            "catboost": CatBoostClassifier,
            "lightgbm": lgb
        },
        "forest":{
            "random": RandomForestClassifier,
        },
        "bagging":{
            "decision-tree": partial(BaggingClassifier, estimator=DecisionTreeClassifier()),
            "logistic-regression": partial(BaggingClassifier, estimator=LogisticRegression()),
            "svm-classifier": partial(BaggingClassifier, estimator=SVC())
        },
        "stacking":{
            "forest": partial(StackingClassifier, estimators=[('forest',RandomForestClassifier())]),
            "logistic": partial(StackingClassifier, estimators=[('logistic',LogisticRegression())]),
            "svm": partial(StackingClassifier, estimators=[('svm',SVC())]),
            "tree": partial(StackingClassifier, estimators=[('tree',DecisionTreeClassifier())]),
            "all": partial(
                    StackingClassifier, 
                    estimators=[
                        ('forest',RandomForestClassifier()),
                        ('logistic',LogisticRegression()),
                        ('svm',SVC()),
                        ('tree',DecisionTreeClassifier())
                    ]
                ),
        }
    }

    return models[key]

def run_default_model(dataset:pd.DataFrame, class_key:str, model_i:int):
    """
    run a single model with default hyperparameters
    """



def hold_out_dataset(dataset:pd.DataFrame, test_size:float=0.2) -> :
    """
    Split the dataset into train and test.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)



def run_default_models(dataset:pd.DataFrame, key:str):
    """
    Run default models based on a models dict key.
    """

    amount_of_models = keys_amounts[key]


    with Pool(processes=amount_of_models) as pool:







