"""
List of models using during this project.

This file, holds each model by category in order
to allow us easily retrieve the models as we wish.
"""

from typing import Dict,Any,Tuple,Callable,List
from functools import partial
from itertools import product
from multiprocessing import Pool, Manager

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, StackingClassifier

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

import numpy as np
import pandas as pd
from tqdm import tqdm

from constants import RANDOM_STATE, TEST_SPLIT
from format_data import (
    load_and_format_dataset,
    get_dataset_target,
    augment_data,
    get_dataset_raw_columns,
    get_dataset_CAIDE_columns,
    get_dataset_feature_engineered_columns
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

def get_models() -> Tuple[Dict[str,Any], int, List[str]]:
    """
    Get the models dict, the amount of models and the key paths in the dictionary
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

    key_paths = []
    for k1 in models.keys():
        for k2 in models[k1].keys():
            key_paths.append(f'{k1}_{k2}')

    TOTAL_MODELS = 16
    return models,TOTAL_MODELS,key_paths

def get_model_by_key(key:str) -> Any:
    """
    Resolve the key path and return the model
    """

    models, _, _ = get_models()

    key_split = key.split('_')
    k1,k2 = key_split[0],key_split[1]

    return models[k1][k2]


Splits = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
def hold_out_dataset(X:pd.DataFrame, y:pd.Series, test_size:float=TEST_SPLIT) -> Splits:
    """
    Split the dataset into train and test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)

def setup_data(columns_func:Callable, test_size:float=TEST_SPLIT) -> Dict[str, Splits]:
    """
    Setup dataset

    it loads, format, normalize and split the data as we need
    for training the models. Both augmented and not augmented.
    """

    df = load_and_format_dataset()
    y = get_dataset_target(df)
    X = columns_func(df)

    not_augmented = hold_out_dataset(X,y, test_size=test_size)

    augmented_X, augmented_y = augment_data(X,y)
    augmented = hold_out_dataset(augmented_X, augmented_y, test_size=test_size)

    return {"not_augmented":not_augmented, "augmented":augmented}

def run_model(data):
    task,dataset = data

    data_type, type_of_model, experiment_name, model_key_path = task
    
    X_train,X_test,y_train,y_test = dataset.augmented \
                                        if data_type == "augmented" \
                                        else \
                                        dataset.not_augmented
    model = get_model_by_key(model_key_path)



def run_default_models():
    """
    Run default models based on a models dict key.
    """

    _,total_models,keys = get_models()

    for name,cols_func in tqdm(
                zip(
                    ('raw', 'ours', 'CAIDE'),
                    (get_dataset_raw_columns, get_dataset_feature_engineered_columns, get_dataset_CAIDE_columns)
                )):

        data = setup_data(cols_func)

        with Manager() as mg:
            dataset = mg.Namespace()
            dataset.not_augmented = data["not_augmented"]
            dataset.augmented = data["augmented"]

            # *2 because we are going to run both augmented and not augmented at once
            with Pool(processes=total_models*2) as pool:
                tasks = list(
                    product(
                        ['augmented', 'not_augmented'], 
                        ['default'], 
                        [name],
                        keys)
                )

                pool.map(run_model, [ (task, dataset) for task in tasks])



if __name__ == "__main__":
    run_default_models()




