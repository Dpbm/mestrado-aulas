"""
List of models using during this project.

This file, holds each model by category in order
to allow us easily retrieve the models as we wish.
"""

from typing import Dict,Any,Tuple,Callable,List
import json
import shutil
import os
import sys
import time
from joblib import dump
from functools import partial
from itertools import product
from multiprocessing import Pool, Manager

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, StackingClassifier

from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import get_scorer

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import shap
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from constants import (
    RANDOM_STATE, 
    OUTPUTS_PATH, 
    RAW_COLUMNS,
    CAIDE_COLUMNS,
    OURS_COLUMNS,
    TARGET
)
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


CV_SCORING = [
    'accuracy',
    'f1',
    'f1_micro',
    'f1_macro',
    'precision',
    'recall',
    'jaccard',
    'roc_auc'
]
CV_FOLDS = 5
TEST_SPLIT = 0.2

def get_models() -> Tuple[Dict[str,Any], int, List[str]]:
    """
    Get the models dict, the amount of models and the key paths in the dictionary
    """

    models = {
        "baseline":{
            "logistic-regression": partial(LogisticRegression,random_state=RANDOM_STATE),
            "svm-classifier": partial(SVC, random_state=RANDOM_STATE),
            "decision-tree": partial(DecisionTreeClassifier,random_state=RANDOM_STATE),
        },
        "boosting":{
            "xgboost": partial(XGBClassifier,random_state=RANDOM_STATE),
            "adaboost":partial(AdaBoostClassifier,random_state=RANDOM_STATE),
            "catboost": partial(CatBoostClassifier, random_seed=RANDOM_STATE),
            "lightgbm": partial(LGBMClassifier, random_state=RANDOM_STATE),
        },
        "forest":{
            "random": partial(RandomForestClassifier,random_state=RANDOM_STATE),
        },
        "bagging":{
            "decision-tree": partial(BaggingClassifier, estimator=DecisionTreeClassifier(random_state=RANDOM_STATE), random_state=RANDOM_STATE),
            "logistic-regression": partial(BaggingClassifier, estimator=LogisticRegression(random_state=RANDOM_STATE), random_state=RANDOM_STATE),
            "svm-classifier": partial(BaggingClassifier, estimator=SVC(random_state=RANDOM_STATE), random_state=RANDOM_STATE)
        },
        "stacking":{
            "forest": partial(StackingClassifier, estimators=[('forest',RandomForestClassifier(random_state=RANDOM_STATE))]),
            "logistic": partial(StackingClassifier, estimators=[('logistic',LogisticRegression(random_state=RANDOM_STATE))]),
            "svm": partial(StackingClassifier, estimators=[('svm',SVC(random_state=RANDOM_STATE))]),
            "tree": partial(StackingClassifier, estimators=[('tree',DecisionTreeClassifier(random_state=RANDOM_STATE))]),
            "all": partial(
                    StackingClassifier, 
                    estimators=[
                        ('forest',RandomForestClassifier(random_state=RANDOM_STATE)),
                        ('logistic',LogisticRegression(random_state=RANDOM_STATE)),
                        ('svm',SVC(random_state=RANDOM_STATE)),
                        ('tree',DecisionTreeClassifier(random_state=RANDOM_STATE))
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

    # this is not efficient but works by now
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

    cols = {
        'raw':RAW_COLUMNS,
        'ours':OURS_COLUMNS,
        'CAIDE':CAIDE_COLUMNS
    }

    # get the partial and instantiate the object
    model = get_model_by_key(model_key_path)()

    start_time_cv = time.perf_counter()
    cv_results = cross_validate(
        model,
        X_train,
        y_train, 
        scoring=CV_SCORING, 
        cv=CV_FOLDS,
        return_train_score=True
        )
    end_time_cv = time.perf_counter()
    
    start_time_inference = time.perf_counter()
    y_pred = model.predict(X_test)
    end_time_inference = time.perf_counter()


    target_folder_path = os.path.join(OUTPUTS_PATH, data_type, type_of_model, experiment_name, model_key_path)
    if os.path.exists(target_folder_path):
        shutil.rmtree(target_folder_path,ignore_errors=True)
    os.makedirs(target_folder_path, exist_ok=True)

    np.save(os.path.join(target_folder_path, 'output_cv_train.npy'),  cv_results)
    np.save(os.path.join(target_folder_path,'y_pred.npy'), y_pred)
    
    with open(os.path.join(target_folder_path, 'time.json'), 'w') as file:
        json.dump({
            'start_train_cv':start_time_cv,
            'end_train_cv':end_time_cv,
            'elapsed_time_cv':end_time_cv-start_time_cv,
            'start_train_inference':start_time_inference,
            'end_train_inference':end_time_inference,
            'elapsed_time_inference':end_time_inference-start_time_inference,
        },file)

    test_metrics = {}
    for scoring_method in CV_SCORING:
        scorer = get_scorer(scoring_method)._score_func
        score = scorer(y_test,y_pred)
        test_metrics[scoring_method] = score
    
    with open(os.path.join(target_folder_path, 'output_test_metrics.json'), 'w') as file:
        json.dump(test_metrics,file)

    if ['tree','forest'] in model_key_path:
        t = model if 'tree' in model_key_path else model.estimators_[0]
        plot_tree(
            t,
            feature_names=cols[experiment_name],
            class_names=TARGET,
            filled=True,  
            rounded=True, 
            fontsize=14, 
        )
        plt.title(f"{model_key_path.replace('_','')} - {data_type} - {type_of_model} model - {experiment_name} dataset columns")
        plt.savefig(os.path.join(target_folder_path, 'tree.png'), dpi=300, bbox_inches='tight')
        plt.close('all')

    explainer = shap.Explainer(model)

    shap_values_train = explainer(X_train)
    shap_values_test = explainer(X_test)

    shap_df_train = pd.DataFrame(shap_values_train.values, columns=X_train.columns)
    shap_df_test = pd.DataFrame(shap_values_test.values, columns=X_test.columns)

    shap_df_train.to_csv(os.path.join(target_folder_path, 'shape_values_train.csv'), index=False)
    shap_df_test.to_csv(os.path.join(target_folder_path, 'shape_values_test.csv'), index=False)

    with open("model.pkl", "wb") as file:
        dump(model, file, protocol=5) 


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
            
            target_experiment_folder = os.path.join(OUTPUTS_PATH, 'df', 'default', name)
            os.makedirs(target_experiment_folder, exist_ok=True)

            # save the modified datasets for later analysis
            for t in ['augmented', 'not_augmented']:
                np.save(os.path.join(target_experiment_folder, f'{t}_x_train.npy'), data[t][0])
                np.save(os.path.join(target_experiment_folder, f'{t}_x_test.npy'), data[t][1])
                np.save(os.path.join(target_experiment_folder, f'{t}_y_train.npy'), data[t][2])
                np.save(os.path.join(target_experiment_folder, f'{t}_y_test.npy'), data[t][3])

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

    if os.path.exists(OUTPUTS_PATH):
        print(f"  THE PATH {OUTPUTS_PATH} already exists")
        print("   This may indicate that You've already run this before!")
        print("   By running it again it will delete all the previous files")
        print("   Are you sure of that?")

        if(input('[y/n]: ').lower() != 'y'):
            sys.exit(0)

        shutil.rmtree(OUTPUTS_PATH)
        os.makedirs(OUTPUTS_PATH, exist_ok=True)

    run_default_models()




