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

from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV
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
            "logistic-regression": {
                "model":partial(LogisticRegression,random_state=RANDOM_STATE),
                "explainer": partial(shap.LinearExplainer, seed=RANDOM_STATE),
                "params":{
                    "penalty": [None, 'l1', 'l2', 'elasticnet'],
                }
            },
            "svm-classifier": {
                "model":partial(SVC, random_state=RANDOM_STATE),
                "explainer":partial(shap.KernelExplainer, seed=RANDOM_STATE),
                "params":{
                    "kernel":['rbf', 'linear', 'poly', 'sigmoid']
                }
            },
            "decision-tree": {
                "model":partial(DecisionTreeClassifier,random_state=RANDOM_STATE),
                "explainer":partial(shap.TreeExplainer, seed=RANDOM_STATE),
                'params':{
                    'criterion': ['gini','entropy', 'log_loss'],
                    'max_depth': [None, 3, 5, 10, 50],
                    'min_samples_split': [2, 10, 20, 30]
                }
            },
        },
        "boosting":{
            "xgboost": {
                "model":partial(XGBClassifier,random_state=RANDOM_STATE, device="cuda"),
                "explainer":partial(shap.TreeExplainer, seed=RANDOM_STATE),
                'params':{
                    'eta':[0.1, 0.3, 0.5, 0.7],
                    'gamma': [0, 0.1, 0.5, 1],
                    'max_depth':[2, 3, 5, 6]
                }
            },
            "adaboost":{
                "model":partial(AdaBoostClassifier,random_state=RANDOM_STATE),
                "explainer":partial(shap.TreeExplainer, seed=RANDOM_STATE),
                'params': {
                    'n_estimators': [30, 50, 70],
                    'learning_rate': [0.5, 0.8, 1]
                }
            },
        },
        "forest":{
            "random": {
                "model":partial(RandomForestClassifier,random_state=RANDOM_STATE),
                "explainer":partial(shap.TreeExplainer, seed=RANDOM_STATE),
                "params":{
                    'criterion': ['gini','entropy', 'log_loss'],
                    'max_depth': [None, 3, 5, 10, 50],
                    'min_samples_split': [2, 10, 20, 30],
                    'max_features':['sqrt','log2',None]
                }
            },
        },
        "bagging":{
            "tree": {
                "model":partial(BaggingClassifier, estimator=DecisionTreeClassifier(random_state=RANDOM_STATE), random_state=RANDOM_STATE),
                "explainer":partial(shap.TreeExplainer, seed=RANDOM_STATE),
                'params':{
                    'n_estimators':[5, 10, 20, 30],
                }
            },
            "logistic": {
                "model":partial(BaggingClassifier, estimator=LogisticRegression(random_state=RANDOM_STATE), random_state=RANDOM_STATE),
                "explainer":partial(shap.LinearExplainer, seed=RANDOM_STATE),
                'params':{
                    'n_estimators':[5, 10, 20, 30],
                }
            },
            "svm": {
                "model":partial(BaggingClassifier, estimator=SVC(random_state=RANDOM_STATE), random_state=RANDOM_STATE),
                "explainer":partial(shap.KernelExplainer, seed=RANDOM_STATE),
                'params':{
                    'n_estimators':[5, 10, 20, 30],
                }
            },
        },
        "stacking":{
            "forest": {
                "model":partial(
                    StackingClassifier, 
                    estimators=[('forest',RandomForestClassifier(random_state=RANDOM_STATE))],
                    cv=CV_FOLDS
                ),
                "explainer":partial(shap.TreeExplainer, seed=RANDOM_STATE)
            },
            "logistic": {
                "model":partial(
                    StackingClassifier, 
                    estimators=[('logistic',LogisticRegression(random_state=RANDOM_STATE))],
                    cv=CV_FOLDS
                ),
                "explainer":partial(shap.LinearExplainer, seed=RANDOM_STATE)
            },
            "svm": {
                "model":partial(
                    StackingClassifier, 
                    estimators=[('svm',SVC(random_state=RANDOM_STATE))],
                    cv=CV_FOLDS
                ),
                "explainer":partial(shap.KernelExplainer, seed=RANDOM_STATE)
            },
            "tree": {
                "model":partial(
                    StackingClassifier, 
                    estimators=[('tree',DecisionTreeClassifier(random_state=RANDOM_STATE))],
                    cv=CV_FOLDS
                ),
                "explainer":partial(shap.TreeExplainer, seed=RANDOM_STATE)
            },
            "all": {
                "model":partial(
                    StackingClassifier, 
                    estimators=[
                        ('forest',RandomForestClassifier(random_state=RANDOM_STATE)),
                        ('logistic',LogisticRegression(random_state=RANDOM_STATE)),
                        ('svm',SVC(random_state=RANDOM_STATE)),
                        ('tree',DecisionTreeClassifier(random_state=RANDOM_STATE))
                    ],
                    cv=CV_FOLDS
                ),
                "explainer":partial(shap.TreeExplainer, seed=RANDOM_STATE)
            },
        }
    }

    key_paths = []
    for k1 in models.keys():
        for k2 in models[k1].keys():
            key_paths.append(f'{k1}_{k2}')

    TOTAL_MODELS = 16
    return models,TOTAL_MODELS,key_paths

def get_model_by_key(key:str) -> Dict[str,Any]:
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

def save_time(
    train_start:float,
    train_end:float,
    inf_start:float,
    inf_end:float,
    path:str
):
    """
    Save time statistics into a json file
    """
    with open(path, 'w') as file:
        json.dump({
            'start_train':train_start,
            'end_train':train_end,
            'elapsed_time':train_end-train_start,
            'start_inference':inf_start,
            'end_inference':inf_end,
            'elapsed_time_inference':inf_end-inf_start,
        },file)

def save_test_metrics(
    y_test:np.ndarray,
    y_pred:np.ndarray,
    path:str
):
    """
    retrieve model test metrics and save into a json file
    """
    with open(path, 'w') as file:
        test_metrics = {}
        for scoring_method in CV_SCORING:
            scorer = get_scorer(scoring_method)._score_func
            score = scorer(y_test,y_pred)
            test_metrics[scoring_method] = score
        json.dump(test_metrics,file)

def save_tree(
    model:Any,
    cols:List[str],
    title:str,
    path:str
):
    """
    Save tree plot for tree and forest based methods
    """
    plot_tree(
        model,
        feature_names=cols,
        class_names=["0","1"],
        filled=True,  
        rounded=True, 
    )
    plt.title(title)
    plt.savefig(
        path, 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.close('all')

def explain_model(
    model:Any,
    explainer:Any,
    X_train:np.ndarray,
    cols:List[str],
    path:str
):
    """
    Use Shap to explain the importante of each feature
    """

    exp = None 
    try:
        exp = explainer(model, X_train, feature_names=cols)
    except TypeError:
        # for errors like:
        # TypeError: The passed model is not callable and cannot be analyzed directly with the given masker! Model: SVC()

        # https://stackoverflow.com/questions/71117308/exception-the-passed-model-is-not-callable-and-cannot-be-analyzed-directly-with
        # https://github.com/shap/shap/issues/2399
        exp = explainer(model.predict, X_train, feature_names=cols)
        
    shap_values = explainer(X_test)
    shap_df = pd.DataFrame(shap_values.values, columns=cols)
    shap_df.to_csv(path, index=False)

def export_model(model:Any, path:str):
    """
    Save model to pickle
    """
    
    with open(path, "wb") as file:
        dump(model, file, protocol=5) 



def run_model(
    model:Any, 
    explainer:Any,
    cols:List[str],
    target_folder_path:str,
    X_train:np.ndarray, 
    X_test:np.ndarray, 
    y_train:np.ndarray, 
    y_test:np.ndarray
):
    """
    Run a single model and extract metrics and 
    other statistics.

    For training, it uses Cross validation.
    """
    start_time_cv = time.perf_counter()
    cv_results = cross_validate(
        model,
        X_train,
        y_train, 
        scoring=CV_SCORING, 
        cv=CV_FOLDS,
        return_train_score=True,
        return_estimator=True,
        return_indices=True,
        )
    end_time_cv = time.perf_counter()

    model_best_precision = int(np.argmax(cv_results['test_precision']))
    model = cv_results["estimator"][model_best_precision]
    
    start_time_inference = time.perf_counter()
    y_pred = model.predict(X_test)
    end_time_inference = time.perf_counter()
    
    np.save(os.path.join(target_folder_path,'y_pred.npy'), y_pred)

    df_cv = pd.DataFrame(cv_results[[key for key in cv_results.keys() if key != 'estimator']])
    df_cv.to_csv(os.path.join(target_folder_path, 'output_cv_train.csv'), index=False)


    save_time(
        start_time_cv, 
        end_time_cv, 
        start_time_inference, 
        end_time_inference, 
        os.path.join(target_folder_path, 'time.json'))

    save_test_metrics(
        y_test,
        y_pred,
        os.path.join(target_folder_path, 'output_test_metrics.json')
    ) 

    if 'tree' in model_key_path or 'forest' in model_key_path:
        selected_tree = model if 'decision-tree' in model_key_path else model.estimators_[0]
        title =  f"{model_key_path.replace('_',' ')} - {data_type} - {type_of_model} model - {experiment_name} dataset columns"
        save_tree(
            selected_tree,
            cols,
            title,
            os.path.join(target_folder_path, 'tree.png')
        )
    
    explain_model(
        model,
        explainer, 
        X_train,
        cols,
        os.path.join(target_folder_path, 'shap_values.csv')
    )

    export_model(
        model,
        os.path.join(target_folder_path,"model.pkl")
    )


def run_default_models(data:Tuple[Any,Any]):
    """
    Run default models based on a mod.
    """
    task,dataset = data
    data_type, type_of_model, experiment_name, model_key_path = task
    
    X_train,X_test,y_train,y_test = dataset.augmented \
                                        if data_type == "augmented" \
                                        else \
                                        dataset.not_augmented

    # get the partial and instantiate the object
    model_data = get_model_by_key(model_key_path)
    model = model_data['model']()
    explainer = model_data['explainer']()

    target_folder_path = os.path.join(OUTPUTS_PATH, data_type, type_of_model, experiment_name, model_key_path)
    if os.path.exists(target_folder_path):
        shutil.rmtree(target_folder_path, ignore_errors=True)
    os.makedirs(target_folder_path, exist_ok=True)

    run_model(
        model,
        explainer,
        dataset.cols,
        target_folder_path,
        X_train,
        X_test,
        y_train,
        y_test
    )

def run_best_params(data:Any):
    """
    Search for each model best set of params
    and then evaluate metrics
    """
    task,dataset = data
    data_type, type_of_model, experiment_name, model_key_path = task
    
    X_train,X_test,y_train,y_test = dataset.augmented \
                                        if data_type == "augmented" \
                                        else \
                                        dataset.not_augmented

    # get the partial and instantiate the object
    model_data = get_model_by_key(model_key_path)
    model = model_data['model']()
    explainer = model_data['explainer']()

    search_cv = GridSearchCV(
        model,
        cv=CV_FOLDS,
        scoring=CV_SCORING,
        param_grid=model_data['params'],
        return_train_score=True,
        refit=True
    )

    start_time_cv = time.perf_counter()
    search_cv.fit(X_train, y_train)
    end_time_cv = time.perf_counter()
    
    target_folder_path = os.path.join(OUTPUTS_PATH, data_type, type_of_model, experiment_name, model_key_path)
    if os.path.exists(target_folder_path):
        shutil.rmtree(target_folder_path, ignore_errors=True)
    os.makedirs(target_folder_path, exist_ok=True)

    df = pd.DataFarme(search_cv.cv_results_)
    df.to_csv(os.path.join(target_folder_path, f'grid_search_output.csv'), index=False)

    best_model = search_cv.best_estimator_
    start_time_inference = time.perf_counter()
    y_pred = best_model.predict(X_test)
    end_time_inference = time.perf_counter()

    np.save(os.path.join(target_folder_path,'y_pred.npy'), y_pred)

    save_time(
        start_time_cv,
        end_time_cv,
        start_time_inference,
        end_time_inference,
        os.path.join(target_folder_path, 'time.json')
    )

    save_test_metrics(
        y_test,
        y_pred,
        os.path.join(target_folder_path, 'output_test_metrics.json')
    )

    if 'tree' in model_key_path or 'forest' in model_key_path:
        selected_tree = model if 'decision-tree' in model_key_path else model.estimators_[0]
        title = f"{model_key_path.replace('_',' ')} - {data_type} - {type_of_model} model - {experiment_name} dataset columns"
        save_tree(
            selected_tree,
            cols,
            title,
            os.path.join(target_folder_path, 'tree.png')
        )

    explain_model(
        model,
        explainer, 
        X_train,
        cols,
        os.path.join(target_folder_path, 'shap_values.csv')

    )

    export_model(
        model,
        os.path.join(target_folder_path,"model.pkl")
    )


def run_all():
    _,total_models,keys = get_models()

    for experiment_type, experiment_func in zip(
        ('default', 'best-params'),
        (run_default_model, run_best_params_model)
        ):
        for name,cols,cols_func in tqdm(
                    zip(
                        ('raw', 'ours', 'CAIDE'),
                        (RAW_COLUMNS, OURS_COLUMNS, CAIDE_COLUMNS),
                        (get_dataset_raw_columns, get_dataset_feature_engineered_columns, get_dataset_CAIDE_columns)
                    )):

            data = setup_data(cols_func)
            with Manager() as mg:
                dataset = mg.Namespace()
                dataset.not_augmented = data["not_augmented"]
                dataset.augmented = data["augmented"]
                dataset.cols = cols
                
                # save the modified datasets for later analysis
                for t in ['augmented', 'not_augmented']:
                    np.save(os.path.join(OUTPUTS_PATH, t, experiment_type, name, f'{t}_x_train.npy'), data[t][0])
                    np.save(os.path.join(OUTPUTS_PATH, t, experiment_type, name, f'{t}_x_test.npy'), data[t][1])
                    np.save(os.path.join(OUTPUTS_PATH, t, experiment_type, name, f'{t}_y_train.npy'), data[t][2])
                    np.save(os.path.join(OUTPUTS_PATH, t, experiment_type, name, f'{t}_y_test.npy'), data[t][3])

                # *2 because we are going to run both augmented and not augmented at once
                with Pool(processes=total_models*2) as pool:
                    tasks = list(
                        product(
                            ['augmented', 'not_augmented'], 
                            [experiment_type], 
                            [name],
                            keys)
                    )
                    pool.map(experiment_func, [ (task, dataset) for task in tasks])

if __name__ == "__main__":
    if os.path.exists(OUTPUTS_PATH):
        print(f"  THE PATH {OUTPUTS_PATH} already exists")
        print("   This may indicate that You've already run this before!")
        print("   By running it again it will delete all the previous files")
        print("   Are you sure of that?")

        if(input('[y (delete) / any key (cancel)]: ').lower() != 'y'):
            sys.exit(0)

        shutil.rmtree(OUTPUTS_PATH)
        os.makedirs(OUTPUTS_PATH, exist_ok=True)

    run_all()



