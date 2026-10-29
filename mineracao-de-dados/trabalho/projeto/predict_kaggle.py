"""
Make predictions with the best model for submission
"""

from joblib import load
import os

import pandas as pd

from format_data import normalize_data, format_columns
from constants import DATASET_EVAL_FILE, RAW_COLUMNS, OUTPUTS_PATH, DATASET_PATH

if __name__ == "__main__":

    model_file_path = os.path.join(
            OUTPUTS_PATH, 
            'augmented', 
            'best-params', 
            'raw', 
            'boosting_xgboost',
            'model.pkl')
    with open(model_file_path, 'rb') as model_file:
        print("[i] Loading parts ...")
        model = load(model_file)
        comp_df = pd.read_csv(DATASET_EVAL_FILE)

        print('[i] Normalizing data ...')
        format_columns(comp_df)
        comp_df = normalize_data(comp_df)

        print("[i] Predicting ...")
        pred = model.predict(comp_df[RAW_COLUMNS])
        comp_df['Diagnosis'] = pred

        print("[i] saving predictions ...")
        csv_path = os.path.join(DATASET_PATH, 'pred.csv')
        comp_df[['PatientID', 'Diagnosis']].to_csv(csv_path, index=False)


