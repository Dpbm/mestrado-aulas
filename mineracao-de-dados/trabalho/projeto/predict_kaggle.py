"""
Make predictions with the best model for submission
"""

from joblib import load
import os

import pandas as pd

from format_data import normalize_data, format_columns
from constants import DATASET_EVAL_FILE, OUTPUTS_PATH, DATASET_PATH, FEATURES_LISTS

if __name__ == "__main__":
    for columns_name,columns in FEATURES_LISTS:
        print(f'[i] Using columns: {columns_name} ')

        model_file_path = os.path.join(
                OUTPUTS_PATH, 
                'augmented', 
                'best-params' if columns_name != 'ours' else 'default', 
                columns_name, 
                'boosting_xgboost' if columns_name != 'ours' else 'stacking_logistic',
                'model.pkl')

        with open(model_file_path, 'rb') as model_file:
            print("[i] Loading parts ...")
            model = load(model_file)
            comp_df = pd.read_csv(DATASET_EVAL_FILE)

            print('[i] Normalizing data ...')
            format_columns(comp_df)
            comp_df = normalize_data(comp_df)

            print("[i] Predicting ...")
            pred = model.predict(comp_df[columns])
            comp_df['Diagnosis'] = pred

            print("[i] saving predictions ...")
            csv_path = os.path.join(DATASET_PATH, f'pred_{columns_name}.csv')
            comp_df[['PatientID', 'Diagnosis']].to_csv(csv_path, index=False)


