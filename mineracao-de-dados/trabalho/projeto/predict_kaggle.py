"""
Make predictions with the best model for submission
"""

from joblib import load
import os

import pandas as pd

from format_data import normalize_data, format_columns
from constants import (
    DATASET_EVAL_FILE, 
    OUTPUTS_PATH, 
    DATASET_PATH, 
    FEATURES_LISTS, 
    SELECTED_BEST_FEATURES,
    BEST_PATH,
    FEATURES_MAPPING
)


if __name__ == "__main__":
    TARGET_DF_TYPE = "augmented"

    df = pd.read_csv(os.path.join(DATASET_PATH, "best_models.csv"))
    df = df.loc[df["dataset_type"] == TARGET_DF_TYPE]

    for row in df.itertuples():
        print(f'[i] Using columns: {row.columns} ')
        path = os.path.join(OUTPUTS_PATH, TARGET_DF_TYPE, row.params_type, row.columns, row.model, "model.pkl") \
                    if row.columns != "best" \
                    else \
                os.path.join(OUTPUTS_PATH, "best-features", TARGET_DF_TYPE, row.params_type, "best-features", row.model, "model.pkl") 

        print(f"model path: {path}")
        with open(path, 'rb') as model_file:
            print("[i] Loading parts ...")
            model = load(model_file)
            comp_df = pd.read_csv(DATASET_EVAL_FILE)

            print('[i] Normalizing data ...')
            format_columns(comp_df)
            comp_df = normalize_data(comp_df)

            print("[i] Predicting ...")
            pred = model.predict(comp_df[FEATURES_MAPPING[row.columns]])
            comp_df['Diagnosis'] = pred

            print("[i] saving predictions ...")
            csv_path = os.path.join(DATASET_PATH, f'pred_{row.columns}.csv')
            comp_df[['PatientID', 'Diagnosis']].to_csv(csv_path, index=False)