"""
Utilities for formating the dataset and applying the things we need.
"""

from typing import Tuple
import os
import shutil

import pandas as pd
import kagglehub
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

from constants import (
    RANDOM_STATE,
    DATASET_HANDLE,
    DATASET_EVAL_FILE,
    DATASET_TRAIN_FILE,
    TARGET, 
    RAW_COLUMNS, 
    OURS_COLUMNS, 
    CAIDE_COLUMNS,
)

def format_columns(df:pd.DataFrame):
    """
    Format columns and data from the given dataset.

    Meant to be used for both training and test CSV.  """
    
    # from table extracted from: https://media.geeksforgeeks.org/wp-content/uploads/20231206171725/BMI-2.png
    # original article at: https://www.geeksforgeeks.org/utilities/bmi-calculator/
    df["BMI_cat"] = 0 #"Underweight"
    df.loc[(df["BMI"] >= 18.5)&(df["BMI"] < 25), "BMI_cat"] = 1 #"Normal"
    df.loc[(df["BMI"] >= 25)&(df["BMI"] < 30), "BMI_cat"] = 2 #"Overweight"
    df.loc[df["BMI"] >= 30, "BMI_cat"] = 3 #"Obese"

    # given that the dataset is from EU, based on https://mindrink.me/alcohol-unit-calculator/
    # an alcohol unit is approximatelly 8g
    df["alcohol_grams_week"] = df["AlcoholConsumption"]*8
    df["alcohol_daily"] = df["alcohol_grams_week"]/7
    
    # for our case we need to have the data in minutes
    df["physical_activity_min"] = df["PhysicalActivity"]*60

    # Cholesterol is divided by 38.67 to convert from mm/dL to mmol/L https://heartcare.sydney/cholesterol-unit-conversion/
    df["cholesterol_mmol_L"] = df["CholesterolTotal"] / 38.67

    # -------------------------- INDICATORS ---------------------------------------------
    # Indicator derived from:
        # J. A. Pillai et al., “Metabolic syndrome biomarkers relate to rate of cognitive decline in MCI and dementia stages of Alzheimer’s disease,” Alz Res Therapy, vol. 15, no. 1, p. 54, Mar. 2023, doi: 10.1186/s13195-023-01203-y.

    # We thought about using the TyG-BMI in this case, as proposed in many articles:
        # Z. Zhang, X. Chen, and Z. Sheng, “Association of triglyceride glucose-body mass index with Alzheimer’s disease pathology, cognition and brain structure in non-demented people,” Sci Rep, vol. 14, no. 1, p. 16097, Jul. 2024, doi: 10.1038/s41598-024-67052-3.
        # Y. Cheng et al., “Association between triglyceride glucose-body mass index and cardiovascular outcomes in patients undergoing percutaneous coronary intervention: a retrospective study,” Cardiovasc Diabetol, vol. 22, no. 1, p. 75, Mar. 2023, doi: 10.1186/s12933-023-01794-8.
        # N. Tian et al., “Triglyceride–glucose index, Alzheimer’s disease plasma biomarkers, and dementia in older adults: The MIND-China study,” Alzheimer’s & Dementia: Diagnosis, Assessment & Disease Monitoring, vol. 15, no. 2, p. e12426, 2023, doi: 10.1002/dad2.12426.

    # However, It requires the glucose levels as well, which were not provided in the datase 
    # since we knew that Tryglicerides and HDL are also related to the disease, we decided to use this indicator  (T/HDL) in our study 

        # K. Nägga et al., “Increased midlife triglycerides predict brain β-amyloid and tau pathology 20 years later,” Neurology, vol. 90, no. 1, pp. e73–e81, Jan. 2018, doi: 10.1212/WNL.0000000000004749.
        # E. D. Moser et al., “Association Between Fluctuations in Blood Lipid Levels Over Time With Incident Alzheimer Disease and Alzheimer Disease–Related Dementias,” Neurology, vol. 101, no. 11, pp. e1127–e1136, Sep. 2023, doi: 10.1212/WNL.0000000000207595.
        # M. M. Bernath et al., “Serum triglycerides in Alzheimer disease,” Neurology, vol. 94, no. 20, pp. e2088–e2098, May 2020, doi: 10.1212/WNL.0000000000009436.
    df["T/HDL"] = df["CholesterolTriglycerides"]/df["CholesterolHDL"]
    
    # points were based on 
        # K. Dhana, D. A. Evans, K. B. Rajan, D. A. Bennett, and M. C. Morris, “Healthy lifestyle and the risk of Alzheimer dementia,” Neurology, vol. 95, no. 4, pp. e374–e383, Jul. 2020, doi: 10.1212/WNL.0000000000009816.

    df["health_points"] = 0
    df.loc[df["Smoking"] == 0, "health_points"] += 1
    df.loc[df["physical_activity_min"] >= 150, "health_points"] += 1
    df.loc[(df["alcohol_grams_week"] <= 15) & (df["Gender"] == 1), "health_points"] += 1 # women
    df.loc[(df["alcohol_grams_week"] <= 30) & (df["Gender"] == 0), "health_points"] += 1 # men

    # based on: 
        # K. M. Hurley, S. E. Oberlander, B. C. Merry, M. M. Wrobleski, A. C. Klassen, and M. M. Black, “The Healthy Eating Index and Youth Healthy Eating Index Are Unique, Nonredundant Measures of Diet Quality among Low-Income, African American Adolescents12,” The Journal of Nutrition, vol. 139, no. 2, pp. 359–364, Feb. 2009, doi: 10.3945/jn.108.097113.
        # and: https://www.dietquality.org/indicators/definitions
    df.loc[df["DietQuality"] >= 8, "health_points"] += 1
    
    # the cut for sleep quality was arbitrary and based on: 
        # Y.-E. S. Ju, B. P. Lucey, and D. M. Holtzman, “Sleep and Alzheimer disease pathology—a bidirectional relationship,” Nat Rev Neurol, vol. 10, no. 2, pp. 115–119, Feb. 2014, doi: 10.1038/nrneurol.2013.269.
    df.loc[df["SleepQuality"] >= 8, "health_points"] += 1
    
    # based on: 
        # T. T. Vu et al., “Favorable Cardiovascular Health at Young and Middle Ages and Dementia in Older Age—The CHA Study,” Journal of the American Heart Association, vol. 8, no. 1, p. e009730, Jan. 2019, doi: 10.1161/JAHA.118.009730.
    df["secondary_indicator"] = 2
    df.loc[(df["SystolicBP"] <= 120) & (df["DiastolicBP"] <= 80) & (df["cholesterol_mmol_L"] < 5.18) & (df["Smoking"] == 0) & (df["Diabetes"] == 0), "secondary_indicator"] = 3
    df.loc[(df["SystolicBP"] >= 140) | (df["DiastolicBP"] >= 90) | (df["cholesterol_mmol_L"] >= 6.22) | (df["Diabetes"] == 1) | (df["Smoking"] == 1), "secondary_indicator"] = 1


    # CAIDE Based on: 
        # “Risk score for the prediction of dementia risk in 20 years among middle aged people: a longitudinal, population-based study - The Lancet Neurology.” Accessed: Oct. 18, 2026. [Online]. Available: https://www.thelancet.com/journals/laneur/article/PIIS1474-4422(06)70537-3/abstract?isEOP=true

    df["CAIDE"] = 5 #everyone is >53 years old

    # since we have no information about education level, we are going to arbitrarily set the points here
    df.loc[df["EducationLevel"] <= 1, "CAIDE"] += 4
    df.loc[df["EducationLevel"] == 2, "CAIDE"] += 3

    df.loc[df["Gender"] == 0, "CAIDE"] += 1
    df.loc[df["SystolicBP"] > 140, "CAIDE"] += 2
    df.loc[df["cholesterol_mmol_L"] > 6.5, "CAIDE"] += 1

    # min is twice a week, 20 min at least each time, so in total 40min
    df.loc[df["physical_activity_min"] < 40, "CAIDE"] += 1
    # In this case CAIDE the more the worse

def normalize_data(dataset:pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline for preprocessing the data
    """

    column_trans = ColumnTransformer([
        ("min_max_scaling", MinMaxScaler(), [
            'Age', 'EducationLevel', 
            "health_points", "secondary_indicator",
            "CAIDE", "BMI_cat"
        ]),
        ("remaining", StandardScaler(),[
            "BMI", "AlcoholConsumption", 'PhysicalActivity', 'DietQuality',
            'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 
            'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 
            'MMSE', 'FunctionalAssessment', 'ADL',  'T/HDL'
        ])
    ],
    remainder="passthrough",
    verbose_feature_names_out=False)

    column_trans.set_output(transform="pandas")

    return column_trans.fit_transform(dataset)



def load_and_format_dataset() -> pd.DataFrame:
    """
    Load dataset from kaggle and format columns.
    """
    if not os.path.exists(DATASET_TRAIN_FILE):
        print("[i] Downloading competition data...")
        comp_path = kagglehub.competition_download(DATASET_HANDLE, force_download=True)
        shutil.move(os.path.join(comp_path, "train.csv"), DATASET_TRAIN_FILE)
        shutil.move(os.path.join(comp_path, "test.csv"), DATASET_EVAL_FILE)
    
    csv_data = pd.read_csv(DATASET_TRAIN_FILE)
    format_columns(csv_data)
    csv_data = normalize_data(csv_data)
    return csv_data

def get_dataset_target(dataset:pd.DataFrame) -> pd.Series:
    """
    Get from within the dataset, only the corresponding
    target column
    """
    return dataset[TARGET]

def augment_data(X:pd.DataFrame, y:pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Augment data using SMOTE
    """
    sm = SMOTE(random_state=RANDOM_STATE)
    return sm.fit_resample(X,y)


def get_dataset_raw_columns(dataset:pd.DataFrame) -> pd.DataFrame:
    """
    Get subdataset based on raw columns data
    """
    return dataset[RAW_COLUMNS]

def get_dataset_feature_engineered_columns(dataset:pd.DataFrame) -> pd.DataFrame:
    """
    Get subdataset based on ours feature engineered columns
    """
    return dataset[OURS_COLUMNS]

def get_dataset_CAIDE_columns(dataset:pd.DataFrame) -> pd.DataFrame:
    """
    Get subdataset based on raw-CAIDE columns data
    """
    return dataset[CAIDE_COLUMNS]

