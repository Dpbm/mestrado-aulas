"""
Code used across multiple notebooks.
"""

import os
from pathlib import Path

RANDOM_STATE = 42

DATASET_HANDLE = 'alzheimers-disease-risk-prediction-eu-business'

current_file = Path(__file__).resolve().parent

OUTPUTS_PATH = os.path.join(current_file, 'outputs')
DATASET_PATH = os.path.join(current_file,"dataset") 

# just to ensure that the dataset/ and outputs/ folders will always exist
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(OUTPUTS_PATH, exist_ok=True)

DATASET_TRAIN_FILE = os.path.join(current_file, "dataset", "train.csv")
DATASET_EVAL_FILE = os.path.join(current_file, "dataset", "comp.csv")


PLOTS_PATH = lambda img_file : os.path.join(current_file, "plots", img_file)


TARGET = "Diagnosis"

RAW_COLUMNS = [
    'Age', 'Gender', 'EducationLevel', 'BMI',
    'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality',
    'SleepQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
    'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 'SystolicBP',
    'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
    'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment',
    'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion',
    'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
    'Forgetfulness',
]
OURS_COLUMNS = ["BMI_cat", "T/HDL", "health_points", "secondary_indicator"]
CAIDE_COLUMNS = [
    'CAIDE', 'BMI', 'Smoking', 'AlcoholConsumption', 'DietQuality',
    'SleepQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
    'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 
    'DiastolicBP', 'CholesterolLDL', 'CholesterolHDL',
    'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment',
    'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion',
    'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
    'Forgetfulness',
]
ALL_COLUMNS = [
    'CAIDE', 'BMI_cat', "T/HDL", "health_points", 
    "secondary_indicator", 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
    'Depression', 'HeadInjury', 'Hypertension', 
    'CholesterolLDL', 'MMSE', 'FunctionalAssessment',
    'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion',
    'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
    'Forgetfulness',
]

FEATURES_NAMES = ('raw', 'ours', 'CAIDE', 'all')
FEATURES_COLUMNS = (RAW_COLUMNS, OURS_COLUMNS, CAIDE_COLUMNS, ALL_COLUMNS)
FEATURES_LISTS = list(zip(('raw', 'ours', 'CAIDE', 'all'), (RAW_COLUMNS, OURS_COLUMNS, CAIDE_COLUMNS, ALL_COLUMNS)))